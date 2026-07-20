import Darwin
import Foundation

enum WorkerProcessError: Error, Sendable, Equatable {
    case executableNotFound(String)
    case launchFailed(String)
    case notRunning
    case inputQueueFull
    case eventBufferOverflow
    case standardOutputClosed
    case unresponsive
    case protocolFailure(WorkerProtocolError)
}

enum WorkerProcessEvent: Sendable, Equatable {
    case envelope(WorkerEnvelope)
    case processExited(status: Int32)
}

protocol WorkerProcessClock: Sendable {
    func timer(interval: Duration) -> AsyncStream<Void>
}

struct ContinuousWorkerProcessClock: WorkerProcessClock {
    func timer(interval: Duration) -> AsyncStream<Void> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    while !Task.isCancelled {
                        try await ContinuousClock().sleep(for: interval)
                        guard !Task.isCancelled else { break }
                        continuation.yield(())
                    }
                } catch {}
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private actor WorkerFrameDecoder {
    private var decoder = LengthPrefixedJSONCodec.Decoder()
    func append(_ data: Data) -> [WorkerFrameDecodeOutcome] { decoder.appendOutcomes(data) }
    func finish() throws { try decoder.finish() }
}

private actor BoundedWorkerInputWriter {
    private struct Write: @unchecked Sendable {
        let data: Data
        let continuation: CheckedContinuation<Void, Error>
    }

    private let handle: FileHandle
    private let cancellationReadHandle: FileHandle
    private let cancellationWriteHandle: FileHandle
    private let maximumQueuedBytes: Int
    private var queue: [Write] = []
    private var queuedBytes = 0
    private var consumer: CheckedContinuation<Write?, Never>?
    private var terminalError: WorkerProcessError?
    private var task: Task<Void, Never>?

    init(handle: FileHandle, maximumQueuedBytes: Int) throws {
        self.handle = handle
        self.maximumQueuedBytes = maximumQueuedBytes
        var cancellationDescriptors: [Int32] = [0, 0]
        guard Darwin.pipe(&cancellationDescriptors) == 0 else {
            throw WorkerProcessError.launchFailed(String(cString: strerror(errno)))
        }
        cancellationReadHandle = FileHandle(fileDescriptor: cancellationDescriptors[0], closeOnDealloc: true)
        cancellationWriteHandle = FileHandle(fileDescriptor: cancellationDescriptors[1], closeOnDealloc: true)
        guard fcntl(handle.fileDescriptor, F_SETNOSIGPIPE, 1) == 0,
              Self.setNonBlocking(handle.fileDescriptor),
              Self.setNonBlocking(cancellationReadHandle.fileDescriptor),
              Self.setNonBlocking(cancellationWriteHandle.fileDescriptor) else {
            throw WorkerProcessError.launchFailed(String(cString: strerror(errno)))
        }
    }

    func start() {
        guard task == nil else { return }
        let cancellationDescriptor = cancellationReadHandle.fileDescriptor
        task = Task.detached(priority: .userInitiated) { [weak self, handle, cancellationDescriptor] in
            defer { handle.closeFile() }
            while let write = await self?.next() {
                do {
                    try Self.writeAll(
                        write.data,
                        to: handle.fileDescriptor,
                        cancellationDescriptor: cancellationDescriptor
                    )
                    write.continuation.resume()
                } catch {
                    write.continuation.resume(throwing: WorkerProcessError.notRunning)
                    await self?.close(with: .notRunning)
                    return
                }
            }
        }
    }

    func enqueue(_ data: Data) async throws {
        if let terminalError { throw terminalError }
        guard data.count <= maximumQueuedBytes, queuedBytes + data.count <= maximumQueuedBytes else {
            throw WorkerProcessError.inputQueueFull
        }
        try await withCheckedThrowingContinuation { continuation in
            let write = Write(data: data, continuation: continuation)
            if let consumer {
                self.consumer = nil
                consumer.resume(returning: write)
            } else {
                queue.append(write)
                queuedBytes += data.count
            }
        }
    }

    func close(with error: WorkerProcessError = .notRunning) {
        guard terminalError == nil else { return }
        terminalError = error
        let pending = queue
        queue.removeAll()
        queuedBytes = 0
        pending.forEach { $0.continuation.resume(throwing: error) }
        consumer?.resume(returning: nil)
        consumer = nil
        var byte: UInt8 = 1
        _ = Darwin.write(cancellationWriteHandle.fileDescriptor, &byte, 1)
    }

    func join() async {
        task?.cancel()
        _ = await task?.value
        task = nil
        handle.closeFile()
        cancellationReadHandle.closeFile()
        cancellationWriteHandle.closeFile()
    }

    private func next() async -> Write? {
        if !queue.isEmpty {
            let next = queue.removeFirst()
            queuedBytes -= next.data.count
            return next
        }
        if terminalError != nil { return nil }
        return await withCheckedContinuation { consumer = $0 }
    }

    private nonisolated static func setNonBlocking(_ descriptor: Int32) -> Bool {
        let flags = fcntl(descriptor, F_GETFL)
        return flags >= 0 && fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0
    }

    private nonisolated static func writeAll(
        _ data: Data,
        to descriptor: Int32,
        cancellationDescriptor: Int32
    ) throws {
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var offset = 0
            while offset < rawBuffer.count {
                var descriptors = [
                    pollfd(fd: descriptor, events: Int16(POLLOUT), revents: 0),
                    pollfd(fd: cancellationDescriptor, events: Int16(POLLIN), revents: 0),
                ]
                let pollResult = Darwin.poll(&descriptors, nfds_t(descriptors.count), -1)
                if pollResult < 0 {
                    if errno == EINTR { continue }
                    throw WorkerProcessError.notRunning
                }
                if descriptors[1].revents != 0 { throw WorkerProcessError.notRunning }
                guard descriptors[0].revents & Int16(POLLOUT) != 0 else {
                    throw WorkerProcessError.notRunning
                }
                let written = Darwin.write(descriptor, baseAddress.advanced(by: offset), rawBuffer.count - offset)
                if written > 0 {
                    offset += written
                } else if written < 0, errno == EINTR || errno == EAGAIN {
                    continue
                } else {
                    throw WorkerProcessError.notRunning
                }
            }
        }
    }
}

actor WorkerProcess {
    private static let launcherAcknowledgement: UInt8 = 0x06
    static let heartbeatInterval: Duration = .seconds(5)
    static let missedHeartbeatLimit = 3
    static let maximumPendingInputBytes = 2 * 1_048_576
    static let maximumDiagnosticChunkBytes = 4_096
    static let standardOutputExitGrace: Duration = .milliseconds(100)

    nonisolated private let eventStream: AsyncThrowingStream<WorkerProcessEvent, Error>
    nonisolated private let diagnosticStream: AsyncStream<String>
    private let eventContinuation: AsyncThrowingStream<WorkerProcessEvent, Error>.Continuation
    private let diagnosticContinuation: AsyncStream<String>.Continuation
    private let process = Process()
    private let standardInput = Pipe()
    private let standardOutput = Pipe()
    private let standardError = Pipe()
    private let decoder = WorkerFrameDecoder()
    private let clock: any WorkerProcessClock
    private let launcherHandshakeTimeout: Duration
    private let writer: BoundedWorkerInputWriter

    private var stdoutTask: Task<Void, Never>?
    private var stderrTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var launcherTimeoutTask: Task<Void, Never>?
    private var escalationTask: Task<Void, Never>?
    private var stdoutEOFTask: Task<Void, Never>?
    private var cleanupTask: Task<Void, Never>?
    private var launcherWaiter: CheckedContinuation<Void, Error>?
    private var terminalWaiters: [CheckedContinuation<Int32, Error>] = []
    private var terminalResult: Result<Int32, WorkerProcessError>?
    private var diagnosticBuffer = Data()
    private var isRunning = false
    private var acceptsWrites = false
    private var launcherReady = false
    private var verifiedProcessGroupID: pid_t?
    private var terminationRequested = false
    private var stdoutEnded = false
    private var processExitStatus: Int32?
    private var terminalFailure: WorkerProcessError?
    private var cleanupStarted = false
    private var didEmitTerminal = false
    private var protocolReady = false
    private var heartbeatObservedSinceLastTick = false
    private var missedHeartbeats = 0

    private init(clock: any WorkerProcessClock, launcherHandshakeTimeout: Duration) throws {
        let events = AsyncThrowingStream.makeStream(
            of: WorkerProcessEvent.self,
            bufferingPolicy: .bufferingOldest(256)
        )
        eventStream = events.stream
        eventContinuation = events.continuation
        let diagnostics = AsyncStream.makeStream(
            of: String.self,
            bufferingPolicy: .bufferingOldest(128)
        )
        diagnosticStream = diagnostics.stream
        diagnosticContinuation = diagnostics.continuation
        self.clock = clock
        self.launcherHandshakeTimeout = launcherHandshakeTimeout
        writer = try BoundedWorkerInputWriter(
            handle: standardInput.fileHandleForWriting,
            maximumQueuedBytes: Self.maximumPendingInputBytes
        )
    }

    static func start(
        executable: URL,
        arguments: [String] = [],
        environment: [String: String] = [:],
        launcherExecutable: URL? = nil,
        launcherHandshakeTimeout: Duration = .seconds(5),
        clock: any WorkerProcessClock = ContinuousWorkerProcessClock()
    ) async throws -> WorkerProcess {
        try validateExecutable(executable)
        let resolvedLauncher = try launcherExecutable ?? bundledLauncherExecutable()
        try validateExecutable(resolvedLauncher)
        let worker = try WorkerProcess(clock: clock, launcherHandshakeTimeout: launcherHandshakeTimeout)
        do {
            try await worker.launch(
                executable: executable,
                arguments: arguments,
                environment: environment,
                launcherExecutable: resolvedLauncher
            )
            try await worker.waitForLauncher()
            return worker
        } catch {
            await worker.abortLaunch()
            throw error
        }
    }

    private static func validateExecutable(_ executable: URL) throws {
        guard executable.isFileURL,
              executable.path.hasPrefix("/"),
              FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw WorkerProcessError.executableNotFound(executable.path)
        }
    }

    private static func bundledLauncherExecutable() throws -> URL {
        let executable = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers/CloudPointWorkerLauncher", isDirectory: false)
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw WorkerProcessError.executableNotFound("CloudPointWorkerLauncher")
        }
        return executable
    }

    nonisolated func events() -> AsyncThrowingStream<WorkerProcessEvent, Error> { eventStream }
    nonisolated func diagnostics() -> AsyncStream<String> { diagnosticStream }

    var processIdentifier: pid_t { process.processIdentifier }
    var processGroupIdentifier: pid_t? {
        verifiedProcessGroupID
    }

    func send(_ envelope: WorkerEnvelope) async throws {
        guard acceptsWrites, isRunning, process.isRunning else { throw WorkerProcessError.notRunning }
        let frame: Data
        do { frame = try LengthPrefixedJSONCodec.encode(envelope) }
        catch let error as WorkerProtocolError { throw WorkerProcessError.protocolFailure(error) }
        try await writer.enqueue(frame)
    }

    func markProtocolReady() {
        guard isRunning,
              launcherReady,
              !protocolReady,
              !cleanupStarted,
              !terminationRequested,
              terminalFailure == nil,
              processExitStatus == nil else {
            return
        }
        protocolReady = true
        heartbeatObservedSinceLastTick = false
        missedHeartbeats = 0
        startHeartbeatSupervision()
    }

#if DEBUG
    var missedHeartbeatCountForTesting: Int { missedHeartbeats }
    var heartbeatObservedSinceLastTickForTesting: Bool { heartbeatObservedSinceLastTick }
#endif

    func terminate() async {
        guard isRunning, !cleanupStarted, !terminationRequested else { return }
        terminationRequested = true
        acceptsWrites = false
        heartbeatTask?.cancel()
        await writer.close()
        signalVerifiedProcessGroup(SIGTERM)
        scheduleEscalation()
    }

    func shutdown() async { await terminate() }

    func waitForTermination() async throws -> Int32 {
        if let terminalResult { return try terminalResult.get() }
        return try await withCheckedThrowingContinuation { terminalWaiters.append($0) }
    }

    private func launch(
        executable: URL,
        arguments: [String],
        environment: [String: String],
        launcherExecutable: URL
    ) async throws {
        process.executableURL = launcherExecutable
        process.arguments = [executable.path] + arguments
        process.environment = environment
        process.standardInput = standardInput
        process.standardOutput = standardOutput
        process.standardError = standardError
        process.terminationHandler = { [weak self] process in
            let status = process.terminationStatus
            Task { await self?.processTerminated(status: status) }
        }

        do { try process.run() }
        catch { throw WorkerProcessError.launchFailed(error.localizedDescription) }

        isRunning = true
        standardInput.fileHandleForReading.closeFile()
        standardOutput.fileHandleForWriting.closeFile()
        standardError.fileHandleForWriting.closeFile()
        startReaders()
        startLauncherTimeout()
    }

    private func startReaders() {
        let stdout = standardOutput.fileHandleForReading
        stdoutTask = Task.detached(priority: .userInitiated) { [weak self, stdout] in
            while !Task.isCancelled {
                let data = stdout.availableData
                if data.isEmpty {
                    await self?.stdoutDidEnd()
                    return
                }
                await self?.receiveStandardOutput(data)
            }
        }
        let stderr = standardError.fileHandleForReading
        stderrTask = Task.detached(priority: .utility) { [weak self, stderr] in
            while !Task.isCancelled {
                let data = stderr.availableData
                if data.isEmpty {
                    await self?.stderrDidEnd()
                    return
                }
                await self?.receiveStandardError(data)
            }
        }
    }

    private func waitForLauncher() async throws {
        if launcherReady { return }
        if let terminalFailure { throw terminalFailure }
        if let processExitStatus { throw WorkerProcessError.launchFailed("launcher exited with status \(processExitStatus)") }
        try await withCheckedThrowingContinuation { launcherWaiter = $0 }
    }

    private func receiveStandardOutput(_ data: Data) async {
        guard !cleanupStarted else { return }
        for outcome in await decoder.append(data) {
            switch outcome {
            case let .envelope(envelope):
                if case .heartbeat? = envelope.event { heartbeatArrived() }
                switch eventContinuation.yield(.envelope(envelope)) {
                case .enqueued: break
                case .dropped, .terminated: await initiateFailure(.eventBufferOverflow)
                @unknown default: await initiateFailure(.eventBufferOverflow)
                }
            case let .failure(error, _):
                await initiateFailure(.protocolFailure(error))
                return
            }
        }
    }

    private func stdoutDidEnd() async {
        guard !stdoutEnded else { return }
        let protocolFailure: WorkerProcessError?
        do {
            try await decoder.finish()
            protocolFailure = nil
        } catch let error as WorkerProtocolError {
            protocolFailure = .protocolFailure(error)
        } catch {
            protocolFailure = .protocolFailure(.malformedEnvelope)
        }
        stdoutEnded = true
        if terminalFailure == nil, let protocolFailure {
            await initiateFailure(protocolFailure)
            return
        }
        if terminalFailure == nil, processExitStatus == nil {
            scheduleStandardOutputClosureCheck()
        }
        await beginCleanupIfReady()
    }

    private func receiveStandardError(_ data: Data) async {
        for byte in data {
            if byte == 0x0A {
                await emitDiagnosticBuffer()
            } else {
                diagnosticBuffer.append(byte)
                if diagnosticBuffer.count == Self.maximumDiagnosticChunkBytes {
                    await emitDiagnosticBuffer()
                }
            }
        }
    }

    private func emitDiagnosticBuffer() async {
        guard !diagnosticBuffer.isEmpty else { return }
        let line = String(decoding: diagnosticBuffer, as: UTF8.self)
        diagnosticBuffer.removeAll(keepingCapacity: true)
        if line.hasPrefix("CLOUDPOINT_LAUNCHER_READY:") {
            guard let pid = pid_t(line.dropFirst("CLOUDPOINT_LAUNCHER_READY:".count)),
                  pid == process.processIdentifier,
                  getpgid(pid) == pid,
                  !launcherReady else {
                await initiateFailure(.launchFailed("launcher did not establish the worker process group"))
                return
            }
            verifiedProcessGroupID = pid
            launcherTimeoutTask?.cancel()
            do { try acknowledgeLauncher() }
            catch {
                await initiateFailure(.launchFailed("launcher acknowledgement failed"))
                return
            }
            launcherReady = true
            await writer.start()
            if isRunning, terminalFailure == nil {
                acceptsWrites = true
            }
            launcherWaiter?.resume()
            launcherWaiter = nil
        } else {
            _ = diagnosticContinuation.yield(line)
        }
    }

    private func stderrDidEnd() async { await emitDiagnosticBuffer() }

    private func processTerminated(status: Int32) async {
        guard processExitStatus == nil else { return }
        processExitStatus = status
        isRunning = false
        acceptsWrites = false
        heartbeatTask?.cancel()
        launcherTimeoutTask?.cancel()
        stdoutEOFTask?.cancel()
        stdoutEOFTask = nil
        signalProcessTreeForCleanup(SIGKILL)
        if !launcherReady {
            launcherWaiter?.resume(throwing: WorkerProcessError.launchFailed("launcher exited with status \(status)"))
            launcherWaiter = nil
        }
        await writer.close()
        await beginCleanupIfReady()
    }

    private func startHeartbeatSupervision() {
        guard protocolReady,
              heartbeatTask == nil,
              isRunning,
              !cleanupStarted,
              !terminationRequested,
              terminalFailure == nil,
              processExitStatus == nil else {
            return
        }
        let ticks = clock.timer(interval: Self.heartbeatInterval)
        heartbeatTask = Task { [weak self] in
            for await _ in ticks {
                guard !Task.isCancelled else { break }
                await self?.heartbeatTick()
            }
        }
    }

    private func heartbeatArrived() {
        guard isRunning,
              !cleanupStarted,
              !terminationRequested,
              terminalFailure == nil,
              processExitStatus == nil else {
            return
        }
        let supervisionWasAlreadyArmed = protocolReady
        protocolReady = true
        missedHeartbeats = 0
        heartbeatObservedSinceLastTick = supervisionWasAlreadyArmed
        startHeartbeatSupervision()
    }

    private func startLauncherTimeout() {
        let ticks = clock.timer(interval: launcherHandshakeTimeout)
        launcherTimeoutTask = Task { [weak self] in
            for await _ in ticks {
                guard !Task.isCancelled else { return }
                await self?.launcherHandshakeTimedOut()
                return
            }
        }
    }

    private func launcherHandshakeTimedOut() async {
        guard !launcherReady, processExitStatus == nil, terminalFailure == nil else { return }
        await initiateFailure(.launchFailed("launcher handshake timed out"))
    }

    private func heartbeatTick() async {
        guard protocolReady, isRunning, !cleanupStarted, !terminationRequested else { return }
        if heartbeatObservedSinceLastTick {
            heartbeatObservedSinceLastTick = false
            missedHeartbeats = 0
            return
        }
        missedHeartbeats += 1
        if missedHeartbeats >= Self.missedHeartbeatLimit { await initiateFailure(.unresponsive) }
    }

    private func scheduleStandardOutputClosureCheck() {
        guard stdoutEOFTask == nil else { return }
        stdoutEOFTask = Task { [weak self] in
            do {
                try await ContinuousClock().sleep(for: Self.standardOutputExitGrace)
            } catch {
                return
            }
            await self?.standardOutputExitGraceElapsed()
        }
    }

    private func standardOutputExitGraceElapsed() async {
        stdoutEOFTask = nil
        guard stdoutEnded,
              processExitStatus == nil,
              terminalFailure == nil,
              isRunning,
              !cleanupStarted,
              !terminationRequested,
              process.isRunning else {
            return
        }
        await initiateFailure(.standardOutputClosed)
    }

    private func initiateFailure(_ error: WorkerProcessError) async {
        guard terminalFailure == nil, !cleanupStarted else { return }
        terminalFailure = error
        acceptsWrites = false
        stdoutEOFTask?.cancel()
        stdoutEOFTask = nil
        if !launcherReady {
            launcherWaiter?.resume(throwing: error)
            launcherWaiter = nil
        }
        await writer.close()
        signalProcessTreeForCleanup(SIGKILL)
        await beginCleanupIfReady()
    }

    private func scheduleEscalation() {
        guard escalationTask == nil else { return }
        let ticks = clock.timer(interval: .seconds(2))
        escalationTask = Task { [weak self] in
            for await _ in ticks {
                guard !Task.isCancelled else { return }
                await self?.escalateIfNeeded()
                return
            }
        }
    }

    private func escalateIfNeeded() {
        guard isRunning, !cleanupStarted else { return }
        signalVerifiedProcessGroup(SIGKILL)
    }

    private func signalVerifiedProcessGroup(_ signal: Int32) {
        guard let processGroupID = verifiedProcessGroupID, processGroupID > 0 else { return }
        _ = Darwin.kill(-processGroupID, signal)
    }

    private func signalProcessTreeForCleanup(_ signal: Int32) {
        if verifiedProcessGroupID != nil {
            signalVerifiedProcessGroup(signal)
            return
        }
        let pid = process.processIdentifier
        guard pid > 0 else { return }
        _ = Darwin.kill(-pid, signal)
        _ = Darwin.kill(pid, signal)
    }

    private func acknowledgeLauncher() throws {
        var acknowledgement = Self.launcherAcknowledgement
        while true {
            let written = Darwin.write(standardInput.fileHandleForWriting.fileDescriptor, &acknowledgement, 1)
            if written == 1 { return }
            if written < 0, errno == EINTR { continue }
            throw WorkerProcessError.launchFailed(String(cString: strerror(errno)))
        }
    }

    private func beginCleanupIfReady() async {
        guard processExitStatus != nil, stdoutEnded, !cleanupStarted else { return }
        cleanupStarted = true
        heartbeatTask?.cancel()
        launcherTimeoutTask?.cancel()
        escalationTask?.cancel()
        stdoutEOFTask?.cancel()
        stdoutEOFTask = nil
        await writer.close()
        standardOutput.fileHandleForReading.closeFile()
        standardError.fileHandleForReading.closeFile()
        stdoutTask?.cancel()
        stderrTask?.cancel()
        let stdoutTask = self.stdoutTask
        let stderrTask = self.stderrTask
        let heartbeatTask = self.heartbeatTask
        let launcherTimeoutTask = self.launcherTimeoutTask
        let escalationTask = self.escalationTask
        cleanupTask = Task { [weak self] in
            _ = await stdoutTask?.value
            _ = await stderrTask?.value
            _ = await heartbeatTask?.value
            _ = await launcherTimeoutTask?.value
            _ = await escalationTask?.value
            await self?.writer.join()
            await self?.completeTerminal()
        }
    }

    private func completeTerminal() {
        guard !didEmitTerminal else { return }
        didEmitTerminal = true
        process.terminationHandler = nil
        stdoutTask = nil
        stderrTask = nil
        heartbeatTask = nil
        launcherTimeoutTask = nil
        escalationTask = nil
        cleanupTask = nil
        if !diagnosticBuffer.isEmpty {
            _ = diagnosticContinuation.yield(String(decoding: diagnosticBuffer, as: UTF8.self))
        }
        diagnosticBuffer.removeAll()
        diagnosticContinuation.finish()
        if let terminalFailure {
            terminalResult = .failure(terminalFailure)
            terminalWaiters.forEach { $0.resume(throwing: terminalFailure) }
            terminalWaiters.removeAll()
            eventContinuation.finish(throwing: terminalFailure)
        }
        else {
            let status = processExitStatus ?? -1
            switch eventContinuation.yield(.processExited(status: status)) {
            case .enqueued:
                terminalResult = .success(status)
                terminalWaiters.forEach { $0.resume(returning: status) }
                terminalWaiters.removeAll()
                eventContinuation.finish()
            case .dropped, .terminated:
                terminalResult = .failure(.eventBufferOverflow)
                terminalWaiters.forEach { $0.resume(throwing: WorkerProcessError.eventBufferOverflow) }
                terminalWaiters.removeAll()
                eventContinuation.finish(throwing: WorkerProcessError.eventBufferOverflow)
            @unknown default:
                terminalResult = .failure(.eventBufferOverflow)
                terminalWaiters.forEach { $0.resume(throwing: WorkerProcessError.eventBufferOverflow) }
                terminalWaiters.removeAll()
                eventContinuation.finish(throwing: WorkerProcessError.eventBufferOverflow)
            }
        }
    }

    private func abortLaunch() async {
        acceptsWrites = false
        isRunning = false
        launcherWaiter?.resume(throwing: WorkerProcessError.launchFailed("launch aborted"))
        launcherWaiter = nil
        signalProcessTreeForCleanup(SIGKILL)
        await writer.close()
        standardOutput.fileHandleForReading.closeFile()
        standardError.fileHandleForReading.closeFile()
        stdoutTask?.cancel()
        stderrTask?.cancel()
        heartbeatTask?.cancel()
        launcherTimeoutTask?.cancel()
        escalationTask?.cancel()
        stdoutEOFTask?.cancel()
        stdoutEOFTask = nil
        _ = await stdoutTask?.value
        _ = await stderrTask?.value
        _ = await heartbeatTask?.value
        _ = await launcherTimeoutTask?.value
        _ = await escalationTask?.value
        await writer.join()
        eventContinuation.finish()
        diagnosticContinuation.finish()
    }

    deinit {
        stdoutTask?.cancel()
        stderrTask?.cancel()
        heartbeatTask?.cancel()
        launcherTimeoutTask?.cancel()
        escalationTask?.cancel()
        stdoutEOFTask?.cancel()
        cleanupTask?.cancel()
        if !didEmitTerminal {
            if let processGroupID = verifiedProcessGroupID {
                _ = Darwin.kill(-processGroupID, SIGKILL)
            } else {
                let pid = process.processIdentifier
                if pid > 0 {
                    _ = Darwin.kill(-pid, SIGKILL)
                    _ = Darwin.kill(pid, SIGKILL)
                }
            }
        }
        standardInput.fileHandleForWriting.closeFile()
        standardOutput.fileHandleForReading.closeFile()
        standardError.fileHandleForReading.closeFile()
        eventContinuation.finish()
        diagnosticContinuation.finish()
    }
}

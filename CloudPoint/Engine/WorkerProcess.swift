import Darwin
import Foundation

enum WorkerProcessError: Error, Sendable, Equatable {
    case executableNotFound(String)
    case launchFailed(String)
    case notRunning
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

    func append(_ data: Data) throws -> [WorkerEnvelope] {
        try decoder.append(data)
    }

    func finish() throws {
        try decoder.finish()
    }
}

actor WorkerProcess {
    static let heartbeatInterval: Duration = .seconds(5)
    static let missedHeartbeatLimit = 3

    nonisolated private let eventStream: AsyncThrowingStream<WorkerProcessEvent, Error>
    nonisolated private let diagnosticStream: AsyncStream<String>
    private let eventContinuation: AsyncThrowingStream<WorkerProcessEvent, Error>.Continuation
    private let diagnosticContinuation: AsyncStream<String>.Continuation
    private let process: Process
    private let standardInput: Pipe
    private let standardOutput: Pipe
    private let standardError: Pipe
    private let decoder = WorkerFrameDecoder()
    private let clock: any WorkerProcessClock
    private var heartbeatTask: Task<Void, Never>?
    private var escalationTask: Task<Void, Never>?
    private var diagnosticBuffer = Data()
    private var isRunning = false
    private var didEmitTerminal = false
    private var missedHeartbeats = 0
    private var processGroupID: pid_t?

    private init(clock: any WorkerProcessClock) {
        let events = AsyncThrowingStream.makeStream(of: WorkerProcessEvent.self)
        eventStream = events.stream
        eventContinuation = events.continuation
        let diagnostics = AsyncStream.makeStream(of: String.self)
        diagnosticStream = diagnostics.stream
        diagnosticContinuation = diagnostics.continuation
        process = Process()
        standardInput = Pipe()
        standardOutput = Pipe()
        standardError = Pipe()
        self.clock = clock
    }

    static func start(
        executable: URL,
        arguments: [String] = [],
        environment: [String: String] = [:],
        clock: any WorkerProcessClock = ContinuousWorkerProcessClock()
    ) async throws -> WorkerProcess {
        guard executable.isFileURL,
              executable.path.hasPrefix("/"),
              FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw WorkerProcessError.executableNotFound(executable.path)
        }
        let worker = WorkerProcess(clock: clock)
        try await worker.launch(executable: executable, arguments: arguments, environment: environment)
        return worker
    }

    nonisolated func events() -> AsyncThrowingStream<WorkerProcessEvent, Error> { eventStream }
    nonisolated func diagnostics() -> AsyncStream<String> { diagnosticStream }

    func send(_ envelope: WorkerEnvelope) throws {
        guard isRunning, process.isRunning else { throw WorkerProcessError.notRunning }
        do {
            try standardInput.fileHandleForWriting.write(contentsOf: LengthPrefixedJSONCodec.encode(envelope))
        } catch let error as WorkerProtocolError {
            throw WorkerProcessError.protocolFailure(error)
        } catch {
            throw WorkerProcessError.notRunning
        }
    }

    func terminate() {
        guard isRunning else { return }
        signalProcessGroup(SIGTERM)
        scheduleEscalation()
    }

    func shutdown() {
        terminate()
    }

    private func launch(executable: URL, arguments: [String], environment: [String: String]) throws {
        process.executableURL = executable
        process.arguments = arguments
        process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, supplied in supplied }
        process.standardInput = standardInput
        process.standardOutput = standardOutput
        process.standardError = standardError

        standardOutput.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            Task { await self?.receiveStandardOutput(data) }
        }
        standardError.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            Task { await self?.receiveStandardError(data) }
        }
        process.terminationHandler = { [weak self] process in
            let status = process.terminationStatus
            Task { await self?.processTerminated(status: status) }
        }

        do {
            try process.run()
        } catch {
            closeResources()
            throw WorkerProcessError.launchFailed(error.localizedDescription)
        }
        isRunning = true
        let pid = process.processIdentifier
        if setpgid(pid, pid) == 0 || getpgid(pid) == pid { processGroupID = pid }
        standardInput.fileHandleForReading.closeFile()
        standardOutput.fileHandleForWriting.closeFile()
        standardError.fileHandleForWriting.closeFile()
        startHeartbeatSupervision()
    }

    private func receiveStandardOutput(_ data: Data) async {
        guard !didEmitTerminal else { return }
        guard !data.isEmpty else { return }
        do {
            for envelope in try await decoder.append(data) {
                if case .heartbeat? = envelope.event { missedHeartbeats = 0 }
                eventContinuation.yield(.envelope(envelope))
            }
        } catch let error as WorkerProtocolError {
            fail(.protocolFailure(error))
        } catch {
            fail(.protocolFailure(.malformedEnvelope))
        }
    }

    private func receiveStandardError(_ data: Data) {
        guard !didEmitTerminal else { return }
        guard !data.isEmpty else { return }
        diagnosticBuffer.append(data)
        while let newline = diagnosticBuffer.firstIndex(of: 0x0A) {
            let lineData = diagnosticBuffer[..<newline]
            diagnosticContinuation.yield(String(decoding: lineData, as: UTF8.self))
            diagnosticBuffer = Data(diagnosticBuffer[diagnosticBuffer.index(after: newline)...])
        }
    }

    private func processTerminated(status: Int32) async {
        guard !didEmitTerminal else { return }
        do { try await decoder.finish() }
        catch let error as WorkerProtocolError {
            fail(.protocolFailure(error))
            return
        } catch {
            fail(.protocolFailure(.malformedEnvelope))
            return
        }

        didEmitTerminal = true
        isRunning = false
        cancelTasks()
        flushDiagnostics()
        closeResources()
        eventContinuation.yield(.processExited(status: status))
        eventContinuation.finish()
        diagnosticContinuation.finish()
    }

    private func startHeartbeatSupervision() {
        let ticks = clock.timer(interval: Self.heartbeatInterval)
        heartbeatTask = Task { [weak self] in
            for await _ in ticks {
                guard !Task.isCancelled else { break }
                await self?.heartbeatTick()
            }
        }
    }

    private func heartbeatTick() {
        guard isRunning, !didEmitTerminal else { return }
        missedHeartbeats += 1
        if missedHeartbeats >= Self.missedHeartbeatLimit { fail(.unresponsive) }
    }

    private func fail(_ error: WorkerProcessError) {
        guard !didEmitTerminal else { return }
        didEmitTerminal = true
        isRunning = false
        cancelTasks()
        signalProcessGroup(SIGKILL)
        flushDiagnostics()
        closeResources()
        eventContinuation.finish(throwing: error)
        diagnosticContinuation.finish()
    }

    private func scheduleEscalation() {
        escalationTask?.cancel()
        escalationTask = Task { [weak self] in
            do { try await ContinuousClock().sleep(for: .seconds(2)) } catch { return }
            guard !Task.isCancelled else { return }
            await self?.escalateIfNeeded()
        }
    }

    private func escalateIfNeeded() {
        guard isRunning, !didEmitTerminal else { return }
        signalProcessGroup(SIGKILL)
    }

    private func signalProcessGroup(_ signal: Int32) {
        if let processGroupID { _ = Darwin.kill(-processGroupID, signal) }
        else if process.isRunning { _ = Darwin.kill(process.processIdentifier, signal) }
    }

    private func cancelTasks() {
        heartbeatTask?.cancel()
        heartbeatTask = nil
        escalationTask?.cancel()
        escalationTask = nil
    }

    private func flushDiagnostics() {
        if !diagnosticBuffer.isEmpty {
            diagnosticContinuation.yield(String(decoding: diagnosticBuffer, as: UTF8.self))
            diagnosticBuffer.removeAll()
        }
    }

    private func closeResources() {
        standardOutput.fileHandleForReading.readabilityHandler = nil
        standardError.fileHandleForReading.readabilityHandler = nil
        standardInput.fileHandleForWriting.closeFile()
        standardOutput.fileHandleForReading.closeFile()
        standardError.fileHandleForReading.closeFile()
        process.terminationHandler = nil
    }

    deinit {
        heartbeatTask?.cancel()
        escalationTask?.cancel()
        if process.isRunning {
            if let processGroupID { _ = Darwin.kill(-processGroupID, SIGKILL) }
            else { _ = Darwin.kill(process.processIdentifier, SIGKILL) }
        }
        standardOutput.fileHandleForReading.readabilityHandler = nil
        standardError.fileHandleForReading.readabilityHandler = nil
        standardInput.fileHandleForWriting.closeFile()
        standardOutput.fileHandleForReading.closeFile()
        standardError.fileHandleForReading.closeFile()
        eventContinuation.finish()
        diagnosticContinuation.finish()
    }
}

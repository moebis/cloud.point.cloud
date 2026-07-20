import Foundation

protocol WorkerTransport: Sendable {
    nonisolated func events() -> AsyncThrowingStream<WorkerProcessEvent, Error>
    func send(_ envelope: WorkerEnvelope) async throws
    func markProtocolReady() async
    func terminate() async
}

extension WorkerProcess: WorkerTransport {}

protocol PythonMLXEngineClock: Sendable {
    func sleep(for duration: Duration) async throws
}

struct ContinuousPythonMLXEngineClock: PythonMLXEngineClock {
    func sleep(for duration: Duration) async throws {
        try await ContinuousClock().sleep(for: duration)
    }
}

struct PythonMLXEngineTimeouts: Sendable, Equatable {
    var commandAcknowledgement: Duration
    var ready: Duration
    var shutdownGrace: Duration

    init(
        commandAcknowledgement: Duration = .seconds(15),
        ready: Duration = .seconds(600),
        shutdownGrace: Duration = .seconds(60)
    ) {
        self.commandAcknowledgement = commandAcknowledgement
        self.ready = ready
        self.shutdownGrace = shutdownGrace
    }
}

enum PythonMLXEngineError: Error, Sendable, Equatable {
    case commandTimedOut(String)
    case readyTimedOut
    case shutdownTimedOut
    case shutdownEndedBeforeCancelled
    case shutdownEndedBeforeExit
    case shutdownExited(status: Int32)
}

struct PythonMLXEngineFactory: ReconstructionEngineFactory, Sendable {
    let runtime: WorkerRuntime
    private let processStarter: PythonMLXEngine.ProcessStarter

    init(
        runtime: WorkerRuntime,
        processStarter: @escaping PythonMLXEngine.ProcessStarter = PythonMLXEngine.liveStarter
    ) {
        self.runtime = runtime
        self.processStarter = processStarter
    }

    func makeEngine(modelDirectory: URL) throws -> any ReconstructionEngine {
        guard modelDirectory.isFileURL, modelDirectory.path.hasPrefix("/") else {
            throw WorkerRuntimeError.runtimeMustBeAbsolute(
                modelDirectory.isFileURL ? modelDirectory.path : modelDirectory.absoluteString
            )
        }
        return PythonMLXEngine(
            runtime: runtime,
            modelDirectory: modelDirectory,
            processStarter: processStarter
        )
    }
}

actor PythonMLXEngine: ReconstructionEngine {
    typealias ProcessStarter = @Sendable (
        _ executable: URL,
        _ arguments: [String],
        _ environment: [String: String]
    ) async throws -> any WorkerTransport

    private enum Lifecycle: Equatable {
        case idle
        case prepared
        case starting(UUID)
        case running
        case shuttingDown
        case shutdown
    }

    private struct PendingCommand {
        let name: String
        let continuation: CheckedContinuation<Void, Error>
        let timeoutTask: Task<Void, Never>
    }

    private struct PendingWait {
        let continuation: CheckedContinuation<Void, Error>
        let timeoutTask: Task<Void, Never>
    }

    static let liveStarter: ProcessStarter = { executable, arguments, environment in
        try await WorkerProcess.start(
            executable: executable,
            arguments: arguments,
            environment: environment
        )
    }

    nonisolated private let eventStream: AsyncThrowingStream<EngineEvent, Error>
    private let eventContinuation: AsyncThrowingStream<EngineEvent, Error>.Continuation
    private let runtime: WorkerRuntime
    private let modelDirectory: URL
    private let processStarter: ProcessStarter
    private let clock: any PythonMLXEngineClock
    private let timeouts: PythonMLXEngineTimeouts

    private var lifecycle = Lifecycle.idle
    private var configuration: EngineConfiguration?
    private var projectID: UUID?
    private var startupTask: Task<any WorkerTransport, Error>?
    private var startupWaiter: CheckedContinuation<any WorkerTransport, Error>?
    private var transport: (any WorkerTransport)?
    private var transportTask: Task<Void, Never>?
    private var pendingCommands: [UUID: PendingCommand] = [:]
    private var readyWaiter: PendingWait?
    private var readyReceived = false
    private var beginInProgress = false
    private var sessionIsActive = false
    private var shutdownRequiresCancelled = false
    private var shutdownReceivedCancelled = false
    private var shutdownStreamEnded = false
    private var shutdownExitStatus: Int32?
    private var shutdownTerminalError: Error?
    private var gracefulShutdownWaiter: PendingWait?
    private var shutdownCompletionWaiters: [CheckedContinuation<Void, Never>] = []
    private var streamFinished = false

    init(
        runtime: WorkerRuntime,
        modelDirectory: URL,
        processStarter: @escaping ProcessStarter = PythonMLXEngine.liveStarter,
        clock: any PythonMLXEngineClock = ContinuousPythonMLXEngineClock(),
        timeouts: PythonMLXEngineTimeouts = PythonMLXEngineTimeouts()
    ) {
        let stream = AsyncThrowingStream.makeStream(
            of: EngineEvent.self,
            bufferingPolicy: .bufferingOldest(256)
        )
        eventStream = stream.stream
        eventContinuation = stream.continuation
        self.runtime = runtime
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.processStarter = processStarter
        self.clock = clock
        self.timeouts = timeouts
    }

    deinit {
        startupTask?.cancel()
        transportTask?.cancel()
        pendingCommands.values.forEach { $0.timeoutTask.cancel() }
        readyWaiter?.timeoutTask.cancel()
        gracefulShutdownWaiter?.timeoutTask.cancel()
        eventContinuation.finish()
    }

    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> {
        eventStream
    }

    func prepare(configuration: EngineConfiguration) async throws {
        guard lifecycle == .idle else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "prepare")
        }
        try configuration.validate()
        self.configuration = configuration
        lifecycle = .prepared
    }

    func begin(project: ProjectDescriptor) async throws {
        guard lifecycle == .prepared, let configuration else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "begin")
        }
        let launch = try WorkerLaunch(
            runtime: runtime,
            project: project.packageURL,
            model: modelDirectory
        )
        let generation = UUID()
        lifecycle = .starting(generation)
        beginInProgress = true
        projectID = project.projectID
        readyReceived = false
        sessionIsActive = false

        let task = Task { [processStarter] in
            try await processStarter(
                launch.executable,
                launch.arguments,
                Self.workerEnvironment()
            )
        }
        startupTask = task
        Task { [weak self] in
            let result = await task.result
            if let self {
                await self.startupCompleted(result, generation: generation)
            } else if case .success(let transport) = result {
                await transport.terminate()
            }
        }

        do {
            let started = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    startupWaiter = continuation
                }
            } onCancel: {
                Task { [weak self] in
                    await self?.interruptStartup(generation: generation)
                }
            }
            guard case .starting(let activeGeneration) = lifecycle,
                  activeGeneration == generation else {
                await started.terminate()
                throw CancellationError()
            }
            startupTask = nil
            transport = started
            startTransportEvents(started)
            lifecycle = .running

            try await sendCommand(.hello(
                clientVersion: "CloudPoint/1.0",
                supportedProtocolVersions: [UInt32(WorkerEnvelope.currentProtocolVersion)]
            ))
            try ensureBeginIsActive()
            try await waitUntilReady()
            try ensureBeginIsActive()
            try await sendCommand(.configure(configuration))
            try ensureBeginIsActive()
            try await sendCommand(.beginSession(
                resumeCheckpoint: project.resumeCheckpoint
            ))
            try ensureBeginIsActive()
            beginInProgress = false
        } catch {
            beginInProgress = false
            switch lifecycle {
            case .starting, .running:
                await stopTransport(finishingWith: error)
            case .idle, .prepared, .shuttingDown, .shutdown:
                break
            }
            throw error
        }
    }

    func enqueue(_ frame: PersistedFrame) async throws {
        try await sendCommand(.enqueueFrame(
            frameIndex: frame.index,
            sourceTimestamp: frame.sourceTimestamp,
            relativePath: frame.relativePath
        ))
    }

    func finishInput() async throws {
        try await sendCommand(.finishInput)
    }

    func pause() async throws {
        try await sendCommand(.pause)
    }

    func resume() async throws {
        try await sendCommand(.resume)
    }

    func cancel() async {
        switch lifecycle {
        case .starting(let generation):
            interruptStartup(generation: generation)
        case .running:
            if beginInProgress {
                await stopTransport(finishingWith: CancellationError())
            } else {
                _ = try? await sendCommand(.cancel)
            }
        case .idle, .prepared, .shuttingDown, .shutdown:
            return
        }
    }

    func shutdown() async {
        switch lifecycle {
        case .shutdown:
            return
        case .shuttingDown:
            await withCheckedContinuation { shutdownCompletionWaiters.append($0) }
            return
        case .starting(let generation):
            interruptStartup(generation: generation)
            return
        case .idle, .prepared:
            lifecycle = .shutdown
            finishStream(nil)
            return
        case .running:
            if beginInProgress {
                await stopTransport(finishingWith: CancellationError())
                lifecycle = .shutdown
                return
            }
            break
        }

        lifecycle = .shuttingDown
        shutdownRequiresCancelled = sessionIsActive
        shutdownReceivedCancelled = !sessionIsActive
        shutdownStreamEnded = false
        shutdownExitStatus = nil
        shutdownTerminalError = nil

        var terminalError: Error?
        do {
            try await sendCommand(.shutdown)
            try await waitForGracefulShutdown()
        } catch {
            terminalError = error
            await stopTransport(finishingWith: error)
        }
        completeShutdown(finishingWith: terminalError)
    }

    private static func workerEnvironment() -> [String: String] {
        [
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            "TMPDIR": NSTemporaryDirectory(),
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
            "LANG": "C",
        ]
    }

    private func startupCompleted(
        _ result: Result<any WorkerTransport, Error>,
        generation: UUID
    ) async {
        guard case .starting(let activeGeneration) = lifecycle,
              activeGeneration == generation,
              let waiter = startupWaiter else {
            if case .success(let lateTransport) = result {
                await lateTransport.terminate()
            }
            return
        }
        startupTask = nil
        startupWaiter = nil
        waiter.resume(with: result)
    }

    private func interruptStartup(generation: UUID) {
        guard case .starting(let activeGeneration) = lifecycle,
              activeGeneration == generation else { return }
        lifecycle = .shutdown
        beginInProgress = false
        startupTask?.cancel()
        startupTask = nil
        let waiter = startupWaiter
        startupWaiter = nil
        waiter?.resume(throwing: CancellationError())
        finishStream(nil)
    }

    private func ensureBeginIsActive() throws {
        try Task.checkCancellation()
        guard lifecycle == .running, beginInProgress else {
            throw CancellationError()
        }
    }

    private func sendCommand(_ command: WorkerCommand) async throws {
        let name = Self.commandName(command)
        let lifecycleAllowsCommand = lifecycle == .running
            || (lifecycle == .shuttingDown && name == "shutdown")
        guard lifecycleAllowsCommand,
              let transport,
              let projectID else {
            throw ReconstructionEngineError.invalidLifecycle(
                operation: name
            )
        }
        try Task.checkCancellation()
        let identifier = UUID()
        let envelope = WorkerEnvelope.command(
            command,
            id: identifier,
            projectId: projectID
        )
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let timeoutTask = Task { [weak self, clock, timeouts] in
                    do {
                        try await clock.sleep(for: timeouts.commandAcknowledgement)
                        try Task.checkCancellation()
                    } catch {
                        return
                    }
                    await self?.commandTimedOut(identifier, name: name)
                }
                pendingCommands[identifier] = PendingCommand(
                    name: name,
                    continuation: continuation,
                    timeoutTask: timeoutTask
                )
                Task { [weak self, transport] in
                    do {
                        try await transport.send(envelope)
                    } catch {
                        await self?.commandSendFailed(identifier, error: error)
                    }
                }
            }
        } onCancel: {
            Task { [weak self] in
                await self?.commandWaitCancelled(identifier)
            }
        }
    }

    private func commandSendFailed(_ identifier: UUID, error: Error) async {
        guard let pending = pendingCommands.removeValue(forKey: identifier) else { return }
        pending.timeoutTask.cancel()
        pending.continuation.resume(throwing: error)
        await stopTransport(finishingWith: error)
    }

    private func commandTimedOut(_ identifier: UUID, name: String) async {
        guard let pending = pendingCommands.removeValue(forKey: identifier) else { return }
        let error = PythonMLXEngineError.commandTimedOut(name)
        pending.continuation.resume(throwing: error)
        await stopTransport(finishingWith: error)
    }

    private func commandWaitCancelled(_ identifier: UUID) async {
        guard let pending = pendingCommands.removeValue(forKey: identifier) else { return }
        pending.timeoutTask.cancel()
        let error = CancellationError()
        pending.continuation.resume(throwing: error)
        await stopTransport(finishingWith: error)
    }

    private func waitUntilReady() async throws {
        if readyReceived { return }
        guard lifecycle == .running else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "ready")
        }
        try Task.checkCancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let timeoutTask = Task { [weak self, clock, timeouts] in
                    do {
                        try await clock.sleep(for: timeouts.ready)
                        try Task.checkCancellation()
                    } catch {
                        return
                    }
                    await self?.readyTimedOut()
                }
                readyWaiter = PendingWait(
                    continuation: continuation,
                    timeoutTask: timeoutTask
                )
            }
        } onCancel: {
            Task { [weak self] in await self?.readyWaitCancelled() }
        }
    }

    private func readyTimedOut() async {
        guard let waiter = readyWaiter else { return }
        readyWaiter = nil
        let error = PythonMLXEngineError.readyTimedOut
        waiter.continuation.resume(throwing: error)
        await stopTransport(finishingWith: error)
    }

    private func readyWaitCancelled() async {
        guard let waiter = readyWaiter else { return }
        readyWaiter = nil
        waiter.timeoutTask.cancel()
        let error = CancellationError()
        waiter.continuation.resume(throwing: error)
        await stopTransport(finishingWith: error)
    }

    private func startTransportEvents(_ transport: any WorkerTransport) {
        transportTask?.cancel()
        let stream = transport.events()
        transportTask = Task { [weak self] in
            do {
                for try await event in stream {
                    guard !Task.isCancelled else { return }
                    await self?.handle(event, transport: transport)
                }
                await self?.transportEnded(nil)
            } catch is CancellationError {
            } catch {
                await self?.transportEnded(error)
            }
        }
    }

    private func handle(
        _ processEvent: WorkerProcessEvent,
        transport: any WorkerTransport
    ) async {
        switch processEvent {
        case let .envelope(envelope):
            guard envelope.projectId == projectID,
                  let event = envelope.event else {
                await stopTransport(finishingWith: WorkerProtocolError.malformedEnvelope)
                return
            }
            switch event {
            case let .ack(commandID, command):
                guard let pending = pendingCommands.removeValue(forKey: commandID) else {
                    await stopTransport(finishingWith: WorkerProtocolError.malformedEnvelope)
                    return
                }
                pending.timeoutTask.cancel()
                guard pending.name == command else {
                    pending.continuation.resume(
                        throwing: WorkerProtocolError.malformedEnvelope
                    )
                    await stopTransport(finishingWith: WorkerProtocolError.malformedEnvelope)
                    return
                }
                if command == "hello" { await transport.markProtocolReady() }
                if command == "beginSession" {
                    sessionIsActive = true
                }
                pending.continuation.resume()

            case let .error(commandID?, payload):
                guard let pending = pendingCommands.removeValue(forKey: commandID) else {
                    await stopTransport(finishingWith: WorkerProtocolError.malformedEnvelope)
                    return
                }
                pending.timeoutTask.cancel()
                pending.continuation.resume(throwing: ReconstructionEngineError.workerFailure(
                    code: payload.code,
                    message: payload.message,
                    recoverable: payload.recoverable,
                    details: payload.details
                ))

            default:
                do {
                    guard let engineEvent = try event.engineEvent() else { return }
                    if case .ready = engineEvent {
                        readyReceived = true
                        readyWaiter?.timeoutTask.cancel()
                        readyWaiter?.continuation.resume()
                        readyWaiter = nil
                    }
                    if case .cancelled = engineEvent {
                        sessionIsActive = false
                        if lifecycle == .shuttingDown {
                            shutdownReceivedCancelled = true
                        }
                    }
                    if case .sessionCompleted = engineEvent {
                        sessionIsActive = false
                        if lifecycle == .shuttingDown {
                            shutdownRequiresCancelled = false
                        }
                    }
                    switch eventContinuation.yield(engineEvent) {
                    case .enqueued: break
                    case .dropped, .terminated:
                        await stopTransport(finishingWith: WorkerProcessError.eventBufferOverflow)
                    @unknown default:
                        await stopTransport(finishingWith: WorkerProcessError.eventBufferOverflow)
                    }
                } catch {
                    await stopTransport(finishingWith: error)
                }
            }

        case let .processExited(status):
            if lifecycle == .shuttingDown {
                shutdownExitStatus = status
                return
            }
            let error: Error? = lifecycle == .shutdown || status == 0
                ? nil
                : WorkerProcessError.launchFailed("worker exited with status \(status)")
            await transportEnded(error)
        }
    }

    private func transportEnded(_ error: Error?) async {
        if lifecycle == .shuttingDown {
            gracefulTransportEnded(error)
            return
        }
        guard lifecycle != .shutdown else {
            finishStream(nil)
            return
        }
        await stopTransport(
            finishingWith: error ?? WorkerProcessError.standardOutputClosed
        )
    }

    private func waitForGracefulShutdown() async throws {
        if shutdownStreamEnded {
            if let shutdownTerminalError { throw shutdownTerminalError }
            return
        }
        try Task.checkCancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let timeoutTask = Task { [weak self, clock, timeouts] in
                    do {
                        try await clock.sleep(for: timeouts.shutdownGrace)
                        try Task.checkCancellation()
                    } catch {
                        return
                    }
                    await self?.gracefulShutdownTimedOut()
                }
                gracefulShutdownWaiter = PendingWait(
                    continuation: continuation,
                    timeoutTask: timeoutTask
                )
            }
        } onCancel: {
            Task { [weak self] in await self?.gracefulShutdownWaitCancelled() }
        }
    }

    private func gracefulShutdownTimedOut() {
        guard let waiter = gracefulShutdownWaiter else { return }
        gracefulShutdownWaiter = nil
        waiter.continuation.resume(throwing: PythonMLXEngineError.shutdownTimedOut)
    }

    private func gracefulShutdownWaitCancelled() {
        guard let waiter = gracefulShutdownWaiter else { return }
        gracefulShutdownWaiter = nil
        waiter.timeoutTask.cancel()
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func gracefulTransportEnded(_ error: Error?) {
        guard !shutdownStreamEnded else { return }
        shutdownStreamEnded = true
        transport = nil
        transportTask = nil

        if let error {
            shutdownTerminalError = error
        } else if shutdownExitStatus == nil {
            shutdownTerminalError = PythonMLXEngineError.shutdownEndedBeforeExit
        } else if let status = shutdownExitStatus, status != 0 {
            shutdownTerminalError = PythonMLXEngineError.shutdownExited(status: status)
        } else if shutdownRequiresCancelled, !shutdownReceivedCancelled {
            shutdownTerminalError = PythonMLXEngineError.shutdownEndedBeforeCancelled
        }

        let pending = pendingCommands.values
        pendingCommands.removeAll()
        let pendingError = shutdownTerminalError ?? WorkerProcessError.standardOutputClosed
        pending.forEach {
            $0.timeoutTask.cancel()
            $0.continuation.resume(throwing: pendingError)
        }
        readyWaiter?.timeoutTask.cancel()
        readyWaiter?.continuation.resume(throwing: pendingError)
        readyWaiter = nil

        guard let waiter = gracefulShutdownWaiter else { return }
        gracefulShutdownWaiter = nil
        waiter.timeoutTask.cancel()
        if let shutdownTerminalError {
            waiter.continuation.resume(throwing: shutdownTerminalError)
        } else {
            waiter.continuation.resume()
        }
    }

    private func completeShutdown(finishingWith error: Error?) {
        lifecycle = .shutdown
        beginInProgress = false
        sessionIsActive = false
        gracefulShutdownWaiter?.timeoutTask.cancel()
        gracefulShutdownWaiter = nil
        finishStream(error)
        let waiters = shutdownCompletionWaiters
        shutdownCompletionWaiters.removeAll()
        waiters.forEach { $0.resume() }
    }

    private func stopTransport(finishingWith error: Error?) async {
        if error != nil { lifecycle = .shutdown }
        beginInProgress = false
        let activeTransport = transport
        transport = nil
        transportTask?.cancel()
        transportTask = nil
        let pending = pendingCommands.values
        pendingCommands.removeAll()
        let terminalError = error ?? ReconstructionEngineError.invalidLifecycle(
            operation: "shutdown"
        )
        pending.forEach {
            $0.timeoutTask.cancel()
            $0.continuation.resume(throwing: terminalError)
        }
        readyWaiter?.timeoutTask.cancel()
        readyWaiter?.continuation.resume(throwing: terminalError)
        readyWaiter = nil
        gracefulShutdownWaiter?.timeoutTask.cancel()
        gracefulShutdownWaiter?.continuation.resume(throwing: terminalError)
        gracefulShutdownWaiter = nil
        finishStream(error)
        if let activeTransport { await activeTransport.terminate() }
    }

    private func finishStream(_ error: Error?) {
        guard !streamFinished else { return }
        streamFinished = true
        if let error { eventContinuation.finish(throwing: error) }
        else { eventContinuation.finish() }
    }

    private static func commandName(_ command: WorkerCommand) -> String {
        switch command {
        case .hello: "hello"
        case .configure: "configure"
        case .beginSession: "beginSession"
        case .enqueueFrame: "enqueueFrame"
        case .finishInput: "finishInput"
        case .pause: "pause"
        case .resume: "resume"
        case .cancel: "cancel"
        case .shutdown: "shutdown"
        }
    }
}

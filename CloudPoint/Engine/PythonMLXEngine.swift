import Foundation

protocol WorkerTransport: Sendable {
    nonisolated func events() -> AsyncThrowingStream<WorkerProcessEvent, Error>
    func send(_ envelope: WorkerEnvelope) async throws
    func markProtocolReady() async
    func terminate() async
}

extension WorkerProcess: WorkerTransport {}

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

    private enum Lifecycle {
        case idle
        case prepared
        case running
        case shutdown
    }

    private struct PendingCommand {
        let name: String
        let continuation: CheckedContinuation<Void, Error>
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

    private var lifecycle = Lifecycle.idle
    private var configuration: EngineConfiguration?
    private var projectID: UUID?
    private var transport: (any WorkerTransport)?
    private var transportTask: Task<Void, Never>?
    private var pendingCommands: [UUID: PendingCommand] = [:]
    private var readyWaiter: CheckedContinuation<Void, Error>?
    private var readyReceived = false
    private var streamFinished = false

    init(
        runtime: WorkerRuntime,
        modelDirectory: URL,
        processStarter: @escaping ProcessStarter = PythonMLXEngine.liveStarter
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
    }

    deinit {
        transportTask?.cancel()
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
        projectID = project.projectID
        readyReceived = false

        do {
            let started = try await processStarter(
                launch.executable,
                launch.arguments,
                Self.workerEnvironment()
            )
            transport = started
            startTransportEvents(started)
            lifecycle = .running

            try await sendCommand(.hello(
                clientVersion: "CloudPoint/1.0",
                supportedProtocolVersions: [UInt32(WorkerEnvelope.currentProtocolVersion)]
            ))
            try await waitUntilReady()
            try await sendCommand(.configure(configuration))
            try await sendCommand(.beginSession(
                resumeCheckpoint: project.resumeCheckpoint
            ))
        } catch {
            await stopTransport(finishingWith: error)
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
        guard lifecycle == .running else { return }
        _ = try? await sendCommand(.cancel)
    }

    func shutdown() async {
        guard lifecycle != .shutdown else { return }
        if lifecycle == .running {
            _ = try? await sendCommand(.shutdown)
        }
        await stopTransport(finishingWith: nil)
        lifecycle = .shutdown
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

    private func sendCommand(_ command: WorkerCommand) async throws {
        guard lifecycle == .running,
              let transport,
              let projectID else {
            throw ReconstructionEngineError.invalidLifecycle(
                operation: Self.commandName(command)
            )
        }
        let identifier = UUID()
        let envelope = WorkerEnvelope.command(
            command,
            id: identifier,
            projectId: projectID
        )
        try await withCheckedThrowingContinuation { continuation in
            pendingCommands[identifier] = PendingCommand(
                name: Self.commandName(command),
                continuation: continuation
            )
            Task { [weak self, transport] in
                do {
                    try await transport.send(envelope)
                } catch {
                    await self?.commandSendFailed(identifier, error: error)
                }
            }
        }
    }

    private func commandSendFailed(_ identifier: UUID, error: Error) {
        pendingCommands.removeValue(forKey: identifier)?.continuation.resume(throwing: error)
    }

    private func waitUntilReady() async throws {
        if readyReceived { return }
        try await withCheckedThrowingContinuation { continuation in
            readyWaiter = continuation
        }
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
                guard pending.name == command else {
                    pending.continuation.resume(
                        throwing: WorkerProtocolError.malformedEnvelope
                    )
                    await stopTransport(finishingWith: WorkerProtocolError.malformedEnvelope)
                    return
                }
                if command == "hello" { await transport.markProtocolReady() }
                pending.continuation.resume()

            case let .error(commandID?, payload):
                guard let pending = pendingCommands.removeValue(forKey: commandID) else {
                    await stopTransport(finishingWith: WorkerProtocolError.malformedEnvelope)
                    return
                }
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
                        readyWaiter?.resume()
                        readyWaiter = nil
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
            let error: Error? = lifecycle == .shutdown || status == 0
                ? nil
                : WorkerProcessError.launchFailed("worker exited with status \(status)")
            await transportEnded(error)
        }
    }

    private func transportEnded(_ error: Error?) async {
        guard lifecycle != .shutdown else {
            finishStream(nil)
            return
        }
        await stopTransport(
            finishingWith: error ?? WorkerProcessError.standardOutputClosed
        )
    }

    private func stopTransport(finishingWith error: Error?) async {
        let activeTransport = transport
        transport = nil
        transportTask?.cancel()
        transportTask = nil
        if let activeTransport { await activeTransport.terminate() }
        let pending = pendingCommands.values
        pendingCommands.removeAll()
        let terminalError = error ?? ReconstructionEngineError.invalidLifecycle(
            operation: "shutdown"
        )
        pending.forEach { $0.continuation.resume(throwing: terminalError) }
        readyWaiter?.resume(throwing: terminalError)
        readyWaiter = nil
        if error != nil { lifecycle = .shutdown }
        finishStream(error)
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

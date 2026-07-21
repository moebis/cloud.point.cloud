import Darwin
import Foundation

enum SharpProcessEvent: Sendable, Equatable {
    case line(Data)
    case exited(Int32, diagnostics: String)
}

private final class SharpDiagnosticBuffer: @unchecked Sendable {
    private static let capacity = 32 * 1_024
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) {
        lock.lock()
        defer { lock.unlock() }
        data.append(chunk)
        if data.count > Self.capacity {
            data.removeFirst(data.count - Self.capacity)
        }
    }

    func text() -> String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

protocol SharpProcessTransport: Sendable {
    nonisolated func events() -> AsyncThrowingStream<SharpProcessEvent, Error>
    func terminate() async
}

actor SharpLineProcess: SharpProcessTransport {
    nonisolated private let stream: AsyncThrowingStream<SharpProcessEvent, Error>
    private let continuation: AsyncThrowingStream<SharpProcessEvent, Error>.Continuation
    private let process = Process()
    private let standardOutput = Pipe()
    private let standardError = Pipe()
    private let diagnosticBuffer = SharpDiagnosticBuffer()
    private var stdoutTask: Task<Void, Never>?
    private var stderrTask: Task<Void, Never>?
    private var didFinish = false

    private init() {
        let pair = AsyncThrowingStream.makeStream(
            of: SharpProcessEvent.self,
            bufferingPolicy: .bufferingOldest(128)
        )
        stream = pair.stream
        continuation = pair.continuation
    }

    static func start(
        executable: URL,
        arguments: [String],
        environment: [String: String]
    ) async throws -> SharpLineProcess {
        guard executable.isFileURL,
              executable.path.hasPrefix("/"),
              FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw WorkerProcessError.executableNotFound(executable.path)
        }
        let transport = SharpLineProcess()
        try await transport.launch(
            executable: executable,
            arguments: arguments,
            environment: environment
        )
        return transport
    }

    nonisolated func events() -> AsyncThrowingStream<SharpProcessEvent, Error> { stream }

    func terminate() {
        guard process.isRunning else {
            finish(nil)
            return
        }
        process.terminate()
        let identifier = process.processIdentifier
        Task.detached {
            try? await ContinuousClock().sleep(for: .seconds(5))
            if Darwin.kill(identifier, 0) == 0 { _ = Darwin.kill(identifier, SIGKILL) }
        }
    }

    private func launch(
        executable: URL,
        arguments: [String],
        environment: [String: String]
    ) throws {
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.standardOutput = standardOutput
        process.standardError = standardError
        do { try process.run() }
        catch { throw WorkerProcessError.launchFailed(error.localizedDescription) }

        let output = standardOutput.fileHandleForReading
        stdoutTask = Task.detached(priority: .userInitiated) { [weak self, output] in
            var buffered = Data()
            do {
                while !Task.isCancelled {
                    let chunk = try output.read(upToCount: 64 * 1_024) ?? Data()
                    if chunk.isEmpty { break }
                    buffered.append(chunk)
                    while let newline = buffered.firstIndex(of: 0x0A) {
                        var line = buffered[..<newline]
                        if line.last == 0x0D { line = line.dropLast() }
                        guard !line.isEmpty else {
                            buffered.removeSubrange(...newline)
                            continue
                        }
                        guard line.count <= SharpWorkerLineCodec.maximumLineBytes else {
                            throw SharpWorkerProtocolError.oversizedLine
                        }
                        await self?.receivedLine(Data(line))
                        buffered.removeSubrange(...newline)
                    }
                    guard buffered.count <= SharpWorkerLineCodec.maximumLineBytes else {
                        throw SharpWorkerProtocolError.oversizedLine
                    }
                }
                guard buffered.isEmpty else {
                    throw SharpWorkerProtocolError.malformedJSON
                }
                await self?.stdoutEnded()
            } catch {
                await self?.finish(error)
            }
        }

        let diagnostics = standardError.fileHandleForReading
        let diagnosticBuffer = self.diagnosticBuffer
        stderrTask = Task.detached(priority: .utility) { [diagnostics, diagnosticBuffer] in
            while !Task.isCancelled {
                let chunk = try? diagnostics.read(upToCount: 4_096)
                if chunk == nil || chunk?.isEmpty == true { break }
                if let chunk { diagnosticBuffer.append(chunk) }
            }
        }
    }

    private func receivedLine(_ data: Data) {
        guard !didFinish else { return }
        guard case .enqueued = continuation.yield(.line(data)) else {
            terminate()
            finish(WorkerProcessError.eventBufferOverflow)
            return
        }
    }

    private func stdoutEnded() async {
        process.waitUntilExit()
        _ = await stderrTask?.value
        let status = process.terminationStatus
        if !didFinish {
            continuation.yield(.exited(status, diagnostics: diagnosticBuffer.text()))
        }
        finish(nil)
    }

    private func finish(_ error: Error?) {
        guard !didFinish else { return }
        didFinish = true
        stdoutTask?.cancel()
        stderrTask?.cancel()
        standardOutput.fileHandleForReading.closeFile()
        standardError.fileHandleForReading.closeFile()
        if let error { continuation.finish(throwing: error) }
        else { continuation.finish() }
    }
}

enum SharpReconstructionEngineError: Error, Sendable, Equatable {
    case invalidMode
    case resumeUnsupported
    case invalidFrame
    case multipleFrames
    case missingFrame
    case unexpectedCompletion
    case workerExited(Int32, String)
}

struct SharpReconstructionEngineFactory: Sendable {
    let runtime: WorkerRuntime
    let installation: SharpModelInstallation

    func makeEngine() -> any ReconstructionEngine {
        SharpReconstructionEngine(runtime: runtime, installation: installation)
    }
}

actor SharpReconstructionEngine: ReconstructionEngine {
    typealias ProcessStarter = @Sendable (
        _ executable: URL,
        _ arguments: [String],
        _ environment: [String: String]
    ) async throws -> any SharpProcessTransport

    private enum Lifecycle: Equatable {
        case idle
        case prepared
        case begun
        case running
        case completed
        case cancelled
        case shutdown
    }

    nonisolated private let stream: AsyncThrowingStream<EngineEvent, Error>
    private let continuation: AsyncThrowingStream<EngineEvent, Error>.Continuation
    private let runtime: WorkerRuntime
    private let installation: SharpModelInstallation
    private let processStarter: ProcessStarter
    private var lifecycle: Lifecycle = .idle
    private var project: ProjectDescriptor?
    private var frame: PersistedFrame?
    private var transport: (any SharpProcessTransport)?
    private var eventTask: Task<Void, Never>?

    init(
        runtime: WorkerRuntime,
        installation: SharpModelInstallation,
        processStarter: @escaping ProcessStarter = { executable, arguments, environment in
            try await SharpLineProcess.start(
                executable: executable,
                arguments: arguments,
                environment: environment
            )
        }
    ) {
        let pair = AsyncThrowingStream.makeStream(
            of: EngineEvent.self,
            bufferingPolicy: .bufferingOldest(128)
        )
        stream = pair.stream
        continuation = pair.continuation
        self.runtime = runtime
        self.installation = installation
        self.processStarter = processStarter
    }

    deinit {
        eventTask?.cancel()
        continuation.finish()
    }

    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> { stream }

    func prepare(configuration: EngineConfiguration) throws {
        guard lifecycle == .idle else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "prepare")
        }
        try configuration.validate()
        lifecycle = .prepared
    }

    func begin(project: ProjectDescriptor) throws {
        guard lifecycle == .prepared else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "begin")
        }
        guard project.modeID == .sharpGaussian else {
            throw SharpReconstructionEngineError.invalidMode
        }
        guard project.resumeCheckpoint == nil else {
            throw SharpReconstructionEngineError.resumeUnsupported
        }
        guard project.packageURL.isFileURL, project.packageURL.path.hasPrefix("/") else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
        self.project = project
        lifecycle = .begun
        continuation.yield(.ready(
            engineVersion: "0.1.0-sharp",
            modelIdentifier: "apple/ml-sharp",
            modelRevision: installation.sourceCommit,
            convertedWeightsSHA256: installation.checkpointSHA256
        ))
    }

    func enqueue(_ frame: PersistedFrame) throws {
        guard lifecycle == .begun else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "enqueue")
        }
        guard self.frame == nil else { throw SharpReconstructionEngineError.multipleFrames }
        guard frame.relativePath == String(format: "Frames/%08u.jpg", frame.index),
              frame.sourceTimestamp.isFinite,
              frame.sourceTimestamp >= 0 else {
            throw SharpReconstructionEngineError.invalidFrame
        }
        self.frame = frame
    }

    func finishInput() async throws {
        guard lifecycle == .begun, let project, let frame else {
            if lifecycle == .begun { throw SharpReconstructionEngineError.missingFrame }
            throw ReconstructionEngineError.invalidLifecycle(operation: "finishInput")
        }
        let output = String(format: "Outputs/Gaussians/%08u.ply", frame.index)
        let preferMPS = project.sharpConfiguration?.preferMPS ?? true
        var arguments = [
            "-I", "-B", "-m", "cloudpoint_worker.sharp.cli",
            "--project", project.packageURL.standardizedFileURL.path,
            "--checkpoint", installation.checkpoint.standardizedFileURL.path,
            "--checkpoint-sha256", installation.checkpointSHA256,
            "--source-commit", installation.sourceCommit,
            "--input-relative-path", frame.relativePath,
            "--output-relative-path", output,
        ]
        arguments.append(preferMPS ? "--prefer-mps" : "--no-prefer-mps")
        let started = try await processStarter(
            runtime.pythonExecutable,
            arguments,
            Self.workerEnvironment()
        )
        guard lifecycle == .begun else {
            await started.terminate()
            throw CancellationError()
        }
        transport = started
        lifecycle = .running
        consumeEvents(started, expectedFrame: frame)
    }

    func pause() throws {
        throw ReconstructionEngineError.invalidLifecycle(operation: "pause")
    }

    func resume() throws {
        throw ReconstructionEngineError.invalidLifecycle(operation: "resume")
    }

    func cancel() async {
        guard ![.completed, .cancelled, .shutdown].contains(lifecycle) else { return }
        lifecycle = .cancelled
        await transport?.terminate()
        continuation.yield(.cancelled(lastCompletedWindowIndex: nil))
        continuation.finish()
    }

    func shutdown() async {
        guard lifecycle != .shutdown else { return }
        lifecycle = .shutdown
        eventTask?.cancel()
        await transport?.terminate()
        continuation.finish()
    }

    private func consumeEvents(
        _ transport: any SharpProcessTransport,
        expectedFrame: PersistedFrame
    ) {
        let events = transport.events()
        eventTask = Task { [weak self] in
            do {
                for try await event in events {
                    guard !Task.isCancelled else { return }
                    await self?.handle(event, expectedFrame: expectedFrame)
                }
            } catch {
                await self?.finish(throwing: error)
            }
        }
    }

    private func handle(_ event: SharpProcessEvent, expectedFrame: PersistedFrame) async {
        guard lifecycle == .running || lifecycle == .completed else { return }
        do {
            switch event {
            case let .line(data):
                switch try SharpWorkerLineCodec.decode(data) {
                case let .progress(stage, fraction):
                    continuation.yield(.gaussianProgress(stage: stage, fraction: fraction))
                case let .heartbeat(seconds):
                    continuation.yield(.heartbeat(
                        busy: true,
                        monotonicSeconds: seconds,
                        queuedFrames: 1,
                        processedFrames: 0,
                        currentWindow: nil
                    ))
                case let .warning(code, message, recoverable):
                    continuation.yield(.warning(
                        code: code,
                        message: message,
                        recoverable: recoverable,
                        details: [:]
                    ))
                case let .completed(result):
                    guard lifecycle == .running,
                          result.sourceFrameIndex == expectedFrame.index else {
                        throw SharpReconstructionEngineError.unexpectedCompletion
                    }
                    lifecycle = .completed
                    continuation.yield(.gaussianCompleted(result))
                case let .failed(code, message, recoverable):
                    throw ReconstructionEngineError.workerFailure(
                        code: code,
                        message: message,
                        recoverable: recoverable,
                        details: [:]
                    )
                case .cancelled:
                    lifecycle = .cancelled
                    continuation.yield(.cancelled(lastCompletedWindowIndex: nil))
                    continuation.finish()
                }
            case let .exited(status, diagnostics):
                if lifecycle == .completed, status == 0 {
                    continuation.finish()
                } else if lifecycle != .cancelled {
                    throw SharpReconstructionEngineError.workerExited(status, diagnostics)
                }
            }
        } catch {
            await transport?.terminate()
            finish(throwing: error)
        }
    }

    private func finish(throwing error: Error) {
        lifecycle = .shutdown
        continuation.finish(throwing: error)
    }

    private static func workerEnvironment() -> [String: String] {
        [
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": FileManager.default.temporaryDirectory.path,
        ]
    }
}

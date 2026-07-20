import Foundation
import XCTest
@testable import CloudPoint

final class PythonMLXEngineTests: XCTestCase {
    func testCancelInterruptsSuspendedStartupAndTerminatesLateTransport() async throws {
        let transport = FakeWorkerTransport()
        let starter = DelayedWorkerStarter(transport: transport)
        let engine = makeEngine(processStarter: starter.start)
        try await engine.prepare(configuration: EngineConfiguration())
        let project = Self.project()

        let completed = expectation(description: "begin returned after cancellation")
        let beginTask = Task<Result<Void, Error>, Never> {
            let result: Result<Void, Error>
            do {
                try await engine.begin(project: project)
                result = .success(())
            } catch {
                result = .failure(error)
            }
            completed.fulfill()
            return result
        }
        await starter.waitUntilStartCount(1)

        await engine.cancel()
        await fulfillment(of: [completed], timeout: 0.5)
        switch await beginTask.value {
        case .failure(let error):
            XCTAssertTrue(error is CancellationError, "Unexpected error: \(error)")
        case .success: XCTFail("Cancelled startup unexpectedly completed")
        }

        await starter.releaseAll()
        await transport.waitUntilTerminated()
        let terminationCount = await transport.terminationCount
        XCTAssertEqual(terminationCount, 1)
    }

    func testConcurrentBeginIsRejectedAndShutdownInterruptsFirstStartup() async throws {
        let transport = FakeWorkerTransport()
        let starter = DelayedWorkerStarter(
            transport: transport,
            rejectUnexpectedStarts: true
        )
        let engine = makeEngine(processStarter: starter.start)
        try await engine.prepare(configuration: EngineConfiguration())
        let project = Self.project()

        let completed = expectation(description: "first begin returned after shutdown")
        let firstBegin = Task<Result<Void, Error>, Never> {
            let result: Result<Void, Error>
            do {
                try await engine.begin(project: project)
                result = .success(())
            } catch {
                result = .failure(error)
            }
            completed.fulfill()
            return result
        }
        await starter.waitUntilStartCount(1)

        do {
            try await engine.begin(project: project)
            XCTFail("Concurrent begin unexpectedly launched")
        } catch let error as ReconstructionEngineError {
            XCTAssertEqual(error, .invalidLifecycle(operation: "begin"))
        } catch {
            XCTFail("Unexpected concurrent begin error: \(error)")
        }
        let startCount = await starter.startCount
        XCTAssertEqual(startCount, 1)

        await engine.shutdown()
        await fulfillment(of: [completed], timeout: 0.5)
        switch await firstBegin.value {
        case .failure(let error): XCTAssertTrue(error is CancellationError)
        case .success: XCTFail("Shutdown startup unexpectedly completed")
        }

        await starter.releaseAll()
        await transport.waitUntilTerminated()
        let terminationCount = await transport.terminationCount
        XCTAssertEqual(terminationCount, 1)
    }

    func testCommandAcknowledgementTimeoutStopsTransport() async throws {
        let transport = FakeWorkerTransport(
            unacknowledgedCommands: ["pause"],
            blockTermination: true
        )
        let clock = ManualPythonMLXEngineClock()
        let engine = makeEngine(transport: transport, clock: clock)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: Self.project())
        let baselineSleeps = await clock.sleepCount
        let completed = expectation(description: "timed-out command returned before termination")

        let pause = Task<Result<Void, Error>, Never> {
            let result: Result<Void, Error>
            do {
                try await engine.pause()
                result = .success(())
            } catch {
                result = .failure(error)
            }
            completed.fulfill()
            return result
        }
        await transport.waitUntilCommand("pause")
        await clock.waitUntilSleepCount(baselineSleeps + 1)
        await clock.fireAll()
        await fulfillment(of: [completed], timeout: 0.5)
        await transport.releaseTermination()

        switch await pause.value {
        case .failure(let error):
            XCTAssertEqual(
                error as? PythonMLXEngineError,
                .commandTimedOut("pause")
            )
        case .success:
            XCTFail("Unacknowledged command unexpectedly completed")
        }
        await transport.waitUntilTerminated()
    }

    func testReadyTimeoutStopsTransport() async throws {
        let transport = FakeWorkerTransport(
            emitReadyAfterHello: false,
            blockTermination: true
        )
        let clock = ManualPythonMLXEngineClock()
        let engine = makeEngine(transport: transport, clock: clock)
        try await engine.prepare(configuration: EngineConfiguration())
        let project = Self.project()
        let completed = expectation(description: "ready timeout returned before termination")

        let begin = Task<Result<Void, Error>, Never> {
            let result: Result<Void, Error>
            do {
                try await engine.begin(project: project)
                result = .success(())
            } catch {
                result = .failure(error)
            }
            completed.fulfill()
            return result
        }
        await transport.waitUntilCommand("hello")
        await clock.waitUntilSleepCount(2)
        await clock.fireAll()
        await fulfillment(of: [completed], timeout: 0.5)
        await transport.releaseTermination()

        switch await begin.value {
        case .failure(let error):
            XCTAssertEqual(error as? PythonMLXEngineError, .readyTimedOut)
        case .success:
            XCTFail("Worker without ready unexpectedly completed startup")
        }
        await transport.waitUntilTerminated()
    }

    func testCancelInterruptsReadyWaitAfterProcessLaunch() async throws {
        let transport = FakeWorkerTransport(emitReadyAfterHello: false)
        let engine = makeEngine(transport: transport)
        try await engine.prepare(configuration: EngineConfiguration())
        let project = Self.project()
        let completed = expectation(description: "ready wait returned after cancellation")

        let begin = Task<Result<Void, Error>, Never> {
            let result: Result<Void, Error>
            do {
                try await engine.begin(project: project)
                result = .success(())
            } catch {
                result = .failure(error)
            }
            completed.fulfill()
            return result
        }
        await transport.waitUntilCommand("hello")

        await engine.cancel()
        await fulfillment(of: [completed], timeout: 0.5)
        begin.cancel()
        switch await begin.value {
        case .failure(let error):
            XCTAssertTrue(error is CancellationError, "Unexpected error: \(error)")
        case .success: XCTFail("Cancelled ready wait unexpectedly completed")
        }
        await transport.waitUntilTerminated()
    }

    func testCancellingCommandWaitStopsTransport() async throws {
        let transport = FakeWorkerTransport(unacknowledgedCommands: ["pause"])
        let engine = makeEngine(transport: transport)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: Self.project())

        let pause = Task<Result<Void, Error>, Never> {
            do {
                try await engine.pause()
                return .success(())
            } catch {
                return .failure(error)
            }
        }
        await transport.waitUntilCommand("pause")
        pause.cancel()

        switch await pause.value {
        case .failure(let error): XCTAssertTrue(error is CancellationError)
        case .success: XCTFail("Cancelled command wait unexpectedly completed")
        }
        await transport.waitUntilTerminated()
    }

    func testShutdownWaitsForCancelledAndNaturalTransportCompletion() async throws {
        let transport = FakeWorkerTransport(shutdownBehavior: .cancelAndFinish)
        let engine = makeEngine(transport: transport)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: Self.project())

        await engine.shutdown()

        let terminationCount = await transport.terminationCount
        let commands = await transport.commands
        XCTAssertEqual(terminationCount, 0)
        XCTAssertEqual(commands.last, "shutdown")
    }

    func testShutdownForcesTerminationAfterGraceTimeout() async throws {
        let transport = FakeWorkerTransport(shutdownBehavior: .cancelAndExitWithoutFinishing)
        let clock = ManualPythonMLXEngineClock()
        let engine = makeEngine(transport: transport, clock: clock)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: Self.project())
        let baselineSleeps = await clock.sleepCount

        let shutdown = Task { await engine.shutdown() }
        await transport.waitUntilCommand("shutdown")
        await clock.waitUntilSleepCount(baselineSleeps + 2)
        await clock.fireAll()
        await shutdown.value

        let terminationCount = await transport.terminationCount
        XCTAssertEqual(terminationCount, 1)
    }

    func testShutdownRejectsNaturalCompletionWithoutRequiredProcessExit() async throws {
        let transport = FakeWorkerTransport(shutdownBehavior: .cancelAndFinishWithoutExit)
        let engine = makeEngine(transport: transport)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: Self.project())

        await engine.shutdown()

        do {
            for try await _ in engine.events() {}
            XCTFail("Missing process-exit event unexpectedly closed cleanly")
        } catch let error as PythonMLXEngineError {
            XCTAssertEqual(error, .shutdownEndedBeforeExit)
        }
    }

    func testShutdownRejectsNaturalCompletionWithoutRequiredCancelledEvent() async throws {
        let transport = FakeWorkerTransport(shutdownBehavior: .finishWithoutCancelled)
        let engine = makeEngine(transport: transport)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: Self.project())

        await engine.shutdown()

        do {
            for try await _ in engine.events() {}
            XCTFail("Missing cancelled event unexpectedly closed cleanly")
        } catch let error as PythonMLXEngineError {
            XCTAssertEqual(error, .shutdownEndedBeforeCancelled)
        }
    }

    func testBeginLaunchesPinnedRuntimeAndCompletesProtocolHandshake() async throws {
        let projectID = UUID(uuidString: "00000000-0000-0000-0000-000000000901")!
        let project = URL(filePath: "/tmp/Real.cloudpoint", directoryHint: .isDirectory)
        let model = URL(filePath: "/tmp/converted", directoryHint: .isDirectory)
        let runtime = WorkerRuntime.unchecked(
            root: URL(filePath: "/opt/cloudpoint/runtime", directoryHint: .isDirectory)
        )
        let transport = FakeWorkerTransport()
        let starter = RecordingWorkerStarter(transport: transport)
        let engine = PythonMLXEngine(
            runtime: runtime,
            modelDirectory: model,
            processStarter: starter.start
        )

        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: ProjectDescriptor(
            projectID: projectID,
            packageURL: project
        ))

        let recordedLaunch = await starter.launch
        let launch = try XCTUnwrap(recordedLaunch)
        XCTAssertEqual(launch.executable, runtime.workerExecutable)
        XCTAssertEqual(launch.arguments, [
            "serve", "--project", project.path,
            "--model", model.path,
        ])
        XCTAssertEqual(Set(launch.environment.keys), [
            "HOME", "TMPDIR", "PATH", "PYTHONNOUSERSITE", "PYTHONHASHSEED",
            "LC_ALL", "LANG",
        ])
        XCTAssertEqual(launch.environment["PATH"], "/usr/bin:/bin")
        let handshakeCommands = await transport.commands
        let protocolReady = await transport.protocolReady
        XCTAssertEqual(handshakeCommands, ["hello", "configure", "beginSession"])
        XCTAssertTrue(protocolReady)

        try await engine.enqueue(PersistedFrame(
            index: 0,
            sourceTimestamp: 0,
            relativePath: "Frames/00000000.jpg"
        ))
        try await engine.finishInput()
        let allCommands = await transport.commands
        XCTAssertEqual(
            allCommands,
            ["hello", "configure", "beginSession", "enqueueFrame", "finishInput"]
        )
        await engine.shutdown()
    }

    func testCommandErrorIsReturnedToTheOwningCall() async throws {
        let transport = FakeWorkerTransport(failingCommand: "pause")
        let starter = RecordingWorkerStarter(transport: transport)
        let engine = PythonMLXEngine(
            runtime: .unchecked(
                root: URL(filePath: "/opt/cloudpoint/runtime", directoryHint: .isDirectory)
            ),
            modelDirectory: URL(filePath: "/tmp/converted", directoryHint: .isDirectory),
            processStarter: starter.start
        )
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: ProjectDescriptor(
            projectID: UUID(),
            packageURL: URL(filePath: "/tmp/Test.cloudpoint", directoryHint: .isDirectory)
        ))

        do {
            try await engine.pause()
            XCTFail("Expected command failure")
        } catch let ReconstructionEngineError.workerFailure(code, _, recoverable, _) {
            XCTAssertEqual(code, "PAUSE_REJECTED")
            XCTAssertTrue(recoverable)
        }
        await engine.shutdown()
    }

    func testMismatchedAcknowledgementTerminatesHandshake() async throws {
        let transport = FakeWorkerTransport(
            acknowledgementOverride: ["configure": "resume"]
        )
        let starter = RecordingWorkerStarter(transport: transport)
        let engine = PythonMLXEngine(
            runtime: .unchecked(
                root: URL(filePath: "/opt/cloudpoint/runtime", directoryHint: .isDirectory)
            ),
            modelDirectory: URL(filePath: "/tmp/converted", directoryHint: .isDirectory),
            processStarter: starter.start
        )
        try await engine.prepare(configuration: EngineConfiguration())

        do {
            try await engine.begin(project: ProjectDescriptor(
                projectID: UUID(),
                packageURL: URL(filePath: "/tmp/Test.cloudpoint", directoryHint: .isDirectory)
            ))
            XCTFail("Expected the mismatched acknowledgement to fail")
        } catch let error as WorkerProtocolError {
            XCTAssertEqual(error, .malformedEnvelope)
        }
    }

    private func makeEngine(
        transport: FakeWorkerTransport = FakeWorkerTransport(),
        clock: any PythonMLXEngineClock = ContinuousPythonMLXEngineClock(),
        processStarter: PythonMLXEngine.ProcessStarter? = nil
    ) -> PythonMLXEngine {
        PythonMLXEngine(
            runtime: .unchecked(
                root: URL(filePath: "/opt/cloudpoint/runtime", directoryHint: .isDirectory)
            ),
            modelDirectory: URL(filePath: "/tmp/converted", directoryHint: .isDirectory),
            processStarter: processStarter ?? { _, _, _ in transport },
            clock: clock,
            timeouts: PythonMLXEngineTimeouts(
                commandAcknowledgement: .seconds(1),
                ready: .seconds(2),
                shutdownGrace: .seconds(3)
            )
        )
    }

    private static func project() -> ProjectDescriptor {
        ProjectDescriptor(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000999")!,
            packageURL: URL(filePath: "/tmp/Test.cloudpoint", directoryHint: .isDirectory)
        )
    }
}

private struct RecordedLaunch: Sendable {
    let executable: URL
    let arguments: [String]
    let environment: [String: String]
}

private actor RecordingWorkerStarter {
    let transport: FakeWorkerTransport
    private(set) var launch: RecordedLaunch?

    init(transport: FakeWorkerTransport) { self.transport = transport }

    func start(
        executable: URL,
        arguments: [String],
        environment: [String: String]
    ) async throws -> any WorkerTransport {
        launch = RecordedLaunch(
            executable: executable,
            arguments: arguments,
            environment: environment
        )
        return transport
    }
}

private enum FakeShutdownBehavior: Sendable {
    case cancelAndFinish
    case cancelAndExitWithoutFinishing
    case cancelAndFinishWithoutExit
    case finishWithoutCancelled
    case stayOpen
}

private actor FakeWorkerTransport: WorkerTransport {
    nonisolated private let stream: AsyncThrowingStream<WorkerProcessEvent, Error>
    private let continuation: AsyncThrowingStream<WorkerProcessEvent, Error>.Continuation
    private let failingCommand: String?
    private let acknowledgementOverride: [String: String]
    private let unacknowledgedCommands: Set<String>
    private let emitReadyAfterHello: Bool
    private let shutdownBehavior: FakeShutdownBehavior
    private let blockTermination: Bool
    private(set) var commands: [String] = []
    private(set) var protocolReady = false
    private(set) var terminationCount = 0
    private var commandWaiters: [(String, CheckedContinuation<Void, Never>)] = []
    private var terminationWaiters: [CheckedContinuation<Void, Never>] = []
    private var terminationReleaseWaiters: [CheckedContinuation<Void, Never>] = []
    private var terminationReleased = false

    init(
        failingCommand: String? = nil,
        acknowledgementOverride: [String: String] = [:],
        unacknowledgedCommands: Set<String> = [],
        emitReadyAfterHello: Bool = true,
        shutdownBehavior: FakeShutdownBehavior = .cancelAndFinish,
        blockTermination: Bool = false
    ) {
        let pair = AsyncThrowingStream.makeStream(of: WorkerProcessEvent.self)
        stream = pair.stream
        continuation = pair.continuation
        self.failingCommand = failingCommand
        self.acknowledgementOverride = acknowledgementOverride
        self.unacknowledgedCommands = unacknowledgedCommands
        self.emitReadyAfterHello = emitReadyAfterHello
        self.shutdownBehavior = shutdownBehavior
        self.blockTermination = blockTermination
    }

    nonisolated func events() -> AsyncThrowingStream<WorkerProcessEvent, Error> { stream }

    func send(_ envelope: WorkerEnvelope) async throws {
        guard let command = envelope.command else { return }
        let name: String
        switch command {
        case .hello: name = "hello"
        case .configure: name = "configure"
        case .beginSession: name = "beginSession"
        case .enqueueFrame: name = "enqueueFrame"
        case .finishInput: name = "finishInput"
        case .pause: name = "pause"
        case .resume: name = "resume"
        case .cancel: name = "cancel"
        case .shutdown: name = "shutdown"
        }
        commands.append(name)
        let readyWaiters = commandWaiters.filter { $0.0 == name }
        commandWaiters.removeAll { $0.0 == name }
        readyWaiters.forEach { $0.1.resume() }
        if unacknowledgedCommands.contains(name) { return }
        if name == failingCommand {
            continuation.yield(.envelope(.event(
                .error(
                    commandId: envelope.id,
                    WorkerErrorPayload(
                        code: "PAUSE_REJECTED",
                        message: "Pause is unavailable",
                        recoverable: true,
                        details: [:]
                    )
                ),
                projectId: envelope.projectId
            )))
            return
        }
        continuation.yield(.envelope(.event(
            .ack(
                commandId: envelope.id,
                command: acknowledgementOverride[name] ?? name
            ),
            projectId: envelope.projectId
        )))
        if name == "hello" {
            continuation.yield(.envelope(.event(
                .heartbeat(
                    busy: false,
                    monotonicSeconds: 1,
                    queuedFrames: 0,
                    processedFrames: 0,
                    currentWindow: nil
                ),
                projectId: envelope.projectId
            )))
            if emitReadyAfterHello {
                continuation.yield(.envelope(.event(
                    .ready(
                        engineVersion: "1.0.0",
                        modelIdentifier: "robbyant/lingbot-map",
                        modelRevision: "204754b",
                        convertedWeightsSHA256: String(repeating: "a", count: 64)
                    ),
                    projectId: envelope.projectId
                )))
            }
        }
        if name == "shutdown" {
            switch shutdownBehavior {
            case .cancelAndFinish:
                continuation.yield(.envelope(.event(
                    .cancelled(lastCompletedWindowIndex: nil),
                    projectId: envelope.projectId
                )))
                continuation.yield(.processExited(status: 0))
                continuation.finish()
            case .cancelAndExitWithoutFinishing:
                continuation.yield(.envelope(.event(
                    .cancelled(lastCompletedWindowIndex: nil),
                    projectId: envelope.projectId
                )))
                continuation.yield(.processExited(status: 0))
            case .cancelAndFinishWithoutExit:
                continuation.yield(.envelope(.event(
                    .cancelled(lastCompletedWindowIndex: nil),
                    projectId: envelope.projectId
                )))
                continuation.finish()
            case .finishWithoutCancelled:
                continuation.yield(.processExited(status: 0))
                continuation.finish()
            case .stayOpen:
                break
            }
        }
    }

    func markProtocolReady() async { protocolReady = true }

    func terminate() async {
        terminationCount += 1
        if blockTermination, !terminationReleased {
            await withCheckedContinuation { terminationReleaseWaiters.append($0) }
        }
        continuation.finish()
        let waiters = terminationWaiters
        terminationWaiters.removeAll()
        waiters.forEach { $0.resume() }
    }

    func waitUntilCommand(_ name: String) async {
        if commands.contains(name) { return }
        await withCheckedContinuation { commandWaiters.append((name, $0)) }
    }

    func waitUntilTerminated() async {
        if terminationCount > 0 { return }
        await withCheckedContinuation { terminationWaiters.append($0) }
    }

    func releaseTermination() {
        terminationReleased = true
        let waiters = terminationReleaseWaiters
        terminationReleaseWaiters.removeAll()
        waiters.forEach { $0.resume() }
    }
}

private enum DelayedStarterError: Error {
    case unexpectedAdditionalStart
}

private actor DelayedWorkerStarter {
    let transport: FakeWorkerTransport
    let rejectUnexpectedStarts: Bool
    private(set) var startCount = 0
    private var startWaiters: [(Int, CheckedContinuation<Void, Never>)] = []
    private var releases: [CheckedContinuation<Void, Never>] = []

    init(transport: FakeWorkerTransport, rejectUnexpectedStarts: Bool = false) {
        self.transport = transport
        self.rejectUnexpectedStarts = rejectUnexpectedStarts
    }

    func start(
        executable _: URL,
        arguments _: [String],
        environment _: [String: String]
    ) async throws -> any WorkerTransport {
        startCount += 1
        let reached = startWaiters.filter { startCount >= $0.0 }
        startWaiters.removeAll { startCount >= $0.0 }
        reached.forEach { $0.1.resume() }
        if rejectUnexpectedStarts, startCount > 1 {
            throw DelayedStarterError.unexpectedAdditionalStart
        }
        await withCheckedContinuation { releases.append($0) }
        return transport
    }

    func waitUntilStartCount(_ expected: Int) async {
        if startCount >= expected { return }
        await withCheckedContinuation { startWaiters.append((expected, $0)) }
    }

    func releaseAll() {
        let pending = releases
        releases.removeAll()
        pending.forEach { $0.resume() }
    }
}

private actor ManualPythonMLXEngineClock: PythonMLXEngineClock {
    private(set) var sleepCount = 0
    private var sleepers: [CheckedContinuation<Void, Error>] = []

    func sleep(for _: Duration) async throws {
        sleepCount += 1
        try await withCheckedThrowingContinuation { sleepers.append($0) }
    }

    func waitUntilSleepCount(_ expected: Int) async {
        while sleepCount < expected { await Task.yield() }
    }

    func fireAll() {
        let pending = sleepers
        sleepers.removeAll()
        pending.forEach { $0.resume() }
    }
}

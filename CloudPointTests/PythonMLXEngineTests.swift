import Foundation
import XCTest
@testable import CloudPoint

final class PythonMLXEngineTests: XCTestCase {
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

private actor FakeWorkerTransport: WorkerTransport {
    nonisolated private let stream: AsyncThrowingStream<WorkerProcessEvent, Error>
    private let continuation: AsyncThrowingStream<WorkerProcessEvent, Error>.Continuation
    private let failingCommand: String?
    private let acknowledgementOverride: [String: String]
    private(set) var commands: [String] = []
    private(set) var protocolReady = false

    init(
        failingCommand: String? = nil,
        acknowledgementOverride: [String: String] = [:]
    ) {
        let pair = AsyncThrowingStream.makeStream(of: WorkerProcessEvent.self)
        stream = pair.stream
        continuation = pair.continuation
        self.failingCommand = failingCommand
        self.acknowledgementOverride = acknowledgementOverride
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

    func markProtocolReady() async { protocolReady = true }

    func terminate() async { continuation.finish() }
}

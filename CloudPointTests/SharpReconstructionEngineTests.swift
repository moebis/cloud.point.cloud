import Foundation
import XCTest
@testable import CloudPoint

final class SharpReconstructionEngineTests: XCTestCase {
    func testEngineLaunchesIsolatedOneShotWorkerAndMapsProgressCompletion() async throws {
        let package = URL(filePath: "/tmp/Scene.cloudpoint", directoryHint: .isDirectory)
        let transport = SharpFakeTransport()
        let starter = SharpStarterRecorder(transport: transport)
        let installation = SharpModelInstallation.fixture()
        let engine = SharpReconstructionEngine(
            runtime: .unchecked(root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)),
            installation: installation,
            processStarter: starter.start
        )
        var iterator = engine.events().makeAsyncIterator()

        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: ProjectDescriptor(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000901")!,
            packageURL: package,
            modeID: .sharpGaussian
        ))
        try await engine.enqueue(PersistedFrame(
            index: 0,
            sourceTimestamp: 1.25,
            relativePath: "Frames/00000000.jpg"
        ))
        try await engine.finishInput()

        guard case let .ready(_, modelIdentifier, revision, digest) = try await iterator.next() else {
            return XCTFail("Expected ready event")
        }
        XCTAssertEqual(modelIdentifier, "apple/ml-sharp")
        XCTAssertEqual(revision, installation.sourceCommit)
        XCTAssertEqual(digest, installation.checkpointSHA256)

        await transport.emit(#"{"fraction":0.5,"protocolVersion":1,"stage":"inference","type":"progress"}"#)
        await transport.emit(#"{"device":"mps","durationSeconds":2.5,"gaussianCount":1179648,"plyRelativePath":"Outputs/Gaussians/00000000.ply","protocolVersion":1,"provenanceRelativePath":"Outputs/Gaussians/00000000.json","sourceFrameIndex":0,"type":"completed","usedCPUFallback":false}"#)
        await transport.exit(0)

        let progress = try await iterator.next()
        XCTAssertEqual(progress, .gaussianProgress(stage: .inference, fraction: 0.5))
        let completed = try await iterator.next()
        XCTAssertEqual(
            completed,
            .gaussianCompleted(SharpWorkerCompletion(
                sourceFrameIndex: 0,
                plyRelativePath: "Outputs/Gaussians/00000000.ply",
                provenanceRelativePath: "Outputs/Gaussians/00000000.json",
                gaussianCount: 1_179_648,
                durationSeconds: 2.5,
                device: "mps",
                usedCPUFallback: false
            ))
        )

        let launch = try XCTUnwrap(starter.launches.first)
        XCTAssertEqual(launch.executable.lastPathComponent, "python3.12")
        XCTAssertEqual(launch.arguments, [
            "-I", "-B", "-m", "cloudpoint_worker.sharp.cli",
            "--project", package.path,
            "--checkpoint", installation.checkpoint.path,
            "--checkpoint-sha256", installation.checkpointSHA256,
            "--source-commit", installation.sourceCommit,
            "--input-relative-path", "Frames/00000000.jpg",
            "--output-relative-path", "Outputs/Gaussians/00000000.ply",
            "--prefer-mps",
        ])
        XCTAssertNil(launch.environment["CLOUDPOINT_SENTINEL_SECRET"])
        XCTAssertEqual(launch.environment["PYTHONNOUSERSITE"], "1")
    }

    func testWorkerFailureTerminatesEventStreamWithStructuredError() async throws {
        let transport = SharpFakeTransport()
        let starter = SharpStarterRecorder(transport: transport)
        let engine = SharpReconstructionEngine(
            runtime: .unchecked(root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)),
            installation: .fixture(),
            processStarter: starter.start
        )
        var iterator = engine.events().makeAsyncIterator()
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: .sharpFixture())
        try await engine.enqueue(.sharpFixture())
        try await engine.finishInput()
        _ = try await iterator.next()

        await transport.emit(#"{"code":"SHARP_INFERENCE_FAILED","message":"bad frame","protocolVersion":1,"recoverable":true,"type":"failed"}"#)

        do {
            _ = try await iterator.next()
            XCTFail("Expected structured SHARP failure")
        } catch let error as ReconstructionEngineError {
            XCTAssertEqual(
                error,
                .workerFailure(
                    code: "SHARP_INFERENCE_FAILED",
                    message: "bad frame",
                    recoverable: true,
                    details: [:]
                )
            )
        }
    }

    func testUnexpectedWorkerExitPreservesBoundedDiagnostics() async throws {
        let transport = SharpFakeTransport()
        let engine = SharpReconstructionEngine(
            runtime: .unchecked(root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)),
            installation: .fixture(),
            processStarter: SharpStarterRecorder(transport: transport).start
        )
        var iterator = engine.events().makeAsyncIterator()
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: .sharpFixture())
        try await engine.enqueue(.sharpFixture())
        try await engine.finishInput()
        _ = try await iterator.next()

        await transport.exit(1, diagnostics: "checkpoint access denied")

        do {
            _ = try await iterator.next()
            XCTFail("Expected worker exit")
        } catch let error as SharpReconstructionEngineError {
            XCTAssertEqual(error, .workerExited(1, "checkpoint access denied"))
        }
    }

    func testCancelTerminatesWorkerAndEmitsCancelled() async throws {
        let transport = SharpFakeTransport()
        let starter = SharpStarterRecorder(transport: transport)
        let engine = SharpReconstructionEngine(
            runtime: .unchecked(root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)),
            installation: .fixture(),
            processStarter: starter.start
        )
        var iterator = engine.events().makeAsyncIterator()
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: .sharpFixture())
        try await engine.enqueue(.sharpFixture())
        try await engine.finishInput()
        _ = try await iterator.next()

        await engine.cancel()

        let terminationCount = await transport.terminationCount
        XCTAssertEqual(terminationCount, 1)
        let cancelled = try await iterator.next()
        XCTAssertEqual(cancelled, .cancelled(lastCompletedWindowIndex: nil))
    }

    func testRejectsWrongModeMultipleFramesAndUnsafeFramePath() async throws {
        let engine = SharpReconstructionEngine(
            runtime: .unchecked(root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)),
            installation: .fixture(),
            processStarter: SharpStarterRecorder(transport: SharpFakeTransport()).start
        )
        try await engine.prepare(configuration: EngineConfiguration())
        await XCTAssertThrowsErrorAsync {
            try await engine.begin(project: ProjectDescriptor(
                projectID: UUID(),
                packageURL: URL(filePath: "/tmp/Scene.cloudpoint", directoryHint: .isDirectory),
                modeID: .lingbotPointCloud
            ))
        }

        let second = SharpReconstructionEngine(
            runtime: .unchecked(root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)),
            installation: .fixture(),
            processStarter: SharpStarterRecorder(transport: SharpFakeTransport()).start
        )
        try await second.prepare(configuration: EngineConfiguration())
        try await second.begin(project: .sharpFixture())
        try await second.enqueue(.sharpFixture())
        await XCTAssertThrowsErrorAsync { try await second.enqueue(.sharpFixture()) }
    }
}

private struct SharpLaunch: Sendable {
    let executable: URL
    let arguments: [String]
    let environment: [String: String]
}

private final class SharpStarterRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let transport: SharpFakeTransport
    private var recorded: [SharpLaunch] = []

    init(transport: SharpFakeTransport) { self.transport = transport }

    func start(
        executable: URL,
        arguments: [String],
        environment: [String: String]
    ) async throws -> any SharpProcessTransport {
        lock.withLock {
            recorded.append(SharpLaunch(
                executable: executable,
                arguments: arguments,
                environment: environment
            ))
        }
        return transport
    }

    var launches: [SharpLaunch] { lock.withLock { recorded } }
}

private actor SharpFakeTransport: SharpProcessTransport {
    nonisolated let stream: AsyncThrowingStream<SharpProcessEvent, Error>
    private let continuation: AsyncThrowingStream<SharpProcessEvent, Error>.Continuation
    private(set) var terminationCount = 0

    init() {
        let pair = AsyncThrowingStream.makeStream(of: SharpProcessEvent.self)
        stream = pair.stream
        continuation = pair.continuation
    }

    nonisolated func events() -> AsyncThrowingStream<SharpProcessEvent, Error> { stream }

    func emit(_ line: String) { continuation.yield(.line(Data(line.utf8))) }

    func exit(_ status: Int32, diagnostics: String = "") {
        continuation.yield(.exited(status, diagnostics: diagnostics))
        continuation.finish()
    }

    func terminate() {
        terminationCount += 1
        continuation.finish()
    }
}

private extension SharpModelInstallation {
    static func fixture() -> SharpModelInstallation {
        let directory = URL(filePath: "/tmp/sharp-model", directoryHint: .isDirectory)
        return SharpModelInstallation(
            directory: directory,
            checkpoint: directory.appending(path: "sharp.pt"),
            checkpointSHA256: String(repeating: "a", count: 64),
            sourceCommit: "sharp-source"
        )
    }
}

private extension ProjectDescriptor {
    static func sharpFixture() -> ProjectDescriptor {
        ProjectDescriptor(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000902")!,
            packageURL: URL(filePath: "/tmp/Scene.cloudpoint", directoryHint: .isDirectory),
            modeID: .sharpGaussian
        )
    }
}

private extension PersistedFrame {
    static func sharpFixture() -> PersistedFrame {
        PersistedFrame(index: 0, sourceTimestamp: 0, relativePath: "Frames/00000000.jpg")
    }
}

private func XCTAssertThrowsErrorAsync(
    _ expression: () async throws -> Void,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        try await expression()
        XCTFail("Expected error", file: file, line: line)
    } catch {}
}

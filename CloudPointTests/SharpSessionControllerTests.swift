import Foundation
import XCTest
@testable import CloudPoint

final class SharpSessionControllerTests: XCTestCase {
    func testGaussianCompletionCommitsOutputAndCompletesProject() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let completion = try writeGaussianArtifacts(in: package.url)
        let manifest = ProjectManifest(
            reconstructionPlan: .sharp(SharpReconstructionConfiguration(inputFrameIndex: 0)),
            outputState: .gaussian(nil),
            frames: [frame],
            sessionState: SessionState(phase: .ready, capturedCount: 1)
        )
        try manifest.writeAtomically(to: package.url)
        let engine = SharpControllerEngine()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                cameraFactory: { _, _ in
                    throw SessionControllerError.cameraFailure(.configurationFailed)
                },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        _ = try await effects.next { $0.phase == .processing && $0.queuedCount == 1 }
        await engine.emit(.gaussianProgress(stage: .inference, fraction: 0.5))
        _ = try await effects.next { $0.setupText?.contains("50%") == true }
        await engine.emit(.gaussianCompleted(completion))
        let completed = try await effects.next { $0.phase == .completed }

        XCTAssertEqual(completed.processedCount, 1)
        XCTAssertFalse(completed.capabilities.canImportRecording)
        let durable = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(durable.sessionState.phase, .completed)
        XCTAssertEqual(durable.outputState, .gaussian(GaussianSceneOutput(
            sourceFrameIndex: 0,
            plyRelativePath: completion.plyRelativePath,
            provenanceRelativePath: completion.provenanceRelativePath,
            gaussianCount: 1,
            modelIdentifier: "apple/ml-sharp",
            modelRevision: String(repeating: "a", count: 40),
            checkpointSHA256: String(repeating: "b", count: 64),
            device: "mps",
            usedCPUFallback: false,
            durationSeconds: 1.1
        )))
    }

    func testInvalidGaussianArtifactFailsProjectWithoutCommittingOutput() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        var completion = try writeGaussianArtifacts(in: package.url, z: .nan)
        completion = SharpWorkerCompletion(
            sourceFrameIndex: completion.sourceFrameIndex,
            plyRelativePath: completion.plyRelativePath,
            provenanceRelativePath: completion.provenanceRelativePath,
            gaussianCount: completion.gaussianCount,
            durationSeconds: completion.durationSeconds,
            device: completion.device,
            usedCPUFallback: completion.usedCPUFallback
        )
        let manifest = ProjectManifest(
            reconstructionPlan: .sharp(SharpReconstructionConfiguration(inputFrameIndex: 0)),
            outputState: .gaussian(nil),
            frames: [frame],
            sessionState: SessionState(phase: .ready, capturedCount: 1)
        )
        try manifest.writeAtomically(to: package.url)
        let engine = SharpControllerEngine()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        _ = try await effects.next { $0.phase == .processing }
        await engine.emit(.gaussianCompleted(completion))
        _ = try await effects.next { $0.phase == .failed }

        let durable = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(durable.outputState, .gaussian(nil))
        XCTAssertEqual(durable.sessionState.processedCount, 0)
    }
}

private actor SharpControllerEngine: ReconstructionEngine {
    nonisolated let stream: AsyncThrowingStream<EngineEvent, Error>
    private let continuation: AsyncThrowingStream<EngineEvent, Error>.Continuation

    init() {
        let pair = AsyncThrowingStream.makeStream(of: EngineEvent.self)
        stream = pair.stream
        continuation = pair.continuation
    }

    func prepare(configuration: EngineConfiguration) throws {}

    func begin(project: ProjectDescriptor) throws {
        continuation.yield(.ready(
            engineVersion: "test-sharp",
            modelIdentifier: "apple/ml-sharp",
            modelRevision: String(repeating: "a", count: 40),
            convertedWeightsSHA256: String(repeating: "b", count: 64)
        ))
    }

    func enqueue(_ frame: PersistedFrame) throws {}
    func finishInput() throws {}
    func pause() throws {}
    func resume() throws {}
    func cancel() { continuation.finish() }
    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> { stream }
    func shutdown() { continuation.finish() }
    func emit(_ event: EngineEvent) { continuation.yield(event) }
}

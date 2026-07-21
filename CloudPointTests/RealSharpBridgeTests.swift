import Foundation
import XCTest
@testable import CloudPoint

final class RealSharpBridgeTests: XCTestCase {
    func testPackagedRuntimeReconstructsAndValidatesRealSharpSceneWhenRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let runtimePath = environment["CLOUDPOINT_WORKER_RUNTIME"],
              let checkpointPath = environment["CLOUDPOINT_REAL_SHARP_CHECKPOINT"],
              let framePath = environment["CLOUDPOINT_REAL_SHARP_FRAME"],
              !runtimePath.isEmpty, !checkpointPath.isEmpty, !framePath.isEmpty else {
            throw XCTSkip(
                "Set CLOUDPOINT_WORKER_RUNTIME, CLOUDPOINT_REAL_SHARP_CHECKPOINT, and CLOUDPOINT_REAL_SHARP_FRAME"
            )
        }
        let runtime = try WorkerRuntime.resolve(
            bundleValue: runtimePath,
            environment: [:],
            bundledResourcesURL: nil
        )
        let checkpoint = URL(filePath: checkpointPath)
        let package = try TemporaryProjectPackage.make()
        try FileManager.default.copyItem(
            at: URL(filePath: framePath),
            to: package.url.appending(path: "Frames/00000000.jpg")
        )
        let installation = SharpModelInstallation(
            directory: checkpoint.deletingLastPathComponent(),
            checkpoint: checkpoint,
            checkpointSHA256: SharpModelRelease.v1.checkpointSHA256,
            sourceCommit: SharpModelRelease.v1.sourceCommit
        )
        let engine = SharpReconstructionEngine(runtime: runtime, installation: installation)
        let completion = try await withThrowingTaskGroup(
            of: SharpWorkerCompletion.self
        ) { group in
            group.addTask {
                var result: SharpWorkerCompletion?
                for try await event in engine.events() {
                    if case let .gaussianCompleted(completion) = event { result = completion }
                }
                guard let result else { throw RealSharpBridgeError.missingCompletion }
                return result
            }
            group.addTask {
                try await ContinuousClock().sleep(for: .seconds(120))
                throw RealSharpBridgeError.timedOut
            }

            try await engine.prepare(configuration: EngineConfiguration())
            try await engine.begin(project: ProjectDescriptor(
                projectID: UUID(),
                packageURL: package.url,
                modeID: .sharpGaussian,
                sharpConfiguration: SharpReconstructionConfiguration(inputFrameIndex: 0)
            ))
            try await engine.enqueue(PersistedFrame(
                index: 0,
                sourceTimestamp: 0,
                relativePath: "Frames/00000000.jpg"
            ))
            try await engine.finishInput()

            let first = try await group.next()
            group.cancelAll()
            guard let first else { throw RealSharpBridgeError.missingCompletion }
            return first
        }

        XCTAssertEqual(completion.gaussianCount, 1_179_648)
        XCTAssertEqual(completion.device, "mps")
        XCTAssertFalse(completion.usedCPUFallback)
        let output = try ProductionGaussianArtifactValidator().validate(
            completion,
            in: package.url
        )
        XCTAssertEqual(output.checkpointSHA256, SharpModelRelease.v1.checkpointSHA256)
        XCTAssertEqual(output.modelRevision, SharpModelRelease.v1.sourceCommit)
    }
}

private enum RealSharpBridgeError: Error {
    case missingCompletion
    case timedOut
}

import Foundation
import XCTest
@testable import CloudPoint

final class RealWorkerBridgeTests: XCTestCase {
    func testPreparedLingbotWorkerCompletesNineFrameFixture() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let runtimePath = environment["CLOUDPOINT_WORKER_RUNTIME"],
              let modelPath = environment["CLOUDPOINT_REAL_MODEL_DIR"] else {
            throw XCTSkip(
                "Set CLOUDPOINT_WORKER_RUNTIME and CLOUDPOINT_REAL_MODEL_DIR for the real MLX bridge test"
            )
        }
        let runtime = try WorkerRuntime.resolve(
            bundleValue: runtimePath,
            environment: environment
        )
        let fixture = try RealWorkerBridgeFixture.make()
        let engine = try PythonMLXEngineFactory(runtime: runtime).makeEngine(
            modelDirectory: URL(filePath: modelPath, directoryHint: .isDirectory)
        )
        let eventTask = Task { () throws -> [EngineEvent] in
            var events: [EngineEvent] = []
            for try await event in engine.events() {
                events.append(event)
                if case .sessionCompleted = event { return events }
            }
            throw RealWorkerBridgeTestError.streamEndedBeforeCompletion
        }

        do {
            try await engine.prepare(configuration: fixture.configuration)
            try await engine.begin(project: fixture.project)
            for frame in fixture.frames { try await engine.enqueue(frame) }
            try await engine.finishInput()
            let events = try await eventTask.value

            XCTAssertEqual(events.compactMap(\.frameIndex).count, 18)
            XCTAssertEqual(events.reduce(into: 0) { count, event in
                if case .frameCompleted = event { count += 1 }
            }, 9)
            let windows = events.compactMap { event -> WindowResult? in
                if case let .windowCompleted(window) = event { window } else { nil }
            }
            XCTAssertEqual(windows.count, 1)
            let window = try XCTUnwrap(windows.first)
            let chunk = try PointChunk.open(
                url: fixture.packageURL.appending(path: window.pointChunkRelativePath)
            )
            XCTAssertEqual(chunk.firstFrame, 0)
            XCTAssertEqual(chunk.lastFrame, 8)
            XCTAssertGreaterThan(chunk.pointCount, 1_000)
        } catch {
            eventTask.cancel()
            await engine.shutdown()
            throw error
        }
        await engine.shutdown()
    }
}

private enum RealWorkerBridgeTestError: Error {
    case streamEndedBeforeCompletion
}

private final class RealWorkerBridgeFixture {
    let packageURL: URL
    let project: ProjectDescriptor
    let frames: [PersistedFrame]
    let configuration = EngineConfiguration()

    private init(
        packageURL: URL,
        project: ProjectDescriptor,
        frames: [PersistedFrame]
    ) {
        self.packageURL = packageURL
        self.project = project
        self.frames = frames
    }

    deinit { try? FileManager.default.removeItem(at: packageURL) }

    static func make() throws -> RealWorkerBridgeFixture {
        let fileManager = FileManager.default
        let packageURL = URL(
            filePath: "/private/tmp/cloudpoint-real-bridge-(UUID().uuidString.lowercased()).cloudpoint",
            directoryHint: .isDirectory
        )
        try fileManager.createDirectory(at: packageURL, withIntermediateDirectories: false)
        for directory in ["Frames", "Predictions", "Points", "Logs"] {
            try fileManager.createDirectory(
                at: packageURL.appending(path: directory),
                withIntermediateDirectories: false
            )
        }
        let repositoryRoot = URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceDirectory = repositoryRoot
            .appending(path: "worker/tests/fixtures/courthouse", directoryHint: .isDirectory)
        var frames: [PersistedFrame] = []
        for index in 0..<9 {
            let source = sourceDirectory.appending(path: String(format: "%06d.png", index))
            let relative = String(format: "Frames/%08d.png", index)
            try fileManager.copyItem(at: source, to: packageURL.appending(path: relative))
            frames.append(PersistedFrame(
                index: UInt32(index),
                sourceTimestamp: Double(index) / 2,
                relativePath: relative
            ))
        }
        let projectID = UUID()
        let manifest = ProjectManifest(
            projectID: projectID,
            frames: frames,
            sessionState: SessionState(
                phase: .processing,
                capturedCount: UInt64(frames.count),
                queuedCount: UInt64(frames.count)
            )
        )
        try manifest.writeAtomically(to: packageURL)
        return RealWorkerBridgeFixture(
            packageURL: packageURL,
            project: ProjectDescriptor(projectID: projectID, packageURL: packageURL),
            frames: frames
        )
    }
}

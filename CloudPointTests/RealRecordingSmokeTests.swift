import AVFoundation
import ImageIO
import XCTest
@testable import CloudPoint

final class RealRecordingSmokeTests: XCTestCase {
    func testConfiguredRealRecordingCompletesAValidatedSchemaV2Project() async throws {
        guard let path = ProcessInfo.processInfo.environment["CLOUDPOINT_SMOKE_VIDEO"],
              !path.isEmpty else {
            throw XCTSkip("Set CLOUDPOINT_SMOKE_VIDEO to run the portable real-recording smoke.")
        }
        let videoURL = URL(filePath: path)
        guard FileManager.default.isReadableFile(atPath: videoURL.path) else {
            XCTFail("CLOUDPOINT_SMOKE_VIDEO is not readable: \(videoURL.path)")
            return
        }

        let duration = try await AVURLAsset(url: videoURL).load(.duration)
        let expectedPlan = try FrameSamplingPlan(duration: duration, framesPerSecond: 1)
        let package = try TemporaryProjectPackage.make()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: ProjectManifest(),
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { MockReconstructionEngine() },
                manifestStore: AtomicManifestStore(),
                recordingImporter: AssetRecordingImporter(),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        _ = try await effects.next(timeout: .seconds(30)) {
            $0.phase == .ready && $0.capabilities.canImportRecording
        }
        try await controller.importRecording(videoURL, framesPerSecond: 1)
        let terminal = try await effects.next(timeout: .seconds(120)) {
            [.completed, .cancelled, .failed].contains($0.phase)
        }
        XCTAssertEqual(
            terminal.phase,
            .completed,
            terminal.errorText ?? "Recording did not reach terminal completion"
        )
        guard terminal.phase == .completed else { return }
        await controller.flush()

        let manifest = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(manifest.formatVersion, 2)
        XCTAssertEqual(manifest.formatVersion, ProjectManifest.currentFormatVersion)
        XCTAssertEqual(manifest.sessionState.phase, .completed)
        XCTAssertEqual(manifest.frames.count, expectedPlan.timestamps.count)
        XCTAssertFalse(manifest.frames.isEmpty)
        XCTAssertEqual(manifest.frames.map(\.index), Array(0..<UInt32(manifest.frames.count)))
        XCTAssertEqual(manifest.sessionState.capturedCount, UInt64(manifest.frames.count))
        XCTAssertEqual(manifest.sessionState.queuedCount, UInt64(manifest.frames.count))
        XCTAssertEqual(manifest.sessionState.processedCount, UInt64(manifest.frames.count))
        XCTAssertEqual(manifest.sessionState.failedCount, 0)

        let artifacts = manifest.completedWindows.flatMap(\.frameArtifacts)
        XCTAssertEqual(manifest.completedWindows.count, manifest.frames.count)
        XCTAssertEqual(artifacts.count, manifest.frames.count)
        XCTAssertEqual(artifacts.map(\.frameIndex), manifest.frames.map(\.index))

        let frameFiles = try regularFilePaths(in: package.url.appending(path: "Frames"))
        let predictionFiles = try regularFilePaths(in: package.url.appending(path: "Predictions"))
        let pointFiles = try regularFilePaths(in: package.url.appending(path: "Points"))
        XCTAssertEqual(
            Set(frameFiles.map { "Frames/\($0)" }),
            Set(manifest.frames.map(\.relativePath))
        )
        XCTAssertEqual(
            Set(predictionFiles.map { "Predictions/\($0)" }),
            Set(artifacts.flatMap {
                [$0.depthRelativePath, $0.confidenceRelativePath, $0.geometryRelativePath]
            })
        )
        XCTAssertEqual(
            Set(pointFiles.map { "Points/\($0)" }),
            Set(manifest.completedWindows.map(\.pointChunkRelativePath))
        )

        for frame in manifest.frames {
            try ProductionJPEGValidator().validate(frame, in: package.url)
            let source = try XCTUnwrap(
                CGImageSourceCreateWithURL(package.url.appending(path: frame.relativePath) as CFURL, nil)
            )
            XCTAssertEqual(CGImageSourceGetCount(source), 1)
        }
        for window in manifest.completedWindows {
            let chunk = try PointChunk.open(
                url: package.url.appending(path: window.pointChunkRelativePath)
            )
            XCTAssertEqual(chunk.firstFrame, window.frameStart)
            XCTAssertEqual(chunk.lastFrame, window.frameEnd)
            XCTAssertGreaterThan(chunk.pointCount, 0)
        }

        let allFiles = try regularFilePaths(in: package.url)
        XCTAssertTrue(allFiles.contains("Manifest.json"))
        XCTAssertEqual(
            allFiles.filter { $0.hasSuffix(".partial") },
            [],
            "A completed smoke project must not retain partial transaction files"
        )
    }

    private func regularFilePaths(in root: URL) throws -> Set<String> {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [],
            errorHandler: { _, _ in false }
        ) else {
            throw CocoaError(.fileReadUnknown)
        }

        let rootPath = root.standardizedFileURL.path
        var paths = Set<String>()
        for case let url as URL in enumerator {
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                continue
            }
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(rootPath + "/") else { throw CocoaError(.fileReadInvalidFileName) }
            paths.insert(String(path.dropFirst(rootPath.count + 1)))
        }
        return paths
    }
}

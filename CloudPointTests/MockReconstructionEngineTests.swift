import Foundation
import XCTest
@testable import CloudPoint

final class MockReconstructionEngineTests: XCTestCase {
    func testMockSeparatesFrameArtifactsFromWindowCPCAndWritesBeforeEvents() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.enqueue(.fixture(index: 7))
        try await engine.finishInput()

        var received: [EngineEvent] = []
        for try await event in events {
            received.append(event)
            switch event {
            case let .frameCompleted(artifacts):
                for path in [
                    artifacts.depthRelativePath,
                    artifacts.confidenceRelativePath,
                    artifacts.geometryRelativePath,
                ] {
                    XCTAssertTrue(FileManager.default.fileExists(atPath: package.url.appending(path: path).path))
                    XCTAssertFalse(path.hasSuffix(".cpc"))
                }
            case let .windowCompleted(result):
                XCTAssertTrue(FileManager.default.fileExists(
                    atPath: package.url.appending(path: result.pointChunkRelativePath).path
                ))
            default: break
            }
        }

        XCTAssertEqual(received.map(\.kind), ["ready", "frameStarted", "frameCompleted", "windowCompleted", "sessionCompleted"])
        guard case let .frameCompleted(artifacts) = received[2],
              case let .windowCompleted(window) = received[3] else {
            return XCTFail("Expected split completion events")
        }
        XCTAssertEqual(artifacts.frameIndex, 7)
        XCTAssertEqual(artifacts.windowIndex, 0)
        XCTAssertEqual(artifacts.depthRelativePath, "Predictions/00000007.depth-f16")
        XCTAssertEqual(window.pointChunkRelativePath, "Points/window-00000000.cpc")
        XCTAssertEqual(window.frameStart, 7)
        XCTAssertEqual(window.frameEnd, 7)

        let data = try Data(contentsOf: package.url.appending(path: window.pointChunkRelativePath))
        XCTAssertEqual(data.count, 32 + (64 * 64 * 24))
        XCTAssertEqual(Array(data.prefix(4)), Array("CPC1".utf8))
        XCTAssertEqual(Self.littleEndianUInt32(data, at: 16), 7)
        XCTAssertEqual(Self.littleEndianUInt32(data, at: 20), 7)
        XCTAssertEqual(Self.littleEndianUInt16(data, at: 48), Float16(2).bitPattern)
        XCTAssertEqual(Self.littleEndianUInt32(data, at: 52), 7)
    }

    func testPauseQueuesFramesUntilResumeAndPreservesGlobalWindowOrder() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.pause()
        try await engine.enqueue(.fixture(index: 8))
        try await engine.enqueue(.fixture(index: 12))
        try await engine.finishInput()
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: package.url.appending(path: "Points/window-00000000.cpc").path
        ))

        try await engine.resume()
        let received = try await Self.collect(events)
        XCTAssertEqual(received.compactMap(\.windowResult).map(\.windowIndex), [0, 1])
        XCTAssertEqual(received.compactMap(\.windowResult).map(\.frameStart), [8, 12])
        XCTAssertEqual(received.last, .sessionCompleted(processedFrames: 2, windowCount: 2, durationSeconds: 0))
    }

    func testCancelReportsLastCompletedWindowAndNeverCompletesSession() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()
        try await engine.enqueue(.fixture(index: 1))

        await engine.cancel()
        let received = try await Self.collect(events)
        XCTAssertEqual(received.last, .cancelled(lastCompletedWindowIndex: 0))
        XCTAssertFalse(received.contains { if case .sessionCompleted = $0 { true } else { false } })
    }

    func testFinishInputWithoutFramesCompletesWithZeroCounts() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()
        try await engine.finishInput()

        let received = try await Self.collect(events)
        XCTAssertEqual(received.map(\.kind), ["ready", "sessionCompleted"])
        XCTAssertEqual(received.last, .sessionCompleted(processedFrames: 0, windowCount: 0, durationSeconds: 0))
    }

    func testOutputDirectorySymlinkIsRejectedBeforeWritingOutsidePackage() async throws {
        let package = try TemporaryProjectPackage.make()
        let external = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: external) }
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        let points = package.url.appending(path: "Points")
        try FileManager.default.removeItem(at: points)
        try FileManager.default.createSymbolicLink(at: points, withDestinationURL: external)

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        await Self.assertReconstructionError(.unsafeOutputPath) {
            try await engine.begin(project: .fixture(packageURL: package.url))
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: external.path), [])
    }

    func testPackageRootSymlinkIsRejected() async throws {
        let package = try TemporaryProjectPackage.make()
        let link = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString)
            .appendingPathExtension("cloudpoint")
        defer { try? FileManager.default.removeItem(at: link) }
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: package.url)

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        await Self.assertReconstructionError(.unsafeOutputPath) {
            try await engine.begin(project: .fixture(packageURL: link))
        }
    }

    func testPreexistingCanonicalOutputSymlinkIsNeverFollowedOrClobbered() async throws {
        let package = try TemporaryProjectPackage.make()
        let external = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: external) }
        try Data("outside".utf8).write(to: external)
        let output = package.url.appending(path: "Predictions/00000014.depth-f16")
        try FileManager.default.createSymbolicLink(at: output, withDestinationURL: external)

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()
        await Self.assertReconstructionError(.outputAlreadyExists("00000014.depth-f16")) {
            try await engine.enqueue(.fixture(index: 14))
        }

        XCTAssertEqual(try Data(contentsOf: external), Data("outside".utf8))
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: output.path), external.path)
        let failure = await Self.collectFailure(events)
        XCTAssertNotNil(failure.error)
        XCTAssertFalse(failure.events.contains { if case .sessionCompleted = $0 { true } else { false } })
        await Self.assertReconstructionError(.invalidLifecycle(operation: "finishInput")) {
            try await engine.finishInput()
        }
        await Self.assertReconstructionError(.invalidLifecycle(operation: "enqueue")) {
            try await engine.enqueue(.fixture(index: 15))
        }
        await Self.assertReconstructionError(.invalidLifecycle(operation: "resume")) {
            try await engine.resume()
        }
    }

    func testMaximumUInt32FrameIndexUsesCanonicalUnsignedPathsAndVisibleConfidence() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.enqueue(.fixture(index: .max))
        try await engine.finishInput()
        let received = try await Self.collect(events)
        let artifacts = try XCTUnwrap(received.compactMap(\.frameArtifacts).first)
        let window = try XCTUnwrap(received.compactMap(\.windowResult).first)

        XCTAssertEqual(artifacts.depthRelativePath, "Predictions/4294967295.depth-f16")
        XCTAssertEqual(window.pointChunkRelativePath, "Points/window-00000000.cpc")
        let chunk = try PointChunk.open(url: package.url.appending(path: window.pointChunkRelativePath))
        XCTAssertGreaterThanOrEqual(try chunk.vertex(at: 0).confidence, Float(EngineConfiguration.fixture().confidenceThreshold))
    }

    func testResumeConsumesCommittedReplayWithoutEventsOrWritesThenUsesNextGlobalWindow() async throws {
        let package = try TemporaryProjectPackage.make()
        let projectID = UUID(uuidString: "00000000-0000-0000-0000-000000000007")!
        let frame = PersistedFrame.fixture(index: 7)
        let first = MockReconstructionEngine(clock: .immediate)
        try await first.prepare(configuration: .fixture())
        try await first.begin(project: ProjectDescriptor(projectID: projectID, packageURL: package.url))
        let firstEvents = first.events()
        try await first.enqueue(frame)
        try await first.finishInput()
        let firstReceived = try await Self.collect(firstEvents)
        let artifacts = try XCTUnwrap(firstReceived.compactMap(\.frameArtifacts).first)
        let result = try XCTUnwrap(firstReceived.compactMap(\.windowResult).first)
        var accumulator = PendingWindowAccumulator()
        try accumulator.add(artifacts)
        let completed = try accumulator.finalize(result, expectedFrameIndices: [7])
        let manifest = ProjectManifest(
            projectID: projectID,
            engineConfiguration: .fixture(),
            frames: [frame],
            completedWindows: [completed],
            sessionState: SessionState(
                phase: .processing,
                capturedCount: 1,
                queuedCount: 1,
                processedCount: 1
            )
        )
        try manifest.writeAtomically(to: package.url)
        let checkpoint = try XCTUnwrap(manifest.resumeCheckpoint())
        let originalBytes = try Self.outputBytes(package: package.url, window: completed)

        let resumed = MockReconstructionEngine(clock: .immediate)
        try await resumed.prepare(configuration: .fixture())
        try await resumed.begin(project: ProjectDescriptor(
            projectID: projectID,
            packageURL: package.url,
            resumeCheckpoint: checkpoint
        ))
        let resumedEvents = resumed.events()
        await Self.assertReconstructionError(.replayOrderViolation) {
            try await resumed.enqueue(.fixture(index: 6))
        }
        try await resumed.enqueue(frame)
        try await resumed.enqueue(.fixture(index: 11))
        try await resumed.finishInput()
        let received = try await Self.collect(resumedEvents)

        XCTAssertEqual(received.compactMap(\.startedFrameIndex), [11])
        XCTAssertEqual(received.compactMap(\.frameArtifacts).map(\.frameIndex), [11])
        XCTAssertEqual(received.compactMap(\.windowResult).map(\.windowIndex), [1])
        XCTAssertEqual(received.last, .sessionCompleted(processedFrames: 1, windowCount: 1, durationSeconds: 0))
        XCTAssertEqual(try Self.outputBytes(package: package.url, window: completed), originalBytes)
    }

    func testManifestSymlinkIsRejectedWithoutReadingOutsidePackage() async throws {
        let package = try TemporaryProjectPackage.make()
        let externalManifest = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: externalManifest) }
        try ProjectManifest.encode(.init()).write(to: externalManifest)
        try FileManager.default.createSymbolicLink(
            at: package.url.appending(path: "Manifest.json"),
            withDestinationURL: externalManifest
        )
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())

        await Self.assertReconstructionError(.unsafeOutputPath) {
            try await engine.begin(project: .fixture(packageURL: package.url))
        }
    }

    func testResumeCanEmitMaximumGlobalWindowOnceThenRejectsOverflow() async throws {
        let package = try TemporaryProjectPackage.make()
        let projectID = UUID(uuidString: "00000000-0000-0000-0000-000000000007")!
        let replayFrame = PersistedFrame.fixture(index: 5)
        let artifacts = FrameArtifacts(
            frameIndex: 5,
            windowIndex: UInt32.max - 1,
            depthRelativePath: WorkerArtifactPath.depth(frameIndex: 5),
            confidenceRelativePath: WorkerArtifactPath.confidence(frameIndex: 5),
            geometryRelativePath: WorkerArtifactPath.geometry(frameIndex: 5),
            durationSeconds: 0
        )
        let completed = CompletedWindow(
            index: UInt32.max - 1,
            inferenceFrameStart: 5,
            frameStart: 5,
            frameEnd: 5,
            pointChunkRelativePath: WorkerArtifactPath.points(windowIndex: UInt32.max - 1),
            alignmentRowMajor: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            lastProcessedFrameIndex: 5,
            inlierCount: 1,
            durationSeconds: 0,
            frameArtifacts: [artifacts]
        )
        for path in [
            artifacts.depthRelativePath,
            artifacts.confidenceRelativePath,
            artifacts.geometryRelativePath,
            completed.pointChunkRelativePath,
        ] {
            try Data("committed".utf8).write(to: package.url.appending(path: path))
        }
        let manifest = ProjectManifest(
            projectID: projectID,
            frames: [replayFrame],
            completedWindows: [completed],
            sessionState: SessionState(
                phase: .processing,
                capturedCount: 1,
                queuedCount: 1,
                processedCount: 1
            )
        )
        try manifest.writeAtomically(to: package.url)
        let checkpoint = try XCTUnwrap(manifest.resumeCheckpoint())
        XCTAssertEqual(checkpoint.nextWindowIndex, .max)

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: ProjectDescriptor(
            projectID: projectID,
            packageURL: package.url,
            resumeCheckpoint: checkpoint
        ))
        let events = engine.events()
        try await engine.enqueue(replayFrame)
        try await engine.pause()
        try await engine.enqueue(.fixture(index: 7))
        await Self.assertReconstructionError(.windowIndexOverflow) {
            try await engine.enqueue(.fixture(index: 9))
        }
        try await engine.resume()
        try await engine.finishInput()
        let received = try await Self.collect(events)

        XCTAssertEqual(received.compactMap(\.windowResult).map(\.windowIndex), [.max])
        XCTAssertEqual(
            received.compactMap(\.windowResult).first?.pointChunkRelativePath,
            "Points/window-4294967295.cpc"
        )
    }

    func testBeginRemovesOnlyExactOrphanOutputPatterns() async throws {
        let package = try TemporaryProjectPackage.make()
        let orphan = package.url.appending(path: "Predictions/00000003.depth-f16")
        let partial = package.url.appending(path: "Predictions/.00000004.depth-f16.aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.partial")
        let uppercasePartial = package.url.appending(path: "Predictions/.00000005.depth-f16.AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA.partial")
        let unknown = package.url.appending(path: "Predictions/notes.txt")
        for url in [orphan, partial, uppercasePartial, unknown] {
            try Data("keep-or-remove".utf8).write(to: url)
        }

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))

        XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: uppercasePartial.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unknown.path))
        await engine.shutdown()
    }

    func testSeparateBeginPassesEachScanFromDirectoryStart() async throws {
        let package = try TemporaryProjectPackage.make()
        let firstOrphan = package.url.appending(path: "Predictions/00000021.depth-f16")
        try Data("first".utf8).write(to: firstOrphan)
        let first = MockReconstructionEngine(clock: .immediate)
        try await first.prepare(configuration: .fixture())
        try await first.begin(project: .fixture(packageURL: package.url))
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstOrphan.path))
        await first.shutdown()

        let secondOrphan = package.url.appending(path: "Predictions/00000022.depth-f16")
        try Data("second".utf8).write(to: secondOrphan)
        let second = MockReconstructionEngine(clock: .immediate)
        try await second.prepare(configuration: .fixture())
        try await second.begin(project: .fixture(packageURL: package.url))
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondOrphan.path))
        await second.shutdown()
    }

    func testInvalidCheckpointDoesNotRunOrphanCleanupOrMutateFiles() async throws {
        let package = try TemporaryProjectPackage.make()
        let orphan = package.url.appending(path: "Predictions/00000031.depth-f16")
        let bytes = Data("must-remain".utf8)
        try bytes.write(to: orphan)
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())

        await Self.assertReconstructionError(.invalidResumeCheckpoint) {
            try await engine.begin(project: ProjectDescriptor(
                projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000007")!,
                packageURL: package.url,
                resumeCheckpoint: ResumeCheckpoint(
                    lastCommittedFrameIndex: 9,
                    replayFromFrameIndex: 8,
                    nextWindowIndex: 1
                )
            ))
        }

        XCTAssertEqual(try Data(contentsOf: orphan), bytes)
    }

    func testInvalidLifecycleMatrixRejectsOperationsWithoutSideEffects() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        await Self.assertReconstructionError(.invalidLifecycle(operation: "begin")) {
            try await engine.begin(project: .fixture(packageURL: package.url))
        }
        await Self.assertReconstructionError(.invalidLifecycle(operation: "enqueue")) {
            try await engine.enqueue(.fixture(index: 1))
        }
        await Self.assertReconstructionError(.invalidLifecycle(operation: "finishInput")) {
            try await engine.finishInput()
        }
        await Self.assertReconstructionError(.invalidLifecycle(operation: "pause")) {
            try await engine.pause()
        }
        await Self.assertReconstructionError(.invalidLifecycle(operation: "resume")) {
            try await engine.resume()
        }

        try await engine.prepare(configuration: .fixture())
        await Self.assertReconstructionError(.invalidLifecycle(operation: "prepare")) {
            try await engine.prepare(configuration: .fixture())
        }
        try await engine.begin(project: .fixture(packageURL: package.url))
        try await engine.finishInput()
        await Self.assertReconstructionError(.invalidLifecycle(operation: "enqueue")) {
            try await engine.enqueue(.fixture(index: 2))
        }
    }

    func testInvalidConfigurationFailsStreamOnceAndRejectsFurtherCalls() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        let events = engine.events()
        var invalid = EngineConfiguration.fixture()
        invalid.voxelSize = 0

        do {
            try await engine.prepare(configuration: invalid)
            XCTFail("Expected invalid configuration")
        } catch let error as EngineConfigurationError {
            XCTAssertEqual(error, .invalidVoxelSize(0))
        }
        let failure = await Self.collectFailure(events)
        XCTAssertNotNil(failure.error)
        XCTAssertEqual(failure.events.count, 0)
        await Self.assertReconstructionError(.invalidLifecycle(operation: "begin")) {
            try await engine.begin(project: .fixture(packageURL: package.url))
        }
    }

    func testNonDirectoryOutputPathFailsBeginAndClosesStream() async throws {
        let package = try TemporaryProjectPackage.make()
        let predictions = package.url.appending(path: "Predictions")
        try FileManager.default.removeItem(at: predictions)
        try Data("not-a-directory".utf8).write(to: predictions)
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        let events = engine.events()

        await Self.assertReconstructionError(.unsafeOutputPath) {
            try await engine.begin(project: .fixture(packageURL: package.url))
        }
        let failure = await Self.collectFailure(events)
        XCTAssertNotNil(failure.error)
        XCTAssertFalse(failure.events.contains { if case .sessionCompleted = $0 { true } else { false } })
    }

    private static func outputBytes(package: URL, window: CompletedWindow) throws -> [String: Data] {
        let paths = [window.pointChunkRelativePath] + window.frameArtifacts.flatMap {
            [$0.depthRelativePath, $0.confidenceRelativePath, $0.geometryRelativePath]
        }
        return try Dictionary(uniqueKeysWithValues: paths.map { ($0, try Data(contentsOf: package.appending(path: $0))) })
    }

    private static func littleEndianUInt16(_ data: Data, at offset: Int) -> UInt16 {
        data[offset..<(offset + 2)].enumerated().reduce(0) { value, byte in
            value | (UInt16(byte.element) << (byte.offset * 8))
        }
    }

    private static func littleEndianUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data[offset..<(offset + 4)].enumerated().reduce(0) { value, byte in
            value | (UInt32(byte.element) << (byte.offset * 8))
        }
    }

    private static func collect(_ stream: AsyncThrowingStream<EngineEvent, Error>) async throws -> [EngineEvent] {
        var events: [EngineEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }

    private static func collectFailure(
        _ stream: AsyncThrowingStream<EngineEvent, Error>
    ) async -> (events: [EngineEvent], error: Error?) {
        var events: [EngineEvent] = []
        do {
            for try await event in stream { events.append(event) }
            return (events, nil)
        } catch {
            return (events, error)
        }
    }

    private static func assertReconstructionError(
        _ expected: ReconstructionEngineError,
        _ expression: () async throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            try await expression()
            XCTFail("Expected \(expected)", file: file, line: line)
        } catch let error as ReconstructionEngineError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("Unexpected error: \(error)", file: file, line: line)
        }
    }
}

private extension EngineConfiguration {
    static func fixture() -> EngineConfiguration { EngineConfiguration() }
}

private extension ProjectDescriptor {
    static func fixture(packageURL: URL) -> ProjectDescriptor {
        ProjectDescriptor(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000007")!,
            packageURL: packageURL
        )
    }
}

private extension PersistedFrame {
    static func fixture(index: UInt32) -> PersistedFrame {
        PersistedFrame(
            index: index,
            sourceTimestamp: Double(index) / 5,
            relativePath: String(format: "Frames/%08u.jpg", index)
        )
    }
}

private extension EngineEvent {
    var kind: String {
        switch self {
        case .ready: "ready"
        case .modelProgress: "modelProgress"
        case .frameStarted: "frameStarted"
        case .frameCompleted: "frameCompleted"
        case .windowCompleted: "windowCompleted"
        case .sessionCompleted: "sessionCompleted"
        case .paused: "paused"
        case .cancelled: "cancelled"
        case .warning: "warning"
        case .heartbeat: "heartbeat"
        }
    }

    var frameArtifacts: FrameArtifacts? {
        guard case let .frameCompleted(value) = self else { return nil }
        return value
    }

    var windowResult: WindowResult? {
        guard case let .windowCompleted(value) = self else { return nil }
        return value
    }

    var startedFrameIndex: UInt32? {
        guard case let .frameStarted(frameIndex, _) = self else { return nil }
        return frameIndex
    }
}

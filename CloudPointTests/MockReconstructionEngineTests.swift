import Foundation
import XCTest
@testable import CloudPoint

final class MockReconstructionEngineTests: XCTestCase {
    func testMockEmitsOneDeterministicChunkPerFrameThenCompletes() async throws {
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
        }

        let result = try XCTUnwrap(received.compactMap { event -> FrameResult? in
            guard case let .frameCompleted(result) = event else { return nil }
            return result
        }.first)
        XCTAssertEqual(result.frameIndex, 7)
        XCTAssertEqual(result.pointChunkPath, "Points/frame-00000007.cpc")
        XCTAssertEqual(result.pointCount, 64 * 64)
        XCTAssertTrue(received.contains(.sessionCompleted))

        let chunkURL = package.url.appending(path: result.pointChunkPath)
        let data = try Data(contentsOf: chunkURL)
        XCTAssertEqual(data.count, 32 + (64 * 64 * 24))
        XCTAssertEqual(Array(data.prefix(4)), Array("CPC1".utf8))
        XCTAssertEqual(Self.littleEndianUInt16(data, at: 4), 1)
        XCTAssertEqual(Self.littleEndianUInt16(data, at: 6), 24)
        XCTAssertEqual(Self.littleEndianUInt64(data, at: 8), 64 * 64)
        XCTAssertEqual(Self.littleEndianUInt32(data, at: 16), 7)
        XCTAssertEqual(Self.littleEndianUInt32(data, at: 20), 7)
        XCTAssertEqual(Array(data[24..<32]), Array(repeating: 0, count: 8))
        XCTAssertEqual(Self.float32(data, at: 32), -1.26, accuracy: 0.000_001)
        XCTAssertEqual(Self.float32(data, at: 36), -1.26, accuracy: 0.000_001)
        XCTAssertEqual(Self.float32(data, at: 40), 0.07, accuracy: 0.000_001)
        XCTAssertEqual(Array(data[44..<48]), [115, 167, 71, 255])
        XCTAssertEqual(Self.littleEndianUInt16(data, at: 48), Float16(2).bitPattern)
        XCTAssertEqual(Self.littleEndianUInt16(data, at: 50), 0)
        XCTAssertEqual(Self.littleEndianUInt32(data, at: 52), 7)
    }

    func testPauseQueuesFramesUntilResumeAndPreservesTheirOrder() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.pause()
        try await engine.enqueue(.fixture(index: 8))
        try await engine.enqueue(.fixture(index: 9))
        try await engine.finishInput()

        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: package.url.appending(path: "Points/frame-00000008.cpc").path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: package.url.appending(path: "Points/frame-00000009.cpc").path
            )
        )

        try await engine.resume()
        let received = try await Self.collect(events)

        XCTAssertEqual(
            received.compactMap { event -> Int? in
                guard case let .frameCompleted(result) = event else { return nil }
                return result.frameIndex
            },
            [8, 9]
        )
        XCTAssertEqual(received.last, .sessionCompleted)
    }

    func testCancelEmitsCancelledThenFinishesTheStream() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        await engine.cancel()
        let received = try await Self.collect(events)

        XCTAssertEqual(received.last, .cancelled)
        XCTAssertFalse(received.contains(.sessionCompleted))
    }

    func testInvalidLifecycleCallsThrowTheOperationThatWasRejected() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)

        await Self.assertLifecycleError("begin") {
            try await engine.begin(project: .fixture(packageURL: package.url))
        }
        await Self.assertLifecycleError("enqueue") {
            try await engine.enqueue(.fixture(index: 1))
        }

        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        try await engine.finishInput()

        await Self.assertLifecycleError("enqueue") {
            try await engine.enqueue(.fixture(index: 2))
        }
    }

    func testFinishInputWithoutFramesCompletesTheStream() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.finishInput()

        let received = try await Self.collect(events)
        XCTAssertEqual(received, [.ready, .sessionCompleted])
    }

    func testWriteFailureFailsTheStreamAndRejectsFurtherLifecycleCalls() async throws {
        let package = try TemporaryProjectPackage.make()
        let pointsURL = package.url.appending(path: "Points")
        try FileManager.default.removeItem(at: pointsURL)
        XCTAssertTrue(FileManager.default.createFile(atPath: pointsURL.path, contents: Data()))

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.pause()
        try await engine.enqueue(.fixture(index: 12))
        var writeFailed = false
        do {
            try await engine.resume()
            XCTFail("Expected the Points file to reject the CPC write")
        } catch {
            writeFailed = true
        }
        if writeFailed {
            // The unfixed engine still accepts finishInput; close that stream so the
            // behavioral assertion below can demonstrate the defect deterministically.
            try? await engine.finishInput()
        }

        let streamFailure = await Self.collectFailure(events)
        XCTAssertNotNil(streamFailure.error)
        XCTAssertFalse(streamFailure.events.contains(.sessionCompleted))

        await Self.assertLifecycleError("finishInput") {
            try await engine.finishInput()
        }
        await Self.assertLifecycleError("enqueue") {
            try await engine.enqueue(.fixture(index: 13))
        }
        await Self.assertLifecycleError("resume") {
            try await engine.resume()
        }
    }

    func testPointsSymlinkToExternalDirectoryIsRejectedWithoutWritingOutsidePackage() async throws {
        let package = try TemporaryProjectPackage.make()
        let externalDirectory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: externalDirectory) }
        try FileManager.default.createDirectory(at: externalDirectory, withIntermediateDirectories: true)

        let pointsURL = package.url.appending(path: "Points")
        try FileManager.default.removeItem(at: pointsURL)
        try FileManager.default.createSymbolicLink(at: pointsURL, withDestinationURL: externalDirectory)

        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        var rejected = false
        do {
            try await engine.enqueue(.fixture(index: 14))
        } catch {
            rejected = true
        }
        if !rejected {
            try await engine.finishInput()
        }

        XCTAssertTrue(rejected, "Expected a package Points symlink to be rejected")
        let streamFailure = await Self.collectFailure(events)
        XCTAssertNotNil(streamFailure.error)
        XCTAssertFalse(streamFailure.events.contains(.sessionCompleted))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: externalDirectory.appending(path: "frame-00000014.cpc").path
            )
        )
    }

    func testMaximumUInt32FrameIndexUsesAnUnsignedPackageRelativeFilename() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()
        let maximumFrameIndex = Int(UInt32.max)

        try await engine.enqueue(.fixture(index: maximumFrameIndex))
        try await engine.finishInput()

        let received = try await Self.collect(events)
        let result = try XCTUnwrap(received.compactMap { event -> FrameResult? in
            guard case let .frameCompleted(result) = event else { return nil }
            return result
        }.first)
        XCTAssertEqual(result.pointChunkPath, "Points/frame-4294967295.cpc")
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: package.url.appending(path: result.pointChunkPath).path
            )
        )
    }

    func testMockChunkConfidenceIsVisibleAtTheDefaultRendererThreshold() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: .fixture())
        try await engine.begin(project: .fixture(packageURL: package.url))
        let events = engine.events()

        try await engine.enqueue(.fixture(index: 15))
        try await engine.finishInput()

        let received = try await Self.collect(events)
        let result = try XCTUnwrap(received.compactMap { event -> FrameResult? in
            guard case let .frameCompleted(result) = event else { return nil }
            return result
        }.first)
        let data = try Data(contentsOf: package.url.appending(path: result.pointChunkPath))
        let confidence = Float(Float16(bitPattern: Self.littleEndianUInt16(data, at: 48)))

        XCTAssertGreaterThanOrEqual(confidence, EngineConfiguration.fixture().confidenceThreshold)
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

    private static func littleEndianUInt64(_ data: Data, at offset: Int) -> UInt64 {
        data[offset..<(offset + 8)].enumerated().reduce(0) { value, byte in
            value | (UInt64(byte.element) << (byte.offset * 8))
        }
    }

    private static func float32(_ data: Data, at offset: Int) -> Float {
        Float(bitPattern: littleEndianUInt32(data, at: offset))
    }

    private static func collect(
        _ stream: AsyncThrowingStream<EngineEvent, Error>
    ) async throws -> [EngineEvent] {
        var events: [EngineEvent] = []
        for try await event in stream {
            events.append(event)
        }
        return events
    }

    private static func collectFailure(
        _ stream: AsyncThrowingStream<EngineEvent, Error>
    ) async -> (events: [EngineEvent], error: Error?) {
        var events: [EngineEvent] = []
        do {
            for try await event in stream {
                events.append(event)
            }
            return (events, nil)
        } catch {
            return (events, error)
        }
    }

    private static func assertLifecycleError(
        _ operation: String,
        _ action: () async throws -> Void
    ) async {
        do {
            try await action()
            XCTFail("Expected \(operation) to be rejected")
        } catch let error as ReconstructionEngineError {
            XCTAssertEqual(error, .invalidLifecycle(operation: operation))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}

private extension EngineConfiguration {
    static func fixture() -> EngineConfiguration {
        EngineConfiguration(windowSize: 32, windowOverlap: 8, confidenceThreshold: 1.5)
    }
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
    static func fixture(index: Int) -> PersistedFrame {
        PersistedFrame(
            index: index,
            sourceTimestamp: Double(index) / 5,
            relativePath: "Frames/\(String(format: "%08d", index)).jpg"
        )
    }
}

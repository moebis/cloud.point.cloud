import Foundation
import XCTest
@testable import CloudPoint

final class PointChunkTests: XCTestCase {
    func testValidPointExposesEveryCPCFieldWithoutNarrowingSourceFrame() throws {
        let sourceFrame = UInt32.max
        let url = try writeFixture(
            points: [
                .init(
                    x: 1.25,
                    y: -2.5,
                    z: 3.75,
                    rgba: [11, 22, 33, 44],
                    confidence: 2.5,
                    flags: 0xA55A,
                    sourceFrame: sourceFrame
                )
            ],
            firstFrame: sourceFrame,
            lastFrame: sourceFrame
        )

        let chunk = try PointChunk.open(url: url)
        let point = try chunk.vertex(at: 0)

        XCTAssertEqual(chunk.pointCount, 1)
        XCTAssertEqual(chunk.firstFrame, sourceFrame)
        XCTAssertEqual(chunk.lastFrame, sourceFrame)
        XCTAssertEqual(point.position, SIMD3<Float>(1.25, -2.5, 3.75))
        XCTAssertEqual(point.rgba, SIMD4<UInt8>(11, 22, 33, 44))
        XCTAssertEqual(point.confidence, 2.5)
        XCTAssertEqual(point.flags, 0xA55A)
        XCTAssertEqual(point.sourceFrame, UInt32.max)
        XCTAssertEqual(try chunk.withVertexBytes { $0.count }, 24)
    }

    func testRejectsBadMagic() throws {
        var data = fixtureData(points: [.fixture])
        data.replaceSubrange(0..<4, with: Array("NOPE".utf8))
        XCTAssertThrowsError(try PointChunk.open(url: write(data)))
    }

    func testRejectsUnsupportedVersion() throws {
        var data = fixtureData(points: [.fixture])
        replaceLittleEndian(UInt16(2), in: &data, at: 4)
        XCTAssertThrowsError(try PointChunk.open(url: write(data)))
    }

    func testRejectsWrongVertexStride() throws {
        var data = fixtureData(points: [.fixture])
        replaceLittleEndian(UInt16(23), in: &data, at: 6)
        XCTAssertThrowsError(try PointChunk.open(url: write(data)))
    }

    func testRejectsDeclaredPointCountThatDoesNotMatchFileSize() throws {
        var data = fixtureData(points: [.fixture])
        replaceLittleEndian(UInt64(2), in: &data, at: 8)
        XCTAssertThrowsError(try PointChunk.open(url: write(data)))
    }

    func testRejectsTrailingBytes() throws {
        var data = fixtureData(points: [.fixture])
        data.append(0)
        XCTAssertThrowsError(try PointChunk.open(url: write(data)))
    }

    func testRejectsCountBeyondConfiguredLimit() throws {
        let url = try writeFixture(points: [.fixture, .fixture])
        XCTAssertThrowsError(
            try PointChunk.open(
                url: url,
                limits: PointChunk.Limits(maxPointCount: 1, maxFileSize: 80)
            )
        )
    }

    func testRejectsHeaderArithmeticOverflowBeforeReadingPayload() throws {
        var data = Data(repeating: 0, count: 32)
        data.replaceSubrange(0..<4, with: Array("CPC1".utf8))
        replaceLittleEndian(UInt16(1), in: &data, at: 4)
        replaceLittleEndian(UInt16(24), in: &data, at: 6)
        replaceLittleEndian(UInt64.max, in: &data, at: 8)

        XCTAssertThrowsError(
            try PointChunk.open(
                url: write(data),
                limits: PointChunk.Limits(maxPointCount: UInt64.max, maxFileSize: UInt64.max)
            )
        ) { error in
            XCTAssertEqual(error as? PointChunkError, .arithmeticOverflow)
        }
    }

    func testRejectsNonFinitePosition() throws {
        var point = PointFixture.fixture
        point.x = .nan
        XCTAssertThrowsError(try PointChunk.open(url: writeFixture(points: [point])))
    }

    func testRejectsNonFiniteFloat16Confidence() throws {
        var point = PointFixture.fixture
        point.confidence = .infinity
        XCTAssertThrowsError(try PointChunk.open(url: writeFixture(points: [point])))
    }

    func testRejectsReversedInclusiveFrameRange() throws {
        XCTAssertThrowsError(
            try PointChunk.open(
                url: writeFixture(points: [.fixture], firstFrame: 9, lastFrame: 8)
            )
        )
    }

    func testRejectsNonZeroReservedHeaderBytes() throws {
        var data = fixtureData(points: [.fixture])
        data[31] = 1
        XCTAssertThrowsError(try PointChunk.open(url: write(data)))
    }

    func testRejectsPointSourceFrameOutsideDeclaredRange() throws {
        var point = PointFixture.fixture
        point.sourceFrame = 10
        XCTAssertThrowsError(
            try PointChunk.open(
                url: writeFixture(points: [point], firstFrame: 4, lastFrame: 9)
            )
        )
    }

    func testOpensChunkProducedByMockEngine() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = MockReconstructionEngine(clock: .immediate)
        try await engine.prepare(configuration: EngineConfiguration())
        try await engine.begin(project: ProjectDescriptor(projectID: UUID(), packageURL: package.url))
        let events = engine.events()

        try await engine.enqueue(
            PersistedFrame(index: 17, sourceTimestamp: 1, relativePath: "Frames/00000017.jpg")
        )
        try await engine.finishInput()

        var result: WindowResult?
        for try await event in events {
            if case let .windowCompleted(value) = event {
                result = value
            }
        }

        let completed = try XCTUnwrap(result)
        let chunk = try PointChunk.open(url: package.url.appending(path: completed.pointChunkRelativePath))
        XCTAssertEqual(chunk.pointCount, 64 * 64)
        XCTAssertEqual(try chunk.vertex(at: 0).sourceFrame, 17)
        XCTAssertGreaterThanOrEqual(
            try chunk.vertex(at: 0).confidence,
            Float(EngineConfiguration().confidenceThreshold)
        )
    }
}

private struct PointFixture {
    var x: Float
    var y: Float
    var z: Float
    var rgba: [UInt8]
    var confidence: Float16
    var flags: UInt16
    var sourceFrame: UInt32

    static let fixture = PointFixture(
        x: 1,
        y: 2,
        z: 3,
        rgba: [10, 20, 30, 255],
        confidence: 2,
        flags: 0,
        sourceFrame: 4
    )
}

private extension PointChunkTests {
    func writeFixture(
        points: [PointFixture],
        firstFrame: UInt32 = 4,
        lastFrame: UInt32 = 4
    ) throws -> URL {
        try write(fixtureData(points: points, firstFrame: firstFrame, lastFrame: lastFrame))
    }

    func fixtureData(
        points: [PointFixture],
        firstFrame: UInt32 = 4,
        lastFrame: UInt32 = 4
    ) -> Data {
        var data = Data()
        data.append(contentsOf: "CPC1".utf8)
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(24))
        data.appendLittleEndian(UInt64(points.count))
        data.appendLittleEndian(firstFrame)
        data.appendLittleEndian(lastFrame)
        data.append(contentsOf: repeatElement(0, count: 8))
        for point in points {
            data.appendLittleEndian(point.x.bitPattern)
            data.appendLittleEndian(point.y.bitPattern)
            data.appendLittleEndian(point.z.bitPattern)
            data.append(contentsOf: point.rgba)
            data.appendLittleEndian(point.confidence.bitPattern)
            data.appendLittleEndian(point.flags)
            data.appendLittleEndian(point.sourceFrame)
        }
        return data
    }

    func write(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "point-chunk-\(UUID().uuidString).cpc")
        try data.write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func replaceLittleEndian<T: FixedWidthInteger>(_ value: T, in data: inout Data, at offset: Int) {
        let bytes = Swift.withUnsafeBytes(of: value.littleEndian, Array.init)
        data.replaceSubrange(offset..<(offset + bytes.count), with: bytes)
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        append(contentsOf: Swift.withUnsafeBytes(of: value.littleEndian, Array.init))
    }
}

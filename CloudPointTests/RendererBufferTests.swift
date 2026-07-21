import Foundation
import Metal
import simd
import XCTest
@testable import CloudPoint

@MainActor
final class RendererBufferTests: XCTestCase {
    func testSwiftAndMetalContractStridesStayExact() {
        XCTAssertEqual(PointChunk.vertexStride, 24)
        XCTAssertEqual(PointCloudRenderer.uniformBufferStride, 80)
        XCTAssertEqual(PointCloudRenderer.maximumDisplayPointCount, 5_000_000)
    }

    func testAppendTracksFullAndDisplayedCountsAndPerChunkDrawRanges() throws {
        let renderer = try makeRenderer(displayLimit: 100)
        let first = try makeChunk(points: [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(2, 0, 0),
        ])
        let second = try makeChunk(points: [
            SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(1, 1, 0),
        ], sourceFrame: 2)

        try renderer.append(first)
        try renderer.append(second)

        XCTAssertEqual(renderer.fullPointCount, 5)
        XCTAssertEqual(renderer.displayedPointCount, 5)
        XCTAssertEqual(renderer.displayedSourceIndices, [0, 1, 2, 3, 4])
        XCTAssertEqual(
            renderer.drawRanges,
            [
                PointDrawRange(chunkIndex: 0, range: 0..<3),
                PointDrawRange(chunkIndex: 1, range: 3..<5),
            ]
        )
    }

    func testSpatialCompactionIsDeterministicBoundedAndNotPrefixSelection() throws {
        var points: [SIMD3<Float>] = []
        for z in 0..<4 {
            for y in 0..<4 {
                for x in 0..<4 {
                    points.append(SIMD3(Float(x), Float(y), Float(z)))
                }
            }
        }
        let chunk = try makeChunk(points: points)
        let first = try makeRenderer(displayLimit: 10)
        let second = try makeRenderer(displayLimit: 10)

        try first.append(chunk)
        try second.append(chunk)

        XCTAssertEqual(first.displayedSourceIndices, second.displayedSourceIndices)
        XCTAssertLessThanOrEqual(first.displayedPointCount, 10)
        XCTAssertGreaterThan(first.displayedPointCount, 1)
        XCTAssertGreaterThan(first.displayedSourceIndices.max() ?? 0, 9)
    }

    func testBufferCapacityGrowsGeometricallyAndNeverPastDisplayLimit() throws {
        let renderer = try makeRenderer(displayLimit: 10)
        try renderer.append(try makeChunk(points: Array(repeating: .zero, count: 3)))
        let firstCapacity = renderer.bufferCapacity

        try renderer.append(try makeChunk(points: Array(repeating: SIMD3<Float>(1, 1, 1), count: 4)))

        XCTAssertEqual(firstCapacity, 4)
        XCTAssertEqual(renderer.bufferCapacity, 8)
        XCTAssertLessThanOrEqual(renderer.bufferCapacity, 10)
    }

    func testCameraControlsAndExtremeInputKeepFiniteMatrices() throws {
        let renderer = try makeRenderer(displayLimit: 10)

        renderer.orbit(deltaX: .greatestFiniteMagnitude, deltaY: -.greatestFiniteMagnitude)
        renderer.pan(deltaX: .greatestFiniteMagnitude, deltaY: -.greatestFiniteMagnitude)
        renderer.zoom(by: .greatestFiniteMagnitude)
        renderer.setPointSize(.infinity)
        renderer.setConfidenceThreshold(.nan)

        XCTAssertTrue(renderer.viewProjectionMatrix(aspectRatio: 16.0 / 9.0).allFinite)
        XCTAssertTrue(renderer.pointSize.isFinite)
        XCTAssertTrue(renderer.confidenceThreshold.isFinite)

        renderer.resetCamera()
        XCTAssertEqual(renderer.cameraState, .default)
    }

    func testDefaultViewProjectsOpenCVCoordinatesRightSideUpAndForward() throws {
        let renderer = try makeRenderer(displayLimit: 10)
        let matrix = renderer.viewProjectionMatrix(aspectRatio: 1)

        let center = try projectedNDC(SIMD3<Float>(0, 0, 1), matrix: matrix)
        let imageRight = try projectedNDC(SIMD3<Float>(1, 0, 1), matrix: matrix)
        let imageDown = try projectedNDC(SIMD3<Float>(0, 1, 1), matrix: matrix)
        let fartherForward = try projectedNDC(SIMD3<Float>(0, 0, 2), matrix: matrix)

        XCTAssertGreaterThan(imageRight.x, center.x)
        XCTAssertLessThan(imageDown.y, center.y)
        XCTAssertGreaterThan(fartherForward.z, center.z)
        XCTAssertTrue((0...1).contains(center.z))

        renderer.orbit(deltaX: 100, deltaY: -50)
        renderer.resetCamera()
        let resetCenter = try projectedNDC(
            SIMD3<Float>(0, 0, 1),
            matrix: renderer.viewProjectionMatrix(aspectRatio: 1)
        )
        XCTAssertEqual(resetCenter.x, center.x, accuracy: 0.000_001)
        XCTAssertEqual(resetCenter.y, center.y, accuracy: 0.000_001)
        XCTAssertEqual(resetCenter.z, center.z, accuracy: 0.000_001)
    }

    func testLargeTargetCloseZoomObliqueOrbitAndSubnormalAspectRemainFinite() throws {
        let renderer = try makeRenderer(displayLimit: 10)

        renderer.zoom(by: 10_000)
        renderer.orbit(deltaX: .pi / 0.005 / 4, deltaY: 0)
        for _ in 0..<15 {
            renderer.pan(deltaX: 10_000, deltaY: 10_000)
        }
        renderer.zoom(by: -10_000)
        renderer.orbit(
            deltaX: 0,
            deltaY: asin(1 / sqrt(Float(3))) / 0.005
        )

        XCTAssertEqual(renderer.cameraState.distance, 0.05)
        XCTAssertEqual(renderer.cameraState.target.x, -1_000_000)
        XCTAssertEqual(renderer.cameraState.target.y, 1_000_000)
        XCTAssertEqual(renderer.cameraState.target.z, 1_000_000)
        XCTAssertTrue(renderer.viewProjectionMatrix(aspectRatio: 1).allFinite)
        XCTAssertTrue(
            renderer.viewProjectionMatrix(aspectRatio: .leastNonzeroMagnitude).allFinite
        )
    }

    func testSpatialCompactionHandlesExtremeFiniteCoordinatesWithoutCollapsing() throws {
        let chunk = try makeChunk(points: [
            SIMD3<Float>(-.greatestFiniteMagnitude, 0, 0),
            .zero,
            SIMD3<Float>(.greatestFiniteMagnitude, 0, 0),
        ])
        let first = try makeRenderer(displayLimit: 2)
        let second = try makeRenderer(displayLimit: 2)

        try first.append(chunk)
        try second.append(chunk)

        XCTAssertEqual(first.displayedPointCount, 2)
        XCTAssertEqual(first.displayedSourceIndices, second.displayedSourceIndices)
        XCTAssertEqual(first.displayedSourceIndices, [0, 1])
    }

    func testPointCloudViewUpdateRebindsGestureRendererAndDevice() throws {
        let firstRenderer = try makeRenderer(displayLimit: 10)
        let replacementRenderer = try makeRenderer(displayLimit: 10)
        let view = InteractivePointCloudView(frame: .zero, device: firstRenderer.device)
        view.pointRenderer = firstRenderer
        view.delegate = firstRenderer
        view.device = nil

        PointCloudView.configure(view, for: replacementRenderer)

        XCTAssertTrue(view.pointRenderer === replacementRenderer)
        XCTAssertTrue((view.delegate as AnyObject?) === replacementRenderer)
        XCTAssertEqual(view.device?.registryID, replacementRenderer.device.registryID)
    }

    private func makeRenderer(displayLimit: Int) throws -> PointCloudRenderer {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        return try PointCloudRenderer(
            device: device,
            displayLimit: displayLimit,
            libraryBundle: Bundle(for: PointCloudRenderer.self)
        )
    }

    private func projectedNDC(
        _ point: SIMD3<Float>,
        matrix: simd_float4x4
    ) throws -> SIMD3<Float> {
        let clip = matrix * SIMD4<Float>(point, 1)
        XCTAssertGreaterThan(clip.w, 0)
        return try XCTUnwrap(
            clip.w.isFinite && clip.w > 0
                ? SIMD3<Float>(clip.x, clip.y, clip.z) / clip.w
                : nil
        )
    }

    private func makeChunk(
        points: [SIMD3<Float>],
        sourceFrame: UInt32 = 1
    ) throws -> PointChunk {
        var data = Data()
        data.append(contentsOf: "CPC1".utf8)
        data.appendInteger(UInt16(1))
        data.appendInteger(UInt16(24))
        data.appendInteger(UInt64(points.count))
        data.appendInteger(sourceFrame)
        data.appendInteger(sourceFrame)
        data.append(contentsOf: repeatElement(0, count: 8))
        for (index, point) in points.enumerated() {
            data.appendInteger(point.x.bitPattern)
            data.appendInteger(point.y.bitPattern)
            data.appendInteger(point.z.bitPattern)
            data.append(contentsOf: [UInt8(truncatingIfNeeded: index), 20, 30, 255])
            data.appendInteger(Float16(2).bitPattern)
            data.appendInteger(UInt16(0))
            data.appendInteger(sourceFrame)
        }

        let url = FileManager.default.temporaryDirectory
            .appending(path: "renderer-chunk-\(UUID().uuidString).cpc")
        try data.write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return try PointChunk.open(url: url)
    }
}

private extension Data {
    mutating func appendInteger<T: FixedWidthInteger>(_ value: T) {
        append(contentsOf: Swift.withUnsafeBytes(of: value.littleEndian, Array.init))
    }
}

private extension simd_float4x4 {
    var allFinite: Bool {
        columns.0.allFinite && columns.1.allFinite && columns.2.allFinite && columns.3.allFinite
    }
}

private extension SIMD4 where Scalar == Float {
    var allFinite: Bool {
        x.isFinite && y.isFinite && z.isFinite && w.isFinite
    }
}

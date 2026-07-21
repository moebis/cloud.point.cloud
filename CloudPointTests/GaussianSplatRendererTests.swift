import Metal
import XCTest
import simd
@testable import CloudPoint

final class GaussianSplatRendererTests: XCTestCase {
    @MainActor
    func testOpenCVDisplayBasisKeepsRightAndFlipsDownForward() {
        let basis = GaussianSplatRenderer.openCVToMetal
        XCTAssertEqual(basis * SIMD4<Float>(1, 0, 0, 1), SIMD4<Float>(1, 0, 0, 1))
        XCTAssertEqual(basis * SIMD4<Float>(0, 1, 0, 1), SIMD4<Float>(0, -1, 0, 1))
        XCTAssertEqual(basis * SIMD4<Float>(0, 0, 1, 1), SIMD4<Float>(0, 0, -1, 1))
    }

    @MainActor
    func testMirroringKeepsAnOffAxisSceneCentered() {
        let openCVCenter = SIMD3<Float>(2, 1, 4)
        let metalCenter = SIMD3<Float>(2, -1, -4)
        let view = GaussianSplatRenderer.modelViewMatrix(
            target: metalCenter,
            distance: 3,
            yaw: 0,
            pitch: 0,
            panOffset: .zero,
            mirrorDisplay: true
        )

        let centered = view * SIMD4<Float>(openCVCenter, 1)

        XCTAssertEqual(centered.x, 0, accuracy: 0.0001)
        XCTAssertEqual(centered.y, 0, accuracy: 0.0001)
        XCTAssertEqual(centered.z, -3, accuracy: 0.0001)
    }

    @MainActor
    func testLoadsSharpCompatiblePLYIntoNativeMetalRenderer() async throws {
        let package = try TemporaryProjectPackage.make()
        let completion = try writeGaussianArtifacts(in: package.url)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is unavailable")
        }
        let renderer = try GaussianSplatRenderer(device: device)
        renderer.load(package.url.appending(path: completion.plyRelativePath))

        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(20)
        while clock.now < deadline {
            switch renderer.state {
            case let .ready(count):
                XCTAssertEqual(count, 1)
                return
            case let .failed(message):
                XCTFail(message)
                return
            case .empty, .loading:
                try await clock.sleep(for: .milliseconds(25))
            }
        }
        XCTFail("Timed out loading synthetic SHARP PLY")
    }

    @MainActor
    func testLoadsRealSharpOutputWhenRequested() async throws {
        guard let path = ProcessInfo.processInfo.environment["CLOUDPOINT_REAL_SHARP_PLY"],
              !path.isEmpty else {
            throw XCTSkip("Set CLOUDPOINT_REAL_SHARP_PLY for the real-checkpoint viewer gate")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is unavailable")
        }
        let renderer = try GaussianSplatRenderer(device: device)
        renderer.load(URL(filePath: path))

        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(120)
        while clock.now < deadline {
            switch renderer.state {
            case let .ready(count):
                XCTAssertEqual(count, 1_179_648)
                return
            case let .failed(message):
                XCTFail(message)
                return
            case .empty, .loading:
                try await clock.sleep(for: .milliseconds(100))
            }
        }
        XCTFail("Timed out loading the real SHARP PLY")
    }
}

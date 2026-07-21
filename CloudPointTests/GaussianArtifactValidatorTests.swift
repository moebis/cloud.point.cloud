import Foundation
import XCTest
@testable import CloudPoint

final class GaussianArtifactValidatorTests: XCTestCase {
    func testValidatesExactSharpPLYAndProvenance() throws {
        let package = try TemporaryProjectPackage.make()
        let completion = try writeGaussianArtifacts(in: package.url)

        let output = try ProductionGaussianArtifactValidator().validate(
            completion,
            in: package.url
        )

        XCTAssertEqual(output.sourceFrameIndex, 0)
        XCTAssertEqual(output.gaussianCount, 1)
        XCTAssertEqual(output.modelIdentifier, "apple/ml-sharp")
        XCTAssertEqual(output.modelRevision, String(repeating: "a", count: 40))
        XCTAssertEqual(output.checkpointSHA256, String(repeating: "b", count: 64))
        XCTAssertEqual(output.device, "mps")
    }

    func testRejectsNonFiniteVertexAndMismatchedProvenance() throws {
        let package = try TemporaryProjectPackage.make()
        let completion = try writeGaussianArtifacts(in: package.url, z: .nan)
        XCTAssertThrowsError(
            try ProductionGaussianArtifactValidator().validate(completion, in: package.url)
        )

        _ = try writeGaussianArtifacts(in: package.url, provenanceCount: 2)
        XCTAssertThrowsError(
            try ProductionGaussianArtifactValidator().validate(completion, in: package.url)
        )
    }

    func testRejectsSymbolicLinkArtifact() throws {
        let package = try TemporaryProjectPackage.make()
        let completion = try writeGaussianArtifacts(in: package.url)
        let ply = package.url.appending(path: completion.plyRelativePath)
        let realPLY = package.url.appending(path: "real.ply")
        try FileManager.default.moveItem(at: ply, to: realPLY)
        try FileManager.default.createSymbolicLink(at: ply, withDestinationURL: realPLY)

        XCTAssertThrowsError(
            try ProductionGaussianArtifactValidator().validate(completion, in: package.url)
        )
    }
}

@discardableResult
func writeGaussianArtifacts(
    in packageURL: URL,
    z: Float = 2,
    provenanceCount: UInt64 = 1
) throws -> SharpWorkerCompletion {
    let plyPath = "Outputs/Gaussians/00000000.ply"
    let provenancePath = "Outputs/Gaussians/00000000.json"
    let header = """
    ply
    format binary_little_endian 1.0
    element vertex 1
    property float x
    property float y
    property float z
    property float f_dc_0
    property float f_dc_1
    property float f_dc_2
    property float opacity
    property float scale_0
    property float scale_1
    property float scale_2
    property float rot_0
    property float rot_1
    property float rot_2
    property float rot_3
    element extrinsic 16
    property float extrinsic
    element intrinsic 9
    property float intrinsic
    element image_size 2
    property uint image_size
    element frame 2
    property int frame
    element disparity 2
    property float disparity
    element color_space 1
    property uchar color_space
    element version 3
    property uchar version
    end_header
    """ + "\n"
    var ply = Data(header.utf8)
    for value in [Float(0), 0, z, 0.1, 0.2, 0.3, 0, -1, -1, -1, 1, 0, 0, 0] {
        ply.appendLittleEndian(value.bitPattern)
    }
    for _ in 0..<16 { ply.appendLittleEndian(Float(0).bitPattern) }
    for _ in 0..<9 { ply.appendLittleEndian(Float(1).bitPattern) }
    ply.appendLittleEndian(UInt32(640))
    ply.appendLittleEndian(UInt32(480))
    ply.appendLittleEndian(Int32(1))
    ply.appendLittleEndian(Int32(1))
    ply.appendLittleEndian(Float(0.1).bitPattern)
    ply.appendLittleEndian(Float(1).bitPattern)
    ply.append(0)
    ply.append(contentsOf: [1, 5, 0])
    try ply.write(to: packageURL.appending(path: plyPath))

    let provenance: [String: Any] = [
        "schemaVersion": 1,
        "modelIdentifier": "apple/ml-sharp",
        "sourceCommit": String(repeating: "a", count: 40),
        "checkpointSHA256": String(repeating: "b", count: 64),
        "sourceFrameIndex": 0,
        "inputRelativePath": "Frames/00000000.jpg",
        "plyRelativePath": plyPath,
        "gaussianCount": provenanceCount,
        "device": "mps",
        "usedCPUFallback": false,
        "focalLengthPixels": 500,
        "imageWidth": 640,
        "imageHeight": 480,
        "durationSeconds": 1.0,
        "generatedAt": "2026-07-21T20:00:00+00:00",
    ]
    try JSONSerialization.data(withJSONObject: provenance, options: [.sortedKeys])
        .write(to: packageURL.appending(path: provenancePath))
    return SharpWorkerCompletion(
        sourceFrameIndex: 0,
        plyRelativePath: plyPath,
        provenanceRelativePath: provenancePath,
        gaussianCount: 1,
        durationSeconds: 1.1,
        device: "mps",
        usedCPUFallback: false
    )
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}

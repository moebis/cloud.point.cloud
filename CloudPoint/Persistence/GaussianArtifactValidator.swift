import Foundation

protocol GaussianArtifactValidating: Sendable {
    func validate(
        _ completion: SharpWorkerCompletion,
        in packageURL: URL
    ) throws -> GaussianSceneOutput
}

enum GaussianArtifactValidationError: Error, Sendable, Equatable {
    case invalidPath
    case invalidProvenance
    case invalidPLY
}

struct ProductionGaussianArtifactValidator: GaussianArtifactValidating {
    private struct Provenance: Decodable {
        let schemaVersion: Int
        let modelIdentifier: String
        let sourceCommit: String
        let checkpointSHA256: String
        let sourceFrameIndex: UInt32
        let inputRelativePath: String
        let plyRelativePath: String
        let gaussianCount: UInt64
        let device: String
        let usedCPUFallback: Bool
        let focalLengthPixels: Double
        let imageWidth: UInt32
        let imageHeight: UInt32
        let durationSeconds: Double
        let generatedAt: String
    }

    func validate(
        _ completion: SharpWorkerCompletion,
        in packageURL: URL
    ) throws -> GaussianSceneOutput {
        let expectedPLY = String(
            format: "Outputs/Gaussians/%08u.ply",
            completion.sourceFrameIndex
        )
        let expectedProvenance = String(
            format: "Outputs/Gaussians/%08u.json",
            completion.sourceFrameIndex
        )
        guard completion.plyRelativePath == expectedPLY,
              completion.provenanceRelativePath == expectedProvenance,
              ProjectRelativePath.isSafe(expectedPLY),
              ProjectRelativePath.isSafe(expectedProvenance) else {
            throw GaussianArtifactValidationError.invalidPath
        }

        let plyURL = packageURL.appending(path: expectedPLY)
        let provenanceURL = packageURL.appending(path: expectedProvenance)
        try requireRegularFile(plyURL, error: .invalidPLY)
        try requireRegularFile(provenanceURL, error: .invalidProvenance)

        let provenanceData = try Data(contentsOf: provenanceURL, options: .mappedIfSafe)
        guard provenanceData.count <= 64 * 1_024,
              let object = try? JSONSerialization.jsonObject(with: provenanceData),
              let dictionary = object as? [String: Any],
              Set(dictionary.keys) == Set([
                "schemaVersion", "modelIdentifier", "sourceCommit",
                "checkpointSHA256", "sourceFrameIndex", "inputRelativePath",
                "plyRelativePath", "gaussianCount", "device", "usedCPUFallback",
                "focalLengthPixels", "imageWidth", "imageHeight", "durationSeconds",
                "generatedAt",
              ]),
              let provenance = try? JSONDecoder().decode(Provenance.self, from: provenanceData),
              provenance.schemaVersion == 1,
              provenance.modelIdentifier == "apple/ml-sharp",
              validLowerHex(provenance.sourceCommit, length: 40),
              validLowerHex(provenance.checkpointSHA256, length: 64),
              provenance.sourceFrameIndex == completion.sourceFrameIndex,
              provenance.inputRelativePath == String(
                format: "Frames/%08u.jpg",
                completion.sourceFrameIndex
              ),
              provenance.plyRelativePath == completion.plyRelativePath,
              provenance.gaussianCount == completion.gaussianCount,
              provenance.device == completion.device,
              provenance.usedCPUFallback == completion.usedCPUFallback,
              provenance.focalLengthPixels.isFinite,
              provenance.focalLengthPixels > 0,
              provenance.imageWidth > 0,
              provenance.imageHeight > 0,
              provenance.durationSeconds.isFinite,
              provenance.durationSeconds >= 0,
              completion.durationSeconds.isFinite,
              completion.durationSeconds >= provenance.durationSeconds,
              !provenance.generatedAt.isEmpty else {
            throw GaussianArtifactValidationError.invalidProvenance
        }

        try validatePLY(plyURL, gaussianCount: completion.gaussianCount)
        return GaussianSceneOutput(
            sourceFrameIndex: completion.sourceFrameIndex,
            plyRelativePath: completion.plyRelativePath,
            provenanceRelativePath: completion.provenanceRelativePath,
            gaussianCount: completion.gaussianCount,
            modelIdentifier: provenance.modelIdentifier,
            modelRevision: provenance.sourceCommit,
            checkpointSHA256: provenance.checkpointSHA256,
            device: completion.device,
            usedCPUFallback: completion.usedCPUFallback,
            durationSeconds: completion.durationSeconds
        )
    }

    private func validatePLY(_ url: URL, gaussianCount: UInt64) throws {
        guard gaussianCount > 0, let handle = try? FileHandle(forReadingFrom: url) else {
            throw GaussianArtifactValidationError.invalidPLY
        }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 64 * 1_024),
              let marker = prefix.range(of: Data("end_header\n".utf8)) else {
            throw GaussianArtifactValidationError.invalidPLY
        }
        let headerEnd = marker.upperBound
        let headerData = prefix[..<headerEnd]
        guard let header = String(data: headerData, encoding: .ascii) else {
            throw GaussianArtifactValidationError.invalidPLY
        }
        let vertexLines = [
            "element vertex \(gaussianCount)",
            "property float x", "property float y", "property float z",
            "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
            "property float opacity", "property float scale_0", "property float scale_1",
            "property float scale_2", "property float rot_0", "property float rot_1",
            "property float rot_2", "property float rot_3",
        ]
        let requiredMetadata = [
            "element extrinsic 16", "property float extrinsic",
            "element intrinsic 9", "property float intrinsic",
            "element image_size 2", "property uint image_size",
            "element frame 2", "property int frame",
            "element disparity 2", "property float disparity",
            "element color_space 1", "property uchar color_space",
            "element version 3", "property uchar version",
        ]
        let lines = header.split(separator: "\n").map(String.init)
        guard lines.first == "ply",
              lines.dropFirst().first == "format binary_little_endian 1.0",
              containsOrdered(vertexLines, in: lines),
              requiredMetadata.allSatisfy(lines.contains) else {
            throw GaussianArtifactValidationError.invalidPLY
        }

        let vertexStride: UInt64 = 14 * 4
        let metadataBytes: UInt64 = 16 * 4 + 9 * 4 + 2 * 4 + 2 * 4 + 2 * 4 + 1 + 3
        let (vertexBytes, overflow) = gaussianCount.multipliedReportingOverflow(by: vertexStride)
        let (payloadBytes, payloadOverflow) = vertexBytes.addingReportingOverflow(metadataBytes)
        let (expectedSize, sizeOverflow) = UInt64(headerEnd)
            .addingReportingOverflow(payloadBytes)
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard !overflow, !payloadOverflow, !sizeOverflow,
              (attributes[.size] as? NSNumber)?.uint64Value == expectedSize else {
            throw GaussianArtifactValidationError.invalidPLY
        }

        try handle.seek(toOffset: UInt64(headerEnd))
        let recordsPerChunk: UInt64 = 4_096
        var remaining = gaussianCount
        while remaining > 0 {
            let recordCount = min(remaining, recordsPerChunk)
            let byteCount = Int(recordCount * vertexStride)
            guard let data = try handle.read(upToCount: byteCount), data.count == byteCount else {
                throw GaussianArtifactValidationError.invalidPLY
            }
            try data.withUnsafeBytes { bytes in
                for record in 0..<Int(recordCount) {
                    let base = record * Int(vertexStride)
                    for component in 0..<14 {
                        let offset = base + component * 4
                        let byte0 = UInt32(bytes[offset])
                        let byte1 = UInt32(bytes[offset + 1]) << 8
                        let byte2 = UInt32(bytes[offset + 2]) << 16
                        let byte3 = UInt32(bytes[offset + 3]) << 24
                        let bits = byte0 | byte1 | byte2 | byte3
                        let value = Float(bitPattern: bits)
                        guard value.isFinite, component != 2 || value > 0 else {
                            throw GaussianArtifactValidationError.invalidPLY
                        }
                    }
                }
            }
            remaining -= recordCount
        }
    }

    private func requireRegularFile(
        _ url: URL,
        error: GaussianArtifactValidationError
    ) throws {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else { throw error }
    }

    private func validLowerHex(_ value: String, length: Int) -> Bool {
        value.count == length && value.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    private func containsOrdered(_ expected: [String], in actual: [String]) -> Bool {
        guard let first = actual.firstIndex(of: expected[0]),
              actual.count >= first + expected.count else { return false }
        return Array(actual[first..<(first + expected.count)]) == expected
    }
}

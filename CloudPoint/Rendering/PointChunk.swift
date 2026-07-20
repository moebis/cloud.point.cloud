import Foundation

enum PointChunkError: Error, Equatable {
    case notARegularFile
    case fileTooSmall(actual: UInt64)
    case fileSizeExceedsLimit(actual: UInt64, limit: UInt64)
    case invalidMagic
    case unsupportedVersion(UInt16)
    case invalidVertexStride(UInt16)
    case pointCountExceedsLimit(actual: UInt64, limit: UInt64)
    case arithmeticOverflow
    case fileSizeMismatch(expected: UInt64, actual: UInt64)
    case invalidFrameRange(first: UInt32, last: UInt32)
    case nonZeroReservedHeader
    case nonFinitePosition(pointIndex: Int)
    case nonFiniteConfidence(pointIndex: Int)
    case sourceFrameOutOfRange(pointIndex: Int, sourceFrame: UInt32)
    case vertexIndexOutOfBounds(Int)
}

struct PointVertex: Sendable, Equatable {
    let position: SIMD3<Float>
    let rgba: SIMD4<UInt8>
    let confidence: Float
    let flags: UInt16
    let sourceFrame: UInt32
}

/// A validated, read-only CPC1 mapping.
///
/// The on-disk record remains packed at 24 bytes. Callers can inspect the mapped
/// payload only inside `withVertexBytes`, keeping the `Data` owner alive for the
/// entire lifetime of the exposed pointer.
final class PointChunk: @unchecked Sendable {
    struct Limits: Sendable, Equatable {
        static let `default` = Limits(
            maxPointCount: 50_000_000,
            maxFileSize: 1_200_000_032
        )

        let maxPointCount: UInt64
        let maxFileSize: UInt64

        init(maxPointCount: UInt64, maxFileSize: UInt64) {
            self.maxPointCount = maxPointCount
            self.maxFileSize = maxFileSize
        }
    }

    static let headerSize = 32
    static let vertexStride = 24

    let url: URL
    let pointCount: Int
    let firstFrame: UInt32
    let lastFrame: UInt32

    private let mappedData: Data

    private init(
        url: URL,
        mappedData: Data,
        pointCount: Int,
        firstFrame: UInt32,
        lastFrame: UInt32
    ) {
        self.url = url
        self.mappedData = mappedData
        self.pointCount = pointCount
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
    }

    static func open(url: URL, limits: Limits = .default) throws -> PointChunk {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true else {
            throw PointChunkError.notARegularFile
        }
        guard let filesystemSize = values.fileSize, filesystemSize >= 0 else {
            throw PointChunkError.notARegularFile
        }
        let actualSize = UInt64(filesystemSize)
        guard actualSize <= limits.maxFileSize else {
            throw PointChunkError.fileSizeExceedsLimit(actual: actualSize, limit: limits.maxFileSize)
        }
        guard actualSize >= UInt64(headerSize) else {
            throw PointChunkError.fileTooSmall(actual: actualSize)
        }

        let mappedData = try Data(contentsOf: url, options: .alwaysMapped)
        guard UInt64(mappedData.count) == actualSize else {
            throw PointChunkError.fileSizeMismatch(
                expected: actualSize,
                actual: UInt64(mappedData.count)
            )
        }

        let header = try mappedData.withUnsafeBytes { bytes -> Header in
            guard bytes.count >= headerSize else {
                throw PointChunkError.fileTooSmall(actual: UInt64(bytes.count))
            }
            guard bytes[0] == 0x43, bytes[1] == 0x50, bytes[2] == 0x43, bytes[3] == 0x31 else {
                throw PointChunkError.invalidMagic
            }

            let version: UInt16 = bytes.littleEndian(at: 4)
            guard version == 1 else {
                throw PointChunkError.unsupportedVersion(version)
            }
            let stride: UInt16 = bytes.littleEndian(at: 6)
            guard stride == UInt16(vertexStride) else {
                throw PointChunkError.invalidVertexStride(stride)
            }

            let pointCount: UInt64 = bytes.littleEndian(at: 8)
            guard pointCount <= limits.maxPointCount else {
                throw PointChunkError.pointCountExceedsLimit(
                    actual: pointCount,
                    limit: limits.maxPointCount
                )
            }
            let firstFrame: UInt32 = bytes.littleEndian(at: 16)
            let lastFrame: UInt32 = bytes.littleEndian(at: 20)
            guard firstFrame <= lastFrame else {
                throw PointChunkError.invalidFrameRange(first: firstFrame, last: lastFrame)
            }
            guard bytes[24..<32].allSatisfy({ $0 == 0 }) else {
                throw PointChunkError.nonZeroReservedHeader
            }
            return Header(pointCount: pointCount, firstFrame: firstFrame, lastFrame: lastFrame)
        }

        let (payloadSize, payloadOverflow) = header.pointCount.multipliedReportingOverflow(
            by: UInt64(vertexStride)
        )
        guard !payloadOverflow else {
            throw PointChunkError.arithmeticOverflow
        }
        let (expectedSize, sizeOverflow) = UInt64(headerSize).addingReportingOverflow(payloadSize)
        guard !sizeOverflow else {
            throw PointChunkError.arithmeticOverflow
        }
        guard expectedSize <= limits.maxFileSize else {
            throw PointChunkError.fileSizeExceedsLimit(actual: expectedSize, limit: limits.maxFileSize)
        }
        guard expectedSize == actualSize else {
            throw PointChunkError.fileSizeMismatch(expected: expectedSize, actual: actualSize)
        }
        guard let pointCount = Int(exactly: header.pointCount) else {
            throw PointChunkError.arithmeticOverflow
        }

        try validateVertices(
            in: mappedData,
            pointCount: pointCount,
            firstFrame: header.firstFrame,
            lastFrame: header.lastFrame
        )

        return PointChunk(
            url: url,
            mappedData: mappedData,
            pointCount: pointCount,
            firstFrame: header.firstFrame,
            lastFrame: header.lastFrame
        )
    }

    func withVertexBytes<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        try mappedData.withUnsafeBytes { bytes in
            let start = bytes.baseAddress?.advanced(by: Self.headerSize)
            return try body(
                UnsafeRawBufferPointer(
                    start: start,
                    count: pointCount * Self.vertexStride
                )
            )
        }
    }

    func vertex(at index: Int) throws -> PointVertex {
        guard index >= 0, index < pointCount else {
            throw PointChunkError.vertexIndexOutOfBounds(index)
        }
        return withVertexBytes { bytes in
            Self.decodeVertex(from: bytes, pointIndex: index)
        }
    }

    func forEachVertex(
        _ body: (_ index: Int, _ vertex: PointVertex) throws -> Void
    ) rethrows {
        try withVertexBytes { bytes in
            for index in 0..<pointCount {
                try body(index, Self.decodeVertex(from: bytes, pointIndex: index))
            }
        }
    }

    private struct Header {
        let pointCount: UInt64
        let firstFrame: UInt32
        let lastFrame: UInt32
    }

    private static func validateVertices(
        in data: Data,
        pointCount: Int,
        firstFrame: UInt32,
        lastFrame: UInt32
    ) throws {
        let validationBatchSize = 65_536
        try data.withUnsafeBytes { fileBytes in
            let vertexBytes = UnsafeRawBufferPointer(
                start: fileBytes.baseAddress?.advanced(by: headerSize),
                count: pointCount * vertexStride
            )
            var batchStart = 0
            while batchStart < pointCount {
                let batchEnd = min(pointCount, batchStart + validationBatchSize)
                for pointIndex in batchStart..<batchEnd {
                    let byteOffset = pointIndex * vertexStride
                    let x = Float(bitPattern: vertexBytes.littleEndian(at: byteOffset))
                    let y = Float(bitPattern: vertexBytes.littleEndian(at: byteOffset + 4))
                    let z = Float(bitPattern: vertexBytes.littleEndian(at: byteOffset + 8))
                    guard x.isFinite, y.isFinite, z.isFinite else {
                        throw PointChunkError.nonFinitePosition(pointIndex: pointIndex)
                    }

                    let confidenceBits: UInt16 = vertexBytes.littleEndian(at: byteOffset + 16)
                    guard Float16(bitPattern: confidenceBits).isFinite else {
                        throw PointChunkError.nonFiniteConfidence(pointIndex: pointIndex)
                    }

                    let sourceFrame: UInt32 = vertexBytes.littleEndian(at: byteOffset + 20)
                    guard sourceFrame >= firstFrame, sourceFrame <= lastFrame else {
                        throw PointChunkError.sourceFrameOutOfRange(
                            pointIndex: pointIndex,
                            sourceFrame: sourceFrame
                        )
                    }
                }
                batchStart = batchEnd
            }
        }
    }

    private static func decodeVertex(
        from bytes: UnsafeRawBufferPointer,
        pointIndex: Int
    ) -> PointVertex {
        let offset = pointIndex * vertexStride
        let x = Float(bitPattern: bytes.littleEndian(at: offset))
        let y = Float(bitPattern: bytes.littleEndian(at: offset + 4))
        let z = Float(bitPattern: bytes.littleEndian(at: offset + 8))
        let rgba = SIMD4<UInt8>(
            bytes[offset + 12],
            bytes[offset + 13],
            bytes[offset + 14],
            bytes[offset + 15]
        )
        let confidenceBits: UInt16 = bytes.littleEndian(at: offset + 16)
        let flags: UInt16 = bytes.littleEndian(at: offset + 18)
        let sourceFrame: UInt32 = bytes.littleEndian(at: offset + 20)
        return PointVertex(
            position: SIMD3(x, y, z),
            rgba: rgba,
            confidence: Float(Float16(bitPattern: confidenceBits)),
            flags: flags,
            sourceFrame: sourceFrame
        )
    }
}

private extension UnsafeRawBufferPointer {
    func littleEndian<T: FixedWidthInteger>(at byteOffset: Int) -> T {
        loadUnaligned(fromByteOffset: byteOffset, as: T.self).littleEndian
    }
}

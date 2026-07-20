import AVFoundation
import CoreVideo
import Foundation

struct VideoFixtureColor: Equatable, Sendable {
    let red: UInt8
    let green: UInt8
    let blue: UInt8
}

struct VideoFixture: Sendable {
    let url: URL
    let presentationTimestamps: [CMTime]
    let colors: [VideoFixtureColor]
}

enum VideoFixtureFactory {
    static let width = 640
    static let height = 360
    static let presentationValues: [CMTimeValue] = [0, 1, 2, 4, 7, 11, 16, 22, 29]
    static let colors: [VideoFixtureColor] = [
        .init(red: 230, green: 24, blue: 24),
        .init(red: 24, green: 230, blue: 24),
        .init(red: 24, green: 24, blue: 230),
        .init(red: 230, green: 230, blue: 24),
        .init(red: 230, green: 24, blue: 230),
        .init(red: 24, green: 230, blue: 230),
        .init(red: 230, green: 112, blue: 24),
        .init(red: 112, green: 24, blue: 230),
        .init(red: 230, green: 230, blue: 230),
    ]

    static func makeVFRMovie(
        in directory: URL,
        fileType: AVFileType = .mov,
        filenameExtension: String = "mov"
    ) async throws -> VideoFixture {
        let url = directory
            .appending(path: "vfr-\(UUID().uuidString)")
            .appendingPathExtension(filenameExtension)
        let writer = try AVAssetWriter(outputURL: url, fileType: fileType)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 4_000_000,
                    AVVideoExpectedSourceFrameRateKey: 30,
                    AVVideoMaxKeyFrameIntervalKey: 1,
                ],
            ]
        )
        input.expectsMediaDataInRealTime = false
        input.transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: CGFloat(height), ty: 0)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        guard writer.canAdd(input) else { throw VideoFixtureError.cannotAddInput }
        writer.add(input)
        guard writer.startWriting() else {
            throw writer.error ?? VideoFixtureError.writerFailed
        }
        writer.startSession(atSourceTime: .zero)

        let timestamps = presentationValues.map { CMTime(value: $0, timescale: 30) }
        for (timestamp, color) in zip(timestamps, colors) {
            while !input.isReadyForMoreMediaData {
                try Task.checkCancellation()
                await Task.yield()
            }
            let pixelBuffer = try makePixelBuffer(color: color)
            guard adaptor.append(pixelBuffer, withPresentationTime: timestamp) else {
                throw writer.error ?? VideoFixtureError.appendFailed
            }
        }

        input.markAsFinished()
        writer.endSession(atSourceTime: CMTime(value: 1, timescale: 1))
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw writer.error ?? VideoFixtureError.writerFailed
        }

        return VideoFixture(url: url, presentationTimestamps: timestamps, colors: colors)
    }

    static func makePixelBuffer(color: VideoFixtureColor) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw VideoFixtureError.pixelBufferCreationFailed(status)
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw VideoFixtureError.missingPixelBufferBaseAddress
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        for row in 0..<height {
            let bytes = baseAddress.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for column in 0..<width {
                let offset = column * 4
                bytes[offset] = color.blue
                bytes[offset + 1] = color.green
                bytes[offset + 2] = color.red
                bytes[offset + 3] = 255
            }
        }
        return buffer
    }
}

private enum VideoFixtureError: Error {
    case cannotAddInput
    case writerFailed
    case appendFailed
    case pixelBufferCreationFailed(CVReturn)
    case missingPixelBufferBaseAddress
}

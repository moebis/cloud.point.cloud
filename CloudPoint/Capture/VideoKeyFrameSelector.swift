@preconcurrency import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct VideoKeyFrameCandidate: Identifiable, Sendable, Equatable {
    var id: Int { index }

    let index: Int
    let timestampSeconds: Double
    let thumbnailJPEG: Data
    let fullResolutionJPEG: Data
    let sharpnessScore: Double
    let exposureScore: Double
    let temporalScore: Double

    var qualityScore: Double {
        sharpnessScore * 0.65 + exposureScore * 0.25 + temporalScore * 0.10
    }
}

protocol VideoKeyFrameSelecting: Sendable {
    func candidates(
        for url: URL,
        durationSeconds: Double,
        count: Int
    ) async throws -> [VideoKeyFrameCandidate]
}

enum VideoKeyFrameSelectorError: Error, LocalizedError, Equatable, Sendable {
    case invalidRequest
    case noFrames
    case imageConversionFailed
    case jpegEncodingFailed

    var errorDescription: String? {
        switch self {
        case .invalidRequest: "The video frame selection request is invalid."
        case .noFrames: "CloudPoint could not find a usable frame in this video."
        case .imageConversionFailed: "CloudPoint could not orient a video frame."
        case .jpegEncodingFailed: "CloudPoint could not prepare the selected frame."
        }
    }
}

struct VideoKeyFrameSelector: VideoKeyFrameSelecting {
    func candidates(
        for url: URL,
        durationSeconds: Double,
        count: Int = 7
    ) async throws -> [VideoKeyFrameCandidate] {
        guard durationSeconds.isFinite,
              durationSeconds > 0,
              (1...15).contains(count) else {
            throw VideoKeyFrameSelectorError.invalidRequest
        }

        let didStartScope = url.startAccessingSecurityScopedResource()
        defer { if didStartScope { url.stopAccessingSecurityScopedResource() } }

        let timestamps = (0..<count).map { ordinal in
            let fraction = (Double(ordinal) + 0.5) / Double(count)
            return CMTime(seconds: durationSeconds * fraction, preferredTimescale: 600)
        }
        let context = CIContext(options: [.cacheIntermediates: false])
        var result: [VideoKeyFrameCandidate] = []
        result.reserveCapacity(count)
        for try await frame in AssetFrameSource(assetURL: url).frames(at: timestamps) {
            try Task.checkCancellation()
            let oriented = try Self.orientedImage(from: frame, context: context)
            let fullJPEG = try Self.jpegData(oriented, quality: 0.92)
            let thumbnail = try Self.thumbnail(of: oriented, maximumDimension: 280)
            let thumbnailJPEG = try Self.jpegData(thumbnail, quality: 0.82)
            let metrics = try Self.metrics(for: thumbnail)
            let timestamp = frame.presentationTimestamp.seconds
            let normalizedTime = min(max(timestamp / durationSeconds, 0), 1)
            let temporalScore = 1 - abs(normalizedTime - 0.5) * 2
            result.append(VideoKeyFrameCandidate(
                index: result.count,
                timestampSeconds: timestamp,
                thumbnailJPEG: thumbnailJPEG,
                fullResolutionJPEG: fullJPEG,
                sharpnessScore: metrics.sharpness,
                exposureScore: metrics.exposure,
                temporalScore: temporalScore
            ))
        }
        guard !result.isEmpty else { throw VideoKeyFrameSelectorError.noFrames }
        return result
    }

    static func recommended(
        in candidates: [VideoKeyFrameCandidate]
    ) -> VideoKeyFrameCandidate? {
        candidates.max {
            if $0.qualityScore != $1.qualityScore {
                return $0.qualityScore < $1.qualityScore
            }
            if $0.temporalScore != $1.temporalScore {
                return $0.temporalScore < $1.temporalScore
            }
            return $0.timestampSeconds > $1.timestampSeconds
        }
    }

    static func selected(
        in candidates: [VideoKeyFrameCandidate],
        preferredIndex: Int?
    ) -> VideoKeyFrameCandidate? {
        if let preferredIndex,
           let preferred = candidates.first(where: { $0.index == preferredIndex }) {
            return preferred
        }
        return recommended(in: candidates)
    }

    private static func orientedImage(
        from frame: CapturedFrame,
        context: CIContext
    ) throws -> CGImage {
        var image = CIImage(cvPixelBuffer: frame.pixelBuffer)
            .transformed(by: frame.orientation)
        let extent = image.extent.integral
        guard extent.width > 0, extent.height > 0 else {
            throw VideoKeyFrameSelectorError.imageConversionFailed
        }
        image = image.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
        guard let result = context.createCGImage(
            image,
            from: CGRect(origin: .zero, size: extent.size)
        ) else {
            throw VideoKeyFrameSelectorError.imageConversionFailed
        }
        return result
    }

    private static func thumbnail(
        of image: CGImage,
        maximumDimension: Int
    ) throws -> CGImage {
        let scale = min(
            1,
            Double(maximumDimension) / Double(max(image.width, image.height))
        )
        let width = max(Int((Double(image.width) * scale).rounded()), 1)
        let height = max(Int((Double(image.height) * scale).rounded()), 1)
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VideoKeyFrameSelectorError.imageConversionFailed
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let result = context.makeImage() else {
            throw VideoKeyFrameSelectorError.imageConversionFailed
        }
        return result
    }

    private static func jpegData(_ image: CGImage, quality: Double) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            throw VideoKeyFrameSelectorError.jpegEncodingFailed
        }
        CGImageDestinationAddImage(
            destination,
            image,
            [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else {
            throw VideoKeyFrameSelectorError.jpegEncodingFailed
        }
        return data as Data
    }

    private static func metrics(for image: CGImage) throws -> (sharpness: Double, exposure: Double) {
        let width = min(image.width, 96)
        let height = min(image.height, 96)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VideoKeyFrameSelectorError.imageConversionFailed
        }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var luminance = [Double](repeating: 0, count: width * height)
        var total = 0.0
        var clipped = 0
        for pixel in 0..<(width * height) {
            let offset = pixel * 4
            let value = (
                0.2126 * Double(pixels[offset])
                    + 0.7152 * Double(pixels[offset + 1])
                    + 0.0722 * Double(pixels[offset + 2])
            ) / 255
            luminance[pixel] = value
            total += value
            if value < 0.02 || value > 0.98 { clipped += 1 }
        }
        let mean = total / Double(luminance.count)
        let clippingRatio = Double(clipped) / Double(luminance.count)
        let exposure = max(0, 1 - abs(mean - 0.5) * 2 - clippingRatio * 0.5)

        var edgeTotal = 0.0
        var edgeCount = 0
        if width > 2, height > 2 {
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let center = luminance[y * width + x]
                    let laplacian = abs(
                        luminance[(y - 1) * width + x]
                            + luminance[(y + 1) * width + x]
                            + luminance[y * width + x - 1]
                            + luminance[y * width + x + 1]
                            - 4 * center
                    )
                    edgeTotal += laplacian
                    edgeCount += 1
                }
            }
        }
        let sharpness = edgeCount > 0
            ? min(edgeTotal / Double(edgeCount) * 4, 1)
            : 0
        return (sharpness, exposure)
    }
}

import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct CapturedFrame: @unchecked Sendable {
    let index: Int
    let presentationTimestamp: CMTime
    let pixelBuffer: CVPixelBuffer
    let orientation: CGAffineTransform
    let sourceSampleSequence: Int
}

enum AssetFrameSourceError: Error, Equatable, Sendable {
    case invalidRequestedTimestamps
    case invalidAssetDuration
    case missingVideoTrack
    case cannotReadVideoTrack
    case missingPixelBuffer
    case readerFailed
}

struct AssetFrameSource: Sendable {
    let assetURL: URL

    func frames(at timestamps: [CMTime]) -> AsyncThrowingStream<CapturedFrame, Error> {
        let state = AssetReaderState(assetURL: assetURL, timestamps: timestamps)
        return AsyncThrowingStream(unfolding: { try await state.next() })
    }
}

protocol FramePersistence: Sendable {
    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame
}

enum FramePersistenceError: Error, Equatable, Sendable {
    case unsafePackageLayout
    case invalidFrame
    case jpegEncodingFailed
    case cannotCreatePartial
}

actor JPEGFramePersistence: FramePersistence {
    private let packageURL: URL
    private let fileManager: FileManager
    private let context = CIContext(options: [.cacheIntermediates: false])

    init(packageURL: URL, fileManager: FileManager = .default) {
        self.packageURL = packageURL
        self.fileManager = fileManager
    }

    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame {
        try Task.checkCancellation()
        guard frame.index >= 0,
              frame.index <= Int(UInt32.max),
              frame.presentationTimestamp.isNumeric,
              CMTimeCompare(frame.presentationTimestamp, .zero) >= 0 else {
            throw FramePersistenceError.invalidFrame
        }

        let framesURL = packageURL.appending(path: "Frames", directoryHint: .isDirectory)
        try validateLayout(packageURL: packageURL, framesURL: framesURL)
        let filename = String(format: "%08u.jpg", UInt32(frame.index))
        let finalURL = framesURL.appending(path: filename)
        let partialURL = framesURL.appending(path: filename + ".partial")
        guard !fileManager.fileExists(atPath: partialURL.path) else {
            throw FramePersistenceError.cannotCreatePartial
        }

        let jpeg = try makeOrientedJPEG(frame)
        try Task.checkCancellation()
        guard fileManager.createFile(atPath: partialURL.path, contents: nil) else {
            throw FramePersistenceError.cannotCreatePartial
        }

        var renamed = false
        let handle: FileHandle
        do {
            handle = try FileHandle(forWritingTo: partialURL)
        } catch {
            try? fileManager.removeItem(at: partialURL)
            throw error
        }

        do {
            try handle.write(contentsOf: jpeg)
            try handle.synchronize()
            try handle.close()
            try Task.checkCancellation()
            try fileManager.moveItem(at: partialURL, to: finalURL)
            renamed = true
            try Task.checkCancellation()
        } catch {
            try? handle.close()
            try? fileManager.removeItem(at: partialURL)
            if renamed { try? fileManager.removeItem(at: finalURL) }
            throw error
        }

        return PersistedFrame(
            index: frame.index,
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: "Frames/\(filename)"
        )
    }

    private func makeOrientedJPEG(_ frame: CapturedFrame) throws -> Data {
        let transformed = CIImage(cvPixelBuffer: frame.pixelBuffer).transformed(by: frame.orientation)
        let extent = transformed.extent.integral
        guard !extent.isNull,
              !extent.isInfinite,
              extent.width > 0,
              extent.height > 0 else {
            throw FramePersistenceError.invalidFrame
        }
        let oriented = transformed.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
        let renderBounds = CGRect(origin: .zero, size: extent.size)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let image = context.createCGImage(oriented, from: renderBounds, format: .RGBA8, colorSpace: colorSpace) else {
            throw FramePersistenceError.jpegEncodingFailed
        }

        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            throw FramePersistenceError.jpegEncodingFailed
        }
        CGImageDestinationAddImage(
            destination,
            image,
            [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else {
            throw FramePersistenceError.jpegEncodingFailed
        }
        return data as Data
    }

    private func validateLayout(packageURL: URL, framesURL: URL) throws {
        guard try isDirectoryWithoutSymlink(packageURL),
              try isDirectoryWithoutSymlink(framesURL) else {
            throw FramePersistenceError.unsafePackageLayout
        }
        let resolvedPackage = packageURL.resolvingSymlinksInPath().standardizedFileURL
        let resolvedFramesParent = framesURL.resolvingSymlinksInPath()
            .deletingLastPathComponent()
            .standardizedFileURL
        guard resolvedPackage == resolvedFramesParent else {
            throw FramePersistenceError.unsafePackageLayout
        }
    }

    private func isDirectoryWithoutSymlink(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        return values.isDirectory == true && values.isSymbolicLink != true
    }
}

private final class AssetReaderState: @unchecked Sendable {
    private struct Sample {
        let timestamp: CMTime
        let pixelBuffer: CVPixelBuffer
        let sequence: Int
    }

    private let assetURL: URL
    private let timestamps: [CMTime]
    private let lock = NSLock()

    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var transform = CGAffineTransform.identity
    private var setupComplete = false
    private var cancelled = false
    private var targetIndex = 0
    private var emittedIndex = 0
    private var decodedSequence = 0
    private var prior: Sample?
    private var current: Sample?

    init(assetURL: URL, timestamps: [CMTime]) {
        self.assetURL = assetURL
        self.timestamps = timestamps
    }

    deinit {
        cancel()
    }

    func next() async throws -> CapturedFrame? {
        try checkCancellation()
        guard targetIndex < timestamps.count else {
            cancelReaderAfterCompletion()
            return nil
        }
        try validateTimestamps()
        if !setupComplete { try await setUpReader() }
        try checkCancellation()
        return try nextSelectedFrame()
    }

    func cancel() {
        let activeReader = lock.withLock { () -> AVAssetReader? in
            cancelled = true
            return reader
        }
        activeReader?.cancelReading()
    }

    private func setUpReader() async throws {
        let asset = AVURLAsset(url: assetURL)
        async let loadedDuration = asset.load(.duration)
        async let loadedTracks = asset.loadTracks(withMediaType: .video)
        let (duration, tracks) = try await (loadedDuration, loadedTracks)
        try checkCancellation()
        guard duration.isNumeric, CMTimeCompare(duration, .zero) > 0 else {
            throw AssetFrameSourceError.invalidAssetDuration
        }
        guard let track = tracks.first else { throw AssetFrameSourceError.missingVideoTrack }
        let preferredTransform = try await track.load(.preferredTransform)
        try checkCancellation()

        let newReader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        let newOutput = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        newOutput.alwaysCopiesSampleData = false
        guard newReader.canAdd(newOutput) else { throw AssetFrameSourceError.cannotReadVideoTrack }
        newReader.add(newOutput)

        let wasCancelled = lock.withLock { () -> Bool in
            guard !cancelled else { return true }
            reader = newReader
            output = newOutput
            transform = preferredTransform
            setupComplete = true
            return false
        }
        guard !wasCancelled else {
            newReader.cancelReading()
            throw CancellationError()
        }
        guard newReader.startReading() else {
            throw newReader.error ?? AssetFrameSourceError.readerFailed
        }
    }

    private func nextSelectedFrame() throws -> CapturedFrame? {
        let target = timestamps[targetIndex]
        if current == nil { current = try readSample() }

        while let sample = current, CMTimeCompare(sample.timestamp, target) < 0 {
            prior = sample
            current = try readSample()
        }

        let selected: Sample
        if let before = prior, let after = current {
            let beforeDistance = CMTimeAbsoluteValue(CMTimeSubtract(target, before.timestamp))
            let afterDistance = CMTimeAbsoluteValue(CMTimeSubtract(after.timestamp, target))
            if CMTimeCompare(beforeDistance, afterDistance) <= 0 {
                selected = before
                prior = nil
            } else {
                selected = after
                prior = nil
                current = nil
            }
        } else if let before = prior {
            selected = before
            prior = nil
        } else if let after = current {
            selected = after
            current = nil
        } else {
            cancelReaderAfterCompletion()
            return nil
        }

        let frame = CapturedFrame(
            index: emittedIndex,
            presentationTimestamp: selected.timestamp,
            pixelBuffer: selected.pixelBuffer,
            orientation: transform,
            sourceSampleSequence: selected.sequence
        )
        emittedIndex += 1
        targetIndex += 1
        if targetIndex == timestamps.count { cancelReaderAfterCompletion() }
        return frame
    }

    private func readSample() throws -> Sample? {
        try checkCancellation()
        guard let output, let reader else { throw AssetFrameSourceError.readerFailed }
        guard let sampleBuffer = output.copyNextSampleBuffer() else {
            switch reader.status {
            case .failed: throw reader.error ?? AssetFrameSourceError.readerFailed
            case .cancelled: throw CancellationError()
            default: return nil
            }
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            throw AssetFrameSourceError.missingPixelBuffer
        }
        let sample = Sample(
            timestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
            pixelBuffer: pixelBuffer,
            sequence: decodedSequence
        )
        decodedSequence += 1
        return sample
    }

    private func validateTimestamps() throws {
        var previous: CMTime?
        for timestamp in timestamps {
            guard timestamp.isNumeric, CMTimeCompare(timestamp, .zero) >= 0 else {
                throw AssetFrameSourceError.invalidRequestedTimestamps
            }
            if let previous, CMTimeCompare(timestamp, previous) <= 0 {
                throw AssetFrameSourceError.invalidRequestedTimestamps
            }
            previous = timestamp
        }
    }

    private func checkCancellation() throws {
        if Task.isCancelled || lock.withLock({ cancelled }) {
            cancel()
            throw CancellationError()
        }
    }

    private func cancelReaderAfterCompletion() {
        let activeReader = lock.withLock { reader }
        if activeReader?.status == .reading { activeReader?.cancelReading() }
    }
}

private func CMTimeAbsoluteValue(_ time: CMTime) -> CMTime {
    CMTimeCompare(time, .zero) < 0 ? CMTimeMultiplyByFloat64(time, multiplier: -1) : time
}

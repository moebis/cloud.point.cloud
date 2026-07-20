import AVFoundation
import CoreImage
import CoreVideo
import Darwin
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
    private let readerFactory: any AssetReaderSessionFactory

    init(
        assetURL: URL,
        readerFactory: any AssetReaderSessionFactory = AVAssetReaderSessionFactory()
    ) {
        self.assetURL = assetURL
        self.readerFactory = readerFactory
    }

    func frames(at timestamps: [CMTime]) -> AsyncThrowingStream<CapturedFrame, Error> {
        let state = AssetReaderState(
            assetURL: assetURL,
            timestamps: timestamps,
            readerFactory: readerFactory
        )
        return AsyncThrowingStream(bufferingPolicy: .bufferingOldest(1)) { continuation in
            let producer = Task {
                do {
                    while let frame = try await state.next() {
                        try Task.checkCancellation()
                        var enqueued = false
                        while !enqueued {
                            try Task.checkCancellation()
                            switch continuation.yield(frame) {
                            case .enqueued:
                                enqueued = true
                            case .dropped:
                                await Task.yield()
                            case .terminated:
                                state.cancel()
                                return
                            @unknown default:
                                state.cancel()
                                return
                            }
                        }
                    }
                    state.cancel()
                    continuation.finish()
                } catch is CancellationError {
                    state.cancel()
                    continuation.finish()
                } catch {
                    state.cancel()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
                state.cancel()
            }
        }
    }
}

struct AssetReaderSample: @unchecked Sendable {
    let timestamp: CMTime
    let pixelBuffer: CVPixelBuffer
    let sequence: Int
}

protocol AssetReaderSession: Sendable {
    var duration: CMTime { get }
    var preferredTransform: CGAffineTransform { get }

    func start() throws
    func nextSample() throws -> AssetReaderSample?
    func cancel()
}

protocol AssetReaderSessionFactory: Sendable {
    func makeSession(for assetURL: URL) async throws -> any AssetReaderSession
}

struct AVAssetReaderSessionFactory: AssetReaderSessionFactory {
    func makeSession(for assetURL: URL) async throws -> any AssetReaderSession {
        let asset = AVURLAsset(url: assetURL)
        async let loadedDuration = asset.load(.duration)
        async let loadedTracks = asset.loadTracks(withMediaType: .video)
        let (duration, tracks) = try await (loadedDuration, loadedTracks)
        guard duration.isNumeric, CMTimeCompare(duration, .zero) > 0 else {
            throw AssetFrameSourceError.invalidAssetDuration
        }
        guard let track = tracks.first else { throw AssetFrameSourceError.missingVideoTrack }
        let preferredTransform = try await track.load(.preferredTransform)

        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AssetFrameSourceError.cannotReadVideoTrack }
        reader.add(output)
        return AVAssetReaderSession(
            reader: reader,
            output: output,
            duration: duration,
            preferredTransform: preferredTransform
        )
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
    case ioFailure(Int32)
}

actor JPEGFramePersistence: FramePersistence {
    private let packageDescriptor: Int32
    private let framesDescriptor: Int32
    private let context = CIContext(options: [.cacheIntermediates: false])

    init(packageURL: URL) throws {
        let packageDescriptor = Darwin.open(
            packageURL.path,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
        )
        guard packageDescriptor >= 0 else {
            throw FramePersistenceError.unsafePackageLayout
        }

        let framesDescriptor = "Frames".withCString {
            Darwin.openat(
                packageDescriptor,
                $0,
                O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
            )
        }
        guard framesDescriptor >= 0 else {
            Darwin.close(packageDescriptor)
            throw FramePersistenceError.unsafePackageLayout
        }

        self.packageDescriptor = packageDescriptor
        self.framesDescriptor = framesDescriptor
    }

    deinit {
        Darwin.close(framesDescriptor)
        Darwin.close(packageDescriptor)
    }

    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame {
        try Task.checkCancellation()
        guard let persistedIndex = UInt32(exactly: frame.index),
              frame.presentationTimestamp.isNumeric,
              CMTimeCompare(frame.presentationTimestamp, .zero) >= 0 else {
            throw FramePersistenceError.invalidFrame
        }

        let filename = String(format: "%08u.jpg", persistedIndex)
        let partialName = filename + ".partial"

        let jpeg = try makeOrientedJPEG(frame)
        try Task.checkCancellation()
        try validateFramesDirectoryIdentity()
        let partialDescriptor = partialName.withCString {
            Darwin.openat(
                framesDescriptor,
                $0,
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                mode_t(S_IRUSR | S_IWUSR)
            )
        }
        guard partialDescriptor >= 0 else {
            if errno == EEXIST { throw FramePersistenceError.cannotCreatePartial }
            throw FramePersistenceError.ioFailure(errno)
        }

        var descriptorIsOpen = true
        var renamed = false
        do {
            try writeAll(jpeg, to: partialDescriptor)
            guard Darwin.fsync(partialDescriptor) == 0 else {
                throw FramePersistenceError.ioFailure(errno)
            }
            guard Darwin.close(partialDescriptor) == 0 else {
                descriptorIsOpen = false
                throw FramePersistenceError.ioFailure(errno)
            }
            descriptorIsOpen = false
            try Task.checkCancellation()
            try validateFramesDirectoryIdentity()
            let renameResult = partialName.withCString { partialPath in
                filename.withCString { finalPath in
                    renameatx_np(
                        framesDescriptor,
                        partialPath,
                        framesDescriptor,
                        finalPath,
                        UInt32(RENAME_EXCL)
                    )
                }
            }
            guard renameResult == 0 else { throw FramePersistenceError.ioFailure(errno) }
            renamed = true
            try validateFramesDirectoryIdentity()
            guard Darwin.fsync(framesDescriptor) == 0 else {
                throw FramePersistenceError.ioFailure(errno)
            }
            try Task.checkCancellation()
            try validateFramesDirectoryIdentity()
        } catch {
            if descriptorIsOpen { Darwin.close(partialDescriptor) }
            partialName.withCString { _ = Darwin.unlinkat(framesDescriptor, $0, 0) }
            if renamed {
                filename.withCString { _ = Darwin.unlinkat(framesDescriptor, $0, 0) }
                _ = Darwin.fsync(framesDescriptor)
            }
            throw error
        }

        return PersistedFrame(
            index: persistedIndex,
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: "Frames/\(filename)"
        )
    }

    private func validateFramesDirectoryIdentity() throws {
        var heldFramesStatus = stat()
        guard Darwin.fstat(framesDescriptor, &heldFramesStatus) == 0,
              heldFramesStatus.st_mode & S_IFMT == S_IFDIR else {
            throw FramePersistenceError.unsafePackageLayout
        }

        var packageFramesStatus = stat()
        let statusResult = "Frames".withCString {
            Darwin.fstatat(
                packageDescriptor,
                $0,
                &packageFramesStatus,
                AT_SYMLINK_NOFOLLOW
            )
        }
        guard statusResult == 0,
              packageFramesStatus.st_mode & S_IFMT == S_IFDIR,
              packageFramesStatus.st_dev == heldFramesStatus.st_dev,
              packageFramesStatus.st_ino == heldFramesStatus.st_ino else {
            throw FramePersistenceError.unsafePackageLayout
        }
    }

    private func writeAll(_ data: Data, to descriptor: Int32) throws {
        try data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            var written = 0
            while written < bytes.count {
                if Task.isCancelled { throw CancellationError() }
                let result = Darwin.write(
                    descriptor,
                    baseAddress.advanced(by: written),
                    bytes.count - written
                )
                if result > 0 {
                    written += result
                } else if result == -1, errno == EINTR {
                    continue
                } else {
                    throw FramePersistenceError.ioFailure(errno)
                }
            }
        }
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

}

private final class AssetReaderState: @unchecked Sendable {
    private struct Sample {
        let timestamp: CMTime
        let pixelBuffer: CVPixelBuffer
        let sequence: Int
    }

    private let assetURL: URL
    private let timestamps: [CMTime]
    private let readerFactory: any AssetReaderSessionFactory
    private let lock = NSLock()

    private var reader: (any AssetReaderSession)?
    private var cancelled = false
    private var targetIndex = 0
    private var emittedIndex = 0
    private var prior: Sample?
    private var current: Sample?

    init(
        assetURL: URL,
        timestamps: [CMTime],
        readerFactory: any AssetReaderSessionFactory
    ) {
        self.assetURL = assetURL
        self.timestamps = timestamps
        self.readerFactory = readerFactory
    }

    deinit {
        cancel()
    }

    func next() async throws -> CapturedFrame? {
        try checkCancellation()
        guard targetIndex < timestamps.count else {
            cancel()
            return nil
        }
        if reader == nil { try await setUpReader() }
        try checkCancellation()
        return try nextSelectedFrame()
    }

    func cancel() {
        let activeReader = lock.withLock { () -> (any AssetReaderSession)? in
            cancelled = true
            return reader
        }
        activeReader?.cancel()
    }

    private func setUpReader() async throws {
        let newReader = try await readerFactory.makeSession(for: assetURL)
        try checkCancellation()
        guard newReader.duration.isNumeric, CMTimeCompare(newReader.duration, .zero) > 0 else {
            throw AssetFrameSourceError.invalidAssetDuration
        }
        try validateTimestamps(against: newReader.duration)

        let wasCancelled = lock.withLock { () -> Bool in
            guard !cancelled else { return true }
            reader = newReader
            return false
        }
        guard !wasCancelled else {
            newReader.cancel()
            throw CancellationError()
        }
        do {
            try newReader.start()
        } catch {
            newReader.cancel()
            throw error
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
            cancel()
            return nil
        }

        guard let reader else { throw AssetFrameSourceError.readerFailed }
        let frame = CapturedFrame(
            index: emittedIndex,
            presentationTimestamp: selected.timestamp,
            pixelBuffer: selected.pixelBuffer,
            orientation: reader.preferredTransform,
            sourceSampleSequence: selected.sequence
        )
        emittedIndex += 1
        targetIndex += 1
        if targetIndex == timestamps.count { cancel() }
        return frame
    }

    private func readSample() throws -> Sample? {
        try checkCancellation()
        guard let reader else { throw AssetFrameSourceError.readerFailed }
        guard let sample = try reader.nextSample() else { return nil }
        return Sample(
            timestamp: sample.timestamp,
            pixelBuffer: sample.pixelBuffer,
            sequence: sample.sequence
        )
    }

    private func validateTimestamps(against duration: CMTime) throws {
        var previous: CMTime?
        for timestamp in timestamps {
            guard timestamp.isNumeric,
                  CMTimeCompare(timestamp, .zero) >= 0,
                  CMTimeCompare(timestamp, duration) < 0 else {
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
}

private final class AVAssetReaderSession: AssetReaderSession, @unchecked Sendable {
    let duration: CMTime
    let preferredTransform: CGAffineTransform

    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private let lock = NSLock()
    private var sequence = 0
    private var cancelled = false

    init(
        reader: AVAssetReader,
        output: AVAssetReaderTrackOutput,
        duration: CMTime,
        preferredTransform: CGAffineTransform
    ) {
        self.reader = reader
        self.output = output
        self.duration = duration
        self.preferredTransform = preferredTransform
    }

    func start() throws {
        guard !lock.withLock({ cancelled }) else { throw CancellationError() }
        guard reader.startReading() else {
            throw reader.error ?? AssetFrameSourceError.readerFailed
        }
    }

    func nextSample() throws -> AssetReaderSample? {
        guard !lock.withLock({ cancelled }) else { throw CancellationError() }
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
        let result = AssetReaderSample(
            timestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
            pixelBuffer: pixelBuffer,
            sequence: sequence
        )
        sequence += 1
        return result
    }

    func cancel() {
        let shouldCancel = lock.withLock { () -> Bool in
            guard !cancelled else { return false }
            cancelled = true
            return true
        }
        if shouldCancel { reader.cancelReading() }
    }
}

private func CMTimeAbsoluteValue(_ time: CMTime) -> CMTime {
    CMTimeCompare(time, .zero) < 0 ? CMTimeMultiplyByFloat64(time, multiplier: -1) : time
}

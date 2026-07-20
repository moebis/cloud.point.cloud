import AVFoundation
import CoreGraphics
import Darwin
import ImageIO
import XCTest
@testable import CloudPoint

final class AssetFrameSourceTests: XCTestCase {
    func testMovMp4AndM4vContainersDecodeVideoFrames() async throws {
        let containers: [(fileType: AVFileType, filenameExtension: String)] = [
            (.mov, "mov"),
            (.mp4, "mp4"),
            (AVFileType(rawValue: "com.apple.m4v-video"), "m4v"),
        ]

        for container in containers {
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let fixture = try await VideoFixtureFactory.makeVFRMovie(
                in: directory,
                fileType: container.fileType,
                filenameExtension: container.filenameExtension
            )
            let source = AssetFrameSource(assetURL: fixture.url)

            let frames = try await Self.collect(source.frames(at: [.zero]))

            XCTAssertEqual(frames.count, 1, "Expected .\(container.filenameExtension) to decode")
            XCTAssertEqual(frames.first?.presentationTimestamp, .zero)
        }
    }

    func testVFRSamplingChoosesNearestUniqueFramesWithEarlierTieBreak() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let source = AssetFrameSource(assetURL: fixture.url)

        let fiveFPS = try FrameSamplingPlan(
            duration: CMTime(value: 1, timescale: 1),
            framesPerSecond: 5
        )
        let fiveFPSFrames = try await Self.collect(source.frames(at: fiveFPS.timestamps))
        XCTAssertEqual(
            fiveFPSFrames.map(\.presentationTimestamp),
            [0, 7, 11, 16, 22].map { CMTime(value: CMTimeValue($0), timescale: 30) }
        )
        XCTAssertEqual(Set(fiveFPSFrames.map { $0.sourceSampleSequence }).count, fiveFPSFrames.count)

        let tieSource = AssetFrameSource(assetURL: fixture.url)
        let tieFrames = try await Self.collect(tieSource.frames(at: [CMTime(value: 3, timescale: 30)]))
        XCTAssertEqual(tieFrames.map(\.presentationTimestamp), [CMTime(value: 2, timescale: 30)])
    }

    func testEmptyTimestampPlanDoesNotOpenMissingAsset() async throws {
        let missingURL = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString)
            .appendingPathExtension("mov")

        let frames = try await Self.collect(AssetFrameSource(assetURL: missingURL).frames(at: []))

        XCTAssertEqual(frames.count, 0)
    }

    func testInvalidOrNonIncreasingRequestedTimestampsAreRejected() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let invalidLists: [[CMTime]] = [
            [.invalid],
            [.indefinite],
            [CMTime(value: -1, timescale: 30)],
            [CMTime(value: 2, timescale: 30), CMTime(value: 1, timescale: 30)],
            [.zero, .zero],
        ]

        for timestamps in invalidLists {
            do {
                _ = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: timestamps))
                XCTFail("Expected invalid timestamp list to fail")
            } catch {
                XCTAssertEqual(error as? AssetFrameSourceError, .invalidRequestedTimestamps)
            }
        }
    }

    func testRequestedTimestampEqualToOrBeyondDurationIsRejectedBeforePersistence() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)

        for timestamp in [CMTime(value: 1, timescale: 1), CMTime(value: 31, timescale: 30)] {
            let package = try TemporaryProjectPackage.make()
            let persistence = try JPEGFramePersistence(packageURL: package.url)
            do {
                for try await frame in AssetFrameSource(assetURL: fixture.url).frames(at: [timestamp]) {
                    _ = try await persistence.persist(frame)
                }
                XCTFail("Expected timestamp \(timestamp) to be rejected")
            } catch {
                XCTAssertEqual(error as? AssetFrameSourceError, .invalidRequestedTimestamps)
            }
            XCTAssertEqual(
                try FileManager.default.contentsOfDirectory(atPath: package.url.appending(path: "Frames").path),
                []
            )
        }
    }

    func testCancellingConsumerCancelsReaderWhileStreamIsRetained() async throws {
        let pixelBuffer = try VideoFixtureFactory.makePixelBuffer(
            color: .init(red: 40, green: 80, blue: 120)
        )
        let reader = BlockingRecordingAssetReaderSession(pixelBuffer: pixelBuffer)
        let source = AssetFrameSource(
            assetURL: URL(filePath: "/unused-by-injected-reader.mov"),
            readerFactory: FixedAssetReaderSessionFactory(reader: reader)
        )
        let stream = source.frames(at: [
            .zero,
            CMTime(value: 1, timescale: 5),
        ])
        let firstFrame = expectation(description: "first frame emitted")

        let consumer = Task { () -> Int in
            var count = 0
            do {
                for try await _ in stream {
                    count += 1
                    firstFrame.fulfill()
                }
            } catch {}
            return count
        }

        await fulfillment(of: [firstFrame], timeout: 1)
        await reader.waitUntilBlockedOnNextSample()
        consumer.cancel()
        await reader.waitUntilCancelled()
        let emittedCount = await consumer.value
        XCTAssertEqual(emittedCount, 1)
        withExtendedLifetime(stream) {}
    }

    func testPreCancelledIterationEmitsNoFrames() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let plan = try FrameSamplingPlan(duration: CMTime(value: 1, timescale: 1), framesPerSecond: 10)
        let assetURL = fixture.url
        let timestamps = plan.timestamps

        let error = await Task { () -> Error? in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                for try await _ in AssetFrameSource(assetURL: assetURL).frames(at: timestamps) {
                    return UnexpectedEmissionError()
                }
                return nil
            } catch { return error }
        }.value

        XCTAssertTrue(error == nil || error is CancellationError)
    }

    func testCancellationAfterPersistedFrameCreatesNoLaterFiles() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let package = try TemporaryProjectPackage.make()
        let persistence = try JPEGFramePersistence(packageURL: package.url)
        let plan = try FrameSamplingPlan(duration: CMTime(value: 1, timescale: 1), framesPerSecond: 10)
        let assetURL = fixture.url
        let timestamps = plan.timestamps

        let emittedCount = await Task { () -> Int in
            var count = 0
            do {
                for try await frame in AssetFrameSource(assetURL: assetURL).frames(at: timestamps) {
                    _ = try await persistence.persist(frame)
                    count += 1
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            } catch {}
            return count
        }.value

        XCTAssertEqual(emittedCount, 1)
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: package.url.appending(path: "Frames").path),
            ["00000000.jpg"]
        )
    }

    func testJPEGAtomicPersistenceAppliesTrackOrientationAndPreservesSelectedColors() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let package = try TemporaryProjectPackage.make()
        let source = AssetFrameSource(assetURL: fixture.url)
        let plan = try FrameSamplingPlan(duration: CMTime(value: 1, timescale: 1), framesPerSecond: 5)
        let persistence = try JPEGFramePersistence(packageURL: package.url)

        var persisted: [PersistedFrame] = []
        for try await frame in source.frames(at: plan.timestamps) {
            let value = try await persistence.persist(frame)
            let finalURL = package.url.appending(path: value.relativePath)
            let partialURL = finalURL.appendingPathExtension("partial")
            XCTAssertTrue(FileManager.default.fileExists(atPath: finalURL.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: partialURL.path))
            persisted.append(value)
        }

        XCTAssertEqual(persisted.map(\.relativePath), (0..<5).map { String(format: "Frames/%08d.jpg", $0) })
        XCTAssertEqual(persisted.map(\.sourceTimestamp), [0, 7, 11, 16, 22].map { Double($0) / 30 }, accuracy: 0.000_001)

        let selectedFixtureIndices = [0, 4, 5, 6, 7]
        for (persistedFrame, fixtureIndex) in zip(persisted, selectedFixtureIndices) {
            let imageURL = package.url.appending(path: persistedFrame.relativePath)
            let decoded = try decodeJPEG(at: imageURL)
            XCTAssertEqual(decoded.width, VideoFixtureFactory.height)
            XCTAssertEqual(decoded.height, VideoFixtureFactory.width)
            assertCenterColor(decoded, approximately: fixture.colors[fixtureIndex])
        }
    }

    func testWriteFailureLeavesNoPartialAndReturnsNoPersistedFrame() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let frame = try XCTUnwrap(frames.first)
        let package = try TemporaryProjectPackage.make()
        let destination = package.url.appending(path: "Frames/00000000.jpg")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)

        do {
            let emitted = try await JPEGFramePersistence(packageURL: package.url).persist(frame)
            XCTFail("Write failure unexpectedly emitted \(emitted)")
        } catch {
            XCTAssertTrue(FileManager.default.fileExists(atPath: destination.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathExtension("partial").path))
        }
    }

    func testPersistenceUsesUnsignedEightDigitMinimumFrameNames() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let sourceFrame = try XCTUnwrap(frames.first)
        let highIndexFrame = CapturedFrame(
            index: Int(UInt32.max),
            presentationTimestamp: sourceFrame.presentationTimestamp,
            pixelBuffer: sourceFrame.pixelBuffer,
            orientation: sourceFrame.orientation,
            sourceSampleSequence: sourceFrame.sourceSampleSequence
        )
        let package = try TemporaryProjectPackage.make()

        let persisted = try await JPEGFramePersistence(packageURL: package.url).persist(highIndexFrame)

        XCTAssertEqual(persisted.relativePath, "Frames/4294967295.jpg")
        XCTAssertTrue(FileManager.default.fileExists(atPath: package.url.appending(path: persisted.relativePath).path))
    }

    func testPersistenceRejectsFrameIndexAboveUInt32Range() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let sourceFrame = try XCTUnwrap(frames.first)
        let outOfRangeFrame = CapturedFrame(
            index: Int(UInt32.max) + 1,
            presentationTimestamp: sourceFrame.presentationTimestamp,
            pixelBuffer: sourceFrame.pixelBuffer,
            orientation: sourceFrame.orientation,
            sourceSampleSequence: sourceFrame.sourceSampleSequence
        )
        let package = try TemporaryProjectPackage.make()

        do {
            _ = try await JPEGFramePersistence(packageURL: package.url).persist(outOfRangeFrame)
            XCTFail("Expected out-of-range frame index to be rejected")
        } catch {
            XCTAssertEqual(error as? FramePersistenceError, .invalidFrame)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: package.url.appending(path: "Frames").path), [])
    }

    func testPreCancelledPersistenceCreatesNoFileOrPartial() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let frame = try XCTUnwrap(frames.first)
        let package = try TemporaryProjectPackage.make()
        let persistence = try JPEGFramePersistence(packageURL: package.url)

        let result = await Task { () -> Result<PersistedFrame, Error> in
            withUnsafeCurrentTask { $0?.cancel() }
            do { return .success(try await persistence.persist(frame)) }
            catch { return .failure(error) }
        }.value

        switch result {
        case let .success(frame): XCTFail("Cancelled persistence emitted \(frame)")
        case let .failure(error): XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: package.url.appending(path: "Frames").path), [])
    }

    func testPackageAndFramesSymlinksCannotEscapePersistenceRoot() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let frame = try XCTUnwrap(frames.first)
        let fileManager = FileManager.default

        let realPackage = try TemporaryProjectPackage.make()
        let packageLink = directory.appending(path: "linked.cloudpoint")
        try fileManager.createSymbolicLink(at: packageLink, withDestinationURL: realPackage.url)
        await assertContainmentRejected(packageURL: packageLink, frame: frame)

        let package = try TemporaryProjectPackage.make()
        let framesURL = package.url.appending(path: "Frames")
        let escaped = directory.appending(path: "escaped", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: escaped, withIntermediateDirectories: false)
        try fileManager.removeItem(at: framesURL)
        try fileManager.createSymbolicLink(at: framesURL, withDestinationURL: escaped)
        await assertContainmentRejected(packageURL: package.url, frame: frame)
        XCTAssertEqual(try fileManager.contentsOfDirectory(atPath: escaped.path), [])
    }

    func testPersistenceRejectsFramesDirectoryRelocatedOutsidePackageBeforeWrite() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let frame = try XCTUnwrap(frames.first)
        let package = try TemporaryProjectPackage.make()
        let fileManager = FileManager.default
        let persistence = try JPEGFramePersistence(packageURL: package.url)
        let framesURL = package.url.appending(path: "Frames", directoryHint: .isDirectory)
        let heldFramesURL = directory.appending(path: "held-frames", directoryHint: .isDirectory)
        let escapedURL = directory.appending(path: "escaped", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: escapedURL, withIntermediateDirectories: false)

        try fileManager.moveItem(at: framesURL, to: heldFramesURL)
        try fileManager.createSymbolicLink(at: framesURL, withDestinationURL: escapedURL)

        do {
            let emitted = try await persistence.persist(frame)
            XCTFail("Relocated Frames directory unexpectedly emitted \(emitted)")
        } catch {
            XCTAssertEqual(error as? FramePersistenceError, .unsafePackageLayout)
        }

        XCTAssertEqual(try fileManager.contentsOfDirectory(atPath: heldFramesURL.path), [])
        XCTAssertEqual(try fileManager.contentsOfDirectory(atPath: escapedURL.path), [])
    }

    func testPersistenceRejectsFramesDirectoryReplacementAfterPartialCreation() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let package = try TemporaryProjectPackage.make()
        let fileManager = FileManager.default
        let persistence = try JPEGFramePersistence(packageURL: package.url)
        let frame = try makeLargeNoisyFrame()
        let framesURL = package.url.appending(path: "Frames", directoryHint: .isDirectory)
        let partialURL = framesURL.appending(path: "00000000.jpg.partial")
        let heldFramesURL = directory.appending(path: "held-frames", directoryHint: .isDirectory)
        let escapedURL = directory.appending(path: "escaped", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: escapedURL, withIntermediateDirectories: false)

        let persistenceTask = Task { () -> Result<PersistedFrame, Error> in
            do { return .success(try await persistence.persist(frame)) }
            catch { return .failure(error) }
        }

        do {
            try await waitForFile(at: partialURL, timeout: .seconds(15))
            try fileManager.moveItem(at: framesURL, to: heldFramesURL)
            try fileManager.createSymbolicLink(at: framesURL, withDestinationURL: escapedURL)
        } catch {
            persistenceTask.cancel()
            _ = await persistenceTask.value
            throw error
        }

        switch await persistenceTask.value {
        case let .success(emitted):
            XCTFail("Replaced Frames directory unexpectedly emitted \(emitted)")
        case let .failure(error):
            XCTAssertEqual(error as? FramePersistenceError, .unsafePackageLayout)
        }
        XCTAssertEqual(try fileManager.contentsOfDirectory(atPath: heldFramesURL.path), [])
        XCTAssertEqual(try fileManager.contentsOfDirectory(atPath: escapedURL.path), [])
    }

    func testConcurrentIndependentStoresForSameIndexProduceExactlyOneValidFinalJPEG() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let frames = try await Self.collect(AssetFrameSource(assetURL: fixture.url).frames(at: [.zero]))
        let frame = try XCTUnwrap(frames.first)
        let package = try TemporaryProjectPackage.make()
        let firstStore = try JPEGFramePersistence(packageURL: package.url)
        let secondStore = try JPEGFramePersistence(packageURL: package.url)

        let outcomes = await withTaskGroup(of: ConcurrentPersistenceOutcome.self) { group in
            for store in [firstStore, secondStore] {
                group.addTask {
                    do { return .success(try await store.persist(frame)) }
                    catch { return .failure }
                }
            }
            var values: [ConcurrentPersistenceOutcome] = []
            for await value in group { values.append(value) }
            return values
        }

        XCTAssertEqual(outcomes.filter(\.isSuccess).count, 1)
        let finalURL = package.url.appending(path: "Frames/00000000.jpg")
        let decoded = try decodeJPEG(at: finalURL)
        XCTAssertGreaterThan(decoded.width, 0)
        XCTAssertGreaterThan(decoded.height, 0)
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: package.url.appending(path: "Frames").path),
            ["00000000.jpg"]
        )
    }

    private func assertContainmentRejected(packageURL: URL, frame: CapturedFrame) async {
        do {
            let emitted = try await JPEGFramePersistence(packageURL: packageURL).persist(frame)
            XCTFail("Escaping path unexpectedly emitted \(emitted)")
        } catch {
            XCTAssertEqual(error as? FramePersistenceError, .unsafePackageLayout)
        }
    }

    private static func collect(_ stream: AsyncThrowingStream<CapturedFrame, Error>) async throws -> [CapturedFrame] {
        var frames: [CapturedFrame] = []
        for try await frame in stream { frames.append(frame) }
        return frames
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeLargeNoisyFrame() throws -> CapturedFrame {
        let width = 4_096
        let height = 4_096
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            ] as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PartialCreationCoordinationError.cannotCreatePixelBuffer(status)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw PartialCreationCoordinationError.missingPixelBufferBaseAddress
        }
        let byteCount = CVPixelBufferGetBytesPerRow(pixelBuffer) * height
        arc4random_buf(baseAddress, byteCount)
        let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
        for offset in stride(from: 3, to: byteCount, by: 4) { bytes[offset] = 255 }

        return CapturedFrame(
            index: 0,
            presentationTimestamp: .zero,
            pixelBuffer: pixelBuffer,
            orientation: .identity,
            sourceSampleSequence: 0
        )
    }

    private func waitForFile(at url: URL, timeout: Duration) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while !FileManager.default.fileExists(atPath: url.path) {
            guard clock.now < deadline else {
                throw PartialCreationCoordinationError.timedOutWaitingForPartial
            }
            await Task.yield()
        }
    }

    private func decodeJPEG(at url: URL) throws -> CGImage {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private func assertCenterColor(
        _ image: CGImage,
        approximately expected: VideoFixtureColor,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        var pixel = [UInt8](repeating: 0, count: 4)
        let context = CGContext(
            data: &pixel,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.interpolationQuality = .none
        context.draw(
            image,
            in: CGRect(x: -CGFloat(image.width / 2), y: -CGFloat(image.height / 2), width: CGFloat(image.width), height: CGFloat(image.height))
        )
        XCTAssertEqual(pixel[0], expected.red, accuracy: 65, file: file, line: line)
        XCTAssertEqual(pixel[1], expected.green, accuracy: 65, file: file, line: line)
        XCTAssertEqual(pixel[2], expected.blue, accuracy: 65, file: file, line: line)
    }
}

private struct UnexpectedEmissionError: Error {}

private enum PartialCreationCoordinationError: Error {
    case cannotCreatePixelBuffer(CVReturn)
    case missingPixelBufferBaseAddress
    case timedOutWaitingForPartial
}

private enum ConcurrentPersistenceOutcome: Sendable {
    case success(PersistedFrame)
    case failure

    var isSuccess: Bool {
        if case .success = self { true } else { false }
    }
}

private struct FixedAssetReaderSessionFactory: AssetReaderSessionFactory {
    let reader: BlockingRecordingAssetReaderSession

    func makeSession(for assetURL: URL) async throws -> any AssetReaderSession {
        reader
    }
}

private final class BlockingRecordingAssetReaderSession: AssetReaderSession, @unchecked Sendable {
    let duration = CMTime(value: 1, timescale: 1)
    let preferredTransform = CGAffineTransform.identity

    private let condition = NSCondition()
    private let pixelBuffer: CVPixelBuffer
    private var hasReturnedFirstSample = false
    private var isBlocked = false
    private var cancelled = false
    private var blockedWaiters: [CheckedContinuation<Void, Never>] = []
    private var cancellationWaiters: [CheckedContinuation<Void, Never>] = []

    init(pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer
    }

    func start() throws {}

    func nextSample() throws -> AssetReaderSample? {
        condition.lock()
        if !hasReturnedFirstSample {
            hasReturnedFirstSample = true
            condition.unlock()
            return AssetReaderSample(timestamp: .zero, pixelBuffer: pixelBuffer, sequence: 0)
        }

        isBlocked = true
        let waiters = blockedWaiters
        blockedWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
        while !cancelled { condition.wait() }
        condition.unlock()
        throw CancellationError()
    }

    func cancel() {
        condition.lock()
        guard !cancelled else {
            condition.unlock()
            return
        }
        cancelled = true
        let waiters = cancellationWaiters
        cancellationWaiters.removeAll()
        condition.broadcast()
        condition.unlock()
        for waiter in waiters { waiter.resume() }
    }

    func waitUntilBlockedOnNextSample() async {
        await withCheckedContinuation { continuation in
            condition.lock()
            if isBlocked {
                condition.unlock()
                continuation.resume()
            } else {
                blockedWaiters.append(continuation)
                condition.unlock()
            }
        }
    }

    func waitUntilCancelled() async {
        await withCheckedContinuation { continuation in
            condition.lock()
            if cancelled {
                condition.unlock()
                continuation.resume()
            } else {
                cancellationWaiters.append(continuation)
                condition.unlock()
            }
        }
    }
}

private func XCTAssertEqual(
    _ actual: [Double],
    _ expected: [Double],
    accuracy: Double,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (lhs, rhs) in zip(actual, expected) {
        XCTAssertEqual(lhs, rhs, accuracy: accuracy, file: file, line: line)
    }
}

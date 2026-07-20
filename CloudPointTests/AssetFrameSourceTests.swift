import AVFoundation
import CoreGraphics
import ImageIO
import XCTest
@testable import CloudPoint

final class AssetFrameSourceTests: XCTestCase {
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

    func testAssetLoadingAndDecodingAreDemandDriven() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try await VideoFixtureFactory.makeVFRMovie(in: directory)
        let source = AssetFrameSource(assetURL: fixture.url)
        let stream = source.frames(at: [.zero])

        try FileManager.default.removeItem(at: fixture.url)

        do {
            _ = try await Self.collect(stream)
            XCTFail("Expected deferred asset loading to fail after the source was removed")
        } catch {
            XCTAssertFalse(error is CancellationError)
        }
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
        let persistence = JPEGFramePersistence(packageURL: package.url)
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
        let persistence = JPEGFramePersistence(packageURL: package.url)

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
        let persistence = JPEGFramePersistence(packageURL: package.url)

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

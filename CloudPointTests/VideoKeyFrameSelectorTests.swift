import AVFoundation
import ImageIO
import XCTest
@testable import CloudPoint

final class VideoKeyFrameSelectorTests: XCTestCase {
    func testRecommendationBalancesSharpnessExposureAndTemporalPreference() throws {
        let candidates = [
            VideoKeyFrameCandidate.fixture(
                index: 0,
                timestamp: 0.1,
                sharpness: 0.95,
                exposure: 0.1,
                temporal: 0.2
            ),
            VideoKeyFrameCandidate.fixture(
                index: 1,
                timestamp: 0.5,
                sharpness: 0.8,
                exposure: 0.9,
                temporal: 1
            ),
            VideoKeyFrameCandidate.fixture(
                index: 2,
                timestamp: 0.9,
                sharpness: 0.2,
                exposure: 1,
                temporal: 0.2
            ),
        ]

        XCTAssertEqual(VideoKeyFrameSelector.recommended(in: candidates)?.index, 1)
        XCTAssertEqual(
            VideoKeyFrameSelector.selected(in: candidates, preferredIndex: 2)?.index,
            2
        )
        XCTAssertEqual(
            VideoKeyFrameSelector.selected(in: candidates, preferredIndex: 99)?.index,
            1
        )
    }

    func testMovMp4AndM4vCandidatesAreOrientedAndContainFullResolutionJPEG() async throws {
        let containers: [(AVFileType, String)] = [
            (.mov, "mov"),
            (.mp4, "mp4"),
            (AVFileType(rawValue: "com.apple.m4v-video"), "m4v"),
        ]

        for (fileType, filenameExtension) in containers {
            let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let fixture = try await VideoFixtureFactory.makeVFRMovie(
                in: directory,
                fileType: fileType,
                filenameExtension: filenameExtension
            )

            let candidates = try await VideoKeyFrameSelector().candidates(
                for: fixture.url,
                durationSeconds: 1,
                count: 5
            )

            XCTAssertFalse(candidates.isEmpty, "Expected .\(filenameExtension) candidates")
            let candidate = try XCTUnwrap(candidates.first)
            let source = try XCTUnwrap(CGImageSourceCreateWithData(candidate.fullResolutionJPEG as CFData, nil))
            let properties = try XCTUnwrap(
                CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            )
            XCTAssertEqual(properties[kCGImagePropertyPixelWidth] as? Int, VideoFixtureFactory.height)
            XCTAssertEqual(properties[kCGImagePropertyPixelHeight] as? Int, VideoFixtureFactory.width)
            XCTAssertFalse(candidate.thumbnailJPEG.isEmpty)
            XCTAssertTrue(candidate.qualityScore.isFinite)
        }
    }

    func testCameraCandidateCopiesPixelBufferIntoDurableJPEGData() throws {
        let pixelBuffer = try VideoFixtureFactory.makePixelBuffer(
            color: VideoFixtureColor(red: 40, green: 120, blue: 220)
        )
        let frame = CapturedFrame(
            index: 0,
            presentationTimestamp: CMTime(seconds: 1.5, preferredTimescale: 600),
            pixelBuffer: pixelBuffer,
            orientation: .identity,
            sourceSampleSequence: 0
        )

        let candidate = try VideoKeyFrameSelector.candidate(from: frame)

        XCTAssertEqual(candidate.timestampSeconds, 1.5, accuracy: 0.001)
        XCTAssertFalse(candidate.fullResolutionJPEG.isEmpty)
        XCTAssertFalse(candidate.thumbnailJPEG.isEmpty)
        XCTAssertNotNil(CGImageSourceCreateWithData(candidate.fullResolutionJPEG as CFData, nil))
    }
}

private extension VideoKeyFrameCandidate {
    static func fixture(
        index: Int,
        timestamp: Double,
        sharpness: Double,
        exposure: Double,
        temporal: Double
    ) -> VideoKeyFrameCandidate {
        VideoKeyFrameCandidate(
            index: index,
            timestampSeconds: timestamp,
            thumbnailJPEG: Data([1]),
            fullResolutionJPEG: Data([2]),
            sharpnessScore: sharpness,
            exposureScore: exposure,
            temporalScore: temporal
        )
    }
}

import AVFoundation
import XCTest
@testable import CloudPoint

final class FrameSamplerTests: XCTestCase {
    func testFiveFPSSamplingUsesExactMediaTimes() throws {
        let plan = try FrameSamplingPlan(
            duration: CMTime(seconds: 1, preferredTimescale: 600),
            framesPerSecond: 5
        )

        XCTAssertEqual(
            plan.timestamps,
            [0, 120, 240, 360, 480].map { CMTime(value: CMTimeValue($0), timescale: 600) }
        )
    }

    func testSamplingStopsStrictlyBeforeDuration() throws {
        let plan = try FrameSamplingPlan(
            duration: CMTime(value: 601, timescale: 600),
            framesPerSecond: 10
        )

        XCTAssertEqual(
            plan.timestamps,
            stride(from: 0, through: 600, by: 60).map {
                CMTime(value: CMTimeValue($0), timescale: 600)
            }
        )
    }

    func testShortPositiveDurationIncludesInitialTimestamp() throws {
        let plan = try FrameSamplingPlan(
            duration: CMTime(value: 1, timescale: 600),
            framesPerSecond: 1
        )

        XCTAssertEqual(plan.timestamps, [.zero])
    }

    func testRatesOutsideOneThroughTenAreRejected() {
        let duration = CMTime(seconds: 1, preferredTimescale: 600)
        XCTAssertThrowsError(try FrameSamplingPlan(duration: duration, framesPerSecond: 0)) {
            XCTAssertEqual($0 as? FrameSamplingError, .invalidRate(0))
        }
        XCTAssertThrowsError(try FrameSamplingPlan(duration: duration, framesPerSecond: 11)) {
            XCTAssertEqual($0 as? FrameSamplingError, .invalidRate(11))
        }
    }

    func testInvalidDurationsAreRejected() {
        let invalidDurations: [CMTime] = [
            .invalid,
            .indefinite,
            .positiveInfinity,
            .negativeInfinity,
            .zero,
            CMTime(value: -1, timescale: 600),
        ]

        for duration in invalidDurations {
            XCTAssertThrowsError(try FrameSamplingPlan(duration: duration, framesPerSecond: 5)) {
                XCTAssertEqual($0 as? FrameSamplingError, .invalidDuration)
            }
        }
    }
}

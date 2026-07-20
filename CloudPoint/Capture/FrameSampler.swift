import AVFoundation
import Foundation

enum FrameSamplingError: Error, Equatable, Sendable {
    case invalidDuration
    case invalidRate(Int)
}

struct FrameSamplingPlan: Sendable, Equatable {
    static let timescale: CMTimeScale = 600

    let timestamps: [CMTime]

    init(duration: CMTime, framesPerSecond: Int) throws {
        guard (1...10).contains(framesPerSecond) else {
            throw FrameSamplingError.invalidRate(framesPerSecond)
        }
        guard duration.isNumeric, CMTimeCompare(duration, .zero) > 0 else {
            throw FrameSamplingError.invalidDuration
        }

        let sampleInterval = CMTimeValue(Self.timescale) / CMTimeValue(framesPerSecond)
        var sampleIndex: CMTimeValue = 0
        var planned: [CMTime] = []

        while true {
            let (value, overflow) = sampleIndex.multipliedReportingOverflow(by: sampleInterval)
            guard !overflow else { throw FrameSamplingError.invalidDuration }
            let timestamp = CMTime(value: value, timescale: Self.timescale)
            guard CMTimeCompare(timestamp, duration) < 0 else { break }
            planned.append(timestamp)
            sampleIndex += 1
        }

        timestamps = planned
    }
}

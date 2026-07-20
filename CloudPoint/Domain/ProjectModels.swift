import Foundation

enum ProjectRelativePath {
    static func isSafe(_ path: String) -> Bool {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.hasPrefix("~"),
              !path.contains("\\"),
              !path.contains("\0") else {
            return false
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        return components.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

enum WorkerArtifactPath {
    static func depth(frameIndex: UInt32) -> String {
        String(format: "Predictions/%08u.depth-f16", frameIndex)
    }

    static func confidence(frameIndex: UInt32) -> String {
        String(format: "Predictions/%08u.confidence-f16", frameIndex)
    }

    static func geometry(frameIndex: UInt32) -> String {
        String(format: "Predictions/%08u.geometry.json", frameIndex)
    }

    static func points(windowIndex: UInt32) -> String {
        String(format: "Points/window-%08u.cpc", windowIndex)
    }
}

struct PersistedFrame: Codable, Sendable, Equatable {
    var index: UInt32
    var sourceTimestamp: Double
    var relativePath: String
}

struct ResumeCheckpoint: Codable, Sendable, Equatable {
    var lastCommittedFrameIndex: UInt32
    var replayFromFrameIndex: UInt32
    var nextWindowIndex: UInt32
}

struct FrameArtifacts: Codable, Sendable, Equatable {
    var frameIndex: UInt32
    var windowIndex: UInt32
    var depthRelativePath: String
    var confidenceRelativePath: String
    var geometryRelativePath: String
    var durationSeconds: Double
}

struct WindowResult: Codable, Sendable, Equatable {
    var windowIndex: UInt32
    var inferenceFrameStart: UInt32
    var frameStart: UInt32
    var frameEnd: UInt32
    var pointChunkRelativePath: String
    var alignmentRowMajor: [Double]
    var lastProcessedFrameIndex: UInt32
    var inlierCount: UInt64
    var durationSeconds: Double
}

struct CompletedWindow: Codable, Sendable, Equatable {
    var index: UInt32
    var inferenceFrameStart: UInt32
    var frameStart: UInt32
    var frameEnd: UInt32
    var pointChunkRelativePath: String
    var alignmentRowMajor: [Double]
    var lastProcessedFrameIndex: UInt32
    var inlierCount: UInt64
    var durationSeconds: Double
    var frameArtifacts: [FrameArtifacts]
}

import Foundation

struct PersistedFrame: Codable, Sendable, Equatable {
    var index: Int
    var sourceTimestamp: Double
    var relativePath: String
}

struct CompletedWindow: Codable, Sendable, Equatable {
    var index: Int
    var frameStart: Int
    var frameEnd: Int
    var pointChunkRelativePath: String
    var alignmentRowMajor: [Float]
}

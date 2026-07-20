import Foundation

protocol ReconstructionEngine: Sendable {
    func prepare(configuration: EngineConfiguration) async throws
    func begin(project: ProjectDescriptor) async throws
    func enqueue(_ frame: PersistedFrame) async throws
    func finishInput() async throws
    func pause() async throws
    func resume() async throws
    func cancel() async
    func events() -> AsyncThrowingStream<EngineEvent, Error>
    func shutdown() async
}

struct EngineConfiguration: Codable, Sendable, Equatable {
    var windowSize: Int
    var windowOverlap: Int
    var confidenceThreshold: Float

    init(
        windowSize: Int = 32,
        windowOverlap: Int = 8,
        confidenceThreshold: Float = 1.5
    ) {
        self.windowSize = windowSize
        self.windowOverlap = windowOverlap
        self.confidenceThreshold = confidenceThreshold
    }
}

struct ProjectDescriptor: Sendable, Equatable {
    var projectID: UUID
    var packageURL: URL
    var resumeAfterFrameIndex: Int?

    init(projectID: UUID, packageURL: URL, resumeAfterFrameIndex: Int? = nil) {
        self.projectID = projectID
        self.packageURL = packageURL
        self.resumeAfterFrameIndex = resumeAfterFrameIndex
    }
}

struct FrameResult: Sendable, Equatable {
    var frameIndex: Int
    var pointChunkPath: String
    var pointCount: Int

    init(frameIndex: Int, pointChunkPath: String, pointCount: Int) {
        self.frameIndex = frameIndex
        self.pointChunkPath = pointChunkPath
        self.pointCount = pointCount
    }
}

enum EngineEvent: Sendable, Equatable {
    case ready
    case frameStarted(frameIndex: Int)
    case frameCompleted(FrameResult)
    case sessionCompleted
    case paused
    case cancelled

    var frameIndex: Int? {
        switch self {
        case let .frameStarted(frameIndex):
            frameIndex
        case let .frameCompleted(result):
            result.frameIndex
        case .ready, .sessionCompleted, .paused, .cancelled:
            nil
        }
    }
}

enum ReconstructionEngineError: Error, Sendable, Equatable {
    case invalidLifecycle(operation: String)
    case invalidFrameIndex(Int)
}

import Foundation

protocol ReconstructionEngine: Sendable {
    func prepare(configuration: EngineConfiguration) async throws
    func begin(project: ProjectDescriptor) async throws
    func enqueue(_ frame: PersistedFrame) async throws
    func finishInput() async throws
    func pause() async throws
    func resume() async throws
    /// May be called concurrently with any suspended startup operation and must
    /// promptly unblock it. CPU-bound preparation belongs in a cancellable child
    /// task or worker process rather than monopolizing actor isolation.
    func cancel() async
    func events() -> AsyncThrowingStream<EngineEvent, Error>
    /// Has the same concurrent-startup guarantee as `cancel()` and is idempotent.
    func shutdown() async
}

protocol ReconstructionEngineFactory: Sendable {
    func makeEngine(modelDirectory: URL) throws -> any ReconstructionEngine
}

enum EngineConfigurationError: Error, Sendable, Equatable {
    case invalidScaleFrames(UInt32)
    case invalidWindowSize(UInt32)
    case invalidWindowOverlap(UInt32)
    case invalidKeyframeInterval(UInt32)
    case invalidCameraRefinementIterations(UInt32)
    case invalidConfidenceThreshold(Double)
    case invalidVoxelSize(Double)
}

struct EngineConfiguration: Codable, Sendable, Equatable {
    var scaleFrames: UInt32
    var windowSize: UInt32
    var windowOverlap: UInt32
    var keyframeInterval: UInt32
    var cameraRefinementIterations: UInt32
    var confidenceThreshold: Double
    var voxelSize: Double

    init(
        scaleFrames: UInt32 = 8,
        windowSize: UInt32 = 32,
        windowOverlap: UInt32 = 8,
        keyframeInterval: UInt32 = 1,
        cameraRefinementIterations: UInt32 = 4,
        confidenceThreshold: Double = 1.5,
        voxelSize: Double = 0.01
    ) {
        self.scaleFrames = scaleFrames
        self.windowSize = windowSize
        self.windowOverlap = windowOverlap
        self.keyframeInterval = keyframeInterval
        self.cameraRefinementIterations = cameraRefinementIterations
        self.confidenceThreshold = confidenceThreshold
        self.voxelSize = voxelSize
    }

    func validate() throws {
        guard scaleFrames > 0, scaleFrames <= windowSize else {
            throw EngineConfigurationError.invalidScaleFrames(scaleFrames)
        }
        guard (1...1_024).contains(windowSize) else {
            throw EngineConfigurationError.invalidWindowSize(windowSize)
        }
        guard windowOverlap < windowSize else {
            throw EngineConfigurationError.invalidWindowOverlap(windowOverlap)
        }
        guard keyframeInterval > 0 else {
            throw EngineConfigurationError.invalidKeyframeInterval(keyframeInterval)
        }
        guard cameraRefinementIterations > 0 else {
            throw EngineConfigurationError.invalidCameraRefinementIterations(cameraRefinementIterations)
        }
        guard confidenceThreshold.isFinite, confidenceThreshold > 0 else {
            throw EngineConfigurationError.invalidConfidenceThreshold(confidenceThreshold)
        }
        guard voxelSize.isFinite, voxelSize > 0 else {
            throw EngineConfigurationError.invalidVoxelSize(voxelSize)
        }
    }
}

struct ProjectDescriptor: Sendable, Equatable {
    var projectID: UUID
    var packageURL: URL
    var resumeCheckpoint: ResumeCheckpoint?

    init(
        projectID: UUID,
        packageURL: URL,
        resumeCheckpoint: ResumeCheckpoint? = nil
    ) {
        self.projectID = projectID
        self.packageURL = packageURL
        self.resumeCheckpoint = resumeCheckpoint
    }
}

enum EngineEvent: Sendable, Equatable {
    case ready(
        engineVersion: String,
        modelIdentifier: String,
        modelRevision: String,
        convertedWeightsSHA256: String
    )
    case modelProgress(phase: WorkerModelProgressPhase, completed: UInt64, total: UInt64)
    case frameStarted(frameIndex: UInt32, windowIndex: UInt32)
    case frameCompleted(FrameArtifacts)
    case windowCompleted(WindowResult)
    case sessionCompleted(processedFrames: UInt64, windowCount: UInt32, durationSeconds: Double)
    case paused(queuedFrames: UInt64, processedFrames: UInt64)
    case cancelled(lastCompletedWindowIndex: UInt32?)
    case warning(code: String, message: String, recoverable: Bool, details: [String: JSONValue])
    case heartbeat(
        busy: Bool,
        monotonicSeconds: Double,
        queuedFrames: UInt64,
        processedFrames: UInt64,
        currentWindow: UInt32?
    )

    var frameIndex: UInt32? {
        switch self {
        case let .frameStarted(frameIndex, _): frameIndex
        case let .frameCompleted(artifacts): artifacts.frameIndex
        case .ready, .modelProgress, .windowCompleted, .sessionCompleted, .paused,
             .cancelled, .warning, .heartbeat:
            nil
        }
    }
}

enum ReconstructionEngineError: Error, Sendable, Equatable {
    case invalidLifecycle(operation: String)
    case unsafeOutputPath
    case outputAlreadyExists(String)
    case invalidResumeCheckpoint
    case missingReplayArtifacts(UInt32)
    case replayOrderViolation
    case windowIndexOverflow
    case workerFailure(code: String, message: String, recoverable: Bool, details: [String: JSONValue])
}

enum PendingWindowAccumulatorError: Error, Sendable, Equatable {
    case invalidArtifact
    case duplicateFrame(UInt32)
    case outOfOrderFrame(previous: UInt32, next: UInt32)
    case crossWindow(expected: UInt32, actual: UInt32)
    case windowMismatch(expected: UInt32, actual: UInt32)
    case artifactFrameIDsMismatch(expected: [UInt32], actual: [UInt32])
    case invalidWindowResult
}

struct PendingWindowAccumulator: Sendable {
    private var pendingWindowIndex: UInt32?
    private var artifacts: [FrameArtifacts] = []

    mutating func add(_ artifact: FrameArtifacts) throws {
        guard artifact.depthRelativePath == WorkerArtifactPath.depth(frameIndex: artifact.frameIndex),
              artifact.confidenceRelativePath == WorkerArtifactPath.confidence(frameIndex: artifact.frameIndex),
              artifact.geometryRelativePath == WorkerArtifactPath.geometry(frameIndex: artifact.frameIndex),
              artifact.durationSeconds.isFinite,
              artifact.durationSeconds >= 0 else {
            throw PendingWindowAccumulatorError.invalidArtifact
        }
        if let pendingWindowIndex, pendingWindowIndex != artifact.windowIndex {
            throw PendingWindowAccumulatorError.crossWindow(
                expected: pendingWindowIndex,
                actual: artifact.windowIndex
            )
        }
        if let previous = artifacts.last?.frameIndex {
            if previous == artifact.frameIndex {
                throw PendingWindowAccumulatorError.duplicateFrame(artifact.frameIndex)
            }
            guard previous < artifact.frameIndex else {
                throw PendingWindowAccumulatorError.outOfOrderFrame(
                    previous: previous,
                    next: artifact.frameIndex
                )
            }
        }
        pendingWindowIndex = artifact.windowIndex
        artifacts.append(artifact)
    }

    mutating func finalize(
        _ result: WindowResult,
        expectedFrameIndices: [UInt32]
    ) throws -> CompletedWindow {
        if let pendingWindowIndex, pendingWindowIndex != result.windowIndex {
            throw PendingWindowAccumulatorError.windowMismatch(
                expected: pendingWindowIndex,
                actual: result.windowIndex
            )
        }
        let actual = artifacts.map(\.frameIndex)
        guard actual == expectedFrameIndices else {
            throw PendingWindowAccumulatorError.artifactFrameIDsMismatch(
                expected: expectedFrameIndices,
                actual: actual
            )
        }
        guard !expectedFrameIndices.isEmpty,
              expectedFrameIndices.first == result.frameStart,
              expectedFrameIndices.last == result.frameEnd,
              result.inferenceFrameStart <= result.frameStart,
              result.frameStart <= result.frameEnd,
              result.frameEnd <= result.lastProcessedFrameIndex,
              result.pointChunkRelativePath == WorkerArtifactPath.points(windowIndex: result.windowIndex),
              result.alignmentRowMajor.count == 16,
              result.alignmentRowMajor.allSatisfy(\.isFinite),
              result.durationSeconds.isFinite,
              result.durationSeconds >= 0 else {
            throw PendingWindowAccumulatorError.invalidWindowResult
        }

        let completed = CompletedWindow(
            index: result.windowIndex,
            inferenceFrameStart: result.inferenceFrameStart,
            frameStart: result.frameStart,
            frameEnd: result.frameEnd,
            pointChunkRelativePath: result.pointChunkRelativePath,
            alignmentRowMajor: result.alignmentRowMajor,
            lastProcessedFrameIndex: result.lastProcessedFrameIndex,
            inlierCount: result.inlierCount,
            durationSeconds: result.durationSeconds,
            frameArtifacts: artifacts
        )
        artifacts.removeAll(keepingCapacity: true)
        pendingWindowIndex = nil
        return completed
    }

    var pendingArtifacts: [FrameArtifacts] { artifacts }
}

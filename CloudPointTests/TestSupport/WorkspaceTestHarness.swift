import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
@testable import CloudPoint

enum WorkspaceHarnessError: Error {
    case injectedWriteFailure
    case injectedEnqueueFailure
    case timeout
}

actor HarnessManifestStore: ManifestStore {
    private(set) var writes: [ProjectManifest] = []
    private var failingWrite: Int?
    private var didFail = false

    init(failingWrite: Int? = nil) {
        self.failingWrite = failingWrite
    }

    func write(_ manifest: ProjectManifest, to packageURL: URL) async throws {
        if !didFail, writes.count + 1 == failingWrite {
            didFail = true
            throw WorkspaceHarnessError.injectedWriteFailure
        }
        try manifest.writeAtomically(to: packageURL)
        writes.append(manifest)
    }
}

actor HarnessEngine: ReconstructionEngine {
    nonisolated let eventStream: AsyncThrowingStream<EngineEvent, Error>
    private let continuation: AsyncThrowingStream<EngineEvent, Error>.Continuation
    private let packageURL: URL
    private let failEnqueue: Bool
    private(set) var enqueued: [PersistedFrame] = []
    private(set) var begunProjects: [ProjectDescriptor] = []
    private(set) var enqueueSawDurableManifest = false
    private(set) var finishInputCount = 0
    private(set) var finishInputManifest: ProjectManifest?

    init(packageURL: URL, failEnqueue: Bool = false) {
        let stream = AsyncThrowingStream.makeStream(of: EngineEvent.self)
        eventStream = stream.stream
        continuation = stream.continuation
        self.packageURL = packageURL
        self.failEnqueue = failEnqueue
    }

    func prepare(configuration: EngineConfiguration) async throws {
        continuation.yield(.ready(
            engineVersion: "harness",
            modelIdentifier: "harness",
            modelRevision: "1",
            convertedWeightsSHA256: String(repeating: "0", count: 64)
        ))
    }

    func begin(project: ProjectDescriptor) async throws { begunProjects.append(project) }

    func enqueue(_ frame: PersistedFrame) async throws {
        let disk = try ProjectManifest.load(from: packageURL)
        enqueueSawDurableManifest = disk.frames.contains(frame)
            && disk.sessionState.capturedCount == UInt64(disk.frames.count)
            && disk.sessionState.queuedCount < disk.sessionState.capturedCount
            && Self.isJPEG(packageURL.appending(path: frame.relativePath))
        if failEnqueue { throw WorkspaceHarnessError.injectedEnqueueFailure }
        enqueued.append(frame)
    }

    func finishInput() async throws {
        finishInputManifest = try ProjectManifest.load(from: packageURL)
        finishInputCount += 1
    }
    func pause() async throws {}
    func resume() async throws {}
    func cancel() async { continuation.finish() }
    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> { eventStream }
    func shutdown() async { continuation.finish() }

    func emit(_ event: EngineEvent) { continuation.yield(event) }
    func fail(_ error: Error) { continuation.finish(throwing: error) }

    private nonisolated static func isJPEG(_ url: URL) -> Bool {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return false }
        return CGImageSourceGetCount(source) == 1 && CGImageSourceCreateImageAtIndex(source, 0, nil) != nil
    }
}

actor BlockingPrepareEngine: ReconstructionEngine {
    nonisolated let eventStream: AsyncThrowingStream<EngineEvent, Error>
    private let eventContinuation: AsyncThrowingStream<EngineEvent, Error>.Continuation
    private var prepareContinuation: CheckedContinuation<Void, Error>?
    private var cancelCompletion: CheckedContinuation<Void, Never>?
    private var prepareWaiters: [CheckedContinuation<Void, Never>] = []
    private let holdCancelCompletion: Bool
    private(set) var prepareStarted = false
    private(set) var beginCount = 0
    private(set) var cancelCount = 0
    private(set) var shutdownCount = 0
    private(set) var waitingForCancelCompletion = false

    init(holdCancelCompletion: Bool = false) {
        let stream = AsyncThrowingStream.makeStream(of: EngineEvent.self)
        eventStream = stream.stream
        eventContinuation = stream.continuation
        self.holdCancelCompletion = holdCancelCompletion
    }

    func waitUntilPrepareStarts() async {
        if prepareStarted { return }
        await withCheckedContinuation { prepareWaiters.append($0) }
    }

    func prepare(configuration: EngineConfiguration) async throws {
        prepareStarted = true
        let waiters = prepareWaiters
        prepareWaiters.removeAll()
        waiters.forEach { $0.resume() }
        try await withCheckedThrowingContinuation { prepareContinuation = $0 }
    }

    func begin(project: ProjectDescriptor) async throws { beginCount += 1 }
    func enqueue(_ frame: PersistedFrame) async throws {}
    func finishInput() async throws {}
    func pause() async throws {}
    func resume() async throws {}

    func cancel() async {
        cancelCount += 1
        eventContinuation.yield(.ready(
            engineVersion: "interrupted",
            modelIdentifier: "interrupted",
            modelRevision: "1",
            convertedWeightsSHA256: String(repeating: "0", count: 64)
        ))
        eventContinuation.yield(.cancelled(lastCompletedWindowIndex: nil))
        eventContinuation.finish()
        releasePreparation(throwing: CancellationError())
        if holdCancelCompletion {
            waitingForCancelCompletion = true
            await withCheckedContinuation { cancelCompletion = $0 }
            waitingForCancelCompletion = false
        }
    }

    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> { eventStream }

    func shutdown() {
        shutdownCount += 1
        eventContinuation.finish()
        releasePreparation(throwing: CancellationError())
    }

    func forceReleasePreparation() {
        guard let continuation = prepareContinuation else { return }
        prepareContinuation = nil
        continuation.resume()
    }

    func releaseCancelCompletion() {
        cancelCompletion?.resume()
        cancelCompletion = nil
    }

    private func releasePreparation(throwing error: Error) {
        guard let continuation = prepareContinuation else { return }
        prepareContinuation = nil
        continuation.resume(throwing: error)
    }
}

final class HarnessEngineFactory: @unchecked Sendable {
    private let lock = NSLock()
    private let engine: HarnessEngine
    private(set) var callCount = 0

    init(engine: HarnessEngine) { self.engine = engine }

    func make() -> any ReconstructionEngine {
        lock.withLock { callCount += 1 }
        return engine
    }

    var calls: Int { lock.withLock { callCount } }
}

final class HarnessEngineSequence: @unchecked Sendable {
    private let lock = NSLock()
    private let engines: [HarnessEngine]
    private var nextIndex = 0

    init(_ engines: [HarnessEngine]) {
        precondition(!engines.isEmpty)
        self.engines = engines
    }

    func make() throws -> any ReconstructionEngine {
        try lock.withLock {
            guard nextIndex < engines.count else { throw SessionControllerError.engineUnavailable }
            defer { nextIndex += 1 }
            return engines[nextIndex]
        }
    }

    var calls: Int { lock.withLock { nextIndex } }
}

actor HarnessEffects {
    private struct Waiter {
        let id: UUID
        let predicate: @Sendable (WorkspaceSnapshot) -> Bool
        let continuation: CheckedContinuation<WorkspaceSnapshot, Error>
    }

    private(set) var adopted: [ProjectManifest] = []
    private(set) var appendedRanges: [ClosedRange<UInt32>] = []
    private(set) var snapshots: [WorkspaceSnapshot] = []
    private(set) var eventLog: [String] = []
    private var waiters: [Waiter] = []

    func adopt(_ manifest: ProjectManifest) {
        adopted.append(manifest)
        eventLog.append("adopt-\(manifest.completedWindows.count)")
    }

    func append(_ chunk: PointChunk) {
        appendedRanges.append(chunk.firstFrame...chunk.lastFrame)
        eventLog.append("append-\(chunk.firstFrame)-\(chunk.lastFrame)")
    }

    func publish(_ snapshot: WorkspaceSnapshot) {
        snapshots.append(snapshot)
        let matches = waiters.filter { $0.predicate(snapshot) }
        let matchedIDs = Set(matches.map(\.id))
        waiters.removeAll { matchedIDs.contains($0.id) }
        matches.forEach { $0.continuation.resume(returning: snapshot) }
    }

    func next(
        timeout: Duration = .seconds(5),
        where predicate: @escaping @Sendable (WorkspaceSnapshot) -> Bool
    ) async throws -> WorkspaceSnapshot {
        if let existing = snapshots.last(where: predicate) { return existing }
        let id = UUID()
        return try await withCheckedThrowingContinuation { continuation in
            waiters.append(Waiter(id: id, predicate: predicate, continuation: continuation))
            Task { [weak self] in
                try? await Task.sleep(for: timeout)
                await self?.expireWaiter(id)
            }
        }
    }

    private func expireWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: WorkspaceHarnessError.timeout)
    }
}

struct HarnessRecordingImporter: RecordingImporting {
    let frames: [PersistedFrame]

    func importFrames(
        from recordingURL: URL,
        into packageURL: URL,
        startingIndex: UInt32,
        framesPerSecond: Int,
        receive: @escaping @Sendable (PersistedFrame) async throws -> Void
    ) async throws {
        for frame in frames { try await receive(frame) }
    }
}

actor BlockingRecordingImporter: RecordingImporting {
    private let frame: PersistedFrame
    private let throwCancellationAfterFrame: Bool
    private var didStart = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    init(frame: PersistedFrame, throwCancellationAfterFrame: Bool = true) {
        self.frame = frame
        self.throwCancellationAfterFrame = throwCancellationAfterFrame
    }

    func waitUntilStarted() async {
        if didStart { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }

    func releasePersistedFrame() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }

    func importFrames(
        from recordingURL: URL,
        into packageURL: URL,
        startingIndex: UInt32,
        framesPerSecond: Int,
        receive: @escaping @Sendable (PersistedFrame) async throws -> Void
    ) async throws {
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
            didStart = true
            let waiters = startWaiters
            startWaiters.removeAll()
            waiters.forEach { $0.resume() }
        }
        try await receive(frame)
        if throwCancellationAfterFrame { try Task.checkCancellation() }
    }
}

actor BlockingStopCameraInput: CameraInput {
    nonisolated let eventStream: AsyncStream<CameraFrameSourceEvent>
    private nonisolated let continuation: AsyncStream<CameraFrameSourceEvent>.Continuation
    private let completion: CameraLifecycleCompletion
    private var stopStarted = false
    private var stopWasReleased = false
    private var stopWaiters: [CheckedContinuation<Void, Never>] = []
    private var stopContinuation: CheckedContinuation<Void, Never>?
    private(set) var state: CameraFrameSourceState = .idle

    init(lifecycleID: UInt64 = 1) {
        let stream = AsyncStream.makeStream(of: CameraFrameSourceEvent.self)
        eventStream = stream.stream
        continuation = stream.continuation
        completion = CameraLifecycleCompletion(
            lifecycleID: lifecycleID,
            durablePersistedEventCount: 0,
            terminalFailure: nil
        )
    }

    func waitUntilStopStarts() async {
        if stopStarted { return }
        await withCheckedContinuation { stopWaiters.append($0) }
    }

    func releaseStop() {
        stopWasReleased = true
        stopContinuation?.resume()
        stopContinuation = nil
    }

    func consumeEvents(
        _ receive: @escaping @Sendable (CameraFrameSourceEvent) async -> Void
    ) async {
        for await event in eventStream { await receive(event) }
    }

    func start(deviceID: String, sampleRate: Int) {
        state = .running(deviceID: deviceID)
    }

    func stop() async -> CameraLifecycleCompletion {
        stopStarted = true
        let waiters = stopWaiters
        stopWaiters.removeAll()
        waiters.forEach { $0.resume() }
        if !stopWasReleased {
            await withCheckedContinuation { stopContinuation = $0 }
        }
        state = .stopped
        return completion
    }

    func shutdown() {
        releaseStop()
        state = .stopped
        continuation.finish()
    }
}

actor ShutdownRequiresCancelEngine: ReconstructionEngine {
    nonisolated let eventStream: AsyncThrowingStream<EngineEvent, Error>
    private let eventContinuation: AsyncThrowingStream<EngineEvent, Error>.Continuation
    private var shutdownWasReleased = false
    private var shutdownContinuations: [CheckedContinuation<Void, Never>] = []
    private(set) var cancelCount = 0
    private(set) var shutdownCount = 0

    init() {
        let stream = AsyncThrowingStream.makeStream(of: EngineEvent.self)
        eventStream = stream.stream
        eventContinuation = stream.continuation
    }

    func prepare(configuration: EngineConfiguration) async throws {
        eventContinuation.yield(.ready(
            engineVersion: "shutdown-gate",
            modelIdentifier: "shutdown-gate",
            modelRevision: "1",
            convertedWeightsSHA256: String(repeating: "0", count: 64)
        ))
    }

    func begin(project: ProjectDescriptor) async throws {}
    func enqueue(_ frame: PersistedFrame) async throws {}
    func finishInput() async throws {}
    func pause() async throws {}
    func resume() async throws {}

    func cancel() {
        cancelCount += 1
        releaseShutdown()
    }

    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> { eventStream }

    func shutdown() async {
        shutdownCount += 1
        if !shutdownWasReleased {
            await withCheckedContinuation { shutdownContinuations.append($0) }
        }
        eventContinuation.finish()
    }

    func forceReleaseShutdown() {
        releaseShutdown()
    }

    private func releaseShutdown() {
        shutdownWasReleased = true
        let continuations = shutdownContinuations
        shutdownContinuations.removeAll()
        continuations.forEach { $0.resume() }
    }
}

actor HarnessCameraInput: CameraInput {
    nonisolated let eventStream: AsyncStream<CameraFrameSourceEvent>
    private nonisolated let continuation: AsyncStream<CameraFrameSourceEvent>.Continuation
    private let completion: CameraLifecycleCompletion
    private let terminalEvents: [CameraFrameSourceEvent]
    private var didEmitTerminalEvents = false
    private(set) var state: CameraFrameSourceState = .idle

    init(
        completion: CameraLifecycleCompletion,
        terminalEvents: [CameraFrameSourceEvent] = []
    ) {
        let stream = AsyncStream.makeStream(of: CameraFrameSourceEvent.self)
        eventStream = stream.stream
        continuation = stream.continuation
        self.completion = completion
        self.terminalEvents = terminalEvents
    }

    func consumeEvents(
        _ receive: @escaping @Sendable (CameraFrameSourceEvent) async -> Void
    ) async {
        for await event in eventStream { await receive(event) }
    }

    func start(deviceID: String, sampleRate: Int) {
        state = .running(deviceID: deviceID)
    }

    func stop() -> CameraLifecycleCompletion {
        emitTerminalEventsIfNeeded()
        state = .stopped
        return completion
    }

    func shutdown() {
        emitTerminalEventsIfNeeded()
        state = .stopped
        continuation.finish()
    }

    nonisolated func emit(_ event: CameraFrameSourceEvent) {
        continuation.yield(event)
    }

    private func emitTerminalEventsIfNeeded() {
        guard !didEmitTerminalEvents else { return }
        didEmitTerminalEvents = true
        terminalEvents.forEach { continuation.yield($0) }
    }
}

enum WorkspaceTestFiles {
    static func writeJPEG(frameIndex: UInt32, in packageURL: URL) throws -> PersistedFrame {
        let width = 2
        let height = 2
        let bytes: [UInt8] = [
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 255, 255,
        ]
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
        let relativePath = String(format: "Frames/%08u.jpg", frameIndex)
        let url = packageURL.appending(path: relativePath)
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return PersistedFrame(index: frameIndex, sourceTimestamp: Double(frameIndex), relativePath: relativePath)
    }

    static func writeArtifacts(
        frameIndex: UInt32,
        windowIndex: UInt32,
        in packageURL: URL
    ) throws -> FrameArtifacts {
        let artifacts = FrameArtifacts(
            frameIndex: frameIndex,
            windowIndex: windowIndex,
            depthRelativePath: WorkerArtifactPath.depth(frameIndex: frameIndex),
            confidenceRelativePath: WorkerArtifactPath.confidence(frameIndex: frameIndex),
            geometryRelativePath: WorkerArtifactPath.geometry(frameIndex: frameIndex),
            durationSeconds: 0
        )
        for path in [
            artifacts.depthRelativePath,
            artifacts.confidenceRelativePath,
            artifacts.geometryRelativePath,
        ] {
            try Data([0]).write(to: packageURL.appending(path: path))
        }
        return artifacts
    }

    static func writePointChunk(
        windowIndex: UInt32,
        firstFrame: UInt32,
        lastFrame: UInt32,
        in packageURL: URL
    ) throws {
        var data = Data()
        data.append(contentsOf: "CPC1".utf8)
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(PointChunk.vertexStride))
        data.appendLittleEndian(UInt64(0))
        data.appendLittleEndian(firstFrame)
        data.appendLittleEndian(lastFrame)
        data.append(contentsOf: repeatElement(0, count: 8))
        try data.write(to: packageURL.appending(path: WorkerArtifactPath.points(windowIndex: windowIndex)))
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}

extension SessionControllerDependencies {
    static func harness(
        engineFactory: HarnessEngineFactory,
        store: HarnessManifestStore,
        importer: HarnessRecordingImporter,
        effects: HarnessEffects,
        cameraFactory: @escaping @Sendable (URL, UInt32) throws -> any CameraInput = { _, _ in
            throw SessionControllerError.cameraFailure(.configurationFailed)
        },
        now: Date = Date(timeIntervalSinceReferenceDate: 9_000)
    ) -> SessionControllerDependencies {
        SessionControllerDependencies(
            engineFactory: { engineFactory.make() },
            cameraFactory: cameraFactory,
            manifestStore: store,
            recordingImporter: importer,
            jpegValidator: ProductionJPEGValidator(),
            pointChunkOpener: ProductionPointChunkOpener(),
            now: { now },
            effects: SessionControllerEffects(
                adoptManifest: { await effects.adopt($0) },
                appendPointChunk: { await effects.append($0) },
                publishSnapshot: { await effects.publish($0) }
            )
        )
    }
}

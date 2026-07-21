@preconcurrency import AVFoundation
import Darwin
import Foundation
import OSLog

private let cloudPointSessionLogger = Logger(
    subsystem: "cloud.point.cloud.CloudPoint",
    category: "ReconstructionSession"
)
import ImageIO

private final class WeakSessionController: @unchecked Sendable {
    weak var value: SessionController?

    init(_ value: SessionController) {
        self.value = value
    }
}

private enum EngineInterruptionIntent: Sendable, Equatable {
    case cancel
    case close

    var strength: Int {
        switch self {
        case .cancel: 1
        case .close: 2
        }
    }
}

private struct EngineInterruptionSignal: Sendable {
    var intent: EngineInterruptionIntent
    var engine: any ReconstructionEngine
    var generation: UInt64
}

private struct EngineInterruptionRequest: Sendable {
    var id: UInt64
    var signal: EngineInterruptionSignal?
}

/// A lock-backed control plane for the only operations allowed to preempt the
/// FIFO mailbox. It never reads or mutates workflow state; it only routes a
/// cancellation or shutdown signal to an engine that is blocked in a control call.
private final class EngineInterruptionCoordinator: @unchecked Sendable {
    private struct ActiveEngine {
        var engine: any ReconstructionEngine
        var generation: UInt64
        var strongestSignal: EngineInterruptionIntent?
        var isInterruptible: Bool
    }

    private let lock = NSLock()
    private var nextRequestID: UInt64 = 0
    private var requests: [UInt64: EngineInterruptionIntent] = [:]
    private var activeEngine: ActiveEngine?

    func register(_ intent: EngineInterruptionIntent) -> EngineInterruptionRequest {
        lock.withLock {
            nextRequestID &+= 1
            let id = nextRequestID
            requests[id] = intent
            return EngineInterruptionRequest(id: id, signal: claimSignalLocked())
        }
    }

    func finish(_ requestID: UInt64) {
        _ = lock.withLock { requests.removeValue(forKey: requestID) }
    }

    func install(_ engine: any ReconstructionEngine, generation: UInt64) {
        lock.withLock {
            activeEngine = ActiveEngine(
                engine: engine,
                generation: generation,
                strongestSignal: nil,
                isInterruptible: true
            )
        }
    }

    @discardableResult
    func setInterruptible(
        _ isInterruptible: Bool,
        generation: UInt64
    ) -> EngineInterruptionSignal? {
        lock.withLock {
            guard activeEngine?.generation == generation else { return nil }
            activeEngine?.isInterruptible = isInterruptible
            return isInterruptible ? claimSignalLocked() : nil
        }
    }

    func claimPendingSignal(
        for generation: UInt64,
        force: Bool = false
    ) -> EngineInterruptionSignal? {
        lock.withLock {
            guard activeEngine?.generation == generation else { return nil }
            return claimSignalLocked(force: force)
        }
    }

    func hasPendingRequest(for generation: UInt64) -> Bool {
        lock.withLock {
            activeEngine?.generation == generation && !requests.isEmpty
        }
    }

    func clearActiveEngine(generation: UInt64) {
        lock.withLock {
            guard activeEngine?.generation == generation else { return }
            activeEngine = nil
        }
    }

    private func claimSignalLocked(force: Bool = false) -> EngineInterruptionSignal? {
        guard var activeEngine,
              force || activeEngine.isInterruptible,
              let requestedIntent = strongestRequestedIntentLocked(),
              (activeEngine.strongestSignal?.strength ?? 0) < requestedIntent.strength else {
            return nil
        }
        activeEngine.strongestSignal = requestedIntent
        self.activeEngine = activeEngine
        return EngineInterruptionSignal(
            intent: requestedIntent,
            engine: activeEngine.engine,
            generation: activeEngine.generation
        )
    }

    private func strongestRequestedIntentLocked() -> EngineInterruptionIntent? {
        if requests.values.contains(where: { $0 == .close }) { return .close }
        if requests.values.contains(where: { $0 == .cancel }) { return .cancel }
        return nil
    }
}

private actor SessionControllerCompletion {
    private var isFinished = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isFinished else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func finish() {
        guard !isFinished else { return }
        isFinished = true
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume() }
    }
}

enum SessionControllerError: Error, Equatable, Sendable, LocalizedError {
    case packageNotSaved
    case packageRelocationWhileActive
    case engineUnavailable
    case controllerClosed
    case invalidSourceFrame
    case invalidJPEG
    case invalidArtifact(String)
    case replayArtifact(UInt32)
    case cameraFailure(CameraFrameSourceFailure)
    case pointChunkRangeMismatch(expected: ClosedRange<UInt32>, actual: ClosedRange<UInt32>)
    case completionCounterMismatch

    var errorDescription: String? {
        switch self {
        case .packageNotSaved: "Save this project first"
        case .packageRelocationWhileActive:
            "Finish or cancel reconstruction before moving this project"
        case .engineUnavailable: "Lingbot engine not installed yet"
        case .controllerClosed: "This project is closed"
        case .invalidSourceFrame: "The recording produced an invalid frame sequence"
        case .invalidJPEG: "A persisted frame is not a decodable JPEG"
        case let .invalidArtifact(path): "The reconstruction output is invalid: \(path)"
        case let .replayArtifact(frame): "The reconstruction worker emitted output for replay frame \(frame)"
        case let .cameraFailure(failure): "Camera input failed: \(String(describing: failure))"
        case .pointChunkRangeMismatch: "The point-cloud frame range does not match the completed window"
        case .completionCounterMismatch: "The reconstruction worker reported inconsistent completion counters"
        }
    }
}

struct WorkspaceCapabilities: Sendable, Equatable {
    var canImportRecording: Bool
    var canUseCamera: Bool
    var canStopCapture: Bool
    var canPause: Bool
    var canResume: Bool
    var canCancel: Bool
    var canEditSamplingRate: Bool

    static let disabled = WorkspaceCapabilities(
        canImportRecording: false,
        canUseCamera: false,
        canStopCapture: false,
        canPause: false,
        canResume: false,
        canCancel: false,
        canEditSamplingRate: false
    )
}

struct WorkspaceSnapshot: Sendable, Equatable {
    var revision: UInt64
    var phase: SessionPhase
    var isCapturing: Bool
    var capturedCount: UInt64
    var queuedCount: UInt64
    var processedCount: UInt64
    var failedCount: UInt64
    var currentWindow: UInt32?
    var expectedInputCount: UInt64?
    var nextInputOrdinal: UInt64?
    var setupText: String?
    var errorText: String?
    var samplingRate: Int
    var pointSize: Float
    var confidenceThreshold: Float
    var capabilities: WorkspaceCapabilities
}

protocol ManifestStore: Sendable {
    func write(_ manifest: ProjectManifest, to packageURL: URL) async throws
}

struct AtomicManifestStore: ManifestStore {
    func write(_ manifest: ProjectManifest, to packageURL: URL) async throws {
        try manifest.writeAtomically(to: packageURL)
    }
}

protocol RecordingImporting: Sendable {
    func importFrames(
        from recordingURL: URL,
        into packageURL: URL,
        startingIndex: UInt32,
        startingSampleOrdinal: UInt64,
        framesPerSecond: Int,
        receive: @escaping @Sendable (PersistedFrame) async throws -> Void
    ) async throws
}

protocol CameraInput: Sendable {
    var state: CameraFrameSourceState { get async }

    func consumeEvents(
        _ receive: @escaping @Sendable (CameraFrameSourceEvent) async -> Void
    ) async
    func start(deviceID: String, sampleRate: Int) async
    func stop() async -> CameraLifecycleCompletion
    func shutdown() async
}

extension CameraFrameSource: CameraInput {
    nonisolated func consumeEvents(
        _ receive: @escaping @Sendable (CameraFrameSourceEvent) async -> Void
    ) async {
        for await event in events() { await receive(event) }
    }
}

protocol SecurityScopedResourceAccessing: Sendable {
    func startAccessing(_ url: URL) -> Bool
    func stopAccessing(_ url: URL)
}

struct SystemSecurityScopedResourceAccess: SecurityScopedResourceAccessing {
    func startAccessing(_ url: URL) -> Bool { url.startAccessingSecurityScopedResource() }
    func stopAccessing(_ url: URL) { url.stopAccessingSecurityScopedResource() }
}

struct AssetRecordingImporter: RecordingImporting {
    private let scope: any SecurityScopedResourceAccessing

    init(scope: any SecurityScopedResourceAccessing = SystemSecurityScopedResourceAccess()) {
        self.scope = scope
    }

    func importFrames(
        from recordingURL: URL,
        into packageURL: URL,
        startingIndex: UInt32,
        startingSampleOrdinal: UInt64,
        framesPerSecond: Int,
        receive: @escaping @Sendable (PersistedFrame) async throws -> Void
    ) async throws {
        let didStartScope = scope.startAccessing(recordingURL)
        defer { if didStartScope { scope.stopAccessing(recordingURL) } }

        let asset = AVURLAsset(url: recordingURL)
        let duration = try await asset.load(.duration)
        let plan = try FrameSamplingPlan(duration: duration, framesPerSecond: framesPerSecond)
        guard let resumeOrdinal = Int(exactly: startingSampleOrdinal),
              resumeOrdinal <= plan.timestamps.count else {
            throw SessionControllerError.invalidSourceFrame
        }
        let source = AssetFrameSource(assetURL: recordingURL)
        let persistence = try JPEGFramePersistence(packageURL: packageURL)

        for try await sourceFrame in source.frames(at: Array(plan.timestamps.dropFirst(resumeOrdinal))) {
            try Task.checkCancellation()
            guard let sourceOffset = UInt32(exactly: sourceFrame.index) else {
                throw SessionControllerError.invalidSourceFrame
            }
            let (index, overflow) = startingIndex.addingReportingOverflow(sourceOffset)
            guard !overflow, let integerIndex = Int(exactly: index) else {
                throw SessionControllerError.invalidSourceFrame
            }
            let frame = CapturedFrame(
                index: integerIndex,
                presentationTimestamp: sourceFrame.presentationTimestamp,
                pixelBuffer: sourceFrame.pixelBuffer,
                orientation: sourceFrame.orientation,
                sourceSampleSequence: sourceFrame.sourceSampleSequence
            )
            try await receive(persistence.persist(frame))
        }
    }
}

protocol JPEGValidating: Sendable {
    func validate(_ frame: PersistedFrame, in packageURL: URL) throws
}

struct ProductionJPEGValidator: JPEGValidating {
    func validate(_ frame: PersistedFrame, in packageURL: URL) throws {
        let expectedPath = String(format: "Frames/%08u.jpg", frame.index)
        guard frame.relativePath == expectedPath,
              ProjectRelativePath.isSafe(frame.relativePath) else {
            throw SessionControllerError.invalidJPEG
        }
        let url = packageURL.appending(path: frame.relativePath)
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) == 1,
              CGImageSourceCreateImageAtIndex(source, 0, nil) != nil else {
            throw SessionControllerError.invalidJPEG
        }
    }
}

protocol PointChunkOpening: Sendable {
    func open(_ url: URL) throws -> PointChunk
}

struct ProductionPointChunkOpener: PointChunkOpening {
    func open(_ url: URL) throws -> PointChunk { try PointChunk.open(url: url) }
}

struct SessionControllerEffects: @unchecked Sendable {
    var adoptManifest: @Sendable (ProjectManifest) async throws -> Void
    var appendPointChunk: @Sendable (PointChunk) async throws -> Void
    var publishSnapshot: @Sendable (WorkspaceSnapshot) async -> Void

    static let none = SessionControllerEffects(
        adoptManifest: { _ in },
        appendPointChunk: { _ in },
        publishSnapshot: { _ in }
    )
}

struct SessionControllerDependencies: Sendable {
    var engineFactory: @Sendable () throws -> any ReconstructionEngine
    var cameraFactory: @Sendable (URL, UInt32) throws -> any CameraInput
    var manifestStore: any ManifestStore
    var recordingImporter: any RecordingImporting
    var jpegValidator: any JPEGValidating
    var pointChunkOpener: any PointChunkOpening
    var now: @Sendable () -> Date
    var effects: SessionControllerEffects

    init(
        engineFactory: @escaping @Sendable () throws -> any ReconstructionEngine,
        cameraFactory: @escaping @Sendable (URL, UInt32) throws -> any CameraInput = { packageURL, startingIndex in
            CameraFrameSource(
                persistence: try JPEGFramePersistence(packageURL: packageURL),
                startingFrameIndex: startingIndex
            )
        },
        manifestStore: any ManifestStore = AtomicManifestStore(),
        recordingImporter: any RecordingImporting = AssetRecordingImporter(),
        jpegValidator: any JPEGValidating = ProductionJPEGValidator(),
        pointChunkOpener: any PointChunkOpening = ProductionPointChunkOpener(),
        now: @escaping @Sendable () -> Date = Date.init,
        effects: SessionControllerEffects = .none
    ) {
        self.engineFactory = engineFactory
        self.cameraFactory = cameraFactory
        self.manifestStore = manifestStore
        self.recordingImporter = recordingImporter
        self.jpegValidator = jpegValidator
        self.pointChunkOpener = pointChunkOpener
        self.now = now
        self.effects = effects
    }
}

/// Owns project workflow state behind one FIFO command pump.
///
/// The controller intentionally is not an actor: an actor method can interleave while
/// awaiting a manifest write or engine call. Every mutation here is instead submitted
/// to `mailbox`, whose single consumer awaits one complete command at a time.
final class SessionController: @unchecked Sendable {
    private typealias MailOperation = @Sendable () async -> Void

    private let dependencies: SessionControllerDependencies
    private let engineInterruptions = EngineInterruptionCoordinator()
    private let mailbox: AsyncStream<MailOperation>
    private let mailContinuation: AsyncStream<MailOperation>.Continuation
    private var pumpTask: Task<Void, Never>?

    private var manifest: ProjectManifest
    private var packageURL: URL?
    private var engine: (any ReconstructionEngine)?
    private var engineEventTask: Task<Void, Never>?
    private var recordingTask: Task<Void, Never>?
    private var recordingTaskSourceGeneration: UInt64?
    private var cameraInput: (any CameraInput)?
    private var cameraEventTask: Task<Void, Never>?
    private var cameraEventCounts: [UInt64: UInt64] = [:]
    private var pendingCameraCompletion: CameraLifecycleCompletion?
    private var generation: UInt64 = 0
    private var sourceGeneration: UInt64 = 0
    private var revision: UInt64 = 0
    private var closed = false
    private var engineReady = false
    private var pendingWindow = PendingWindowAccumulator()
    private var expectedOutputFrameIDs: [UInt32] = []
    private var replayFrameIDs = Set<UInt32>()
    private var invocationQueuedCount: UInt64 = 0
    private var invocationProcessedCount: UInt64 = 0
    private var invocationWindowCount: UInt32 = 0
    private var setupText: String?
    private var errorText: String?
    private var openRequested = false
    private var openedPackageURL: URL?
    private var finiteResumeGeneration: UInt64?
    private var pendingPackageRelocationURL: URL?
    private var pendingCancelCompletion: SessionControllerCompletion?
    private var pendingCloseCompletion: SessionControllerCompletion?
    private var samplingRate = 2
    private var pointSize: Float = 3
    private var confidenceThreshold: Float = 1.5

    init(
        manifest: ProjectManifest,
        packageURL: URL?,
        dependencies: SessionControllerDependencies
    ) {
        self.manifest = manifest
        self.packageURL = packageURL
        self.dependencies = dependencies
        let stream = AsyncStream.makeStream(of: MailOperation.self)
        mailbox = stream.stream
        mailContinuation = stream.continuation
        pumpTask = Task { [mailbox] in
            for await operation in mailbox { await operation() }
        }
    }

    deinit {
        mailContinuation.finish()
        pumpTask?.cancel()
        engineEventTask?.cancel()
        recordingTask?.cancel()
        cameraEventTask?.cancel()
    }

    func open() async throws {
        try await submit { try await $0.openFromMailbox() }
    }

    func updatePackageURL(_ url: URL?) async {
        _ = try? await submit { controller in
            let currentURL = controller.packageURL?.standardizedFileURL
            let requestedURL = url?.standardizedFileURL
            let isRelocation = currentURL != nil && currentURL != requestedURL
            if isRelocation, let requestedURL {
                if controller.recordingTask != nil {
                    controller.pendingPackageRelocationURL = requestedURL
                    controller.recordingTask?.cancel()
                    controller.setupText = "Finishing the current frame before moving this project"
                    await controller.publishSnapshot()
                    return
                }
                do {
                    try await controller.relocateProject(to: requestedURL)
                } catch {
                    controller.errorText = error.localizedDescription
                    await controller.publishSnapshot()
                }
                return
            }
            if isRelocation {
                controller.errorText = SessionControllerError
                    .packageRelocationWhileActive
                    .localizedDescription
                await controller.publishSnapshot()
                return
            }
            let shouldOpenAfterFirstSave = controller.openRequested
                && controller.packageURL == nil
                && url != nil
            controller.packageURL = url
            controller.setupText = url == nil ? SessionControllerError.packageNotSaved.localizedDescription : nil
            if shouldOpenAfterFirstSave {
                do {
                    try await controller.openFromMailbox()
                } catch {
                    controller.errorText = error.localizedDescription
                    await controller.publishSnapshot()
                }
            } else {
                await controller.publishSnapshot()
            }
        }
    }

    private func relocateProject(to requestedURL: URL) async throws {
        guard let originalURL = packageURL else { throw SessionControllerError.packageNotSaved }
        if cameraInput != nil, manifest.sessionState.isCapturing {
            try await stopCameraFromMailbox()
        }
        let manifestToRelocate = manifest

        let originalGeneration = generation
        sourceGeneration &+= 1
        engineEventTask?.cancel()
        engineEventTask = nil
        recordingTask?.cancel()
        recordingTask = nil
        recordingTaskSourceGeneration = nil
        cameraEventTask?.cancel()
        cameraEventTask = nil
        if let cameraInput { await cameraInput.shutdown() }
        cameraInput = nil
        if let engine {
            let pendingDelivery = engineInterruptions
                .setInterruptible(true, generation: originalGeneration)
                .map { signal in Task { await self.deliver(signal) } }
            await engine.shutdown()
            await pendingDelivery?.value
        }
        engineInterruptions.clearActiveEngine(generation: originalGeneration)
        generation &+= 1
        self.engine = nil
        engineReady = false
        finiteResumeGeneration = nil
        openedPackageURL = nil
        packageURL = requestedURL
        setupText = nil
        errorText = nil

        do {
            try synchronizeDurableProjectFiles(
                manifestToRelocate,
                from: originalURL,
                to: requestedURL
            )
            try await dependencies.manifestStore.write(manifestToRelocate, to: requestedURL)
            manifest = manifestToRelocate
            try await dependencies.effects.adoptManifest(manifestToRelocate)
            try await openFromMailbox(restoreCommittedChunks: false)
        } catch {
            if let engine { await engine.shutdown() }
            self.engine = nil
            await failDurably(error, generation: generation)
            throw error
        }
    }

    private func synchronizeDurableProjectFiles(
        _ manifest: ProjectManifest,
        from sourcePackageURL: URL,
        to destinationPackageURL: URL
    ) throws {
        let fileManager = FileManager.default
        for directory in ["Frames", "Predictions", "Points", "Logs"] {
            try fileManager.createDirectory(
                at: destinationPackageURL.appending(path: directory),
                withIntermediateDirectories: true
            )
        }

        var relativePaths = Set(manifest.frames.map(\.relativePath))
        for window in manifest.completedWindows {
            relativePaths.insert(window.pointChunkRelativePath)
            for artifact in window.frameArtifacts {
                relativePaths.insert(artifact.depthRelativePath)
                relativePaths.insert(artifact.confidenceRelativePath)
                relativePaths.insert(artifact.geometryRelativePath)
            }
        }

        for relativePath in relativePaths.sorted() {
            guard ProjectRelativePath.isSafe(relativePath) else {
                throw SessionControllerError.invalidArtifact(relativePath)
            }
            let sourceURL = sourcePackageURL.appending(path: relativePath)
            let sourceValues = try sourceURL.resourceValues(
                forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
            )
            guard sourceValues.isRegularFile == true,
                  sourceValues.isSymbolicLink != true else {
                throw SessionControllerError.invalidArtifact(relativePath)
            }

            let destinationURL = destinationPackageURL.appending(path: relativePath)
            try fileManager.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let partialURL = destinationURL.deletingLastPathComponent().appending(
                path: ".\(destinationURL.lastPathComponent).\(UUID().uuidString.lowercased()).partial"
            )
            defer { try? fileManager.removeItem(at: partialURL) }
            try fileManager.copyItem(at: sourceURL, to: partialURL)
            let renameResult = partialURL.path.withCString { sourcePath in
                destinationURL.path.withCString { destinationPath in
                    Darwin.rename(sourcePath, destinationPath)
                }
            }
            guard renameResult == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        }
    }

    func importRecording(_ url: URL, framesPerSecond: Int) async throws {
        try await submit { try await $0.startRecordingImport(url, framesPerSecond: framesPerSecond) }
    }

    func useCamera(deviceID: String, sampleRate: Int) async throws {
        try await submit {
            try await $0.startCamera(deviceID: deviceID, sampleRate: sampleRate)
        }
    }

    func stopCamera() async throws {
        try await submit { try await $0.stopCameraFromMailbox() }
    }

    func pause() async throws {
        try await submit { try await $0.pauseFromMailbox() }
    }

    func resume() async throws {
        try await submit { try await $0.resumeFromMailbox() }
    }

    func cancel() async {
        let request = engineInterruptions.register(.cancel)
        defer { engineInterruptions.finish(request.id) }
        let deliveryTask = request.signal.map { signal in
            Task { await self.deliver(signal) }
        }
        let deferredCompletion = try? await submit { await $0.cancelFromMailbox() }
        if let deferredCompletion { await deferredCompletion.wait() }
        await deliveryTask?.value
    }

    func close() async {
        let request = engineInterruptions.register(.close)
        defer { engineInterruptions.finish(request.id) }
        let deliveryTask = request.signal.map { signal in
            Task { await self.deliver(signal) }
        }
        let deferredCompletion = try? await submit { await $0.closeFromMailbox() }
        if let deferredCompletion { await deferredCompletion.wait() }
        await deliveryTask?.value
    }

    func flush() async {
        _ = try? await submit { _ in }
    }

    func currentManifest() async throws -> ProjectManifest {
        try await submit { $0.manifest }
    }

    func replaceRecordingSource(_ replacement: RecordingSourceReference) async throws {
        try await submit { controller in
            guard let current = controller.manifest.recordingSource,
                  replacement.fingerprint == current.fingerprint,
                  replacement.durationSeconds == current.durationSeconds,
                  replacement.framesPerSecond == current.framesPerSecond,
                  replacement.expectedSampleCount == current.expectedSampleCount,
                  replacement.nextSampleOrdinal == current.nextSampleOrdinal else {
                throw RecordingSourceAccessError.changed
            }
            var staged = controller.manifest
            staged.recordingSource = replacement
            staged.updatedAt = controller.dependencies.now()
            try await controller.commit(staged)
        }
    }

    func setSamplingRate(_ value: Int) async {
        _ = try? await submit { controller in
            guard (1...10).contains(value), controller.capabilities.canEditSamplingRate else { return }
            controller.samplingRate = value
            await controller.publishSnapshot()
        }
    }

    func setPointSize(_ value: Float) async {
        _ = try? await submit { controller in
            guard value.isFinite else { return }
            controller.pointSize = min(max(value, 1), 64)
            await controller.publishSnapshot()
        }
    }

    func setConfidenceThreshold(_ value: Float) async {
        _ = try? await submit { controller in
            guard value.isFinite else { return }
            controller.confidenceThreshold = min(max(value, 0), Float(Float16.greatestFiniteMagnitude))
            await controller.publishSnapshot()
        }
    }

    func setCameraMirrorDisplay(_ value: Bool) async throws {
        try await submit { controller in
            guard var source = controller.manifest.cameraSource else { return }
            guard source.mirrorDisplay != value else { return }
            source.mirrorDisplay = value
            var staged = controller.manifest
            staged.cameraSource = source
            staged.updatedAt = controller.dependencies.now()
            try await controller.commit(staged)
        }
    }

    private func submit<Result: Sendable>(
        _ operation: @escaping @Sendable (SessionController) async throws -> Result
    ) async throws -> Result {
        try await withCheckedThrowingContinuation { resultContinuation in
            let result = mailContinuation.yield { [weak self] in
                guard let self else {
                    resultContinuation.resume(throwing: SessionControllerError.controllerClosed)
                    return
                }
                do { resultContinuation.resume(returning: try await operation(self)) }
                catch { resultContinuation.resume(throwing: error) }
            }
            if case .terminated = result {
                resultContinuation.resume(throwing: SessionControllerError.controllerClosed)
            }
        }
    }

    private func enqueue(_ operation: @escaping @Sendable (SessionController) async -> Void) {
        mailContinuation.yield { [weak self] in
            guard let self else { return }
            await operation(self)
        }
    }

    private func deliver(_ signal: EngineInterruptionSignal) async {
        switch signal.intent {
        case .cancel:
            await signal.engine.cancel()
        case .close:
            await signal.engine.shutdown()
        }
    }

    private func checkForStartupInterruption(
        engine: any ReconstructionEngine,
        generation: UInt64
    ) async throws {
        guard engineInterruptions.hasPendingRequest(for: generation) else { return }
        if let signal = engineInterruptions.claimPendingSignal(for: generation) {
            await deliver(signal)
        }
        throw CancellationError()
    }

    private func openFromMailbox(restoreCommittedChunks: Bool = true) async throws {
        guard !closed else { throw SessionControllerError.controllerClosed }
        openRequested = true
        guard let packageURL else {
            setupText = SessionControllerError.packageNotSaved.localizedDescription
            await publishSnapshot()
            return
        }
        guard openedPackageURL != packageURL.standardizedFileURL else {
            await publishSnapshot()
            return
        }
        _ = try ProjectManifest.validate(manifest)

        for window in manifest.completedWindows where restoreCommittedChunks {
            let chunk = try dependencies.pointChunkOpener.open(
                packageURL.appending(path: window.pointChunkRelativePath)
            )
            guard chunk.firstFrame == window.frameStart, chunk.lastFrame == window.frameEnd else {
                throw SessionControllerError.pointChunkRangeMismatch(
                    expected: window.frameStart...window.frameEnd,
                    actual: chunk.firstFrame...chunk.lastFrame
                )
            }
            try await dependencies.effects.appendPointChunk(chunk)
        }

        if [.completed, .cancelled].contains(manifest.sessionState.phase) {
            openedPackageURL = packageURL.standardizedFileURL
            engineReady = false
            await publishSnapshot()
            return
        }

        let createdEngine: any ReconstructionEngine
        do { createdEngine = try dependencies.engineFactory() }
        catch {
            openedPackageURL = packageURL.standardizedFileURL
            setupText = SessionControllerError.engineUnavailable.localizedDescription
            await publishSnapshot()
            return
        }
        openedPackageURL = packageURL.standardizedFileURL
        engine = createdEngine
        engineReady = false
        generation &+= 1
        let activeGeneration = generation
        engineInterruptions.install(createdEngine, generation: activeGeneration)
        let hasDurableWork = !manifest.frames.isEmpty
            || !manifest.completedWindows.isEmpty
            || manifest.sessionState.capturedCount > 0
            || manifest.sessionState.queuedCount > 0
            || manifest.sessionState.processedCount > 0
        let hasIncompleteRecording = manifest.recordingSource.map {
            $0.nextSampleOrdinal < $0.expectedSampleCount
        } ?? false
        finiteResumeGeneration = hasDurableWork && !hasIncompleteRecording
            ? activeGeneration
            : nil
        startEngineEvents(createdEngine, generation: activeGeneration)

        var staged = manifest
        if staged.sessionState.phase == .empty {
            staged.sessionState = try staged.sessionState.applying(.prepare)
        } else {
            staged.sessionState = SessionState(
                phase: .preparing,
                capturedCount: staged.sessionState.capturedCount,
                queuedCount: staged.sessionState.queuedCount,
                processedCount: staged.sessionState.processedCount,
                failedCount: staged.sessionState.failedCount
            )
        }
        staged.updatedAt = dependencies.now()
        try await commit(staged)

        do {
            try await checkForStartupInterruption(
                engine: createdEngine,
                generation: activeGeneration
            )
            try await createdEngine.prepare(configuration: manifest.engineConfiguration)
            try await checkForStartupInterruption(
                engine: createdEngine,
                generation: activeGeneration
            )
            let checkpoint = try manifest.resumeCheckpoint()
            try await createdEngine.begin(project: ProjectDescriptor(
                projectID: manifest.projectID,
                packageURL: packageURL,
                resumeCheckpoint: checkpoint
            ))
            try await checkForStartupInterruption(
                engine: createdEngine,
                generation: activeGeneration
            )

            expectedOutputFrameIDs.removeAll()
            replayFrameIDs.removeAll()
            invocationQueuedCount = 0
            invocationProcessedCount = 0
            invocationWindowCount = 0
            let replayStart = checkpoint?.replayFromFrameIndex
            let replayEnd = checkpoint?.lastCommittedFrameIndex
            let persistedFrames = manifest.frames
            let previouslyQueuedCount = manifest.sessionState.queuedCount
            for (ordinal, frame) in persistedFrames.enumerated()
                where replayStart.map({ frame.index >= $0 }) ?? true {
                try await checkForStartupInterruption(
                    engine: createdEngine,
                    generation: activeGeneration
                )
                try dependencies.jpegValidator.validate(frame, in: packageURL)
                try await createdEngine.enqueue(frame)
                try await checkForStartupInterruption(
                    engine: createdEngine,
                    generation: activeGeneration
                )
                let isReplay = replayEnd.map { frame.index <= $0 } ?? false
                if isReplay {
                    replayFrameIDs.insert(frame.index)
                } else {
                    expectedOutputFrameIDs.append(frame.index)
                    invocationQueuedCount = try checkedAdd(invocationQueuedCount, 1)
                }
                if UInt64(ordinal) >= previouslyQueuedCount {
                    var admitted = manifest
                    admitted.sessionState = try admitted.sessionState.applying(.frameAdmitted)
                    admitted.updatedAt = dependencies.now()
                    try await commit(admitted)
                }
            }
            if finiteResumeGeneration == activeGeneration {
                try await createdEngine.finishInput()
                try await checkForStartupInterruption(
                    engine: createdEngine,
                    generation: activeGeneration
                )
            }
            engineInterruptions.setInterruptible(false, generation: activeGeneration)
        } catch {
            if engineInterruptions.hasPendingRequest(for: activeGeneration) {
                engineEventTask?.cancel()
                engineEventTask = nil
                throw CancellationError()
            }
            await failDurably(error, generation: activeGeneration)
            throw error
        }
    }

    private func startEngineEvents(_ engine: any ReconstructionEngine, generation: UInt64) {
        engineEventTask?.cancel()
        let stream = engine.events()
        engineEventTask = Task { [weak self] in
            do {
                for try await event in stream {
                    guard !Task.isCancelled else { return }
                    self?.enqueue { controller in
                        await controller.handleEngineEvent(event, generation: generation)
                    }
                }
                self?.enqueue { controller in
                    await controller.handleEngineStreamEnd(nil, generation: generation)
                }
            } catch is CancellationError {
            } catch {
                self?.enqueue { controller in
                    await controller.handleEngineStreamEnd(error, generation: generation)
                }
            }
        }
    }

    private func startRecordingImport(_ url: URL, framesPerSecond: Int) async throws {
        guard !closed else { throw SessionControllerError.controllerClosed }
        guard let packageURL else { throw SessionControllerError.packageNotSaved }
        guard engineReady, engine != nil else { throw SessionControllerError.engineUnavailable }
        guard (1...10).contains(framesPerSecond) else {
            throw FrameSamplingError.invalidRate(framesPerSecond)
        }

        var staged = manifest
        staged.sessionState = try staged.sessionState.applying(.startImport)
        staged.updatedAt = dependencies.now()
        samplingRate = framesPerSecond
        try await commit(staged)

        sourceGeneration &+= 1
        let activeSourceGeneration = sourceGeneration
        let startingIndex: UInt32
        if let finalIndex = manifest.frames.last?.index {
            let (next, overflow) = finalIndex.addingReportingOverflow(1)
            guard !overflow else { throw SessionControllerError.invalidSourceFrame }
            startingIndex = next
        } else {
            startingIndex = 0
        }
        let importer = dependencies.recordingImporter
        let startingSampleOrdinal = manifest.recordingSource?.nextSampleOrdinal ?? 0
        cloudPointSessionLogger.info(
            "Starting recording import generation \(activeSourceGeneration) at frame \(startingIndex), sample \(startingSampleOrdinal)"
        )
        recordingTask?.cancel()
        recordingTaskSourceGeneration = activeSourceGeneration
        recordingTask = Task { [weak self] in
            guard let controller = self else { return }
            do {
                try await importer.importFrames(
                    from: url,
                    into: packageURL,
                    startingIndex: startingIndex,
                    startingSampleOrdinal: startingSampleOrdinal,
                    framesPerSecond: framesPerSecond
                ) { frame in
                    try await controller.submit { mailboxOwner in
                        try await mailboxOwner.acceptPersistedFrame(
                            frame,
                            sourceGeneration: activeSourceGeneration,
                            advancesRecordingCursor: true
                        )
                    }
                }
                controller.enqueue { mailboxOwner in
                    await mailboxOwner.recordingSourceEnded(activeSourceGeneration)
                }
            } catch is CancellationError {
                controller.enqueue { mailboxOwner in
                    await mailboxOwner.recordingSourceEnded(activeSourceGeneration)
                }
            } catch {
                controller.enqueue { mailboxOwner in
                    await mailboxOwner.recordingSourceFailed(
                        error,
                        sourceGeneration: activeSourceGeneration
                    )
                }
            }
        }
    }

    private func startCamera(deviceID: String, sampleRate: Int) async throws {
        guard !closed else { throw SessionControllerError.controllerClosed }
        guard let packageURL else { throw SessionControllerError.packageNotSaved }
        guard engineReady, engine != nil else { throw SessionControllerError.engineUnavailable }
        guard (1...10).contains(sampleRate) else {
            throw FrameSamplingError.invalidRate(sampleRate)
        }

        let startingIndex: UInt32
        if let finalIndex = manifest.frames.last?.index {
            let (next, overflow) = finalIndex.addingReportingOverflow(1)
            guard !overflow else { throw SessionControllerError.invalidSourceFrame }
            startingIndex = next
        } else {
            startingIndex = 0
        }
        let input = try dependencies.cameraFactory(packageURL, startingIndex)

        var staged = manifest
        staged.sessionState = try staged.sessionState.applying(.startCapture)
        staged.updatedAt = dependencies.now()
        samplingRate = sampleRate
        try await commit(staged)

        sourceGeneration &+= 1
        let activeSourceGeneration = sourceGeneration
        cameraInput = input
        cameraEventCounts.removeAll()
        pendingCameraCompletion = nil
        startCameraEvents(input, sourceGeneration: activeSourceGeneration)
        await input.start(deviceID: deviceID, sampleRate: sampleRate)
        switch await input.state {
        case .running:
            return
        case let .failed(failure):
            let error = SessionControllerError.cameraFailure(failure)
            await input.shutdown()
            await failDurably(error, generation: generation)
            throw error
        case .idle, .starting, .stopped:
            let error = SessionControllerError.cameraFailure(.configurationFailed)
            await input.shutdown()
            await failDurably(error, generation: generation)
            throw error
        }
    }

    private func startCameraEvents(
        _ input: any CameraInput,
        sourceGeneration: UInt64
    ) {
        cameraEventTask?.cancel()
        let weakController = WeakSessionController(self)
        cameraEventTask = Task {
            await input.consumeEvents { event in
                guard !Task.isCancelled, let controller = weakController.value else { return }
                _ = try? await controller.submit { controller in
                    await controller.handleCameraEvent(
                        event,
                        sourceGeneration: sourceGeneration
                    )
                }
            }
        }
    }

    private func handleCameraEvent(
        _ event: CameraFrameSourceEvent,
        sourceGeneration: UInt64
    ) async {
        guard sourceGeneration == self.sourceGeneration, !closed else { return }
        switch event {
        case let .persisted(payload):
            let received = cameraEventCounts[payload.lifecycleID, default: 0]
            if payload.sequence < received {
                guard manifest.frames.contains(payload.frame) else {
                    await failDurably(SessionControllerError.invalidSourceFrame, generation: generation)
                    return
                }
                return
            }
            guard payload.sequence == received else {
                await failDurably(SessionControllerError.invalidSourceFrame, generation: generation)
                return
            }
            do {
                try await acceptPersistedFrame(
                    payload.frame,
                    sourceGeneration: sourceGeneration,
                    advancesRecordingCursor: false
                )
                cameraEventCounts[payload.lifecycleID] = try checkedAdd(received, 1)
                await finishCameraIfDrained(sourceGeneration: sourceGeneration)
            } catch {
                await failDurably(error, generation: generation)
            }

        case .dropped:
            await publishSnapshot()

        case let .failed(payload):
            let received = cameraEventCounts[payload.lifecycleID, default: 0]
            guard payload.sequence == received else {
                await failDurably(SessionControllerError.invalidSourceFrame, generation: generation)
                return
            }
            await failDurably(
                SessionControllerError.cameraFailure(payload.failure),
                generation: generation
            )
        }
    }

    private func stopCameraFromMailbox() async throws {
        guard let input = cameraInput,
              manifest.sessionState.isCapturing else {
            throw SessionTransitionError.illegal(
                from: manifest.sessionState.phase,
                event: .stopCapture
            )
        }
        let activeSourceGeneration = sourceGeneration
        let completion = await input.stop()
        await input.shutdown()
        try await reconcileCameraCompletion(
            completion,
            sourceGeneration: activeSourceGeneration
        )

        var staged = manifest
        staged.sessionState = try staged.sessionState.applying(.stopCapture)
        staged.updatedAt = dependencies.now()
        try await commit(staged)
        await finishCameraIfDrained(sourceGeneration: activeSourceGeneration)
    }

    private func reconcileCameraCompletion(
        _ completion: CameraLifecycleCompletion,
        sourceGeneration: UInt64
    ) async throws {
        guard sourceGeneration == self.sourceGeneration,
              UInt64(exactly: completion.durablePersistedEvents.count)
                == completion.durablePersistedEventCount else {
            throw SessionControllerError.invalidSourceFrame
        }
        var received = cameraEventCounts[completion.lifecycleID, default: 0]
        for payload in completion.durablePersistedEvents {
            guard payload.lifecycleID == completion.lifecycleID else {
                throw SessionControllerError.invalidSourceFrame
            }
            if payload.sequence < received {
                guard manifest.frames.contains(payload.frame) else {
                    throw SessionControllerError.invalidSourceFrame
                }
                continue
            }
            guard payload.sequence == received else {
                throw SessionControllerError.invalidSourceFrame
            }
            try await acceptPersistedFrame(
                payload.frame,
                sourceGeneration: sourceGeneration,
                advancesRecordingCursor: false
            )
            received = try checkedAdd(received, 1)
            cameraEventCounts[completion.lifecycleID] = received
        }
        guard received == completion.durablePersistedEventCount else {
            throw SessionControllerError.invalidSourceFrame
        }
        pendingCameraCompletion = completion
    }

    private func finishCameraIfDrained(sourceGeneration: UInt64) async {
        guard sourceGeneration == self.sourceGeneration,
              let completion = pendingCameraCompletion else { return }
        let received = cameraEventCounts[completion.lifecycleID, default: 0]
        guard received <= completion.durablePersistedEventCount else {
            await failDurably(SessionControllerError.invalidSourceFrame, generation: generation)
            return
        }
        guard received == completion.durablePersistedEventCount else { return }
        pendingCameraCompletion = nil
        if let failure = completion.terminalFailure {
            await failDurably(SessionControllerError.cameraFailure(failure), generation: generation)
            return
        }
        await finishSource(sourceGeneration)
        cameraInput = nil
    }

    private func acceptPersistedFrame(
        _ frame: PersistedFrame,
        sourceGeneration: UInt64,
        advancesRecordingCursor: Bool
    ) async throws {
        guard sourceGeneration == self.sourceGeneration, !closed else { return }
        guard let packageURL, let engine else { throw SessionControllerError.engineUnavailable }
        guard manifest.frames.last.map({ frame.index > $0.index }) ?? true else {
            throw SessionControllerError.invalidSourceFrame
        }
        try dependencies.jpegValidator.validate(frame, in: packageURL)

        var captured = manifest
        captured.frames.append(frame)
        captured.sessionState = try captured.sessionState.applying(.durableFrameCommitted)
        if advancesRecordingCursor, var source = captured.recordingSource {
            let (nextOrdinal, overflow) = source.nextSampleOrdinal.addingReportingOverflow(1)
            guard !overflow, nextOrdinal <= source.expectedSampleCount else {
                throw SessionControllerError.invalidSourceFrame
            }
            source.nextSampleOrdinal = nextOrdinal
            captured.recordingSource = source
        }
        captured.updatedAt = dependencies.now()
        do {
            try await commit(captured)
        } catch {
            await failDurably(error, generation: generation)
            throw error
        }

        do {
            try await engine.enqueue(frame)
        } catch {
            await failDurably(error, generation: generation)
            throw error
        }

        var admitted = manifest
        admitted.sessionState = try admitted.sessionState.applying(.frameAdmitted)
        admitted.updatedAt = dependencies.now()
        do {
            try await commit(admitted)
            expectedOutputFrameIDs.append(frame.index)
            invocationQueuedCount = try checkedAdd(invocationQueuedCount, 1)
        } catch {
            await failDurably(error, generation: generation)
            throw error
        }
    }

    private func finishSource(_ sourceGeneration: UInt64) async {
        guard sourceGeneration == self.sourceGeneration, !closed, let engine else { return }
        do {
            try await engine.finishInput()
            var staged = manifest
            staged.sessionState = try staged.sessionState.applying(.finishInput)
            staged.updatedAt = dependencies.now()
            try await commit(staged)
        } catch {
            await failDurably(error, generation: generation)
        }
    }

    private func recordingSourceEnded(_ sourceGeneration: UInt64) async {
        guard recordingTaskSourceGeneration == sourceGeneration else { return }
        recordingTask = nil
        recordingTaskSourceGeneration = nil
        guard !closed else { return }
        if let cancellation = pendingCancelCompletion {
            pendingCancelCompletion = nil
            await cancelImmediatelyFromMailbox()
            await performDeferredSourceActions()
            await cancellation.finish()
            return
        }
        guard sourceGeneration == self.sourceGeneration else {
            await performDeferredSourceActions()
            return
        }
        await finishSource(sourceGeneration)
        await performDeferredSourceActions()
    }

    private func recordingSourceFailed(
        _ error: Error,
        sourceGeneration: UInt64
    ) async {
        guard recordingTaskSourceGeneration == sourceGeneration else { return }
        recordingTask = nil
        recordingTaskSourceGeneration = nil
        guard !closed else { return }
        if let cancellation = pendingCancelCompletion {
            pendingCancelCompletion = nil
            await cancelImmediatelyFromMailbox()
            await performDeferredSourceActions()
            await cancellation.finish()
            return
        }
        guard sourceGeneration == self.sourceGeneration else {
            await performDeferredSourceActions()
            return
        }
        await failDurably(error, generation: generation)
        await performDeferredSourceActions()
    }

    private func performDeferredSourceActions() async {
        if let relocationURL = pendingPackageRelocationURL {
            pendingPackageRelocationURL = nil
            do {
                try await relocateProject(to: relocationURL)
            } catch {
                errorText = error.localizedDescription
                await publishSnapshot()
            }
        }
        if let completion = pendingCloseCompletion {
            pendingCloseCompletion = nil
            await closeImmediatelyFromMailbox()
            await completion.finish()
        }
    }

    private func handleEngineEvent(_ event: EngineEvent, generation: UInt64) async {
        guard generation == self.generation, !closed else { return }
        guard !engineInterruptions.hasPendingRequest(for: generation) else { return }
        do {
            switch event {
            case .ready:
                guard manifest.sessionState.phase == .preparing else { return }
                engineReady = true
                setupText = nil
                var staged = manifest
                staged.sessionState = try staged.sessionState.applying(.ready)
                if finiteResumeGeneration == generation {
                    staged.sessionState = try staged.sessionState
                        .applying(.startImport)
                        .applying(.finishInput)
                    finiteResumeGeneration = nil
                }
                staged.updatedAt = dependencies.now()
                try await commit(staged)

            case let .frameStarted(_, windowIndex):
                var staged = manifest
                staged.sessionState = try staged.sessionState.applying(
                    .frameStarted(windowIndex: windowIndex)
                )
                staged.updatedAt = dependencies.now()
                try await commit(staged)

            case let .frameCompleted(artifacts):
                guard !replayFrameIDs.contains(artifacts.frameIndex) else {
                    throw SessionControllerError.replayArtifact(artifacts.frameIndex)
                }
                try validate(artifacts)
                try pendingWindow.add(artifacts)

            case let .windowCompleted(result):
                try await completeWindow(result)

            case let .sessionCompleted(processedFrames, windowCount, _):
                guard processedFrames == invocationProcessedCount,
                      windowCount == invocationWindowCount,
                      invocationProcessedCount == invocationQueuedCount,
                      expectedOutputFrameIDs.isEmpty,
                      pendingWindow.pendingArtifacts.isEmpty else {
                    throw SessionControllerError.completionCounterMismatch
                }
                var staged = manifest
                staged.sessionState = try staged.sessionState
                    .applying(.beginFinalizing)
                    .applying(.complete)
                staged.updatedAt = dependencies.now()
                try await commit(staged)
                engineReady = false
                engineInterruptions.clearActiveEngine(generation: generation)

            case .cancelled:
                guard manifest.sessionState.phase != .cancelled else { return }
                var staged = manifest
                staged.sessionState = try staged.sessionState.applying(.cancel)
                staged.updatedAt = dependencies.now()
                try await commit(staged)
                engineReady = false
                engineInterruptions.clearActiveEngine(generation: generation)

            case .modelProgress, .paused, .warning, .heartbeat:
                await publishSnapshot()
            }
        } catch {
            await failDurably(error, generation: generation)
        }
    }

    private func completeWindow(_ result: WindowResult) async throws {
        guard let packageURL else { throw SessionControllerError.packageNotSaved }
        let expected = expectedOutputFrameIDs.filter {
            $0 >= result.frameStart && $0 <= result.frameEnd
        }
        var stagedAccumulator = pendingWindow
        let completed = try stagedAccumulator.finalize(
            result,
            expectedFrameIndices: expected
        )
        let chunk = try dependencies.pointChunkOpener.open(
            packageURL.appending(path: completed.pointChunkRelativePath)
        )
        let expectedRange = result.frameStart...result.frameEnd
        let actualRange = chunk.firstFrame...chunk.lastFrame
        guard expectedRange == actualRange else {
            throw SessionControllerError.pointChunkRangeMismatch(
                expected: expectedRange,
                actual: actualRange
            )
        }

        var staged = manifest
        staged.completedWindows.append(completed)
        staged.sessionState = try staged.sessionState.applying(
            .windowCommitted(
                windowIndex: completed.index,
                processedFrames: UInt64(completed.frameArtifacts.count)
            )
        )
        staged.updatedAt = dependencies.now()
        try await writeAndAdopt(staged)
        pendingWindow = stagedAccumulator
        let committedIDs = Set(expected)
        expectedOutputFrameIDs.removeAll { committedIDs.contains($0) }
        invocationProcessedCount = try checkedAdd(
            invocationProcessedCount,
            UInt64(completed.frameArtifacts.count)
        )
        let (nextWindowCount, overflow) = invocationWindowCount.addingReportingOverflow(1)
        guard !overflow else { throw SessionControllerError.completionCounterMismatch }
        invocationWindowCount = nextWindowCount
        try await dependencies.effects.appendPointChunk(chunk)
        await publishSnapshot()
    }

    private func validate(_ artifacts: FrameArtifacts) throws {
        guard let packageURL else { throw SessionControllerError.packageNotSaved }
        let paths = [
            (artifacts.depthRelativePath, WorkerArtifactPath.depth(frameIndex: artifacts.frameIndex)),
            (artifacts.confidenceRelativePath, WorkerArtifactPath.confidence(frameIndex: artifacts.frameIndex)),
            (artifacts.geometryRelativePath, WorkerArtifactPath.geometry(frameIndex: artifacts.frameIndex)),
        ]
        for (path, expected) in paths {
            guard path == expected, ProjectRelativePath.isSafe(path) else {
                throw SessionControllerError.invalidArtifact(path)
            }
            let values = try packageURL.appending(path: path).resourceValues(
                forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
            )
            guard values.isRegularFile == true, values.isSymbolicLink != true else {
                throw SessionControllerError.invalidArtifact(path)
            }
        }
    }

    private func pauseFromMailbox() async throws {
        guard let engine else { throw SessionControllerError.engineUnavailable }
        try await engine.pause()
        var staged = manifest
        staged.sessionState = try staged.sessionState.applying(.pause)
        staged.updatedAt = dependencies.now()
        try await commit(staged)
    }

    private func resumeFromMailbox() async throws {
        guard let engine else { throw SessionControllerError.engineUnavailable }
        try await engine.resume()
        var staged = manifest
        staged.sessionState = try staged.sessionState.applying(.resume)
        staged.updatedAt = dependencies.now()
        try await commit(staged)
    }

    private func cancelFromMailbox() async -> SessionControllerCompletion? {
        guard !closed else { return nil }
        guard manifest.sessionState.isProcessing
                || manifest.sessionState.phase == .ready
                || manifest.sessionState.phase == .preparing else {
            return nil
        }
        if recordingTask != nil {
            let completion = pendingCancelCompletion ?? SessionControllerCompletion()
            pendingCancelCompletion = completion
            recordingTask?.cancel()
            return completion
        }
        await cancelImmediatelyFromMailbox()
        await performDeferredSourceActions()
        return nil
    }

    private func cancelImmediatelyFromMailbox() async {
        guard !closed else { return }
        guard manifest.sessionState.isProcessing
                || manifest.sessionState.phase == .ready
                || manifest.sessionState.phase == .preparing else {
            return
        }
        if cameraInput != nil, manifest.sessionState.isCapturing {
            do {
                try await stopCameraFromMailbox()
            } catch {
                errorText = error.localizedDescription
            }
        }
        let activeGeneration = generation
        defer { engineInterruptions.clearActiveEngine(generation: activeGeneration) }
        recordingTask?.cancel()
        recordingTask = nil
        recordingTaskSourceGeneration = nil
        cameraEventTask?.cancel()
        if let cameraInput { await cameraInput.shutdown() }
        cameraInput = nil
        sourceGeneration &+= 1
        if let signal = engineInterruptions.claimPendingSignal(
            for: activeGeneration,
            force: true
        ) {
            await deliver(signal)
        } else if !engineInterruptions.hasPendingRequest(for: activeGeneration),
                  let engine {
            await engine.cancel()
        }
        do {
            var staged = manifest
            staged.sessionState = try staged.sessionState.applying(.cancel)
            staged.updatedAt = dependencies.now()
            try await commit(staged)
        } catch {
            errorText = error.localizedDescription
            await publishSnapshot()
        }
        engineReady = false
        finiteResumeGeneration = nil
    }

    private func closeFromMailbox() async -> SessionControllerCompletion? {
        guard !closed else { return nil }
        if recordingTask != nil {
            let completion = pendingCloseCompletion ?? SessionControllerCompletion()
            pendingCloseCompletion = completion
            recordingTask?.cancel()
            return completion
        }
        await closeImmediatelyFromMailbox()
        return nil
    }

    private func closeImmediatelyFromMailbox() async {
        guard !closed else { return }
        let activeGeneration = generation
        defer { engineInterruptions.clearActiveEngine(generation: activeGeneration) }
        if cameraInput != nil, manifest.sessionState.isCapturing {
            do {
                try await stopCameraFromMailbox()
            } catch {
                errorText = error.localizedDescription
            }
        }
        closed = true
        sourceGeneration &+= 1
        generation &+= 1
        recordingTask?.cancel()
        recordingTask = nil
        recordingTaskSourceGeneration = nil
        cameraEventTask?.cancel()
        cameraEventTask = nil
        if let cameraInput { await cameraInput.shutdown() }
        cameraInput = nil
        engineEventTask?.cancel()
        engineEventTask = nil
        if let signal = engineInterruptions.claimPendingSignal(
            for: activeGeneration,
            force: true
        ) {
            await deliver(signal)
        } else if !engineInterruptions.hasPendingRequest(for: activeGeneration),
                  let engine {
            await engine.shutdown()
        }
        self.engine = nil
        engineReady = false
        finiteResumeGeneration = nil
        await publishSnapshot()
        mailContinuation.finish()
    }

    private func handleEngineStreamEnd(_ error: Error?, generation: UInt64) async {
        guard generation == self.generation, !closed else { return }
        guard !engineInterruptions.hasPendingRequest(for: generation) else { return }
        if let error {
            await failDurably(error, generation: generation)
        } else if ![.completed, .cancelled, .failed].contains(manifest.sessionState.phase) {
            await failDurably(SessionControllerError.engineUnavailable, generation: generation)
        }
    }

    private func failDurably(_ error: Error, generation: UInt64) async {
        guard generation == self.generation, !closed else { return }
        guard !engineInterruptions.hasPendingRequest(for: generation) else { return }
        let diagnostic = String(reflecting: error)
        cloudPointSessionLogger.error(
            "Reconstruction failed: \(diagnostic, privacy: .public)"
        )
        errorText = error.localizedDescription
        engineReady = false
        engineInterruptions.clearActiveEngine(generation: generation)
        self.generation &+= 1
        engineEventTask?.cancel()
        recordingTask?.cancel()
        cameraEventTask?.cancel()
        if let cameraInput { await cameraInput.shutdown() }
        cameraInput = nil
        sourceGeneration &+= 1

        guard ![.completed, .cancelled, .failed, .empty].contains(manifest.sessionState.phase) else {
            await publishSnapshot()
            return
        }
        do {
            var failed = manifest
            failed.sessionState = try failed.sessionState.applying(.fail)
            failed.updatedAt = dependencies.now()
            try await commit(failed)
        } catch {
            errorText = error.localizedDescription
            await publishSnapshot()
        }
    }

    private func commit(_ staged: ProjectManifest) async throws {
        try await writeAndAdopt(staged)
        await publishSnapshot()
    }

    private func writeAndAdopt(_ staged: ProjectManifest) async throws {
        guard let packageURL else { throw SessionControllerError.packageNotSaved }
        _ = try ProjectManifest.validate(staged)
        try await dependencies.manifestStore.write(staged, to: packageURL)
        manifest = staged
        try await dependencies.effects.adoptManifest(staged)
    }

    private var capabilities: WorkspaceCapabilities {
        guard !closed, packageURL != nil else { return .disabled }
        let phase = manifest.sessionState.phase
        return WorkspaceCapabilities(
            canImportRecording: engineReady && phase == .ready,
            canUseCamera: engineReady && phase == .ready,
            canStopCapture: manifest.sessionState.isCapturing,
            canPause: engineReady && [.importing, .capturing, .processing].contains(phase),
            canResume: engineReady && phase == .paused,
            canCancel: phase == .preparing || engineReady && [.ready, .importing, .capturing, .processing, .paused, .finalizing].contains(phase),
            canEditSamplingRate: engineReady && phase == .ready
        )
    }

    private func publishSnapshot() async {
        if revision < .max { revision += 1 }
        let state = manifest.sessionState
        await dependencies.effects.publishSnapshot(WorkspaceSnapshot(
            revision: revision,
            phase: state.phase,
            isCapturing: state.isCapturing,
            capturedCount: state.capturedCount,
            queuedCount: state.queuedCount,
            processedCount: state.processedCount,
            failedCount: state.failedCount,
            currentWindow: state.currentWindow,
            expectedInputCount: manifest.recordingSource?.expectedSampleCount,
            nextInputOrdinal: manifest.recordingSource?.nextSampleOrdinal,
            setupText: packageURL == nil ? SessionControllerError.packageNotSaved.localizedDescription : setupText,
            errorText: errorText,
            samplingRate: samplingRate,
            pointSize: pointSize,
            confidenceThreshold: confidenceThreshold,
            capabilities: capabilities
        ))
    }

    private func checkedAdd(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw SessionTransitionError.counterOverflow }
        return result
    }
}

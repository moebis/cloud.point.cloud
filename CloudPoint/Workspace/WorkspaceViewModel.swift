@preconcurrency import AVFoundation
import Metal
import SwiftUI

enum WorkspaceSourceMode: Sendable, Equatable {
    case recording
    case cameraPreflight
    case camera
    case project
}

enum WorkspaceRecoveryAction: Sendable, Equatable {
    case locateOriginal
    case chooseAnotherVideo
    case repairModel
    case resumeFromCheckpoint

    var title: String {
        switch self {
        case .locateOriginal: "Locate Original…"
        case .chooseAnotherVideo: "Choose Another Video…"
        case .repairModel: "Repair Model"
        case .resumeFromCheckpoint: "Resume from Last Checkpoint"
        }
    }
}

struct WorkspaceProgressPresentation: Sendable, Equatable {
    let title: String
    let completedCount: UInt64
    let totalCount: UInt64?

    static func make(
        phase: SessionPhase,
        source: WorkspaceSourceMode,
        sampledCount: UInt64,
        queuedCount: UInt64,
        processedCount: UInt64,
        expectedCount: UInt64?
    ) -> WorkspaceProgressPresentation {
        switch phase {
        case .empty where source == .cameraPreflight,
             .ready where source == .cameraPreflight:
            WorkspaceProgressPresentation(
                title: "Ready to capture",
                completedCount: 0,
                totalCount: nil
            )
        case .empty, .ready:
            WorkspaceProgressPresentation(
                title: "Ready",
                completedCount: 0,
                totalCount: expectedCount
            )
        case .preparing:
            WorkspaceProgressPresentation(
                title: "Preparing reconstruction model",
                completedCount: 0,
                totalCount: nil
            )
        case .importing:
            WorkspaceProgressPresentation(
                title: "Reading video",
                completedCount: sampledCount,
                totalCount: expectedCount
            )
        case .capturing:
            WorkspaceProgressPresentation(
                title: "Capturing live frames",
                completedCount: sampledCount,
                totalCount: nil
            )
        case .processing where source == .camera || source == .cameraPreflight:
            WorkspaceProgressPresentation(
                title: "Processing remaining camera frames",
                completedCount: processedCount,
                totalCount: queuedCount
            )
        case .processing:
            WorkspaceProgressPresentation(
                title: "Reconstructing scene",
                completedCount: processedCount,
                totalCount: expectedCount ?? queuedCount
            )
        case .paused:
            WorkspaceProgressPresentation(
                title: "Paused",
                completedCount: processedCount,
                totalCount: expectedCount ?? queuedCount
            )
        case .finalizing:
            WorkspaceProgressPresentation(
                title: "Finalizing point cloud",
                completedCount: processedCount,
                totalCount: expectedCount ?? queuedCount
            )
        case .completed:
            WorkspaceProgressPresentation(
                title: "Complete",
                completedCount: processedCount,
                totalCount: expectedCount ?? queuedCount
            )
        case .cancelled:
            WorkspaceProgressPresentation(
                title: "Cancelled",
                completedCount: processedCount,
                totalCount: expectedCount ?? queuedCount
            )
        case .failed:
            WorkspaceProgressPresentation(
                title: "Reconstruction stopped",
                completedCount: processedCount,
                totalCount: expectedCount ?? queuedCount
            )
        }
    }
}

@MainActor
final class WorkspaceViewModel: ObservableObject {
    @Published private(set) var snapshot: WorkspaceSnapshot
    @Published private(set) var cameras: [CameraDescriptor] = []
    @Published var selectedCameraID: String?
    @Published private(set) var sourceMode: WorkspaceSourceMode
    @Published private(set) var recoveryAction: WorkspaceRecoveryAction?
    @Published private(set) var sourceErrorText: String?
    @Published private(set) var mirrorDisplay: Bool

    /// Stable UI-owned capture session. It deliberately is not part of the
    /// equatable WorkspaceSnapshot value.
    let previewSession: AVCaptureSession
    let preflightPreviewSession: AVCaptureSession
    let renderer: PointCloudRenderer?

    private let document: CloudPointDocument
    private let preflightPreview: CameraPreflightPreviewController
    private let recordingSources: any RecordingSourceManaging
    private let panelPresenter: any InputPanelPresenting
    private let onChooseAnotherVideo: () -> Void
    private let onRepairModel: () -> Void
    private let onRetryProject: () -> Void
    private let packageScope: any SecurityScopedResourceAccessing
    private let packageScopeURL: URL
    private var packageScopeIsActive: Bool
    private var controller: SessionController?
    private var initialSource: WorkspaceInitialSource?
    private var didResolvePersistedRecording = false
    private var didStart = false
    private var didClose = false

    var requiresCloseConfirmation: Bool { snapshot.isCapturing }
    var projectURL: URL { packageScopeURL }

    var presentedErrorText: String? { sourceErrorText ?? snapshot.errorText }

    var progress: WorkspaceProgressPresentation {
        WorkspaceProgressPresentation.make(
            phase: snapshot.phase,
            source: sourceMode,
            sampledCount: snapshot.nextInputOrdinal ?? snapshot.capturedCount,
            queuedCount: snapshot.queuedCount,
            processedCount: snapshot.processedCount,
            expectedCount: snapshot.expectedInputCount
        )
    }

    init(
        document: CloudPointDocument,
        packageURL: URL,
        packageBookmarkData: Data? = nil,
        initialSource: WorkspaceInitialSource? = nil,
        recordingSources: any RecordingSourceManaging = SystemRecordingSourceManager(),
        panelPresenter: any InputPanelPresenting = SystemInputPanelPresenter(),
        bookmarks: any SecurityScopedBookmarking = SystemSecurityScopedBookmarks(),
        packageScope: any SecurityScopedResourceAccessing = SystemSecurityScopedResourceAccess(),
        engineFactory injectedEngineFactory: (@Sendable () throws -> any ReconstructionEngine)? = nil,
        onChooseAnotherVideo: @escaping () -> Void = {},
        onRepairModel: @escaping () -> Void = {},
        onRetryProject: @escaping () -> Void = {},
        arguments: [String] = ProcessInfo.processInfo.arguments
    ) {
        self.document = document
        self.recordingSources = recordingSources
        self.panelPresenter = panelPresenter
        self.onChooseAnotherVideo = onChooseAnotherVideo
        self.onRepairModel = onRepairModel
        self.onRetryProject = onRetryProject
        self.packageScope = packageScope
        let initialMirrorDisplay = document.manifest.cameraSource?.mirrorDisplay ?? false
        mirrorDisplay = initialMirrorDisplay
        let resolvedPackageURL: URL
        if let packageBookmarkData,
           let resolution = try? bookmarks.resolve(packageBookmarkData) {
            resolvedPackageURL = resolution.url
        } else {
            resolvedPackageURL = packageURL
        }
        packageScopeURL = resolvedPackageURL
        packageScopeIsActive = packageScope.startAccessing(resolvedPackageURL)
        self.initialSource = initialSource
        recoveryAction = nil
        sourceErrorText = nil
        switch initialSource {
        case let .camera(deviceID, _):
            sourceMode = .cameraPreflight
            selectedCameraID = deviceID
            self.initialSource = nil
        case .recording:
            sourceMode = .recording
        case nil where document.manifest.recordingSource != nil:
            sourceMode = .recording
        case nil where document.manifest.cameraSource != nil:
            sourceMode = Self.stateMode(for: document.manifest.sessionState)
        case nil:
            sourceMode = .project
        }
        let preflightPreview = CameraPreflightPreviewController()
        self.preflightPreview = preflightPreview
        preflightPreviewSession = preflightPreview.session
        let cameraCaptureSession = AVCameraCaptureSession()
        previewSession = cameraCaptureSession.previewSession
        let pointCloudRenderer: PointCloudRenderer?
        if let device = MTLCreateSystemDefaultDevice() {
            pointCloudRenderer = try? PointCloudRenderer(device: device)
        } else {
            pointCloudRenderer = nil
        }
        pointCloudRenderer?.setMirrorDisplay(initialMirrorDisplay)
        renderer = pointCloudRenderer
        let state = document.manifest.sessionState
        snapshot = WorkspaceSnapshot(
            revision: 0,
            phase: state.phase,
            isCapturing: state.isCapturing,
            capturedCount: state.capturedCount,
            queuedCount: state.queuedCount,
            processedCount: state.processedCount,
            failedCount: state.failedCount,
            currentWindow: state.currentWindow,
            expectedInputCount: document.manifest.recordingSource?.expectedSampleCount,
            nextInputOrdinal: document.manifest.recordingSource?.nextSampleOrdinal,
            setupText: nil,
            errorText: nil,
            samplingRate: 2,
            pointSize: 3,
            confidenceThreshold: 1.5,
            capabilities: .disabled
        )

#if DEBUG
        let useMock = Self.shouldUseMockEngine(arguments: arguments)
#endif
        let effects = SessionControllerEffects(
            adoptManifest: { [weak self] manifest in
                await MainActor.run { self?.document.adoptCommittedManifest(manifest) }
            },
            appendPointChunk: { [weak self] chunk in
                try await MainActor.run {
                    guard let renderer = self?.renderer else { return }
                    try renderer.append(chunk)
                }
            },
            publishSnapshot: { [weak self] snapshot in
                await MainActor.run {
                    guard let self else { return }
                    self.snapshot = snapshot
                    self.renderer?.setPointSize(snapshot.pointSize)
                    self.renderer?.setConfidenceThreshold(snapshot.confidenceThreshold)
                    if self.sourceErrorText == nil {
                        if snapshot.setupText != nil {
                            self.recoveryAction = .repairModel
                        } else if snapshot.phase == .failed {
                            self.recoveryAction = .resumeFromCheckpoint
                        } else if self.recoveryAction == .repairModel
                                    || self.recoveryAction == .resumeFromCheckpoint {
                            self.recoveryAction = nil
                        }
                    }
                    self.beginInitialRecordingIfReady()
                }
            }
        )
        let reconstructionEngineFactory: @Sendable () throws -> any ReconstructionEngine
        if let injectedEngineFactory {
            reconstructionEngineFactory = injectedEngineFactory
        } else {
            reconstructionEngineFactory = {
#if DEBUG
                guard useMock else { throw SessionControllerError.engineUnavailable }
                return MockReconstructionEngine()
#else
                throw SessionControllerError.engineUnavailable
#endif
            }
        }
        controller = SessionController(
            manifest: document.manifest,
            packageURL: resolvedPackageURL,
            dependencies: SessionControllerDependencies(
                engineFactory: reconstructionEngineFactory,
                cameraFactory: { packageURL, startingIndex in
                    CameraFrameSource(
                        persistence: try JPEGFramePersistence(packageURL: packageURL),
                        captureSession: cameraCaptureSession,
                        startingFrameIndex: startingIndex
                    )
                },
                effects: effects
            )
        )
    }

    deinit {
        let controllerToClose = controller
        let previewToClose = preflightPreview
        let scope = packageScope
        let scopeURL = packageScopeURL
        let shouldStopScope = packageScopeIsActive
        Task {
            await controllerToClose?.close()
            await previewToClose.stop()
            if shouldStopScope { scope.stopAccessing(scopeURL) }
        }
    }

    static func shouldUseMockEngine(arguments: [String]) -> Bool {
#if DEBUG
        arguments.contains("--mock-engine")
#else
        false
#endif
    }

    func start() {
        guard !didStart else { return }
        didStart = true
        Task { [controller] in
            do { try await controller?.open() }
            catch is CancellationError {}
            catch { await MainActor.run { self.report(error) } }
        }
        Task {
            await resolvePersistedRecordingIfNeeded()
        }
        guard sourceMode == .camera || sourceMode == .cameraPreflight else { return }
        Task {
            let values = await CameraCatalog.devices()
            await MainActor.run {
                cameras = values
                selectedCameraID = selectedCameraID ?? values.first?.id
            }
            if let selectedCameraID {
                await showCameraPreflight(deviceID: selectedCameraID)
            }
        }
    }

    func close() async {
        guard !didClose else { return }
        didClose = true
        await controller?.close()
        await preflightPreview.stop()
        if packageScopeIsActive {
            packageScope.stopAccessing(packageScopeURL)
            packageScopeIsActive = false
        }
    }

    private func beginInitialRecordingIfReady() {
        guard snapshot.capabilities.canImportRecording,
              case let .recording(url, framesPerSecond, _) = initialSource else {
            return
        }
        initialSource = nil
        sourceErrorText = nil
        recoveryAction = nil
        Task { [controller] in
            do {
                try await controller?.importRecording(
                    url,
                    framesPerSecond: framesPerSecond
                )
            }
            catch { await MainActor.run { self.report(error) } }
        }
    }

    func useCamera() {
        guard snapshot.capabilities.canUseCamera,
              let selectedCameraID else { return }
        let rate = snapshot.samplingRate
        sourceMode = .camera
        Task { [controller, preflightPreview] in
            await preflightPreview.stop()
            do {
                try await controller?.useCamera(
                    deviceID: selectedCameraID,
                    sampleRate: rate
                )
            } catch {
                await MainActor.run {
                    self.sourceMode = .cameraPreflight
                    self.report(error)
                }
            }
        }
    }

    func selectCamera(_ deviceID: String?) {
        guard selectedCameraID != deviceID else { return }
        selectedCameraID = deviceID
        guard sourceMode == .cameraPreflight, let deviceID else { return }
        Task { await showCameraPreflight(deviceID: deviceID) }
    }

    func performRecoveryAction() {
        switch recoveryAction {
        case .locateOriginal:
            guard let url = panelPresenter.chooseVideo() else { return }
            Task { await locateOriginal(at: url) }
        case .chooseAnotherVideo:
            onChooseAnotherVideo()
        case .repairModel:
            onRepairModel()
        case .resumeFromCheckpoint:
            onRetryProject()
        case nil:
            break
        }
    }

    func locateOriginal(at url: URL) async {
        guard let source = document.manifest.recordingSource else { return }
        do {
            let replacement = try await recordingSources.replacement(
                for: url,
                preserving: source
            )
            try await controller?.replaceRecordingSource(replacement)
            initialSource = .recording(
                url,
                framesPerSecond: replacement.framesPerSecond,
                expectedSampleCount: replacement.expectedSampleCount
            )
            sourceErrorText = nil
            recoveryAction = nil
            beginInitialRecordingIfReady()
        } catch {
            sourceErrorText = error.localizedDescription
            recoveryAction = .locateOriginal
        }
    }

    func stopCapture() {
        Task { [controller] in
            do { try await controller?.stopCamera() }
            catch { await MainActor.run { self.report(error) } }
        }
    }

    func stopCapture(then action: @escaping @MainActor () -> Void) {
        Task { [controller] in
            do {
                try await controller?.stopCamera()
                await MainActor.run { action() }
            } catch {
                await MainActor.run { self.report(error) }
            }
        }
    }

    func pause() {
        Task { [controller] in
            do { try await controller?.pause() }
            catch { await MainActor.run { self.report(error) } }
        }
    }

    func resume() {
        Task { [controller] in
            do { try await controller?.resume() }
            catch { await MainActor.run { self.report(error) } }
        }
    }

    func cancel() {
        Task { [controller] in await controller?.cancel() }
    }

    func setSamplingRate(_ value: Int) {
        Task { [controller] in await controller?.setSamplingRate(value) }
    }

    func setPointSize(_ value: Float) {
        renderer?.setPointSize(value)
        Task { [controller] in await controller?.setPointSize(value) }
    }

    func setConfidenceThreshold(_ value: Float) {
        renderer?.setConfidenceThreshold(value)
        Task { [controller] in await controller?.setConfidenceThreshold(value) }
    }

    func setMirrorDisplay(_ value: Bool) {
        mirrorDisplay = value
        renderer?.setMirrorDisplay(value)
        Task { [controller] in
            do { try await controller?.setCameraMirrorDisplay(value) }
            catch { await MainActor.run { self.report(error) } }
        }
    }

    func resetView() { renderer?.resetCamera() }

    private func report(_ error: Error) {
        snapshot.errorText = error.localizedDescription
        if error is RecordingSourceAccessError {
            recoveryAction = .locateOriginal
        } else {
            recoveryAction = sourceMode == .recording
                ? .chooseAnotherVideo
                : .resumeFromCheckpoint
        }
    }

    private func resolvePersistedRecordingIfNeeded() async {
        guard !didResolvePersistedRecording,
              initialSource == nil,
              let source = document.manifest.recordingSource,
              source.nextSampleOrdinal < source.expectedSampleCount else {
            return
        }
        didResolvePersistedRecording = true
        do {
            let url = try await recordingSources.resolve(source)
            initialSource = .recording(
                url,
                framesPerSecond: source.framesPerSecond,
                expectedSampleCount: source.expectedSampleCount
            )
            beginInitialRecordingIfReady()
        } catch {
            sourceErrorText = error.localizedDescription
            recoveryAction = .locateOriginal
        }
    }

    private func showCameraPreflight(deviceID: String) async {
        do {
            try await preflightPreview.show(deviceID: deviceID)
            sourceErrorText = nil
        } catch {
            sourceErrorText = "Camera preview is unavailable. Choose another camera and try again."
        }
    }

    private static func stateMode(for state: SessionState) -> WorkspaceSourceMode {
        state.isCapturing || state.capturedCount > 0 ? .camera : .cameraPreflight
    }
}

private final class CameraPreflightPreviewController: @unchecked Sendable {
    let session = AVCaptureSession()

    private let queue = DispatchQueue(label: "cloud.point.cloud.camera.preflight")
    private let queueKey = DispatchSpecificKey<UInt8>()

    init() {
        queue.setSpecific(key: queueKey, value: 1)
    }

    func show(deviceID: String) async throws {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    tearDown()
                    let discovery = AVCaptureDevice.DiscoverySession(
                        deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
                        mediaType: .video,
                        position: .unspecified
                    )
                    guard let device = discovery.devices.first(where: { $0.uniqueID == deviceID }) else {
                        throw CameraPreflightError.noCamera
                    }
                    let input = try AVCaptureDeviceInput(device: device)
                    session.beginConfiguration()
                    session.sessionPreset = .high
                    guard session.canAddInput(input) else {
                        session.commitConfiguration()
                        throw CameraCaptureSessionError.cannotAddInput
                    }
                    session.addInput(input)
                    session.commitConfiguration()
                    session.startRunning()
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func stop() async {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                tearDown()
                continuation.resume()
            }
        }
    }

    deinit {
        if DispatchQueue.getSpecific(key: queueKey) != nil {
            tearDown()
        } else {
            queue.sync { tearDown() }
        }
    }

    private func tearDown() {
        if session.isRunning { session.stopRunning() }
        session.beginConfiguration()
        for output in session.outputs { session.removeOutput(output) }
        for input in session.inputs { session.removeInput(input) }
        session.commitConfiguration()
    }
}

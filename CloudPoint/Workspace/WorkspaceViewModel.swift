@preconcurrency import AVFoundation
import AppKit
import Metal
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class WorkspaceViewModel: ObservableObject {
    static let recordingContentTypes: [UTType] = [
        .movie,
        .quickTimeMovie,
        .mpeg4Movie,
    ]

    @Published private(set) var snapshot: WorkspaceSnapshot
    @Published private(set) var cameras: [CameraDescriptor] = []
    @Published var selectedCameraID: String?

    /// Stable UI-owned capture session. It deliberately is not part of the
    /// equatable WorkspaceSnapshot value.
    let previewSession: AVCaptureSession
    let renderer: PointCloudRenderer?

    private let document: CloudPointDocument
    private var controller: SessionController?
    private var didStart = false

    init(
        document: CloudPointDocument,
        packageURL: URL?,
        arguments: [String] = ProcessInfo.processInfo.arguments
    ) {
        self.document = document
        let cameraCaptureSession = AVCameraCaptureSession()
        previewSession = cameraCaptureSession.previewSession
        if let device = MTLCreateSystemDefaultDevice() {
            renderer = try? PointCloudRenderer(device: device)
        } else {
            renderer = nil
        }
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
            setupText: packageURL == nil ? SessionControllerError.packageNotSaved.localizedDescription : nil,
            errorText: nil,
            samplingRate: 2,
            pointSize: 3,
            confidenceThreshold: 1.5,
            capabilities: .disabled
        )

        let useMock = Self.shouldUseMockEngine(arguments: arguments)
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
                }
            }
        )
        controller = SessionController(
            manifest: document.manifest,
            packageURL: packageURL,
            dependencies: SessionControllerDependencies(
                engineFactory: {
                    guard useMock else { throw SessionControllerError.engineUnavailable }
                    return MockReconstructionEngine()
                },
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
        Task { await controllerToClose?.close() }
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
            let values = await CameraCatalog.devices()
            await MainActor.run {
                cameras = values
                selectedCameraID = selectedCameraID ?? values.first?.id
            }
        }
    }

    func updatePackageURL(_ url: URL?) {
        Task { [controller] in await controller?.updatePackageURL(url) }
    }

    func openRecording() {
        guard snapshot.capabilities.canImportRecording else { return }
        let panel = NSOpenPanel()
        panel.title = "Open Recording"
        panel.prompt = "Open Recording"
        panel.allowedContentTypes = Self.recordingContentTypes
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rate = snapshot.samplingRate
        Task { [controller] in
            do { try await controller?.importRecording(url, framesPerSecond: rate) }
            catch { await MainActor.run { self.report(error) } }
        }
    }

    func useCamera() {
        guard snapshot.capabilities.canUseCamera,
              let selectedCameraID else { return }
        let rate = snapshot.samplingRate
        Task { [controller] in
            do {
                try await controller?.useCamera(
                    deviceID: selectedCameraID,
                    sampleRate: rate
                )
            } catch {
                await MainActor.run { self.report(error) }
            }
        }
    }

    func stopCapture() {
        Task { [controller] in
            do { try await controller?.stopCamera() }
            catch { await MainActor.run { self.report(error) } }
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

    func resetView() { renderer?.resetCamera() }

    private func report(_ error: Error) {
        snapshot.errorText = error.localizedDescription
    }
}

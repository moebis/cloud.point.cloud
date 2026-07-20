import AppKit
import SwiftUI

struct WorkspaceView: View {
    @StateObject private var viewModel: WorkspaceViewModel
    @State private var inspectorPresented = true
    @State private var closeConfirmationPresented = false
    @State private var confirmedClose: (() -> Void)?

    let sourceTitle: String
    let onOpenVideo: () -> Void
    let onShowWelcome: () -> Void

    init(
        document: CloudPointDocument,
        packageURL: URL,
        packageBookmarkData: Data? = nil,
        initialSource: WorkspaceInitialSource? = nil,
        sourceTitle: String = "CloudPoint Project",
        onOpenVideo: @escaping () -> Void = {},
        onRepairModel: @escaping () -> Void = {},
        onShowWelcome: @escaping () -> Void = {}
    ) {
        self.sourceTitle = sourceTitle
        self.onOpenVideo = onOpenVideo
        self.onShowWelcome = onShowWelcome
        _viewModel = StateObject(
            wrappedValue: WorkspaceViewModel(
                document: document,
                packageURL: packageURL,
                packageBookmarkData: packageBookmarkData,
                initialSource: initialSource,
                onChooseAnotherVideo: onOpenVideo,
                onRepairModel: onRepairModel
            )
        )
    }

    var body: some View {
        HSplitView {
            sourceAndProgressSidebar
                .frame(minWidth: 236, idealWidth: 260, maxWidth: 310)

            viewport
                .frame(minWidth: 520)
        }
        .frame(minWidth: 820, minHeight: 600)
        .background(Color(nsColor: .windowBackgroundColor))
        .inspector(isPresented: $inspectorPresented) { inspector }
        .task { viewModel.start() }
        .background {
            WorkspaceWindowCloseGuard(isEnabled: viewModel.requiresCloseConfirmation) { close in
                requestClose(confirmedAction: close)
            }
        }
        .confirmationDialog(
            "Stop camera capture and close?",
            isPresented: $closeConfirmationPresented,
            titleVisibility: .visible
        ) {
            Button("Stop Capture and Close", role: .destructive) {
                let action = confirmedClose
                confirmedClose = nil
                viewModel.stopCapture { action?() }
            }
            Button("Keep Capturing", role: .cancel) { confirmedClose = nil }
        } message: {
            Text("Captured frames are already autosaved. Frames still in the reconstruction queue will be resumed from the last checkpoint.")
        }
        .toolbar { toolbarContent }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Button("All Projects", systemImage: "chevron.left") {
                if viewModel.requiresCloseConfirmation {
                    requestClose(confirmedAction: onShowWelcome)
                } else {
                    onShowWelcome()
                }
            }
            .help("Return to CloudPoint projects")
        }

        ToolbarItem(placement: .principal) {
            VStack(spacing: 1) {
                Text(sourceTitle)
                    .font(.headline)
                    .lineLimit(1)
                Text(viewModel.progress.title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
        }

        ToolbarItemGroup(placement: .primaryAction) {
            if viewModel.snapshot.capabilities.canResume {
                Button("Resume", systemImage: "play.fill") { viewModel.resume() }
                    .help("Resume reconstruction")
            } else {
                Button("Pause", systemImage: "pause.fill") { viewModel.pause() }
                    .disabled(!viewModel.snapshot.capabilities.canPause)
                    .help("Pause reconstruction")
            }

            Button("Cancel", systemImage: "xmark.circle") { viewModel.cancel() }
                .disabled(!viewModel.snapshot.capabilities.canCancel)
                .help("Cancel this reconstruction")

            Button("Reset View", systemImage: "viewfinder") { viewModel.resetView() }
                .help("Reset the point-cloud camera")

            Button("Inspector", systemImage: "sidebar.trailing") {
                inspectorPresented.toggle()
            }
            .help(inspectorPresented ? "Hide inspector" : "Show inspector")

            ShareLink(item: viewModel.projectURL) {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(viewModel.snapshot.phase != .completed)
            .help("Export or share the completed CloudPoint project")
        }
    }

    private var sourceAndProgressSidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    sourceHeader

                    switch viewModel.sourceMode {
                    case .recording:
                        recordingControls
                    case .cameraPreflight, .camera:
                        cameraControls
                    case .project:
                        projectControls
                    }

                    Divider()
                    progressSection

                    if viewModel.presentedErrorText != nil || viewModel.snapshot.setupText != nil {
                        recoveryCard
                    }
                }
                .padding(18)
            }

            autosaveFooter
        }
        .background(.regularMaterial)
    }

    private var sourceHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Source", systemImage: sourceIcon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Text(sourceTitle)
                .font(.title3.weight(.semibold))
                .lineLimit(2)
                .help(sourceTitle)
        }
    }

    private var recordingControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Video recording", systemImage: "film.stack")
                .font(.headline)

            if let total = viewModel.snapshot.expectedInputCount {
                Text("CloudPoint sampled \(total.formatted()) frames from this recording. The source link and resume position are stored with the project.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button("Open Another Video…", systemImage: "plus.rectangle.on.rectangle") {
                onOpenVideo()
            }
            .buttonStyle(.bordered)
        }
    }

    private var cameraControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(
                viewModel.snapshot.isCapturing ? "Live camera" : "Camera preflight",
                systemImage: viewModel.snapshot.isCapturing ? "record.circle.fill" : "video"
            )
            .font(.headline)
            .foregroundStyle(viewModel.snapshot.isCapturing ? .red : .primary)

            if !viewModel.snapshot.isCapturing {
                Text("Choose a camera and sampling quality. Capture starts only when you press Start Capture.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Picker(
                "Camera",
                selection: Binding(
                    get: { viewModel.selectedCameraID },
                    set: { viewModel.selectCamera($0) }
                )
            ) {
                Text("Choose a camera").tag(String?.none)
                ForEach(viewModel.cameras) { camera in
                    Text(camera.name).tag(Optional(camera.id))
                }
            }
            .disabled(viewModel.snapshot.isCapturing)

            Stepper(
                "Sampling: (viewModel.snapshot.samplingRate) fps",
                value: Binding(
                    get: { viewModel.snapshot.samplingRate },
                    set: { viewModel.setSamplingRate($0) }
                ),
                in: 1...10
            )
            .disabled(!viewModel.snapshot.capabilities.canEditSamplingRate)

            if viewModel.snapshot.isCapturing {
                Button("Stop Capture", systemImage: "stop.fill") { viewModel.stopCapture() }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .disabled(!viewModel.snapshot.capabilities.canStopCapture)
            } else {
                Button("Start Capture", systemImage: "record.circle") { viewModel.useCamera() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(
                        !viewModel.snapshot.capabilities.canUseCamera
                            || viewModel.selectedCameraID == nil
                    )
            }
        }
    }

    private var projectControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("CloudPoint project", systemImage: "shippingbox")
                .font(.headline)
            Text("Committed point-cloud windows are restored automatically when this project opens.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button("Open a Video…", systemImage: "film") { onOpenVideo() }
                .buttonStyle(.borderedProminent)
        }
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text(viewModel.progress.title)
                    .font(.headline)
                Spacer()
                if let total = viewModel.progress.totalCount {
                    Text("\(viewModel.progress.completedCount.formatted()) / \(total.formatted())")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            if let total = viewModel.progress.totalCount, total > 0 {
                ProgressView(
                    value: Double(min(viewModel.progress.completedCount, total)),
                    total: Double(total)
                )
                .progressViewStyle(.linear)
            } else if [.preparing, .finalizing].contains(viewModel.snapshot.phase) {
                ProgressView()
                    .controlSize(.small)
            }


            if viewModel.sourceMode == .recording,
               viewModel.snapshot.phase == .importing,
               let total = viewModel.snapshot.expectedInputCount,
               total > 0 {
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Scene reconstruction")
                        Spacer()
                        Text("\(viewModel.snapshot.processedCount.formatted()) / \(total.formatted())")
                            .monospacedDigit()
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    ProgressView(
                        value: Double(min(viewModel.snapshot.processedCount, total)),
                        total: Double(total)
                    )
                }
            }

            VStack(spacing: 8) {
                ProgressMetric(
                    title: viewModel.sourceMode == .recording ? "Read" : "Captured",
                    value: viewModel.snapshot.capturedCount,
                    systemImage: viewModel.sourceMode == .recording ? "film" : "camera"
                )
                ProgressMetric(
                    title: "Queued",
                    value: viewModel.snapshot.queuedCount,
                    systemImage: "tray.full"
                )
                ProgressMetric(
                    title: "Reconstructed",
                    value: viewModel.snapshot.processedCount,
                    systemImage: "point.3.connected.trianglepath.dotted"
                )
            }
        }
    }

    private var recoveryCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(
                viewModel.snapshot.setupText == nil ? "Needs attention" : "Model setup",
                systemImage: viewModel.snapshot.setupText == nil
                    ? "exclamationmark.triangle.fill"
                    : "cpu"
            )
            .font(.headline)

            Text(viewModel.presentedErrorText ?? viewModel.snapshot.setupText ?? "CloudPoint needs an action before it can continue.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)

            if let action = viewModel.recoveryAction {
                Button(action.title) { viewModel.performRecoveryAction() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(12)
        .background(.orange.opacity(0.11), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(.orange.opacity(0.28))
        }
    }

    private var autosaveFooter: some View {
        HStack(spacing: 7) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text("Autosaved")
            Spacer()
            if let window = viewModel.snapshot.currentWindow {
                Text("Window \(window)")
                    .monospacedDigit()
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 18)
        .frame(height: 38)
        .background(.bar)
    }

    private var viewport: some View {
        ZStack {
            if let renderer = viewModel.renderer {
                PointCloudView(renderer: renderer)
            } else {
                ContentUnavailableView(
                    "Metal is unavailable",
                    systemImage: "display.trianglebadge.exclamationmark"
                )
            }

            if viewModel.sourceMode == .cameraPreflight,
               !viewModel.snapshot.isCapturing,
               viewModel.snapshot.capturedCount == 0 {
                cameraPreflightSurface
            } else if viewModel.snapshot.isCapturing {
                CameraPreviewView(session: viewModel.previewSession)
                    .background(.black)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay {
                        RoundedRectangle(cornerRadius: 14)
                            .strokeBorder(.white.opacity(0.14))
                    }
                    .shadow(radius: 18)
                    .frame(width: 320, height: 200)
                    .padding(20)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            } else if viewModel.snapshot.capturedCount == 0,
                      viewModel.snapshot.phase != .failed {
                ContentUnavailableView(
                    emptyViewportTitle,
                    systemImage: "point.3.connected.trianglepath.dotted",
                    description: Text(emptyViewportDescription)
                )
                .allowsHitTesting(false)
            }
        }
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.018, green: 0.025, blue: 0.04),
                    Color(red: 0.035, green: 0.05, blue: 0.075),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
    }

    private var cameraPreflightSurface: some View {
        VStack(spacing: 18) {
            ZStack {
                RoundedRectangle(cornerRadius: 22)
                    .fill(.black.opacity(0.5))
                CameraPreviewView(session: viewModel.preflightPreviewSession)
                    .clipShape(RoundedRectangle(cornerRadius: 22))
            }
            .aspectRatio(16 / 9, contentMode: .fit)
            .frame(maxWidth: 640)
            .overlay(alignment: .bottomLeading) {
                Text(selectedCameraName)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(12)
            }

            VStack(spacing: 5) {
                Text("Camera is ready")
                    .font(.title2.bold())
                Text("Review the camera and sampling settings, then start capture from the Source panel.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var inspector: some View {
        Form {
            Section("Reconstruction quality") {
                Stepper(
                    "Sampling: \(viewModel.snapshot.samplingRate) fps",
                    value: Binding(
                        get: { viewModel.snapshot.samplingRate },
                        set: { viewModel.setSamplingRate($0) }
                    ),
                    in: 1...10
                )
                .disabled(!viewModel.snapshot.capabilities.canEditSamplingRate)

                LabeledContent("Confidence") {
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.snapshot.confidenceThreshold) },
                            set: { viewModel.setConfidenceThreshold(Float($0)) }
                        ),
                        in: 0...5
                    )
                    .frame(minWidth: 120)
                }
            }

            Section("Point-cloud display") {
                LabeledContent("Point size") {
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.snapshot.pointSize) },
                            set: { viewModel.setPointSize(Float($0)) }
                        ),
                        in: 1...12
                    )
                    .frame(minWidth: 120)
                }

                Button("Reset View", systemImage: "viewfinder") { viewModel.resetView() }
            }

            Section("Project") {
                LabeledContent("Status", value: viewModel.progress.title)
                LabeledContent("Captured", value: viewModel.snapshot.capturedCount.formatted())
                LabeledContent("Reconstructed", value: viewModel.snapshot.processedCount.formatted())
            }
        }
        .formStyle(.grouped)
        .inspectorColumnWidth(min: 250, ideal: 290, max: 360)
    }

    private var sourceIcon: String {
        switch viewModel.sourceMode {
        case .recording: "film.stack"
        case .cameraPreflight, .camera: "video"
        case .project: "shippingbox"
        }
    }

    private var emptyViewportTitle: String {
        viewModel.snapshot.phase == .preparing ? "Preparing the model" : "Building your 3D scene"
    }

    private var emptyViewportDescription: String {
        if viewModel.snapshot.phase == .preparing {
            return "CloudPoint is loading the local MLX reconstruction model."
        }
        return "Geometry appears here as reconstruction windows finish."
    }

    private var selectedCameraName: String {
        guard let selectedCameraID = viewModel.selectedCameraID else { return "Choose a camera" }
        return viewModel.cameras.first { $0.id == selectedCameraID }?.name ?? "Selected camera"
    }

    private func requestClose(confirmedAction: @escaping () -> Void) {
        confirmedClose = confirmedAction
        closeConfirmationPresented = true
    }
}

private struct ProgressMetric: View {
    let title: String
    let value: UInt64
    let systemImage: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
                .frame(width: 18)
            Text(title)
            Spacer()
            Text(value.formatted())
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }
}

@MainActor
private struct WorkspaceWindowCloseGuard: NSViewRepresentable {
    let isEnabled: Bool
    let onCloseRequested: (@escaping () -> Void) -> Void

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        Task { @MainActor in context.coordinator.attach(to: view.window) }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        context.coordinator.isEnabled = isEnabled
        context.coordinator.onCloseRequested = onCloseRequested
        Task { @MainActor in context.coordinator.attach(to: view.window) }
    }

    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) {
        coordinator.detach()
    }

    @MainActor
    final class Coordinator: NSObject, NSWindowDelegate {
        weak var window: NSWindow?
        nonisolated(unsafe) weak var previousDelegate: NSWindowDelegate?
        var isEnabled = false
        var onCloseRequested: ((@escaping () -> Void) -> Void)?
        private var allowsNextClose = false

        func attach(to window: NSWindow?) {
            guard let window, self.window !== window else { return }
            detach()
            self.window = window
            previousDelegate = window.delegate
            window.delegate = self
        }

        func detach() {
            if window?.delegate === self { window?.delegate = previousDelegate }
            window = nil
            previousDelegate = nil
        }

        func windowShouldClose(_ sender: NSWindow) -> Bool {
            if allowsNextClose {
                allowsNextClose = false
                return previousDelegate?.windowShouldClose?(sender) ?? true
            }
            guard isEnabled else {
                return previousDelegate?.windowShouldClose?(sender) ?? true
            }
            onCloseRequested? { [weak self] in
                guard let self, let window = self.window else { return }
                allowsNextClose = true
                window.performClose(nil)
            }
            return false
        }

        override func responds(to selector: Selector!) -> Bool {
            super.responds(to: selector) || previousDelegate?.responds(to: selector) == true
        }

        override func forwardingTarget(for selector: Selector!) -> Any? {
            previousDelegate?.responds(to: selector) == true
                ? previousDelegate
                : super.forwardingTarget(for: selector)
        }
    }
}

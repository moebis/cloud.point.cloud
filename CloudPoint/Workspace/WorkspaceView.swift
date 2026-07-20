import SwiftUI

struct WorkspaceView: View {
    @StateObject private var viewModel: WorkspaceViewModel
    let sourceTitle: String
    let onOpenVideo: () -> Void
    let onShowWelcome: () -> Void

    init(
        document: CloudPointDocument,
        packageURL: URL,
        initialSource: WorkspaceInitialSource? = nil,
        sourceTitle: String = "CloudPoint Project",
        onOpenVideo: @escaping () -> Void = {},
        onShowWelcome: @escaping () -> Void = {}
    ) {
        self.sourceTitle = sourceTitle
        self.onOpenVideo = onOpenVideo
        self.onShowWelcome = onShowWelcome
        _viewModel = StateObject(
            wrappedValue: WorkspaceViewModel(
                document: document,
                packageURL: packageURL,
                initialSource: initialSource
            )
        )
    }

    var body: some View {
        HSplitView {
            sourceSidebar
                .frame(minWidth: 220, idealWidth: 250, maxWidth: 320)

            viewport
                .frame(minWidth: 520)

            inspector
                .frame(minWidth: 230, idealWidth: 260, maxWidth: 320)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { timeline }
        .frame(minWidth: 1_000, minHeight: 680)
        .task { viewModel.start() }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button("Projects", systemImage: "chevron.left") { onShowWelcome() }
            }
            ToolbarItem(placement: .principal) {
                Text(sourceTitle)
                    .font(.headline)
                    .lineLimit(1)
            }
            ToolbarItem(placement: .primaryAction) {
                Button("Open Video", systemImage: "film") { onOpenVideo() }
                    .keyboardShortcut("o", modifiers: .command)
            }
        }
    }

    private var sourceSidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Source")
                .font(.title2.bold())

            Button("Open Another Video…", systemImage: "film") { onOpenVideo() }
            .buttonStyle(.borderedProminent)

            Picker("Camera", selection: $viewModel.selectedCameraID) {
                Text("Choose a camera").tag(String?.none)
                ForEach(viewModel.cameras) { camera in
                    Text(camera.name).tag(Optional(camera.id))
                }
            }

            Button("Use Camera", systemImage: "video") {
                viewModel.useCamera()
            }
            .disabled(!viewModel.snapshot.capabilities.canUseCamera || viewModel.selectedCameraID == nil)

            Button("Stop Capture", systemImage: "stop.fill") {
                viewModel.stopCapture()
            }
            .disabled(!viewModel.snapshot.capabilities.canStopCapture)

            Divider()

            LabeledContent("Phase") {
                Text(viewModel.snapshot.phase.rawValue.capitalized)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Captured", value: viewModel.snapshot.capturedCount.formatted())
            LabeledContent("Admitted", value: viewModel.snapshot.queuedCount.formatted())
            LabeledContent("Processed", value: viewModel.snapshot.processedCount.formatted())

            if let setup = viewModel.snapshot.setupText {
                Label(setup, systemImage: "info.circle")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if let error = viewModel.snapshot.errorText {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            Spacer()
        }
        .padding(18)
        .background(.regularMaterial)
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

            if viewModel.snapshot.isCapturing {
                CameraPreviewView(session: viewModel.previewSession)
                    .background(.black.opacity(0.6))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .frame(width: 260, height: 160)
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                    .allowsHitTesting(false)
            }

            if viewModel.snapshot.capturedCount == 0,
               !viewModel.snapshot.isCapturing,
               viewModel.snapshot.phase != .failed {
                ContentUnavailableView(
                    "Create a 3D map",
                    systemImage: "point.3.connected.trianglepath.dotted",
                    description: Text("Open a recording or use a camera.")
                )
                .allowsHitTesting(false)
            }
        }
        .background(Color(red: 0.025, green: 0.03, blue: 0.04))
    }

    private var inspector: some View {
        Form {
            Section("Reconstruction") {
                Stepper(
                    "Sampling: \(viewModel.snapshot.samplingRate) fps",
                    value: Binding(
                        get: { viewModel.snapshot.samplingRate },
                        set: { viewModel.setSamplingRate($0) }
                    ),
                    in: 1...10
                )
                .disabled(!viewModel.snapshot.capabilities.canEditSamplingRate)
            }

            Section("Display") {
                LabeledContent("Point size") {
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.snapshot.pointSize) },
                            set: { viewModel.setPointSize(Float($0)) }
                        ),
                        in: 1...12
                    )
                    .frame(width: 120)
                }
                LabeledContent("Confidence") {
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.snapshot.confidenceThreshold) },
                            set: { viewModel.setConfidenceThreshold(Float($0)) }
                        ),
                        in: 0...5
                    )
                    .frame(width: 120)
                }
                Button("Reset View", systemImage: "viewfinder") {
                    viewModel.resetView()
                }
            }

            Section("Session") {
                Button("Pause", systemImage: "pause.fill") { viewModel.pause() }
                    .disabled(!viewModel.snapshot.capabilities.canPause)
                Button("Resume", systemImage: "play.fill") { viewModel.resume() }
                    .disabled(!viewModel.snapshot.capabilities.canResume)
                Button("Cancel", systemImage: "xmark.circle") { viewModel.cancel() }
                    .disabled(!viewModel.snapshot.capabilities.canCancel)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(.regularMaterial)
    }

    private var timeline: some View {
        VStack(spacing: 8) {
            Divider()
            HStack(spacing: 14) {
                Image(systemName: "timeline.selection")
                ProgressView(
                    value: Double(viewModel.snapshot.processedCount),
                    total: Double(max(viewModel.snapshot.queuedCount, 1))
                )
                Text("\(viewModel.snapshot.processedCount) / \(viewModel.snapshot.queuedCount) frames")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                if let window = viewModel.snapshot.currentWindow {
                    Text("Window \(window)")
                        .font(.caption.monospacedDigit())
                }
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 10)
        }
        .background(.bar)
    }
}

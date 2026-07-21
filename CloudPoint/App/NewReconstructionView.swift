import AppKit
import SwiftUI

struct NewReconstructionView: View {
    @ObservedObject var coordinator: AppCoordinator
    let request: PendingReconstructionRequest

    @State private var candidates: [VideoKeyFrameCandidate] = []
    @State private var selectedFrameIndex: Int?
    @State private var isLoadingFrames = false
    @State private var frameError: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    sourceSummary
                    modeCards
                    if case .video = request.source {
                        framePicker
                    }
                    if let error = frameError ?? coordinator.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.callout)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                    }
                }
                .padding(28)
            }
        }
        .frame(width: 820, height: 680)
        .background(.ultraThickMaterial)
        .task(id: request.id) { await loadFramesIfNeeded() }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 13)
                    .fill(.blue.gradient)
                Image(systemName: "view.3d")
                    .font(.system(size: 23, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .frame(width: 48, height: 48)

            VStack(alignment: .leading, spacing: 3) {
                Text("Create a reconstruction")
                    .font(.title2.bold())
                Text("Choose the spatial representation that fits this source.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Cancel", role: .cancel) {
                coordinator.cancelPendingReconstruction()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(22)
    }

    private var sourceSummary: some View {
        HStack(spacing: 12) {
            Image(systemName: sourceIcon)
                .font(.title3)
                .frame(width: 30)
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text(sourceTitle)
                    .font(.headline)
                    .lineLimit(1)
                Text(sourceDetail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(14)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
        .accessibilityElement(children: .combine)
    }

    private var modeCards: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Reconstruction mode")
                .font(.headline)
            HStack(alignment: .top, spacing: 14) {
                ReconstructionModeCard(
                    icon: "circle.hexagongrid.fill",
                    tint: .cyan,
                    title: "Point Cloud",
                    badge: "MULTI-FRAME",
                    description: "Samples the full recording or live feed and reconstructs colored 3D points with LingBot on MLX.",
                    details: "Best for moving through a scene",
                    actionTitle: "Create Point Cloud",
                    isEnabled: !coordinator.isBusy
                ) {
                    Task {
                        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)
                    }
                }

                ReconstructionModeCard(
                    icon: "sparkles.rectangle.stack.fill",
                    tint: .purple,
                    title: "Gaussian Scene",
                    badge: "APPLE SHARP · EXPERIMENTAL",
                    description: sharpDescription,
                    details: sharpDetail,
                    actionTitle: sharpActionTitle,
                    isEnabled: selectedFrame != nil && !coordinator.isBusy
                ) {
                    guard let selectedFrame else { return }
                    Task {
                        await coordinator.createPendingSharpReconstruction(
                            selectedFrame: selectedFrame
                        )
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var framePicker: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Choose the SHARP source frame")
                        .font(.headline)
                    Text("CloudPoint recommends a clear, balanced frame. Pick another if you prefer.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if isLoadingFrames { ProgressView().controlSize(.small) }
            }

            if !candidates.isEmpty {
                ScrollView(.horizontal) {
                    HStack(spacing: 10) {
                        ForEach(candidates) { candidate in
                            Button {
                                selectedFrameIndex = candidate.index
                            } label: {
                                KeyFrameThumbnail(
                                    candidate: candidate,
                                    isSelected: selectedFrameIndex == candidate.index,
                                    isRecommended: recommendedFrame?.index == candidate.index
                                )
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(
                                "Frame at \(candidate.timestampSeconds.formatted(.number.precision(.fractionLength(1)))) seconds"
                            )
                            .accessibilityValue(
                                selectedFrameIndex == candidate.index ? "Selected" : "Not selected"
                            )
                        }
                    }
                    .padding(.vertical, 2)
                }
                .scrollIndicators(.hidden)
            } else if !isLoadingFrames, frameError == nil {
                ContentUnavailableView(
                    "No source frames",
                    systemImage: "film",
                    description: Text("Try another video.")
                )
                .frame(maxWidth: .infinity)
            }
        }
        .padding(16)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 16))
    }

    private var selectedFrame: VideoKeyFrameCandidate? {
        VideoKeyFrameSelector.selected(
            in: candidates,
            preferredIndex: selectedFrameIndex
        )
    }

    private var recommendedFrame: VideoKeyFrameCandidate? {
        VideoKeyFrameSelector.recommended(in: candidates)
    }

    private func loadFramesIfNeeded() async {
        guard case .video = request.source else { return }
        isLoadingFrames = true
        frameError = nil
        do {
            let loaded = try await coordinator.loadPendingVideoKeyFrames()
            guard !Task.isCancelled else { return }
            candidates = loaded
            selectedFrameIndex = VideoKeyFrameSelector.recommended(in: loaded)?.index
        } catch is CancellationError {
        } catch {
            frameError = error.localizedDescription
        }
        isLoadingFrames = false
    }

    private var sourceIcon: String {
        switch request.source {
        case .video: "film.fill"
        case .camera: "video.fill"
        }
    }

    private var sourceTitle: String {
        switch request.source {
        case let .video(url, _): url.lastPathComponent
        case let .camera(preflight): preflight.deviceName
        }
    }

    private var sourceDetail: String {
        switch request.source {
        case let .video(_, probe):
            "\(probe.durationSeconds.formatted(.number.precision(.fractionLength(1)))) sec video · \(probe.sampledFrameCount) point-cloud samples"
        case .camera:
            "Live camera source"
        }
    }

    private var sharpDescription: String {
        switch request.source {
        case .video:
            "Builds a dense metric 3D Gaussian representation from one selected frame using Apple SHARP."
        case .camera:
            "Single-snapshot SHARP capture is being connected to this camera workflow."
        }
    }

    private var sharpDetail: String {
        switch request.source {
        case .video: "Best for nearby novel views from one image"
        case .camera: "Choose a video source in this build"
        }
    }

    private var sharpActionTitle: String {
        switch request.source {
        case .video: "Create Gaussian Scene"
        case .camera: "Coming next"
        }
    }
}

private struct ReconstructionModeCard: View {
    let icon: String
    let tint: Color
    let title: String
    let badge: String
    let description: String
    let details: String
    let actionTitle: String
    let isEnabled: Bool
    let action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack(alignment: .top) {
                Image(systemName: icon)
                    .font(.system(size: 27, weight: .semibold))
                    .foregroundStyle(tint)
                    .frame(width: 38, height: 38)
                Spacer()
                Text(badge)
                    .font(.caption2.bold())
                    .foregroundStyle(tint)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(tint.opacity(0.12), in: Capsule())
            }
            Text(title)
                .font(.title3.bold())
            Text(description)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Label(details, systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 2)
            Button(actionTitle, action: action)
                .buttonStyle(.borderedProminent)
                .tint(tint)
                .frame(maxWidth: .infinity)
                .disabled(!isEnabled)
        }
        .padding(18)
        .frame(maxWidth: .infinity, minHeight: 250, alignment: .topLeading)
        .background(.background.opacity(0.45), in: RoundedRectangle(cornerRadius: 18))
        .overlay {
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(tint.opacity(0.25))
        }
        .accessibilityElement(children: .contain)
    }
}

private struct KeyFrameThumbnail: View {
    let candidate: VideoKeyFrameCandidate
    let isSelected: Bool
    let isRecommended: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack(alignment: .topLeading) {
                if let image = NSImage(data: candidate.thumbnailJPEG) {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 128, height: 82)
                        .clipped()
                } else {
                    Rectangle()
                        .fill(.quaternary)
                        .frame(width: 128, height: 82)
                        .overlay { Image(systemName: "photo") }
                }
                if isRecommended {
                    Text("BEST")
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.purple, in: Capsule())
                        .foregroundStyle(.white)
                        .padding(6)
                }
            }
            Text("\(candidate.timestampSeconds.formatted(.number.precision(.fractionLength(1)))) sec")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .padding(.horizontal, 2)
        }
        .padding(6)
        .background(
            isSelected ? Color.purple.opacity(0.14) : Color.clear,
            in: RoundedRectangle(cornerRadius: 12)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(isSelected ? Color.purple : Color.secondary.opacity(0.18), lineWidth: isSelected ? 2 : 1)
        }
    }
}

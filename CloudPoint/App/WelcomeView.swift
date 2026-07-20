import AppKit
import SwiftUI

struct WelcomeView: View {
    @ObservedObject var coordinator: AppCoordinator
    @State private var isDropTargeted = false

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.025, green: 0.035, blue: 0.07),
                    Color(red: 0.055, green: 0.075, blue: 0.13),
                    Color(red: 0.035, green: 0.04, blue: 0.075),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            Circle()
                .fill(Color.cyan.opacity(0.12))
                .frame(width: 520, height: 520)
                .blur(radius: 90)
                .offset(x: -350, y: -260)

            Circle()
                .fill(Color.indigo.opacity(0.18))
                .frame(width: 480, height: 480)
                .blur(radius: 100)
                .offset(x: 420, y: 300)

            VStack(spacing: 22) {
                brandHeader

                HStack(alignment: .top, spacing: 22) {
                    actionCard
                    recentCard
                }
                .frame(maxWidth: 980)

                dropHint
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 24)

            if coordinator.isBusy {
                Color.black.opacity(0.24).ignoresSafeArea()
                ProgressView("Preparing project…")
                    .controlSize(.large)
                    .padding(.horizontal, 28)
                    .padding(.vertical, 20)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
            }
        }
        .frame(minWidth: 860, minHeight: 620)
        .dropDestination(for: URL.self) { urls, _ in
            Task { await coordinator.openDroppedItems(urls) }
            return urls.contains { CloudPointInputRouter.kind(for: $0) != nil }
        } isTargeted: { isDropTargeted = $0 }
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 22)
                    .strokeBorder(Color.cyan, style: StrokeStyle(lineWidth: 3, dash: [9, 7]))
                    .padding(16)
                    .allowsHitTesting(false)
            }
        }
    }

    private var brandHeader: some View {
        VStack(spacing: 10) {
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .frame(width: 82, height: 82)
                .shadow(color: .cyan.opacity(0.22), radius: 24)

            Text("CloudPoint")
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
            Text("Turn video and live capture into a spatial point cloud.")
                .font(.title3)
                .foregroundStyle(.white.opacity(0.68))

            EngineStateBadge(state: coordinator.engineState)
        }
    }

    private var actionCard: some View {
        VStack(spacing: 12) {
            WelcomeActionButton(
                title: "Open Video…",
                subtitle: "MOV, MP4, or M4V",
                systemImage: "film.stack",
                prominent: true,
                action: coordinator.chooseVideo
            )
            .keyboardShortcut("o", modifiers: .command)

            WelcomeActionButton(
                title: "Use Camera",
                subtitle: "Create a live spatial capture",
                systemImage: "video.fill",
                prominent: false,
                action: { Task { await coordinator.useCamera() } }
            )

            WelcomeActionButton(
                title: "Open CloudPoint Project…",
                subtitle: "Continue an autosaved project",
                systemImage: "shippingbox",
                prominent: false,
                action: coordinator.chooseProject
            )

            if let error = coordinator.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(Color(red: 1, green: 0.65, blue: 0.55))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
                    .textSelection(.enabled)
            }
        }
        .padding(20)
        .frame(width: 390)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
        .overlay {
            RoundedRectangle(cornerRadius: 22)
                .strokeBorder(.white.opacity(0.09))
        }
    }

    private var recentCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Recent Projects")
                    .font(.headline)
                    .foregroundStyle(.white)
                Spacer()
                Text(coordinator.recentProjects.count.formatted())
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.48))
            }

            if coordinator.recentProjects.isEmpty {
                ContentUnavailableView(
                    "No Recent Projects",
                    systemImage: "clock.arrow.circlepath",
                    description: Text("Projects are created and autosaved when you open a video.")
                )
                .foregroundStyle(.white.opacity(0.62))
                .frame(maxWidth: .infinity, minHeight: 190)
            } else {
                ScrollView {
                    LazyVStack(spacing: 7) {
                        ForEach(coordinator.recentProjects) { recent in
                            Button { coordinator.openRecent(recent) } label: {
                                RecentProjectRow(project: recent)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .frame(maxHeight: 260)
            }
        }
        .padding(20)
        .frame(width: 390)
        .frame(minHeight: 270, alignment: .top)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
        .overlay {
            RoundedRectangle(cornerRadius: 22)
                .strokeBorder(.white.opacity(0.09))
        }
    }

    private var dropHint: some View {
        Label(
            isDropTargeted ? "Release to open" : "Drop a video or .cloudpoint project anywhere",
            systemImage: isDropTargeted ? "arrow.down.circle.fill" : "square.and.arrow.down"
        )
        .font(.callout.weight(.medium))
        .foregroundStyle(isDropTargeted ? Color.cyan : .white.opacity(0.52))
        .animation(.easeOut(duration: 0.16), value: isDropTargeted)
    }
}

private struct WelcomeActionButton: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let prominent: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: systemImage)
                    .font(.system(size: 19, weight: .semibold))
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(
                            prominent
                                ? Color.white.opacity(0.74)
                                : Color(red: 0.24, green: 0.28, blue: 0.38)
                        )
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.bold())
                    .opacity(0.62)
            }
            .foregroundStyle(
                prominent
                    ? Color.white
                    : Color(red: 0.08, green: 0.11, blue: 0.2)
            )
            .padding(.horizontal, 16)
            .frame(height: 68)
            .background(
                prominent
                    ? AnyShapeStyle(LinearGradient(
                        colors: [Color.blue, Color.indigo],
                        startPoint: .leading,
                        endPoint: .trailing
                    ))
                    : AnyShapeStyle(Color.white.opacity(0.82)),
                in: RoundedRectangle(cornerRadius: 14)
            )
        }
        .buttonStyle(.plain)
    }
}

private struct RecentProjectRow: View {
    let project: RecentProject

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "point.3.filled.connected.trianglepath.dotted")
                .foregroundStyle(.cyan)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 3) {
                Text(project.displayName)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                Text("\(project.phase.displayName) · \(project.lastOpenedAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.48))
                    .lineLimit(1)
            }
            Spacer()
            Image(systemName: "chevron.right")
                .font(.caption.bold())
                .foregroundStyle(.white.opacity(0.32))
        }
        .padding(.horizontal, 12)
        .frame(height: 58)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct EngineStateBadge: View {
    let state: WelcomeEngineState

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: state.systemImage)
                .foregroundStyle(state.tint)
            Text(state.label)
                .foregroundStyle(.white.opacity(0.72))
            if let progress = state.progress {
                ProgressView(value: progress)
                    .frame(width: 72)
                    .tint(state.tint)
            }
        }
        .font(.caption.weight(.medium))
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(.white.opacity(0.07), in: Capsule())
    }
}

private extension WelcomeEngineState {
    var label: String {
        switch self {
        case .ready: "Reconstruction engine ready"
        case .setupRequired: "Model setup required"
        case .downloading: "Downloading model"
        case .converting: "Converting for Apple Silicon"
        case .repairRequired: "Model repair required"
        }
    }

    var systemImage: String {
        switch self {
        case .ready: "checkmark.circle.fill"
        case .setupRequired: "arrow.down.circle"
        case .downloading: "arrow.down.circle.fill"
        case .converting: "cpu"
        case .repairRequired: "wrench.and.screwdriver.fill"
        }
    }

    var tint: Color {
        switch self {
        case .ready: .green
        case .setupRequired: .yellow
        case .downloading, .converting: .cyan
        case .repairRequired: .orange
        }
    }

    var progress: Double? {
        switch self {
        case let .downloading(progress), let .converting(progress):
            min(max(progress, 0), 1)
        case .ready, .setupRequired, .repairRequired:
            nil
        }
    }
}

private extension SessionPhase {
    var displayName: String {
        switch self {
        case .empty: "Ready"
        case .preparing: "Preparing"
        case .ready: "Ready"
        case .importing: "Reading video"
        case .capturing: "Capturing"
        case .processing: "Reconstructing"
        case .paused: "Paused"
        case .finalizing: "Finalizing"
        case .completed: "Complete"
        case .cancelled: "Cancelled"
        case .failed: "Failed"
        }
    }
}

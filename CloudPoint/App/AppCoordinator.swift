@preconcurrency import AVFoundation
import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum CloudPointInputKind: Sendable, Equatable {
    case video
    case project
}

struct CloudPointInputRouter {
    static let movieExtensions = Set(["mov", "mp4", "m4v"])
    static let movieContentTypes: [UTType] = [
        .quickTimeMovie,
        .mpeg4Movie,
        UTType("com.apple.m4v-video")!,
    ]
    static let droppedContentTypes: [UTType] = movieContentTypes + [.cloudPointProject]

    static func kind(for url: URL) -> CloudPointInputKind? {
        let pathExtension = url.pathExtension.lowercased()
        if movieExtensions.contains(pathExtension) { return .video }
        if pathExtension == "cloudpoint" { return .project }
        return nil
    }
}

struct VideoProbeResult: Sendable, Equatable {
    let durationSeconds: Double
    let sampledFrameCount: Int
}

protocol VideoMetadataProbing: Sendable {
    func probe(_ url: URL, framesPerSecond: Int) async throws -> VideoProbeResult
}

enum VideoProbeError: Error, LocalizedError, Equatable {
    case noVideoTrack
    case invalidDuration
    case noSampledFrames

    var errorDescription: String? {
        switch self {
        case .noVideoTrack: "The selected file does not contain a readable video track."
        case .invalidDuration: "The selected video has an invalid duration."
        case .noSampledFrames: "The selected video does not contain any usable frames."
        }
    }
}

struct AVFoundationVideoMetadataProbe: VideoMetadataProbing {
    func probe(_ url: URL, framesPerSecond: Int) async throws -> VideoProbeResult {
        let didStartScope = url.startAccessingSecurityScopedResource()
        defer { if didStartScope { url.stopAccessingSecurityScopedResource() } }

        let asset = AVURLAsset(url: url)
        async let durationValue = asset.load(.duration)
        async let videoTracks = asset.loadTracks(withMediaType: .video)
        let (duration, tracks) = try await (durationValue, videoTracks)
        guard !tracks.isEmpty else { throw VideoProbeError.noVideoTrack }
        guard duration.isValid,
              !duration.isIndefinite,
              duration.seconds.isFinite,
              duration.seconds > 0 else {
            throw VideoProbeError.invalidDuration
        }
        let plan = try FrameSamplingPlan(
            duration: duration,
            framesPerSecond: framesPerSecond
        )
        guard !plan.timestamps.isEmpty else { throw VideoProbeError.noSampledFrames }
        return VideoProbeResult(
            durationSeconds: duration.seconds,
            sampledFrameCount: plan.timestamps.count
        )
    }
}

enum WorkspaceInitialSource: Sendable, Equatable {
    case recording(URL, framesPerSecond: Int)
    case camera
}

struct WorkspaceLaunch: Identifiable, Sendable, Equatable {
    var id: UUID { projectID }

    let projectID: UUID
    let sourceTitle: String
    let packageURL: URL
    let manifest: ProjectManifest
    let initialSource: WorkspaceInitialSource?
}

enum WelcomeEngineState: Sendable, Equatable {
    case ready
    case setupRequired
    case downloading(progress: Double)
    case converting(progress: Double)
    case repairRequired
}

@MainActor
protocol InputPanelPresenting {
    func chooseVideo() -> URL?
    func chooseProject() -> URL?
}

@MainActor
struct SystemInputPanelPresenter: InputPanelPresenting {
    func chooseVideo() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Open Video"
        panel.prompt = "Open Video"
        panel.allowedContentTypes = CloudPointInputRouter.movieContentTypes
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    func chooseProject() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Open CloudPoint Project"
        panel.prompt = "Open Project"
        panel.allowedContentTypes = [.cloudPointProject]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        return panel.runModal() == .OK ? panel.url : nil
    }
}

@MainActor
final class AppCoordinator: ObservableObject {
    enum Destination: Sendable, Equatable {
        case welcome
        case workspace(WorkspaceLaunch)
    }

    static let defaultSamplingRate = 2

    @Published private(set) var destination: Destination = .welcome
    @Published private(set) var recentProjects: [RecentProject] = []
    @Published private(set) var isBusy = false
    @Published private(set) var errorMessage: String?
    @Published var engineState: WelcomeEngineState

    private let projectStore: any ManagedProjectStoring
    private let videoProbe: any VideoMetadataProbing
    private let panelPresenter: any InputPanelPresenting
    private var didStart = false

    init(
        projectStore: any ManagedProjectStoring,
        videoProbe: any VideoMetadataProbing,
        panelPresenter: any InputPanelPresenting = SystemInputPanelPresenter(),
        engineState: WelcomeEngineState = .setupRequired,
        initialError: String? = nil
    ) {
        self.projectStore = projectStore
        self.videoProbe = videoProbe
        self.panelPresenter = panelPresenter
        self.engineState = engineState
        errorMessage = initialError
    }

    static func live() -> AppCoordinator {
        do {
            return AppCoordinator(
                projectStore: try ManagedProjectStore.live(),
                videoProbe: AVFoundationVideoMetadataProbe()
            )
        } catch {
            return AppCoordinator(
                projectStore: UnavailableManagedProjectStore(error: error),
                videoProbe: AVFoundationVideoMetadataProbe(),
                initialError: error.localizedDescription
            )
        }
    }

    func start() async {
        guard !didStart else { return }
        didStart = true
        await refreshRecentProjects()
    }

    func chooseVideo() {
        guard let url = panelPresenter.chooseVideo() else { return }
        Task { await openInput(url) }
    }

    func chooseProject() {
        guard let url = panelPresenter.chooseProject() else { return }
        Task { await openInput(url) }
    }

    func useCamera() {
        Task { await createCameraProject() }
    }

    func openInput(_ url: URL) async {
        switch CloudPointInputRouter.kind(for: url) {
        case .video:
            await openVideo(url)
        case .project:
            await openProject(url)
        case nil:
            errorMessage = "Choose a MOV, MP4, M4V, or CloudPoint project."
        }
    }

    func openDroppedItems(_ urls: [URL]) async {
        guard let url = urls.first(where: { CloudPointInputRouter.kind(for: $0) != nil }) else {
            errorMessage = "Drop a MOV, MP4, M4V, or CloudPoint project."
            return
        }
        await openInput(url)
    }

    func openExternalURL(_ url: URL) async {
        await openInput(url)
    }

    func openRecent(_ recent: RecentProject) {
        Task { await openProject(recent.packageURL) }
    }

    func showWelcome() {
        destination = .welcome
        Task { await refreshRecentProjects() }
    }

    private func openVideo(_ url: URL) async {
        await performRoute {
            _ = try await videoProbe.probe(
                url,
                framesPerSecond: Self.defaultSamplingRate
            )
            let project = try await projectStore.createProject(sourceName: url.lastPathComponent)
            destination = .workspace(Self.launch(
                project,
                initialSource: .recording(
                    url,
                    framesPerSecond: Self.defaultSamplingRate
                )
            ))
        }
    }

    private func openProject(_ url: URL) async {
        await performRoute {
            let project = try await projectStore.openProject(at: url)
            destination = .workspace(Self.launch(project, initialSource: nil))
        }
    }

    private func createCameraProject() async {
        await performRoute {
            let project = try await projectStore.createProject(sourceName: "Camera Capture")
            destination = .workspace(Self.launch(project, initialSource: .camera))
        }
    }

    private func performRoute(_ operation: () async throws -> Void) async {
        guard !isBusy else { return }
        isBusy = true
        errorMessage = nil
        do {
            try await operation()
            await refreshRecentProjects()
        } catch {
            errorMessage = error.localizedDescription
        }
        isBusy = false
    }

    private func refreshRecentProjects() async {
        do { recentProjects = try await projectStore.recentProjects() }
        catch { errorMessage = error.localizedDescription }
    }

    private static func launch(
        _ project: ManagedProject,
        initialSource: WorkspaceInitialSource?
    ) -> WorkspaceLaunch {
        WorkspaceLaunch(
            projectID: project.id,
            sourceTitle: project.displayName,
            packageURL: project.packageURL,
            manifest: project.manifest,
            initialSource: initialSource
        )
    }
}

private actor UnavailableManagedProjectStore: ManagedProjectStoring {
    let error: Error

    init(error: Error) { self.error = error }

    func createProject(sourceName: String) throws -> ManagedProject { throw error }
    func openProject(at url: URL) throws -> ManagedProject { throw error }
    func recentProjects() throws -> [RecentProject] { throw error }
}

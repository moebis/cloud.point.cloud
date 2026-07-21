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
    case recording(URL, framesPerSecond: Int, expectedSampleCount: UInt64)
    case camera(deviceID: String, deviceName: String)
}

struct CameraPreflightResult: Sendable, Equatable {
    let deviceID: String
    let deviceName: String
}

protocol CameraPreflighting: Sendable {
    func preflight() async throws -> CameraPreflightResult
}

enum CameraPreflightError: Error, LocalizedError, Equatable, Sendable {
    case permissionDenied
    case noCamera

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            "Camera access is off. Allow CloudPoint in System Settings, then try again."
        case .noCamera:
            "No camera is available. Connect a camera, then try again."
        }
    }
}

struct SystemCameraPreflight: CameraPreflighting {
    private let authority: any CameraAuthorizing

    init(authority: any CameraAuthorizing = SystemCameraAuthority()) {
        self.authority = authority
    }

    func preflight() async throws -> CameraPreflightResult {
        guard case .authorized = await authority.requestAccess() else {
            throw CameraPreflightError.permissionDenied
        }
        guard let camera = await CameraCatalog.devices().first else {
            throw CameraPreflightError.noCamera
        }
        return CameraPreflightResult(deviceID: camera.id, deviceName: camera.name)
    }
}

struct WorkspaceLaunch: Identifiable, Sendable, Equatable {
    var id: UUID { sessionID }

    let sessionID: UUID
    let projectID: UUID
    let sourceTitle: String
    let packageURL: URL
    let packageBookmarkData: Data?
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

    private enum PendingModelRoute {
        case recording(URL, VideoProbeResult)
        case camera(CameraPreflightResult)
        case project(URL)
        case recent(RecentProject)
    }

    private struct WorkspaceIdentity: Equatable {
        let projectID: UUID
        let packageURL: URL

        init(projectID: UUID, packageURL: URL) {
            self.projectID = projectID
            self.packageURL = packageURL.standardizedFileURL
        }
    }

    private struct OwnedWorkspace {
        let identity: WorkspaceIdentity
        let viewModel: WorkspaceViewModel
    }

    @Published private(set) var destination: Destination = .welcome
    @Published private(set) var recentProjects: [RecentProject] = []
    @Published private(set) var isBusy = false
    @Published private(set) var errorMessage: String?
    @Published var engineState: WelcomeEngineState
    @Published var isModelSetupPresented = false

    let modelSetupViewModel: ModelSetupViewModel?

    private let projectStore: any ManagedProjectStoring
    private let videoProbe: any VideoMetadataProbing
    private let recordingSources: any RecordingSourceManaging
    private let cameraPreflight: any CameraPreflighting
    private let panelPresenter: any InputPanelPresenting
    private let modelInstaller: (any ModelInstalling)?
    private let engineContext: ProductionReconstructionContext?
    private var didStart = false
    private var pendingModelRoute: PendingModelRoute?
    private var ownedWorkspace: OwnedWorkspace?

    init(
        projectStore: any ManagedProjectStoring,
        videoProbe: any VideoMetadataProbing,
        recordingSources: any RecordingSourceManaging = SystemRecordingSourceManager(),
        cameraPreflight: any CameraPreflighting = SystemCameraPreflight(),
        panelPresenter: any InputPanelPresenting = SystemInputPanelPresenter(),
        modelInstaller: (any ModelInstalling)? = nil,
        engineContext: ProductionReconstructionContext? = nil,
        engineState: WelcomeEngineState = .setupRequired,
        initialError: String? = nil
    ) {
        self.projectStore = projectStore
        self.videoProbe = videoProbe
        self.recordingSources = recordingSources
        self.cameraPreflight = cameraPreflight
        self.panelPresenter = panelPresenter
        self.modelInstaller = modelInstaller
        self.engineContext = engineContext
        modelSetupViewModel = modelInstaller.map(ModelSetupViewModel.init(installer:))
        self.engineState = engineState
        errorMessage = initialError
    }

    static func live() -> AppCoordinator {
        do {
            let store = try ManagedProjectStore.live()
            do {
                let runtime = try WorkerRuntime.resolve(
                    bundleValue: Bundle.main.object(
                        forInfoDictionaryKey: "CloudPointWorkerRuntime"
                    ) as? String,
                    environment: ProcessInfo.processInfo.environment
                )
                let directories = try ModelDirectories.live()
                let installer = ModelInstaller(
                    directories: directories,
                    downloader: URLSessionModelDownloader(),
                    converter: ProcessModelConverter(runtime: runtime)
                )
                return AppCoordinator(
                    projectStore: store,
                    videoProbe: AVFoundationVideoMetadataProbe(),
                    modelInstaller: installer,
                    engineContext: ProductionReconstructionContext(
                        runtime: runtime,
                        modelDirectory: directories.converted
                    )
                )
            } catch {
                return AppCoordinator(
                    projectStore: store,
                    videoProbe: AVFoundationVideoMetadataProbe(),
                    modelInstaller: UnavailableModelInstaller(error: error),
                    engineState: .repairRequired,
                    initialError: "The CloudPoint MLX runtime is unavailable. Reinstall or rebuild the app."
                )
            }
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
        await refreshEngineHealth()
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

    func useCamera() async {
        await createCameraProject()
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
        Task { await openRecentProject(recent) }
    }

    func showWelcome() async {
        await closeOwnedWorkspace()
        destination = .welcome
        await refreshRecentProjects()
    }

    func repairModel() {
        if case let .workspace(launch) = destination {
            pendingModelRoute = .project(launch.packageURL)
        } else {
            pendingModelRoute = nil
        }
        isModelSetupPresented = true
    }

    func retryCurrentProject() {
        guard case let .workspace(launch) = destination else { return }
        pendingModelRoute = .project(launch.packageURL)
        Task {
            await refreshEngineHealth()
            guard engineState == .ready, engineContext != nil else {
                isModelSetupPresented = true
                return
            }
            let route = pendingModelRoute
            pendingModelRoute = nil
            guard case let .project(url) = route else { return }
            await performRoute {
                try await finishOpenProject(url, replacingExistingWorkspace: true)
            }
        }
    }

    func continueAfterModelSetup() async {
        await refreshEngineHealth()
        guard engineState == .ready, engineContext != nil else { return }
        isModelSetupPresented = false
        let route = pendingModelRoute
        pendingModelRoute = nil
        guard let route else { return }
        await performRoute {
            switch route {
            case let .recording(url, probe):
                try await finishOpenVideo(url, probe: probe)
            case let .camera(preflight):
                try await finishCameraProject(preflight)
            case let .project(url):
                try await finishOpenProject(url, replacingExistingWorkspace: true)
            case let .recent(recent):
                try await finishOpenRecentProject(recent)
            }
        }
    }

    func workspaceViewModel(for launch: WorkspaceLaunch) -> WorkspaceViewModel {
        let identity = WorkspaceIdentity(
            projectID: launch.projectID,
            packageURL: launch.packageURL
        )
        if let existing = ownedWorkspace, existing.identity == identity {
            return existing.viewModel
        }
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: launch.manifest),
            packageURL: launch.packageURL,
            packageBookmarkData: launch.packageBookmarkData,
            initialSource: launch.initialSource,
            engineFactory: reconstructionEngineFactory,
            onChooseAnotherVideo: { [weak self] in self?.chooseVideo() },
            onRepairModel: { [weak self] in self?.repairModel() },
            onRetryProject: { [weak self] in self?.retryCurrentProject() }
        )
        if let existing = ownedWorkspace {
            Task { await existing.viewModel.close() }
        }
        ownedWorkspace = OwnedWorkspace(identity: identity, viewModel: viewModel)
        return viewModel
    }

    var reconstructionEngineFactory: @Sendable () throws -> any ReconstructionEngine {
        guard let engineContext else {
            return { throw SessionControllerError.engineUnavailable }
        }
        return { try engineContext.makeEngine() }
    }

    private func openVideo(_ url: URL) async {
        await performRoute {
            let probe = try await videoProbe.probe(
                url,
                framesPerSecond: Self.defaultSamplingRate
            )
            guard await requireReadyModel(for: .recording(url, probe)) else { return }
            try await finishOpenVideo(url, probe: probe)
        }
    }

    private func finishOpenVideo(_ url: URL, probe: VideoProbeResult) async throws {
        let reference = try await recordingSources.makeReference(
            for: url,
            probe: probe,
            framesPerSecond: Self.defaultSamplingRate
        )
        let project = try await projectStore.createRecordingProject(
            sourceName: url.lastPathComponent,
            source: reference
        )
        await presentWorkspace(Self.launch(
            project,
            initialSource: .recording(
                url,
                framesPerSecond: Self.defaultSamplingRate,
                expectedSampleCount: UInt64(probe.sampledFrameCount)
            )
        ))
    }

    private func openProject(_ url: URL) async {
        await performRoute {
            let project = try await projectStore.openProject(at: url)
            let phase = project.manifest.sessionState.phase
            if ![SessionPhase.completed, .cancelled].contains(phase) {
                guard await requireReadyModel(for: .project(url)) else { return }
            }
            await presentWorkspace(Self.launch(project, initialSource: nil))
        }
    }

    private func openRecentProject(_ recent: RecentProject) async {
        await performRoute {
            let project = try await projectStore.openRecentProject(recent)
            let phase = project.manifest.sessionState.phase
            if ![SessionPhase.completed, .cancelled].contains(phase) {
                guard await requireReadyModel(for: .recent(recent)) else { return }
            }
            await presentWorkspace(Self.launch(project, initialSource: nil))
        }
    }

    private func finishOpenProject(
        _ url: URL,
        replacingExistingWorkspace: Bool = false
    ) async throws {
        let project = try await projectStore.openProject(at: url)
        await presentWorkspace(
            Self.launch(project, initialSource: nil),
            replacingExistingWorkspace: replacingExistingWorkspace
        )
    }

    private func finishOpenRecentProject(_ recent: RecentProject) async throws {
        let project = try await projectStore.openRecentProject(recent)
        await presentWorkspace(
            Self.launch(project, initialSource: nil),
            replacingExistingWorkspace: true
        )
    }

    private func createCameraProject() async {
        await performRoute {
            let preflight = try await cameraPreflight.preflight()
            guard await requireReadyModel(for: .camera(preflight)) else { return }
            try await finishCameraProject(preflight)
        }
    }

    private func finishCameraProject(_ preflight: CameraPreflightResult) async throws {
        let source = CameraSourceReference(
            deviceID: preflight.deviceID,
            deviceName: preflight.deviceName
        )
        let project = try await projectStore.createCameraProject(
            sourceName: "\(preflight.deviceName) Capture",
            source: source
        )
        await presentWorkspace(Self.launch(
            project,
            initialSource: .camera(
                deviceID: preflight.deviceID,
                deviceName: preflight.deviceName
            )
        ))
    }

    private func requireReadyModel(for route: PendingModelRoute) async -> Bool {
        guard modelInstaller != nil else { return true }
        await refreshEngineHealth()
        guard engineState == .ready, engineContext != nil else {
            pendingModelRoute = route
            isModelSetupPresented = true
            return false
        }
        return true
    }

    private func refreshEngineHealth() async {
        guard let modelInstaller else { return }
        engineState = Self.welcomeState(for: await modelInstaller.health())
    }

    private static func welcomeState(for health: ModelHealth) -> WelcomeEngineState {
        switch health {
        case .absent: .setupRequired
        case let .preparing(event):
            switch event {
            case let .downloading(received, expected),
                 let .verifying(received, expected):
                .downloading(progress: expected > 0 ? Double(received) / Double(expected) : 0)
            case .converting, .validating: .converting(progress: 0)
            case .ready: .ready
            }
        case .ready: .ready
        case .invalid: .repairRequired
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

    private func presentWorkspace(
        _ launch: WorkspaceLaunch,
        replacingExistingWorkspace: Bool = false
    ) async {
        let identity = WorkspaceIdentity(
            projectID: launch.projectID,
            packageURL: launch.packageURL
        )
        if replacingExistingWorkspace || ownedWorkspace?.identity != identity {
            await closeOwnedWorkspace()
        }
        destination = .workspace(launch)
    }

    private func closeOwnedWorkspace() async {
        guard let existing = ownedWorkspace else { return }
        ownedWorkspace = nil
        await existing.viewModel.close()
    }

    private static func launch(
        _ project: ManagedProject,
        initialSource: WorkspaceInitialSource?
    ) -> WorkspaceLaunch {
        WorkspaceLaunch(
            sessionID: UUID(),
            projectID: project.id,
            sourceTitle: project.displayName,
            packageURL: project.packageURL,
            packageBookmarkData: project.packageBookmarkData,
            manifest: project.manifest,
            initialSource: initialSource
        )
    }
}

private actor UnavailableManagedProjectStore: ManagedProjectStoring {
    let error: Error

    init(error: Error) { self.error = error }

    func createProject(sourceName: String) throws -> ManagedProject { throw error }
    func createRecordingProject(
        sourceName: String,
        source: RecordingSourceReference
    ) throws -> ManagedProject { throw error }
    func createCameraProject(
        sourceName: String,
        source: CameraSourceReference
    ) throws -> ManagedProject { throw error }
    func openProject(at url: URL) throws -> ManagedProject { throw error }
    func recentProjects() throws -> [RecentProject] { throw error }
}

private actor UnavailableModelInstaller: ModelInstalling {
    let error: Error

    init(error: Error) { self.error = error }

    func health() -> ModelHealth {
        .invalid(.operationFailed(String(describing: error)))
    }

    func prepare() -> AsyncThrowingStream<ModelSetupEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish(throwing: error) }
    }

    func cancel() {}
}

import XCTest
@testable import CloudPoint

@MainActor
final class AppCoordinatorTests: XCTestCase {
    func testInputRouterSupportsMovMp4M4vAndCloudPointOnly() {
        XCTAssertEqual(CloudPointInputRouter.kind(for: URL(filePath: "/tmp/a.mov")), .video)
        XCTAssertEqual(CloudPointInputRouter.kind(for: URL(filePath: "/tmp/a.MP4")), .video)
        XCTAssertEqual(CloudPointInputRouter.kind(for: URL(filePath: "/tmp/a.m4v")), .video)
        XCTAssertEqual(CloudPointInputRouter.kind(for: URL(filePath: "/tmp/a.cloudpoint")), .project)
        XCTAssertNil(CloudPointInputRouter.kind(for: URL(filePath: "/tmp/a.txt")))
    }

    func testVideoRouteProbesBeforePresentingModeAndCreatesNoProjectUntilConfirmation() async throws {
        let video = URL(filePath: "/tmp/IMG_2285.MOV")
        let package = URL(filePath: "/tmp/IMG_2285.cloudpoint", directoryHint: .isDirectory)
        let project = ManagedProject(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000121")!,
            displayName: "IMG_2285",
            packageURL: package,
            manifest: ProjectManifest(),
            lastOpenedAt: Date(timeIntervalSinceReferenceDate: 42)
        )
        let store = CoordinatorStore(project: project)
        let probe = CoordinatorVideoProbe(
            result: VideoProbeResult(durationSeconds: 10.6, sampledFrameCount: 22)
        )
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: probe,
            recordingSources: CoordinatorRecordingSources()
        )

        await coordinator.openInput(video)

        let probedURLs = await probe.requestedURLs()
        let sourceNamesBeforeConfirmation = await store.requestedSourceNames()
        XCTAssertEqual(probedURLs, [video])
        XCTAssertEqual(sourceNamesBeforeConfirmation, [])
        XCTAssertEqual(
            coordinator.pendingReconstruction?.source,
            .video(video, VideoProbeResult(durationSeconds: 10.6, sampledFrameCount: 22))
        )
        XCTAssertEqual(coordinator.destination, .welcome)

        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)

        let sourceNamesAfterConfirmation = await store.requestedSourceNames()
        XCTAssertEqual(sourceNamesAfterConfirmation, ["IMG_2285.MOV"])
        XCTAssertNil(coordinator.pendingReconstruction)
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected a workspace destination")
        }
        XCTAssertEqual(launch.projectID, project.id)
        XCTAssertEqual(launch.packageURL, package)
        XCTAssertEqual(launch.sourceTitle, "IMG_2285")
        XCTAssertEqual(
            launch.initialSource,
            .recording(video, framesPerSecond: 2, expectedSampleCount: 22)
        )
        let recordingSources = await store.requestedRecordingSources()
        XCTAssertEqual(recordingSources.first?.expectedSampleCount, 22)
        XCTAssertEqual(recordingSources.first?.nextSampleOrdinal, 0)
    }

    func testCameraRoutePreflightsBeforeModeConfirmationAndCarriesSelectedDevice() async {
        let store = CoordinatorStore(project: .fixture())
        let preflight = CoordinatorCameraPreflight(
            result: CameraPreflightResult(
                deviceID: "continuity-camera",
                deviceName: "Moebis’s iPhone"
            )
        )
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 1, sampledFrameCount: 2)
            ),
            cameraPreflight: preflight
        )

        await coordinator.useCamera()

        let preflightRequestCount = await preflight.requestCount()
        let sourceNamesBeforeConfirmation = await store.requestedSourceNames()
        XCTAssertEqual(preflightRequestCount, 1)
        XCTAssertEqual(sourceNamesBeforeConfirmation, [])
        XCTAssertEqual(
            coordinator.pendingReconstruction?.source,
            .camera(CameraPreflightResult(
                deviceID: "continuity-camera",
                deviceName: "Moebis’s iPhone"
            ))
        )

        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)

        let sourceNamesAfterConfirmation = await store.requestedSourceNames()
        XCTAssertEqual(sourceNamesAfterConfirmation, ["Moebis’s iPhone Capture"])
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected camera preflight workspace")
        }
        XCTAssertEqual(
            launch.initialSource,
            .camera(deviceID: "continuity-camera", deviceName: "Moebis’s iPhone")
        )
    }

    func testSharpVideoConfirmationCreatesProjectWithSelectedOrientedFrameWithoutLingBotSetup() async throws {
        let video = URL(filePath: "/tmp/scene.mp4")
        let store = CoordinatorStore(project: .fixture())
        let selected = VideoKeyFrameCandidate(
            index: 2,
            timestampSeconds: 1.25,
            thumbnailJPEG: Data("thumbnail".utf8),
            fullResolutionJPEG: Data("oriented-full-frame".utf8),
            sharpnessScore: 0.9,
            exposureScore: 0.8,
            temporalScore: 0.7
        )
        let selector = CoordinatorKeyFrameSelector(candidates: [selected])
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 3, sampledFrameCount: 6)
            ),
            recordingSources: CoordinatorRecordingSources(),
            videoKeyFrameSelector: selector,
            modelInstaller: CoordinatorModelInstaller(health: .absent),
            engineContext: nil
        )

        await coordinator.openInput(video)
        let candidates = try await coordinator.loadPendingVideoKeyFrames()
        await coordinator.createPendingSharpReconstruction(selectedFrame: selected)
        let requestedKeyFrameURLs = await selector.requestedURLs()
        let requestedSharpFrames = await store.requestedSharpFrames()

        XCTAssertEqual(candidates, [selected])
        XCTAssertEqual(requestedKeyFrameURLs, [video])
        XCTAssertEqual(requestedSharpFrames, [selected])
        XCTAssertFalse(coordinator.isModelSetupPresented)
        XCTAssertNil(coordinator.pendingReconstruction)
        guard case .workspace = coordinator.destination else {
            return XCTFail("Expected SHARP workspace")
        }
    }

    func testCameraFailureDoesNotCreateOrListAnEmptyProject() async {
        let store = CoordinatorStore(project: .fixture())
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 1, sampledFrameCount: 2)
            ),
            cameraPreflight: CoordinatorCameraPreflight(error: CameraPreflightError.permissionDenied)
        )

        await coordinator.useCamera()

        let requestedSourceNames = await store.requestedSourceNames()
        XCTAssertEqual(requestedSourceNames, [])
        XCTAssertEqual(coordinator.destination, .welcome)
        XCTAssertEqual(coordinator.errorMessage, CameraPreflightError.permissionDenied.localizedDescription)
    }

    func testDroppedAndFinderMoviesUseTheSameValidatedVideoRoute() async {
        let first = URL(filePath: "/tmp/first.mov")
        let second = URL(filePath: "/tmp/second.mp4")
        let store = CoordinatorStore(project: .fixture())
        let probe = CoordinatorVideoProbe(
            result: VideoProbeResult(durationSeconds: 1, sampledFrameCount: 2)
        )
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: probe,
            recordingSources: CoordinatorRecordingSources()
        )

        await coordinator.openDroppedItems([first])
        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)
        await coordinator.openExternalURL(second)
        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)

        let probedURLs = await probe.requestedURLs()
        let createdSourceNames = await store.requestedSourceNames()
        XCTAssertEqual(probedURLs, [first, second])
        XCTAssertEqual(createdSourceNames, ["first.mov", "second.mp4"])
    }

    func testCloudPointInputLoadsExistingProjectWithoutVideoProbe() async {
        let package = URL(filePath: "/tmp/Existing.cloudpoint", directoryHint: .isDirectory)
        let project = ManagedProject.fixture(packageURL: package)
        let store = CoordinatorStore(project: project)
        let probe = CoordinatorVideoProbe(
            result: VideoProbeResult(durationSeconds: 1, sampledFrameCount: 2)
        )
        let coordinator = AppCoordinator(projectStore: store, videoProbe: probe)

        await coordinator.openInput(package)

        let probedURLs = await probe.requestedURLs()
        let openedURLs = await store.requestedProjectURLs()
        XCTAssertEqual(probedURLs, [])
        XCTAssertEqual(openedURLs, [package])
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected a workspace destination")
        }
        XCTAssertNil(launch.initialSource)
    }

    func testMissingModelDefersVideoProjectCreationAndPresentsSetup() async {
        let video = URL(filePath: "/tmp/pending.mov")
        let store = CoordinatorStore(project: .fixture())
        let installer = CoordinatorModelInstaller(health: .absent)
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            recordingSources: CoordinatorRecordingSources(),
            modelInstaller: installer,
            engineContext: .fixture()
        )
        await coordinator.start()

        await coordinator.openInput(video)
        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)

        let requestedSourceNames = await store.requestedSourceNames()
        XCTAssertEqual(requestedSourceNames, [])
        XCTAssertEqual(coordinator.destination, .welcome)
        XCTAssertEqual(coordinator.engineState, .setupRequired)
        XCTAssertTrue(coordinator.isModelSetupPresented)
    }

    func testMissingModelDefersIncompleteProjectWorkspaceAndPresentsSetup() async {
        let package = URL(filePath: "/tmp/Pending.cloudpoint", directoryHint: .isDirectory)
        let store = CoordinatorStore(project: .fixture(packageURL: package))
        let installer = CoordinatorModelInstaller(health: .absent)
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: installer,
            engineContext: .fixture()
        )
        await coordinator.start()

        await coordinator.openInput(package)

        let openedURLs = await store.requestedProjectURLs()
        XCTAssertEqual(openedURLs, [package])
        XCTAssertEqual(coordinator.destination, .welcome)
        XCTAssertEqual(coordinator.engineState, .setupRequired)
        XCTAssertTrue(coordinator.isModelSetupPresented)
    }

    func testCompletedProjectOpensForViewingWithoutInstalledModel() async {
        let package = URL(filePath: "/tmp/Complete.cloudpoint", directoryHint: .isDirectory)
        var manifest = ProjectManifest()
        manifest.sessionState = SessionState(phase: .completed)
        let store = CoordinatorStore(
            project: .fixture(packageURL: package, manifest: manifest)
        )
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: CoordinatorModelInstaller(health: .absent),
            engineContext: .fixture()
        )
        await coordinator.start()

        await coordinator.openInput(package)

        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected the completed project workspace")
        }
        XCTAssertEqual(launch.packageURL, package)
        XCTAssertFalse(coordinator.isModelSetupPresented)
    }

    func testFailedProjectRequiresReadyModelBeforeRetrying() async {
        let package = URL(filePath: "/tmp/Failed.cloudpoint", directoryHint: .isDirectory)
        var manifest = ProjectManifest()
        manifest.sessionState = SessionState(phase: .failed, failedCount: 1)
        let store = CoordinatorStore(
            project: .fixture(packageURL: package, manifest: manifest)
        )
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: CoordinatorModelInstaller(health: .absent),
            engineContext: .fixture()
        )
        await coordinator.start()

        await coordinator.openInput(package)

        XCTAssertEqual(coordinator.destination, .welcome)
        XCTAssertTrue(coordinator.isModelSetupPresented)
    }

    func testRepairModelReopensCurrentProjectAfterSetupCompletes() async {
        let package = URL(filePath: "/tmp/Repair.cloudpoint", directoryHint: .isDirectory)
        let store = CoordinatorStore(project: .fixture(packageURL: package))
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: CoordinatorModelInstaller(health: .ready(.fixture())),
            engineContext: .fixture()
        )
        await coordinator.start()
        await coordinator.openInput(package)

        coordinator.repairModel()
        XCTAssertTrue(coordinator.isModelSetupPresented)
        await coordinator.continueAfterModelSetup()

        let openedURLs = await store.requestedProjectURLs()
        XCTAssertEqual(openedURLs, [package, package])
        XCTAssertFalse(coordinator.isModelSetupPresented)
    }

    func testWorkspaceViewModelIsOwnedOncePerProjectAcrossSceneRecreation() async {
        let package = URL(filePath: "/tmp/Owned.cloudpoint", directoryHint: .isDirectory)
        let coordinator = AppCoordinator(
            projectStore: CoordinatorStore(project: .fixture(packageURL: package)),
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: CoordinatorModelInstaller(health: .ready(.fixture())),
            engineContext: .fixture()
        )
        await coordinator.openInput(package)
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected workspace destination")
        }

        let first = coordinator.workspaceViewModel(for: launch)
        let second = coordinator.workspaceViewModel(for: launch)

        XCTAssertTrue(first === second)
        await first.close()
    }

    func testShowingWelcomeClosesAndEvictsTheCurrentWorkspace() async {
        let package = URL(filePath: "/tmp/Welcome-Eviction.cloudpoint", directoryHint: .isDirectory)
        let coordinator = AppCoordinator(
            projectStore: CoordinatorStore(project: .fixture(packageURL: package)),
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: CoordinatorModelInstaller(health: .ready(.fixture())),
            engineContext: .fixture()
        )
        await coordinator.openInput(package)
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected workspace destination")
        }
        weak var releasedViewModel: WorkspaceViewModel?
        do {
            let viewModel = coordinator.workspaceViewModel(for: launch)
            releasedViewModel = viewModel
        }

        await coordinator.showWelcome()

        XCTAssertEqual(coordinator.destination, .welcome)
        XCTAssertNil(releasedViewModel)
    }

    func testCopiedPackagesWithTheSameManifestUUIDNeverShareAWorkspaceViewModel() async {
        let firstPackage = URL(
            filePath: "/tmp/Original.cloudpoint",
            directoryHint: .isDirectory
        )
        let copiedPackage = URL(
            filePath: "/tmp/Copy.cloudpoint",
            directoryHint: .isDirectory
        )
        let projectID = UUID(uuidString: "00000000-0000-0000-0000-000000000199")!
        let manifest = ProjectManifest(projectID: projectID)
        let firstProject = ManagedProject.fixture(
            packageURL: firstPackage,
            manifest: manifest
        )
        let copiedProject = ManagedProject.fixture(
            packageURL: copiedPackage,
            manifest: manifest
        )
        let coordinator = AppCoordinator(
            projectStore: CoordinatorStore(projectsByURL: [
                firstPackage: firstProject,
                copiedPackage: copiedProject,
            ]),
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            modelInstaller: CoordinatorModelInstaller(health: .ready(.fixture())),
            engineContext: .fixture()
        )
        await coordinator.openInput(firstPackage)
        guard case let .workspace(firstLaunch) = coordinator.destination else {
            return XCTFail("Expected first workspace destination")
        }
        weak var releasedFirstViewModel: WorkspaceViewModel?
        do {
            let firstViewModel = coordinator.workspaceViewModel(for: firstLaunch)
            releasedFirstViewModel = firstViewModel
            XCTAssertEqual(firstViewModel.projectURL, firstPackage)
        }

        await coordinator.openInput(copiedPackage)
        guard case let .workspace(copiedLaunch) = coordinator.destination else {
            return XCTFail("Expected copied workspace destination")
        }
        let copiedViewModel = coordinator.workspaceViewModel(for: copiedLaunch)

        XCTAssertEqual(copiedViewModel.projectURL, copiedPackage)
        XCTAssertNil(releasedFirstViewModel)
        await copiedViewModel.close()
    }

    func testReadyModelPublishesEngineStateAndAllowsVideoRoute() async {
        let video = URL(filePath: "/tmp/ready.mov")
        let store = CoordinatorStore(project: .fixture())
        let installer = CoordinatorModelInstaller(health: .ready(.fixture()))
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: CoordinatorVideoProbe(
                result: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4)
            ),
            recordingSources: CoordinatorRecordingSources(),
            modelInstaller: installer,
            engineContext: .fixture()
        )
        await coordinator.start()

        await coordinator.openInput(video)
        await coordinator.createPendingReconstruction(mode: .lingbotPointCloud)

        let requestedSourceNames = await store.requestedSourceNames()
        XCTAssertEqual(requestedSourceNames, ["ready.mov"])
        XCTAssertEqual(coordinator.engineState, .ready)
        XCTAssertFalse(coordinator.isModelSetupPresented)
    }
}

private actor CoordinatorStore: ManagedProjectStoring {
    let project: ManagedProject
    let projectsByURL: [URL: ManagedProject]
    private(set) var createdSourceNames: [String] = []
    private(set) var openedURLs: [URL] = []
    private(set) var recordingSources: [RecordingSourceReference] = []
    private(set) var sharpFrames: [VideoKeyFrameCandidate] = []
    private(set) var cameraSources: [CameraSourceReference] = []

    init(project: ManagedProject) {
        self.project = project
        projectsByURL = [:]
    }

    init(projectsByURL: [URL: ManagedProject]) {
        precondition(!projectsByURL.isEmpty)
        project = projectsByURL.values.first!
        self.projectsByURL = projectsByURL
    }

    func createProject(sourceName: String) -> ManagedProject {
        createdSourceNames.append(sourceName)
        return project
    }

    func createRecordingProject(
        sourceName: String,
        source: RecordingSourceReference
    ) -> ManagedProject {
        createdSourceNames.append(sourceName)
        recordingSources.append(source)
        return project
    }

    func createSharpRecordingProject(
        sourceName: String,
        source: RecordingSourceReference,
        selectedFrame: VideoKeyFrameCandidate
    ) -> ManagedProject {
        createdSourceNames.append(sourceName)
        recordingSources.append(source)
        sharpFrames.append(selectedFrame)
        return project
    }

    func createCameraProject(
        sourceName: String,
        source: CameraSourceReference
    ) -> ManagedProject {
        createdSourceNames.append(sourceName)
        cameraSources.append(source)
        return project
    }

    func openProject(at url: URL) -> ManagedProject {
        openedURLs.append(url)
        return projectsByURL[url] ?? project
    }

    func recentProjects() -> [RecentProject] { [] }

    func requestedSourceNames() -> [String] { createdSourceNames }

    func requestedProjectURLs() -> [URL] { openedURLs }

    func requestedRecordingSources() -> [RecordingSourceReference] { recordingSources }

    func requestedSharpFrames() -> [VideoKeyFrameCandidate] { sharpFrames }
}

private actor CoordinatorKeyFrameSelector: VideoKeyFrameSelecting {
    let providedCandidates: [VideoKeyFrameCandidate]
    private var urls: [URL] = []

    init(candidates: [VideoKeyFrameCandidate]) {
        providedCandidates = candidates
    }

    func candidates(
        for url: URL,
        durationSeconds: Double,
        count: Int
    ) -> [VideoKeyFrameCandidate] {
        urls.append(url)
        return providedCandidates
    }

    func requestedURLs() -> [URL] { urls }
}

private actor CoordinatorVideoProbe: VideoMetadataProbing {
    let result: VideoProbeResult
    private(set) var probedURLs: [URL] = []

    init(result: VideoProbeResult) { self.result = result }

    func probe(_ url: URL, framesPerSecond: Int) async throws -> VideoProbeResult {
        probedURLs.append(url)
        return result
    }

    func requestedURLs() -> [URL] { probedURLs }
}

private struct CoordinatorRecordingSources: RecordingSourceManaging {
    func makeReference(
        for url: URL,
        probe: VideoProbeResult,
        framesPerSecond: Int
    ) async throws -> RecordingSourceReference {
        RecordingSourceReference(
            bookmarkData: Data(url.path.utf8),
            originalFilename: url.lastPathComponent,
            fingerprint: RecordingSourceFingerprint(
                byteCount: 1,
                sha256: String(repeating: "a", count: 64)
            ),
            durationSeconds: probe.durationSeconds,
            framesPerSecond: framesPerSecond,
            expectedSampleCount: UInt64(probe.sampledFrameCount),
            nextSampleOrdinal: 0
        )
    }

    func resolve(_ reference: RecordingSourceReference) async throws -> URL {
        URL(filePath: String(decoding: reference.bookmarkData, as: UTF8.self))
    }

    func replacement(
        for url: URL,
        preserving reference: RecordingSourceReference
    ) async throws -> RecordingSourceReference {
        var result = reference
        result.bookmarkData = Data(url.path.utf8)
        return result
    }
}

private actor CoordinatorCameraPreflight: CameraPreflighting {
    private let outcome: Result<CameraPreflightResult, Error>
    private var requests = 0

    init(result: CameraPreflightResult) { outcome = .success(result) }
    init(error: Error) { outcome = .failure(error) }

    func preflight() async throws -> CameraPreflightResult {
        requests += 1
        return try outcome.get()
    }

    func requestCount() -> Int { requests }
}

private actor CoordinatorModelInstaller: ModelInstalling {
    let currentHealth: ModelHealth

    init(health: ModelHealth) { currentHealth = health }

    func health() -> ModelHealth { currentHealth }

    func prepare() -> AsyncThrowingStream<ModelSetupEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}

private extension ModelInstallation {
    static func fixture() -> ModelInstallation {
        ModelInstallation(
            directory: URL(filePath: "/tmp/model", directoryHint: .isDirectory),
            sourceRevision: "fixture",
            sourceSHA256: String(repeating: "a", count: 64),
            convertedSHA256: String(repeating: "b", count: 64),
            engineVersion: "fixture"
        )
    }
}

private extension ProductionReconstructionContext {
    static func fixture() -> ProductionReconstructionContext {
        ProductionReconstructionContext(
            runtime: .unchecked(
                root: URL(filePath: "/tmp/runtime", directoryHint: .isDirectory)
            ),
            modelDirectory: URL(filePath: "/tmp/model", directoryHint: .isDirectory)
        )
    }
}

private extension ManagedProject {
    static func fixture(
        packageURL: URL = URL(filePath: "/tmp/Test.cloudpoint", directoryHint: .isDirectory),
        manifest: ProjectManifest = ProjectManifest()
    ) -> ManagedProject {
        ManagedProject(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000122")!,
            displayName: "Test",
            packageURL: packageURL,
            manifest: manifest,
            lastOpenedAt: Date(timeIntervalSinceReferenceDate: 1)
        )
    }
}

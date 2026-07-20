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

    func testVideoRouteProbesThenCreatesDurableProjectAndCarriesInitialSource() async throws {
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
        let coordinator = AppCoordinator(projectStore: store, videoProbe: probe)

        await coordinator.openInput(video)

        let probedURLs = await probe.requestedURLs()
        let createdSourceNames = await store.requestedSourceNames()
        XCTAssertEqual(probedURLs, [video])
        XCTAssertEqual(createdSourceNames, ["IMG_2285.MOV"])
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Expected a workspace destination")
        }
        XCTAssertEqual(launch.projectID, project.id)
        XCTAssertEqual(launch.packageURL, package)
        XCTAssertEqual(launch.sourceTitle, "IMG_2285")
        XCTAssertEqual(launch.initialSource, .recording(video, framesPerSecond: 2))
    }

    func testDroppedAndFinderMoviesUseTheSameValidatedVideoRoute() async {
        let first = URL(filePath: "/tmp/first.mov")
        let second = URL(filePath: "/tmp/second.mp4")
        let store = CoordinatorStore(project: .fixture())
        let probe = CoordinatorVideoProbe(
            result: VideoProbeResult(durationSeconds: 1, sampledFrameCount: 2)
        )
        let coordinator = AppCoordinator(projectStore: store, videoProbe: probe)

        await coordinator.openDroppedItems([first])
        await coordinator.openExternalURL(second)

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
}

private actor CoordinatorStore: ManagedProjectStoring {
    let project: ManagedProject
    private(set) var createdSourceNames: [String] = []
    private(set) var openedURLs: [URL] = []

    init(project: ManagedProject) { self.project = project }

    func createProject(sourceName: String) -> ManagedProject {
        createdSourceNames.append(sourceName)
        return project
    }

    func openProject(at url: URL) -> ManagedProject {
        openedURLs.append(url)
        return project
    }

    func recentProjects() -> [RecentProject] { [] }

    func requestedSourceNames() -> [String] { createdSourceNames }

    func requestedProjectURLs() -> [URL] { openedURLs }
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

private extension ManagedProject {
    static func fixture(
        packageURL: URL = URL(filePath: "/tmp/Test.cloudpoint", directoryHint: .isDirectory)
    ) -> ManagedProject {
        ManagedProject(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000122")!,
            displayName: "Test",
            packageURL: packageURL,
            manifest: ProjectManifest(),
            lastOpenedAt: Date(timeIntervalSinceReferenceDate: 1)
        )
    }
}

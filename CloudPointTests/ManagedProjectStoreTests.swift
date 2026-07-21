import XCTest
@testable import CloudPoint

final class ManagedProjectStoreTests: XCTestCase {
    func testExternalRecentProjectPersistsAndResolvesSecurityScopedBookmark() async throws {
        let support = try TemporaryDirectory.make()
        let external = try TemporaryProjectPackage.make()
        let manifest = ProjectManifest.fixture()
        try manifest.writeAtomically(to: external.url)
        let bookmarks = ProjectBookmarkHarness()
        let store = ManagedProjectStore(
            applicationSupportDirectory: support.url,
            bookmarks: bookmarks
        )

        let opened = try await store.openProject(at: external.url)
        let recent = try await store.recentProjects()

        XCTAssertEqual(opened.packageBookmarkData, Data(external.url.path.utf8))
        XCTAssertEqual(recent.first?.packageBookmarkData, opened.packageBookmarkData)
        XCTAssertEqual(recent.first?.packageURL.standardizedFileURL, external.url.standardizedFileURL)
        XCTAssertGreaterThanOrEqual(bookmarks.resolveCount, 1)
    }

    @MainActor
    func testAppCoordinatorReopensExternalRecentUsingRetainedBookmarkCapability() async throws {
        let support = try TemporaryDirectory.make()
        let external = try TemporaryProjectPackage.make()
        var manifest = ProjectManifest.fixture()
        manifest.sessionState = SessionState(phase: .completed)
        try manifest.writeAtomically(to: external.url)
        let bookmarks = ProjectBookmarkHarness()
        let store = ManagedProjectStore(
            applicationSupportDirectory: support.url,
            bookmarks: bookmarks
        )
        _ = try await store.openProject(at: external.url)
        guard let recent = try await store.recentProjects().first else {
            return XCTFail("Expected external recent project")
        }
        let movedURL = external.url.deletingLastPathComponent().appending(
            path: "Moved-\(UUID().uuidString).cloudpoint",
            directoryHint: .isDirectory
        )
        try FileManager.default.moveItem(at: external.url, to: movedURL)
        defer { try? FileManager.default.removeItem(at: movedURL) }
        bookmarks.resolveTo(movedURL)
        let resolveCountBeforeReopen = bookmarks.resolveCount
        let coordinator = AppCoordinator(
            projectStore: store,
            videoProbe: UnusedProjectVideoProbe()
        )

        coordinator.openRecent(recent)

        let didFinish = await waitForRecentOpen {
            if case .workspace = coordinator.destination { return true }
            return coordinator.errorMessage != nil
        }
        guard case let .workspace(launch) = coordinator.destination else {
            return XCTFail("Recent project did not reopen: \(coordinator.errorMessage ?? "unknown error")")
        }
        XCTAssertTrue(didFinish)
        XCTAssertGreaterThan(bookmarks.resolveCount, resolveCountBeforeReopen)
        XCTAssertEqual(launch.packageURL.standardizedFileURL, movedURL.standardizedFileURL)
        XCTAssertEqual(launch.packageBookmarkData, Data(movedURL.path.utf8))
    }

    func testCreateProjectAtomicallyBuildsAutosavedPackageUnderApplicationSupport() async throws {
        let support = try TemporaryDirectory.make()
        let projectID = UUID(uuidString: "00000000-0000-0000-0000-000000000101")!
        let instant = Date(timeIntervalSinceReferenceDate: 12_345)
        let store = ManagedProjectStore(
            applicationSupportDirectory: support.url,
            now: { instant },
            makeUUID: { projectID }
        )

        let project = try await store.createProject(sourceName: "IMG_2285.MOV")

        XCTAssertEqual(project.id, projectID)
        XCTAssertEqual(project.displayName, "IMG_2285")
        XCTAssertEqual(
            project.packageURL.deletingLastPathComponent(),
            support.url.appending(path: "CloudPoint/Projects", directoryHint: .isDirectory)
        )
        XCTAssertEqual(project.packageURL.pathExtension, "cloudpoint")
        XCTAssertEqual(project.lastOpenedAt, instant)
        let manifest = try ProjectManifest.load(from: project.packageURL)
        XCTAssertEqual(manifest.projectID, projectID)
        XCTAssertEqual(manifest.sessionState, .empty)
        XCTAssertEqual(
            Set(try FileManager.default.contentsOfDirectory(atPath: project.packageURL.path)),
            ["Manifest.json", "Frames", "Predictions", "Points", "Outputs", "Logs"]
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: project.packageURL.appending(path: "Outputs/Gaussians").path
            )
        )
        XCTAssertEqual(try partialFiles(beneath: support.url), [])
    }

    func testRecordingProjectAtomicallyPersistsBookmarkFingerprintAndSamplingPlan() async throws {
        let support = try TemporaryDirectory.make()
        let store = ManagedProjectStore(applicationSupportDirectory: support.url)
        let source = RecordingSourceReference(
            bookmarkData: Data("video-bookmark".utf8),
            originalFilename: "Atrium.mov",
            fingerprint: RecordingSourceFingerprint(
                byteCount: 1_024,
                sha256: String(repeating: "d", count: 64)
            ),
            durationSeconds: 11,
            framesPerSecond: 2,
            expectedSampleCount: 22,
            nextSampleOrdinal: 0
        )

        let project = try await store.createRecordingProject(
            sourceName: "Atrium.mov",
            source: source
        )

        XCTAssertEqual(project.manifest.recordingSource, source)
        XCTAssertEqual(try ProjectManifest.load(from: project.packageURL).recordingSource, source)
    }

    func testSharpRecordingProjectAtomicallyPersistsSelectedFullResolutionFrame() async throws {
        let support = try TemporaryDirectory.make()
        let store = ManagedProjectStore(applicationSupportDirectory: support.url)
        let source = RecordingSourceReference(
            bookmarkData: Data("video-bookmark".utf8),
            originalFilename: "Atrium.mov",
            fingerprint: RecordingSourceFingerprint(
                byteCount: 1_024,
                sha256: String(repeating: "d", count: 64)
            ),
            durationSeconds: 11,
            framesPerSecond: 2,
            expectedSampleCount: 22,
            nextSampleOrdinal: 0
        )
        let selected = VideoKeyFrameCandidate(
            index: 3,
            timestampSeconds: 4.25,
            thumbnailJPEG: Data("thumbnail".utf8),
            fullResolutionJPEG: Data("full-resolution-jpeg".utf8),
            sharpnessScore: 0.8,
            exposureScore: 0.9,
            temporalScore: 0.7
        )

        let project = try await store.createSharpRecordingProject(
            sourceName: "Atrium.mov",
            source: source,
            selectedFrame: selected
        )
        let manifest = try ProjectManifest.load(from: project.packageURL)

        XCTAssertEqual(manifest.reconstructionPlan.modeID, .sharpGaussian)
        XCTAssertEqual(manifest.outputState, .gaussian(nil))
        XCTAssertEqual(manifest.frames, [PersistedFrame(
            index: 0,
            sourceTimestamp: 4.25,
            relativePath: "Frames/00000000.jpg"
        )])
        XCTAssertEqual(manifest.recordingSource?.expectedSampleCount, 1)
        XCTAssertEqual(manifest.recordingSource?.nextSampleOrdinal, 1)
        XCTAssertEqual(manifest.sessionState.phase, .ready)
        XCTAssertEqual(manifest.sessionState.capturedCount, 1)
        XCTAssertEqual(
            try Data(contentsOf: project.packageURL.appending(path: "Frames/00000000.jpg")),
            selected.fullResolutionJPEG
        )
        XCTAssertEqual(try partialFiles(beneath: support.url), [])
    }

    func testSharpCameraProjectPersistsSnapshotAndCameraIdentity() async throws {
        let support = try TemporaryDirectory.make()
        let store = ManagedProjectStore(applicationSupportDirectory: support.url)
        let source = CameraSourceReference(
            deviceID: "camera-42",
            deviceName: "Studio Camera",
            mirrorDisplay: true
        )
        let snapshot = VideoKeyFrameCandidate(
            index: 0,
            timestampSeconds: 0,
            thumbnailJPEG: Data("thumbnail".utf8),
            fullResolutionJPEG: Data("camera-jpeg".utf8),
            sharpnessScore: 0.8,
            exposureScore: 0.9,
            temporalScore: 1
        )

        let project = try await store.createSharpCameraProject(
            sourceName: "Studio Camera Snapshot",
            source: source,
            selectedFrame: snapshot
        )
        let manifest = try ProjectManifest.load(from: project.packageURL)

        XCTAssertEqual(manifest.reconstructionPlan.modeID, .sharpGaussian)
        XCTAssertEqual(manifest.cameraSource, source)
        XCTAssertNil(manifest.recordingSource)
        XCTAssertEqual(manifest.sessionState.phase, .ready)
        XCTAssertEqual(manifest.sessionState.capturedCount, 1)
        XCTAssertEqual(manifest.frames.count, 1)
        XCTAssertEqual(
            try Data(contentsOf: project.packageURL.appending(path: "Frames/00000000.jpg")),
            snapshot.fullResolutionJPEG
        )
        XCTAssertEqual(try partialFiles(beneath: support.url), [])
    }

    func testCameraProjectPersistsPreflightSelection() async throws {
        let support = try TemporaryDirectory.make()
        let store = ManagedProjectStore(applicationSupportDirectory: support.url)
        let source = CameraSourceReference(deviceID: "camera-42", deviceName: "Studio Camera")

        let project = try await store.createCameraProject(
            sourceName: "Studio Camera Capture",
            source: source
        )

        XCTAssertEqual(project.manifest.cameraSource, source)
        XCTAssertEqual(try ProjectManifest.load(from: project.packageURL).cameraSource, source)
    }

    func testOpenProjectLoadsCommittedManifestAndRefreshesRecentState() async throws {
        let support = try TemporaryDirectory.make()
        let clock = ValueSequence([
            Date(timeIntervalSinceReferenceDate: 100),
            Date(timeIntervalSinceReferenceDate: 200),
        ])
        let store = ManagedProjectStore(
            applicationSupportDirectory: support.url,
            now: { clock.next() },
            makeUUID: { UUID(uuidString: "00000000-0000-0000-0000-000000000102")! }
        )
        let created = try await store.createProject(sourceName: "Atrium.mp4")
        var committed = created.manifest
        committed.sessionState = SessionState(phase: .completed)
        committed.updatedAt = Date(timeIntervalSinceReferenceDate: 150)
        try committed.writeAtomically(to: created.packageURL)

        let reopened = try await store.openProject(at: created.packageURL)

        XCTAssertEqual(reopened.manifest, committed)
        XCTAssertEqual(reopened.displayName, "Atrium")
        XCTAssertEqual(reopened.lastOpenedAt, Date(timeIntervalSinceReferenceDate: 200))
        let recent = try await store.recentProjects()
        XCTAssertEqual(recent.map(\.id), [created.id])
        XCTAssertEqual(recent.first?.phase, .completed)
        XCTAssertEqual(recent.first?.lastOpenedAt, reopened.lastOpenedAt)
    }

    func testRecentProjectsAreOrderedByMostRecentlyOpened() async throws {
        let support = try TemporaryDirectory.make()
        let clock = ValueSequence([
            Date(timeIntervalSinceReferenceDate: 10),
            Date(timeIntervalSinceReferenceDate: 20),
            Date(timeIntervalSinceReferenceDate: 30),
        ])
        let identifiers = ValueSequence([
            UUID(uuidString: "00000000-0000-0000-0000-000000000111")!,
            UUID(uuidString: "00000000-0000-0000-0000-000000000112")!,
        ])
        let store = ManagedProjectStore(
            applicationSupportDirectory: support.url,
            now: { clock.next() },
            makeUUID: { identifiers.next() }
        )
        let first = try await store.createProject(sourceName: "First.mov")
        let second = try await store.createProject(sourceName: "Second.m4v")

        _ = try await store.openProject(at: first.packageURL)
        let recent = try await store.recentProjects()

        XCTAssertEqual(recent.map(\.id), [first.id, second.id])
        XCTAssertEqual(recent.map(\.displayName), ["First", "Second"])
        XCTAssertEqual(recent.map(\.lastOpenedAt), [
            Date(timeIntervalSinceReferenceDate: 30),
            Date(timeIntervalSinceReferenceDate: 20),
        ])
    }

    private func partialFiles(beneath root: URL) throws -> [String] {
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            throw CocoaError(.fileReadUnknown)
        }
        return enumerator.compactMap { ($0 as? URL)?.lastPathComponent }
            .filter { $0.hasSuffix(".partial") }
            .sorted()
    }
}

private final class ProjectBookmarkHarness: SecurityScopedBookmarking, @unchecked Sendable {
    private let lock = NSLock()
    private var storedResolveCount = 0
    private var resolvedURL: URL?

    var resolveCount: Int { lock.withLock { storedResolveCount } }

    func resolveTo(_ url: URL) {
        lock.withLock { resolvedURL = url }
    }

    func makeBookmark(for url: URL) throws -> Data { Data(url.path.utf8) }

    func resolve(_ bookmark: Data) throws -> SecurityScopedBookmarkResolution {
        let url = lock.withLock {
            storedResolveCount += 1
            return resolvedURL ?? URL(
                filePath: String(decoding: bookmark, as: UTF8.self),
                directoryHint: .isDirectory
            )
        }
        return SecurityScopedBookmarkResolution(
            url: url,
            isStale: false
        )
    }
}

@MainActor
private func waitForRecentOpen(
    timeout: Duration = .seconds(2),
    condition: @escaping @MainActor () -> Bool
) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while clock.now < deadline {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}

private struct UnusedProjectVideoProbe: VideoMetadataProbing {
    func probe(_ url: URL, framesPerSecond: Int) async throws -> VideoProbeResult {
        throw VideoProbeError.noVideoTrack
    }
}

private final class TemporaryDirectory {
    let url: URL

    private init(url: URL) { self.url = url }

    deinit { try? FileManager.default.removeItem(at: url) }

    static func make() throws -> TemporaryDirectory {
        let url = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return TemporaryDirectory(url: url)
    }
}

private final class ValueSequence<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Value]

    init(_ values: [Value]) { self.values = values }

    func next() -> Value {
        lock.withLock { values.removeFirst() }
    }
}

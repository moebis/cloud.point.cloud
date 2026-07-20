import XCTest
@testable import CloudPoint

final class ManagedProjectStoreTests: XCTestCase {
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
            ["Manifest.json", "Frames", "Predictions", "Points", "Logs"]
        )
        XCTAssertEqual(try partialFiles(beneath: support.url), [])
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

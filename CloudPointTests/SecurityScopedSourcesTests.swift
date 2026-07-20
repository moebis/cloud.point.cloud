import XCTest
@testable import CloudPoint

final class SecurityScopedSourcesTests: XCTestCase {
    func testRecordingFingerprintDetectsSameSizeContentReplacement() async throws {
        let root = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString,
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let video = root.appending(path: "clip.mov")
        try Data("abc".utf8).write(to: video)
        let bookmarks = SourceBookmarkHarness(url: video)
        let scope = SourceScopeHarness()
        let manager = SystemRecordingSourceManager(bookmarks: bookmarks, scope: scope)

        let reference = try await manager.makeReference(
            for: video,
            probe: VideoProbeResult(durationSeconds: 2, sampledFrameCount: 4),
            framesPerSecond: 2
        )

        XCTAssertEqual(
            reference.fingerprint.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
        let resolvedURL = try await manager.resolve(reference)
        XCTAssertEqual(resolvedURL, video.standardizedFileURL)

        try Data("abd".utf8).write(to: video)
        do {
            _ = try await manager.resolve(reference)
            XCTFail("Expected a stable fingerprint mismatch")
        } catch {
            XCTAssertEqual(error as? RecordingSourceAccessError, .changed)
        }
        XCTAssertEqual(scope.startCount, scope.stopCount)
    }
}

private final class SourceBookmarkHarness: SecurityScopedBookmarking, @unchecked Sendable {
    let url: URL

    init(url: URL) { self.url = url }

    func makeBookmark(for url: URL) throws -> Data { Data("bookmark".utf8) }

    func resolve(_ bookmark: Data) throws -> SecurityScopedBookmarkResolution {
        SecurityScopedBookmarkResolution(url: url, isStale: false)
    }
}

private final class SourceScopeHarness: SecurityScopedResourceAccessing, @unchecked Sendable {
    private let lock = NSLock()
    private var starts = 0
    private var stops = 0

    var startCount: Int { lock.withLock { starts } }
    var stopCount: Int { lock.withLock { stops } }

    func startAccessing(_ url: URL) -> Bool {
        lock.withLock { starts += 1 }
        return true
    }

    func stopAccessing(_ url: URL) {
        lock.withLock { stops += 1 }
    }
}

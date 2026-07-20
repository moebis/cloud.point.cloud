import CryptoKit
import Foundation

struct SecurityScopedBookmarkResolution: Sendable, Equatable {
    let url: URL
    let isStale: Bool
}

protocol SecurityScopedBookmarking: Sendable {
    func makeBookmark(for url: URL) throws -> Data
    func resolve(_ bookmark: Data) throws -> SecurityScopedBookmarkResolution
}

struct SystemSecurityScopedBookmarks: SecurityScopedBookmarking {
    func makeBookmark(for url: URL) throws -> Data {
        try url.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
    }

    func resolve(_ bookmark: Data) throws -> SecurityScopedBookmarkResolution {
        var stale = false
        let url = try URL(
            resolvingBookmarkData: bookmark,
            options: [.withSecurityScope],
            relativeTo: nil,
            bookmarkDataIsStale: &stale
        )
        return SecurityScopedBookmarkResolution(url: url.standardizedFileURL, isStale: stale)
    }
}

protocol RecordingSourceManaging: Sendable {
    func makeReference(
        for url: URL,
        probe: VideoProbeResult,
        framesPerSecond: Int
    ) async throws -> RecordingSourceReference
    func resolve(_ reference: RecordingSourceReference) async throws -> URL
    func replacement(
        for url: URL,
        preserving reference: RecordingSourceReference
    ) async throws -> RecordingSourceReference
}

enum RecordingSourceAccessError: Error, LocalizedError, Equatable, Sendable {
    case unavailable
    case changed
    case unreadable

    var errorDescription: String? {
        switch self {
        case .unavailable:
            "CloudPoint can’t access the original video. Locate it to resume from the last frame."
        case .changed:
            "That video does not match the original recording for this project."
        case .unreadable:
            "CloudPoint couldn’t read the selected video."
        }
    }
}

struct SystemRecordingSourceManager: RecordingSourceManaging {
    private let bookmarks: any SecurityScopedBookmarking
    private let scope: any SecurityScopedResourceAccessing

    init(
        bookmarks: any SecurityScopedBookmarking = SystemSecurityScopedBookmarks(),
        scope: any SecurityScopedResourceAccessing = SystemSecurityScopedResourceAccess()
    ) {
        self.bookmarks = bookmarks
        self.scope = scope
    }

    func makeReference(
        for url: URL,
        probe: VideoProbeResult,
        framesPerSecond: Int
    ) async throws -> RecordingSourceReference {
        let fingerprint = try await fingerprint(url)
        let bookmark: Data
        do { bookmark = try bookmarks.makeBookmark(for: url) }
        catch { throw RecordingSourceAccessError.unavailable }
        guard let expectedCount = UInt64(exactly: probe.sampledFrameCount) else {
            throw RecordingSourceAccessError.unreadable
        }
        return RecordingSourceReference(
            bookmarkData: bookmark,
            originalFilename: url.lastPathComponent,
            fingerprint: fingerprint,
            durationSeconds: probe.durationSeconds,
            framesPerSecond: framesPerSecond,
            expectedSampleCount: expectedCount,
            nextSampleOrdinal: 0
        )
    }

    func resolve(_ reference: RecordingSourceReference) async throws -> URL {
        let resolution: SecurityScopedBookmarkResolution
        do { resolution = try bookmarks.resolve(reference.bookmarkData) }
        catch { throw RecordingSourceAccessError.unavailable }
        let candidate = resolution.url
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            throw RecordingSourceAccessError.unavailable
        }
        guard try await fingerprint(candidate) == reference.fingerprint else {
            throw RecordingSourceAccessError.changed
        }
        return candidate
    }

    func replacement(
        for url: URL,
        preserving reference: RecordingSourceReference
    ) async throws -> RecordingSourceReference {
        guard try await fingerprint(url) == reference.fingerprint else {
            throw RecordingSourceAccessError.changed
        }
        var replacement = reference
        do { replacement.bookmarkData = try bookmarks.makeBookmark(for: url) }
        catch { throw RecordingSourceAccessError.unavailable }
        replacement.originalFilename = url.lastPathComponent
        return replacement
    }

    private func fingerprint(_ url: URL) async throws -> RecordingSourceFingerprint {
        let didStartScope = scope.startAccessing(url)
        defer { if didStartScope { scope.stopAccessing(url) } }
        do {
            return try await Task.detached(priority: .userInitiated) {
                let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                guard values.isRegularFile == true,
                      let size = values.fileSize,
                      size > 0,
                      let byteCount = UInt64(exactly: size) else {
                    throw RecordingSourceAccessError.unreadable
                }
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                var digest = SHA256()
                while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
                    try Task.checkCancellation()
                    digest.update(data: data)
                }
                return RecordingSourceFingerprint(
                    byteCount: byteCount,
                    sha256: digest.finalize().map { String(format: "%02x", $0) }.joined()
                )
            }.value
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as RecordingSourceAccessError {
            throw error
        } catch {
            throw RecordingSourceAccessError.unreadable
        }
    }
}

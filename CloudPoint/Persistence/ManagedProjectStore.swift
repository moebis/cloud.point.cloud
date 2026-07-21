import Darwin
import Foundation

struct ManagedProject: Identifiable, Sendable, Equatable {
    let id: UUID
    let displayName: String
    let packageURL: URL
    let packageBookmarkData: Data?
    let manifest: ProjectManifest
    let lastOpenedAt: Date

    init(
        id: UUID,
        displayName: String,
        packageURL: URL,
        packageBookmarkData: Data? = nil,
        manifest: ProjectManifest,
        lastOpenedAt: Date
    ) {
        self.id = id
        self.displayName = displayName
        self.packageURL = packageURL
        self.packageBookmarkData = packageBookmarkData
        self.manifest = manifest
        self.lastOpenedAt = lastOpenedAt
    }
}

struct RecentProject: Identifiable, Sendable, Equatable {
    let id: UUID
    let displayName: String
    let packageURL: URL
    let packageBookmarkData: Data?
    let phase: SessionPhase
    let lastOpenedAt: Date

    init(
        id: UUID,
        displayName: String,
        packageURL: URL,
        packageBookmarkData: Data? = nil,
        phase: SessionPhase,
        lastOpenedAt: Date
    ) {
        self.id = id
        self.displayName = displayName
        self.packageURL = packageURL
        self.packageBookmarkData = packageBookmarkData
        self.phase = phase
        self.lastOpenedAt = lastOpenedAt
    }
}

protocol ManagedProjectStoring: Sendable {
    func createProject(sourceName: String) async throws -> ManagedProject
    func createRecordingProject(
        sourceName: String,
        source: RecordingSourceReference
    ) async throws -> ManagedProject
    func createSharpRecordingProject(
        sourceName: String,
        source: RecordingSourceReference,
        selectedFrame: VideoKeyFrameCandidate
    ) async throws -> ManagedProject
    func createCameraProject(
        sourceName: String,
        source: CameraSourceReference
    ) async throws -> ManagedProject
    func openProject(at url: URL) async throws -> ManagedProject
    func openRecentProject(_ recent: RecentProject) async throws -> ManagedProject
    func recentProjects() async throws -> [RecentProject]
}

extension ManagedProjectStoring {
    func openRecentProject(_ recent: RecentProject) async throws -> ManagedProject {
        try await openProject(at: recent.packageURL)
    }
}

enum ManagedProjectStoreError: Error, LocalizedError, Equatable {
    case applicationSupportUnavailable
    case invalidProjectPackage
    case atomicMoveFailed

    var errorDescription: String? {
        switch self {
        case .applicationSupportUnavailable:
            "CloudPoint could not access Application Support."
        case .invalidProjectPackage:
            "The selected item is not a valid CloudPoint project."
        case .atomicMoveFailed:
            "CloudPoint could not create the project atomically."
        }
    }
}

actor ManagedProjectStore: ManagedProjectStoring {
    private struct RecentRecord: Codable, Sendable, Equatable {
        let id: UUID
        var displayName: String
        var packagePath: String
        var packageBookmarkData: Data?
        var phase: SessionPhase
        var lastOpenedAt: Date
    }

    private let fileManager = FileManager.default
    private let applicationSupportDirectory: URL
    private let bookmarks: any SecurityScopedBookmarking
    private let scope: any SecurityScopedResourceAccessing
    private let now: @Sendable () -> Date
    private let makeUUID: @Sendable () -> UUID

    init(
        applicationSupportDirectory: URL,
        bookmarks: any SecurityScopedBookmarking = SystemSecurityScopedBookmarks(),
        scope: any SecurityScopedResourceAccessing = SystemSecurityScopedResourceAccess(),
        now: @escaping @Sendable () -> Date = Date.init,
        makeUUID: @escaping @Sendable () -> UUID = UUID.init
    ) {
        self.applicationSupportDirectory = applicationSupportDirectory.standardizedFileURL
        self.bookmarks = bookmarks
        self.scope = scope
        self.now = now
        self.makeUUID = makeUUID
    }

    static func live() throws -> ManagedProjectStore {
        guard let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            throw ManagedProjectStoreError.applicationSupportUnavailable
        }
        return ManagedProjectStore(
            applicationSupportDirectory: support
        )
    }

    func createProject(sourceName: String) throws -> ManagedProject {
        try createProject(
            sourceName: sourceName,
            recordingSource: nil,
            cameraSource: nil,
            reconstructionPlan: nil,
            selectedFrame: nil
        )
    }

    func createRecordingProject(
        sourceName: String,
        source: RecordingSourceReference
    ) async throws -> ManagedProject {
        try createProject(
            sourceName: sourceName,
            recordingSource: source,
            cameraSource: nil,
            reconstructionPlan: nil,
            selectedFrame: nil
        )
    }

    func createSharpRecordingProject(
        sourceName: String,
        source: RecordingSourceReference,
        selectedFrame: VideoKeyFrameCandidate
    ) async throws -> ManagedProject {
        var singleFrameSource = source
        singleFrameSource.framesPerSecond = 1
        singleFrameSource.expectedSampleCount = 1
        singleFrameSource.nextSampleOrdinal = 1
        return try createProject(
            sourceName: sourceName,
            recordingSource: singleFrameSource,
            cameraSource: nil,
            reconstructionPlan: .sharp(
                SharpReconstructionConfiguration(inputFrameIndex: 0)
            ),
            selectedFrame: selectedFrame
        )
    }

    func createCameraProject(
        sourceName: String,
        source: CameraSourceReference
    ) async throws -> ManagedProject {
        try createProject(
            sourceName: sourceName,
            recordingSource: nil,
            cameraSource: source,
            reconstructionPlan: nil,
            selectedFrame: nil
        )
    }

    private func createProject(
        sourceName: String,
        recordingSource: RecordingSourceReference?,
        cameraSource: CameraSourceReference?,
        reconstructionPlan: ReconstructionPlan?,
        selectedFrame: VideoKeyFrameCandidate?
    ) throws -> ManagedProject {
        let projectID = makeUUID()
        let openedAt = now()
        let displayName = Self.displayName(for: sourceName)
        let projects = projectsDirectory
        try fileManager.createDirectory(at: projects, withIntermediateDirectories: true)

        let packageName = "\(Self.safeFilename(displayName))-\(projectID.uuidString.lowercased()).cloudpoint"
        let packageURL = projects.appending(path: packageName, directoryHint: .isDirectory)
        let stagingURL = projects.appending(
            path: ".\(packageName).\(UUID().uuidString.lowercased()).partial",
            directoryHint: .isDirectory
        )
        var packageCommitted = false
        defer {
            if !packageCommitted { try? fileManager.removeItem(at: stagingURL) }
        }

        try fileManager.createDirectory(at: stagingURL, withIntermediateDirectories: false)
        for directory in ["Frames", "Predictions", "Points", "Outputs/Gaussians", "Logs"] {
            try fileManager.createDirectory(
                at: stagingURL.appending(path: directory, directoryHint: .isDirectory),
                withIntermediateDirectories: true
            )
        }
        let frames: [PersistedFrame]
        let sessionState: SessionState
        if let selectedFrame {
            let frame = PersistedFrame(
                index: 0,
                sourceTimestamp: selectedFrame.timestampSeconds,
                relativePath: "Frames/00000000.jpg"
            )
            try selectedFrame.fullResolutionJPEG.write(
                to: stagingURL.appending(path: frame.relativePath),
                options: .withoutOverwriting
            )
            frames = [frame]
            sessionState = SessionState(phase: .ready, capturedCount: 1)
        } else {
            frames = []
            sessionState = .empty
        }
        let manifest = ProjectManifest(
            projectID: projectID,
            createdAt: openedAt,
            updatedAt: openedAt,
            reconstructionPlan: reconstructionPlan,
            recordingSource: recordingSource,
            cameraSource: cameraSource,
            frames: frames,
            sessionState: sessionState
        )
        try manifest.writeAtomically(to: stagingURL)
        try Self.renameExclusively(from: stagingURL, to: packageURL)
        packageCommitted = true

        do {
            try upsertRecent(RecentRecord(
                id: projectID,
                displayName: displayName,
                packagePath: packageURL.path,
                packageBookmarkData: nil,
                phase: manifest.sessionState.phase,
                lastOpenedAt: openedAt
            ))
        } catch {
            try? fileManager.removeItem(at: packageURL)
            throw error
        }

        return ManagedProject(
            id: projectID,
            displayName: displayName,
            packageURL: packageURL,
            manifest: manifest,
            lastOpenedAt: openedAt
        )
    }

    func openProject(at url: URL) throws -> ManagedProject {
        let packageURL = url.standardizedFileURL
        return try openProjectWithSecurityScope(at: packageURL)
    }

    func openRecentProject(_ recent: RecentProject) throws -> ManagedProject {
        guard let bookmarkData = recent.packageBookmarkData else {
            return try openProject(at: recent.packageURL)
        }
        let resolution = try bookmarks.resolve(bookmarkData)
        return try openProjectWithSecurityScope(
            at: resolution.url.standardizedFileURL,
            expectedProjectID: recent.id
        )
    }

    private func openProjectWithSecurityScope(
        at packageURL: URL,
        expectedProjectID: UUID? = nil
    ) throws -> ManagedProject {
        let didStartScope = scope.startAccessing(packageURL)
        defer { if didStartScope { scope.stopAccessing(packageURL) } }
        return try openProjectWithinScope(
            at: packageURL,
            expectedProjectID: expectedProjectID
        )
    }

    private func openProjectWithinScope(
        at packageURL: URL,
        expectedProjectID: UUID?
    ) throws -> ManagedProject {
        guard packageURL.pathExtension.lowercased() == "cloudpoint",
              (try packageURL.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true else {
            throw ManagedProjectStoreError.invalidProjectPackage
        }
        let manifest = try ProjectManifest.load(from: packageURL)
        guard expectedProjectID == nil || manifest.projectID == expectedProjectID else {
            throw ManagedProjectStoreError.invalidProjectPackage
        }
        let openedAt = now()
        let records = try loadRecentRecords()
        let prior = records.first { $0.id == manifest.projectID }
        let displayName = prior?.displayName ?? Self.displayName(
            forProjectPackage: packageURL,
            projectID: manifest.projectID
        )
        let bookmarkData = Self.isDescendant(packageURL, of: projectsDirectory)
            ? nil
            : try bookmarks.makeBookmark(for: packageURL)
        try upsertRecent(RecentRecord(
            id: manifest.projectID,
            displayName: displayName,
            packagePath: packageURL.path,
            packageBookmarkData: bookmarkData,
            phase: manifest.sessionState.phase,
            lastOpenedAt: openedAt
        ))
        return ManagedProject(
            id: manifest.projectID,
            displayName: displayName,
            packageURL: packageURL,
            packageBookmarkData: bookmarkData,
            manifest: manifest,
            lastOpenedAt: openedAt
        )
    }

    func recentProjects() throws -> [RecentProject] {
        let records = try loadRecentRecords()
        var refreshed: [RecentRecord] = []
        for var record in records {
            let packageURL: URL
            if let bookmark = record.packageBookmarkData,
               let resolution = try? bookmarks.resolve(bookmark) {
                packageURL = resolution.url.standardizedFileURL
                record.packagePath = packageURL.path
                if resolution.isStale {
                    record.packageBookmarkData = try? bookmarks.makeBookmark(for: packageURL)
                }
            } else {
                packageURL = URL(
                    filePath: record.packagePath,
                    directoryHint: .isDirectory
                ).standardizedFileURL
            }
            let didStartScope = scope.startAccessing(packageURL)
            defer { if didStartScope { scope.stopAccessing(packageURL) } }
            guard fileManager.fileExists(atPath: packageURL.path),
                  let manifest = try? ProjectManifest.load(from: packageURL),
                  manifest.projectID == record.id else {
                continue
            }
            record.phase = manifest.sessionState.phase
            refreshed.append(record)
        }
        let sorted = Self.sortRecent(refreshed)
        if sorted != records { try writeRecentRecords(sorted) }
        return sorted.map {
            RecentProject(
                id: $0.id,
                displayName: $0.displayName,
                packageURL: URL(filePath: $0.packagePath, directoryHint: .isDirectory),
                packageBookmarkData: $0.packageBookmarkData,
                phase: $0.phase,
                lastOpenedAt: $0.lastOpenedAt
            )
        }
    }

    private var cloudPointDirectory: URL {
        applicationSupportDirectory.appending(path: "CloudPoint", directoryHint: .isDirectory)
    }

    private var projectsDirectory: URL {
        cloudPointDirectory.appending(path: "Projects", directoryHint: .isDirectory)
    }

    private var recentURL: URL {
        cloudPointDirectory.appending(path: "RecentProjects.json")
    }

    private func upsertRecent(_ record: RecentRecord) throws {
        var records = try loadRecentRecords()
        records.removeAll { $0.id == record.id || $0.packagePath == record.packagePath }
        records.append(record)
        records = Array(Self.sortRecent(records).prefix(20))
        try writeRecentRecords(records)
    }

    private func loadRecentRecords() throws -> [RecentRecord] {
        guard fileManager.fileExists(atPath: recentURL.path) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        return try decoder.decode([RecentRecord].self, from: Data(contentsOf: recentURL))
    }

    private func writeRecentRecords(_ records: [RecentRecord]) throws {
        try fileManager.createDirectory(at: cloudPointDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .millisecondsSince1970
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(records)
        let partialURL = cloudPointDirectory.appending(
            path: ".RecentProjects.json.\(UUID().uuidString.lowercased()).partial"
        )
        defer { try? fileManager.removeItem(at: partialURL) }
        try data.write(to: partialURL, options: .withoutOverwriting)
        let handle = try FileHandle(forWritingTo: partialURL)
        do {
            try handle.synchronize()
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
        let renameResult = partialURL.path.withCString { sourcePath in
            recentURL.path.withCString { destinationPath in
                Darwin.rename(sourcePath, destinationPath)
            }
        }
        guard renameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    private static func sortRecent(_ records: [RecentRecord]) -> [RecentRecord] {
        records.sorted {
            if $0.lastOpenedAt != $1.lastOpenedAt { return $0.lastOpenedAt > $1.lastOpenedAt }
            if $0.displayName != $1.displayName { return $0.displayName < $1.displayName }
            return $0.id.uuidString < $1.id.uuidString
        }
    }

    private static func displayName(for sourceName: String) -> String {
        let source = URL(filePath: sourceName).deletingPathExtension().lastPathComponent
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return source.isEmpty ? "CloudPoint Project" : source
    }

    private static func displayName(forProjectPackage url: URL, projectID: UUID) -> String {
        var base = url.deletingPathExtension().lastPathComponent
        let suffix = "-\(projectID.uuidString.lowercased())"
        if base.lowercased().hasSuffix(suffix) { base.removeLast(suffix.count) }
        return base.isEmpty ? "CloudPoint Project" : base
    }

    private static func safeFilename(_ displayName: String) -> String {
        let disallowed = CharacterSet(charactersIn: "/:\\\0")
        let mapped = displayName.unicodeScalars.map {
            disallowed.contains($0) ? "-" : String($0)
        }.joined()
        let trimmed = mapped.trimmingCharacters(in: .whitespacesAndNewlines)
        return String((trimmed.isEmpty ? "CloudPoint Project" : trimmed).prefix(80))
    }

    private static func isDescendant(_ candidate: URL, of directory: URL) -> Bool {
        let candidateComponents = candidate.standardizedFileURL.pathComponents
        let directoryComponents = directory.standardizedFileURL.pathComponents
        return candidateComponents.starts(with: directoryComponents)
    }

    private static func renameExclusively(from source: URL, to destination: URL) throws {
        let result = source.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                renameatx_np(
                    AT_FDCWD,
                    sourcePath,
                    AT_FDCWD,
                    destinationPath,
                    UInt32(RENAME_EXCL)
                )
            }
        }
        guard result == 0 else { throw ManagedProjectStoreError.atomicMoveFailed }
    }
}

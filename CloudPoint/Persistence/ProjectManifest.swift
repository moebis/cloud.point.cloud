import Foundation

enum ProjectManifestError: Error, Equatable, Sendable {
    case unsupportedFormatVersion(Int)
}

struct ProjectManifest: Codable, Sendable, Equatable {
    static let currentFormatVersion = 1

    var formatVersion: Int
    var projectID: UUID
    var createdAt: Date
    var updatedAt: Date
    var frames: [PersistedFrame]
    var completedWindows: [CompletedWindow]
    var sessionState: SessionState

    init(
        formatVersion: Int = ProjectManifest.currentFormatVersion,
        projectID: UUID = UUID(),
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        frames: [PersistedFrame] = [],
        completedWindows: [CompletedWindow] = [],
        sessionState: SessionState = .empty
    ) {
        self.formatVersion = formatVersion
        self.projectID = projectID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.frames = frames
        self.completedWindows = completedWindows
        self.sessionState = sessionState
    }

    static func load(from packageURL: URL) throws -> ProjectManifest {
        try removeStalePartials(beneath: packageURL)
        let data = try Data(contentsOf: manifestURL(in: packageURL))
        let manifest = try decode(data)

        guard manifest.formatVersion == currentFormatVersion else {
            throw ProjectManifestError.unsupportedFormatVersion(manifest.formatVersion)
        }

        return manifest
    }

    func writeAtomically(to packageURL: URL, fileManager: FileManager = .default) throws {
        let manifestURL = Self.manifestURL(in: packageURL)
        let partialURL = packageURL.appending(path: "Manifest.json.partial")
        let data = try Self.encode(self)

        try data.write(to: partialURL, options: .atomic)
        let partialHandle = try FileHandle(forWritingTo: partialURL)
        try partialHandle.synchronize()
        try partialHandle.close()

        if fileManager.fileExists(atPath: manifestURL.path) {
            _ = try fileManager.replaceItemAt(manifestURL, withItemAt: partialURL)
        } else {
            try fileManager.moveItem(at: partialURL, to: manifestURL)
        }
    }

    static func encode(_ manifest: ProjectManifest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(manifest)
    }

    static func decode(_ data: Data) throws -> ProjectManifest {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(ProjectManifest.self, from: data)
    }

    private static func manifestURL(in packageURL: URL) -> URL {
        packageURL.appending(path: "Manifest.json")
    }

    private static func removeStalePartials(beneath packageURL: URL) throws {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: packageURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        for case let url as URL in enumerator where url.pathExtension == "partial" {
            try fileManager.removeItem(at: url)
        }
    }
}

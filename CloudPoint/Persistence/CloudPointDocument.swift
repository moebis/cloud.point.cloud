import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static let cloudPointProject = UTType(
        exportedAs: "cloud.point.cloud.project",
        conformingTo: .package
    )
}

final class CloudPointDocument: ReferenceFileDocument, @unchecked Sendable {
    typealias Snapshot = ProjectManifest

    static var readableContentTypes: [UTType] {
        [.cloudPointProject]
    }

    private let manifestLock = NSLock()
    private var storedManifest: ProjectManifest

    var manifest: ProjectManifest {
        manifestLock.withLock { storedManifest }
    }

    init() {
        storedManifest = ProjectManifest()
    }

    required init(configuration: ReadConfiguration) throws {
        storedManifest = try Self.loadManifest(from: configuration.file)
    }

    /// Mirrors a manifest transaction that the active SessionController has
    /// already committed to disk. This deliberately does not publish
    /// `objectWillChange`: doing so would schedule a stale second writer through
    /// SwiftUI document autosave.
    func adoptCommittedManifest(_ committed: ProjectManifest) {
        manifestLock.withLock { storedManifest = committed }
    }

    static func loadManifest(from package: FileWrapper) throws -> ProjectManifest {
        guard let manifestData = package.fileWrappers?["Manifest.json"]?.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }

        return try ProjectManifest.decode(manifestData)
    }

    func snapshot(contentType: UTType) throws -> ProjectManifest {
        manifest
    }

    func fileWrapper(
        snapshot: ProjectManifest,
        configuration: WriteConfiguration
    ) throws -> FileWrapper {
        try Self.packageWrapper(for: snapshot, preserving: configuration.existingFile)
    }

    static func packageWrapper(
        for manifest: ProjectManifest,
        preserving existingPackage: FileWrapper? = nil
    ) throws -> FileWrapper {
        var packageContents = existingPackage?.fileWrappers ?? [:]

        for name in ["Frames", "Predictions", "Points", "Logs"] where packageContents[name] == nil {
            packageContents[name] = FileWrapper(directoryWithFileWrappers: [:])
        }

        packageContents["Manifest.json"] = FileWrapper(
            regularFileWithContents: try ProjectManifest.encode(manifest)
        )

        return FileWrapper(directoryWithFileWrappers: packageContents)
    }
}

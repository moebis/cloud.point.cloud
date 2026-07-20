import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static let cloudPointProject = UTType(
        exportedAs: "cloud.point.cloud.project",
        conformingTo: .package
    )
}

@MainActor
final class CloudPointDocument: @preconcurrency ReferenceFileDocument {
    typealias Snapshot = ProjectManifest

    static var readableContentTypes: [UTType] {
        [.cloudPointProject]
    }

    var manifest: ProjectManifest

    init() {
        manifest = ProjectManifest()
    }

    required init(configuration: ReadConfiguration) throws {
        manifest = try Self.loadManifest(from: configuration.file)
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

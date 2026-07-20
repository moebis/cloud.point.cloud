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

    var manifest: ProjectManifest

    init() {
        manifest = ProjectManifest()
    }

    required init(configuration: ReadConfiguration) throws {
        guard let manifestData = configuration.file.fileWrappers?["Manifest.json"]?.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }

        manifest = try ProjectManifest.decode(manifestData)
    }

    func snapshot(contentType: UTType) throws -> ProjectManifest {
        manifest
    }

    func fileWrapper(
        snapshot: ProjectManifest,
        configuration: WriteConfiguration
    ) throws -> FileWrapper {
        try Self.packageWrapper(for: snapshot)
    }

    static func packageWrapper(for manifest: ProjectManifest) throws -> FileWrapper {
        let manifestWrapper = FileWrapper(regularFileWithContents: try ProjectManifest.encode(manifest))
        let packageDirectories = ["Frames", "Predictions", "Points", "Logs"].reduce(
            into: [String: FileWrapper]()
        ) { directories, name in
            directories[name] = FileWrapper(directoryWithFileWrappers: [:])
        }
        var packageContents = packageDirectories
        packageContents["Manifest.json"] = manifestWrapper

        return FileWrapper(directoryWithFileWrappers: packageContents)
    }
}

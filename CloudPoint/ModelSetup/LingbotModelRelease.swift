import Foundation

struct LingbotModelRelease: Sendable, Equatable {
    let repository: String
    let revision: String
    let sourceCommit: String
    let filename: String
    let sourceBytes: Int64
    let sourceSHA256: String
    let convertedBytes: Int64
    let convertedSHA256: String
    let tensorCount: Int
    let mlxVersion: String
    let engineVersion: String
    let downloadURL: URL

    static let v1 = Self(
        repository: "robbyant/lingbot-map",
        revision: "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9",
        sourceCommit: "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2",
        filename: "lingbot-map-long.pt",
        sourceBytes: 4_632_303_465,
        sourceSHA256: "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
        convertedBytes: 2_316_040_080,
        convertedSHA256: "eb966484923b5a205677b3ce7316d079c46fc6503bc9b6ac256b6e11560ea2e5",
        tensorCount: 1_342,
        mlxVersion: "0.32.0",
        engineVersion: "0.1.0",
        downloadURL: URL(
            string: "https://huggingface.co/robbyant/lingbot-map/resolve/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/lingbot-map-long.pt?download=true"
        )!
    )

    static func fixture(sourceBytes: Int64, sourceSHA256: String) -> Self {
        Self(
            repository: "fixture/model",
            revision: "fixture-revision",
            sourceCommit: "fixture-source",
            filename: "fixture.pt",
            sourceBytes: sourceBytes,
            sourceSHA256: sourceSHA256,
            convertedBytes: 9,
            convertedSHA256: "ba451247dcf0d65bb50c654ae2ebfb3e3173ec730bd5174a4b9eef5b3dc7c6da",
            tensorCount: 1,
            mlxVersion: "fixture-mlx",
            engineVersion: "fixture",
            downloadURL: URL(string: "https://example.invalid/fixture.pt")!
        )
    }
}

struct ModelDirectories: Sendable, Equatable {
    let root: URL
    let releaseRoot: URL
    let source: URL
    let converted: URL
    let sourceCheckpoint: URL
    let sourcePartial: URL
    let convertedPartial: URL
    let resumeData: URL

    init(root: URL, release: LingbotModelRelease) {
        let standardized = root.standardizedFileURL
        let slug = release.repository.replacingOccurrences(of: "/", with: "-")
        let releaseRoot = standardized
            .appending(path: slug, directoryHint: .isDirectory)
            .appending(path: release.revision, directoryHint: .isDirectory)
        self.root = standardized
        self.releaseRoot = releaseRoot
        source = releaseRoot.appending(path: "source", directoryHint: .isDirectory)
        converted = releaseRoot.appending(path: "converted", directoryHint: .isDirectory)
        sourceCheckpoint = source.appending(path: release.filename)
        sourcePartial = releaseRoot.appending(path: ".checkpoint.download.partial")
        convertedPartial = releaseRoot.appending(path: ".converted.partial", directoryHint: .isDirectory)
        resumeData = releaseRoot.appending(path: "download.resume")
    }

    var convertedWeights: URL {
        converted.appending(path: "lingbot-map-long-f16.safetensors")
    }

    var weightsManifest: URL { converted.appending(path: "weights-manifest.json") }
    var modelManifest: URL { converted.appending(path: "model-manifest.json") }

    static func live(
        release: LingbotModelRelease = .v1,
        fileManager: FileManager = .default
    ) throws -> Self {
        let support = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return Self(
            root: support
                .appending(path: "cloud.point.cloud.CloudPoint", directoryHint: .isDirectory)
                .appending(path: "Models", directoryHint: .isDirectory),
            release: release
        )
    }
}

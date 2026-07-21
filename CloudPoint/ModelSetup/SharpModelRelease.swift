import CryptoKit
import Foundation

enum SharpModelLicenseKind: String, Codable, Sendable, Equatable {
    case appleMachineLearningResearchModel
}

struct SharpModelRelease: Sendable, Equatable {
    let identifier: String
    let sourceCommit: String
    let filename: String
    let checkpointBytes: Int64
    let checkpointSHA256: String
    let requiredFreeBytes: Int64
    let downloadURL: URL
    let licenseKind: SharpModelLicenseKind

    static let v1 = Self(
        identifier: "apple-sharp-2572gikvuh",
        sourceCommit: "1eaa046834b81852261262b41b0919f5c1efdd2e",
        filename: "sharp_2572gikvuh.pt",
        checkpointBytes: 2_809_738_232,
        checkpointSHA256: "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d",
        requiredFreeBytes: 6_156_347_648,
        downloadURL: URL(
            string: "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
        )!,
        licenseKind: .appleMachineLearningResearchModel
    )
}

struct SharpModelDirectories: Sendable, Equatable {
    let root: URL
    let releaseRoot: URL
    let installation: URL
    let checkpoint: URL
    let license: URL
    let acceptance: URL
    let provenance: URL
    let partial: URL
    let resumeData: URL

    init(root: URL, release: SharpModelRelease) {
        let root = root.standardizedFileURL
        let releaseRoot = root
            .appending(path: "Apple-SHARP", directoryHint: .isDirectory)
            .appending(path: release.identifier, directoryHint: .isDirectory)
        let installation = releaseRoot.appending(path: "verified", directoryHint: .isDirectory)
        self.root = root
        self.releaseRoot = releaseRoot
        self.installation = installation
        checkpoint = installation.appending(path: release.filename)
        license = installation.appending(path: "LICENSE_MODEL.txt")
        acceptance = installation.appending(path: "license-acceptance.json")
        provenance = installation.appending(path: "model-provenance.json")
        partial = releaseRoot.appending(path: ".checkpoint.download.partial")
        resumeData = releaseRoot.appending(path: "download.resume")
    }

    static func live(
        release: SharpModelRelease = .v1,
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

enum SharpModelLicenseAgreement {
    static let sha256 = "e177a921ef25a523c429c6916e819c769bb4942619aadd16afe0108105a2baaf"

    static func load(bundle: Bundle = .main) throws -> String {
        let candidates = [
            bundle.url(
                forResource: "LICENSE_MODEL",
                withExtension: "txt",
                subdirectory: "Sharp"
            ),
            bundle.url(forResource: "LICENSE_MODEL", withExtension: "txt"),
        ]
        guard let url = candidates.compactMap({ $0 }).first else {
            throw SharpModelInstallerError.invalidInstallation
        }
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == sha256, let text = String(data: data, encoding: .utf8) else {
            throw SharpModelInstallerError.invalidInstallation
        }
        return text
    }
}

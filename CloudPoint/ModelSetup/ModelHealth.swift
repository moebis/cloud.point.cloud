import CryptoKit
import Darwin
import Foundation

struct ModelInstallation: Sendable, Equatable {
    let directory: URL
    let sourceRevision: String
    let sourceSHA256: String
    let convertedSHA256: String
    let engineVersion: String
}

enum ModelHealthFailure: Error, Sendable, Equatable {
    case missingConvertedArtifact(String)
    case manifestMismatch(String)
    case operationFailed(String)
}

enum ModelHealth: Sendable, Equatable {
    case absent
    case preparing(ModelSetupEvent)
    case ready(ModelInstallation)
    case invalid(ModelHealthFailure)
}

enum SecureModelFileError: Error, Sendable, Equatable {
    case unavailable
    case notRegular
    case tooLarge
    case sizeMismatch
}

private struct ConvertedModelManifest: Decodable {
    let schemaVersion: Int
    let modelIdentifier: String
    let modelRevision: String
    let sourceSHA256: String
    let convertedSha256: String
    let tensorCount: Int
    let mlxVersion: String
    let engineVersion: String
    let sourceCommit: String
    let modelFilename: String
    let modelSize: Int64
    let conversionUTC: String
}

enum ModelHealthInspector {
    static let convertedFilename = "lingbot-map-long-f16.safetensors"
    static let weightsManifestFilename = "weights-manifest.json"
    static let modelManifestFilename = "model-manifest.json"
    static let maximumManifestBytes = 16 * 1_048_576

    static func inspect(
        release: LingbotModelRelease,
        directory: URL
    ) throws -> ModelInstallation? {
        let modelManifestURL = directory.appending(path: modelManifestFilename)
        let weightsManifestURL = directory.appending(path: weightsManifestFilename)
        let convertedURL = directory.appending(path: convertedFilename)
        let fileManager = FileManager.default
        let expected = [modelManifestURL, weightsManifestURL, convertedURL]
        let existingCount = expected.reduce(into: 0) { count, url in
            if fileManager.fileExists(atPath: url.path) { count += 1 }
        }
        if existingCount == 0 { return nil }
        for url in expected where !fileManager.fileExists(atPath: url.path) {
            throw ModelHealthFailure.missingConvertedArtifact(url.lastPathComponent)
        }

        let manifestData = try readRegularFile(
            modelManifestURL,
            maximumBytes: maximumManifestBytes
        )
        let weightsManifestData = try readRegularFile(
            weightsManifestURL,
            maximumBytes: maximumManifestBytes
        )
        let manifest: ConvertedModelManifest
        do {
            let object = try JSONSerialization.jsonObject(with: manifestData)
            guard let dictionary = object as? [String: Any],
                  Set(dictionary.keys) == [
                      "schemaVersion", "modelIdentifier", "modelRevision",
                      "sourceSHA256", "convertedSha256", "tensorCount",
                      "mlxVersion", "engineVersion", "sourceCommit",
                      "modelFilename", "modelSize", "conversionUTC",
                  ] else {
                throw ModelHealthFailure.manifestMismatch(
                    "model-manifest.json fields are not exact"
                )
            }
            manifest = try JSONDecoder().decode(ConvertedModelManifest.self, from: manifestData)
        }
        catch { throw ModelHealthFailure.manifestMismatch("model-manifest.json is invalid") }

        let weights: [Any]
        do {
            guard let decoded = try JSONSerialization.jsonObject(
                with: weightsManifestData
            ) as? [Any] else {
                throw ModelHealthFailure.manifestMismatch("weights-manifest.json is invalid")
            }
            weights = decoded
        } catch {
            throw ModelHealthFailure.manifestMismatch("weights-manifest.json is invalid")
        }

        guard manifest.schemaVersion == 1,
              manifest.modelIdentifier == release.repository,
              manifest.modelRevision == release.revision,
              manifest.sourceSHA256 == release.sourceSHA256,
              manifest.sourceCommit == release.sourceCommit,
              manifest.modelFilename == release.filename,
              manifest.modelSize == release.sourceBytes,
              manifest.tensorCount == release.tensorCount,
              weights.count == release.tensorCount,
              manifest.mlxVersion == release.mlxVersion,
              manifest.engineVersion == release.engineVersion,
              !manifest.conversionUTC.isEmpty,
              isLowercaseSHA256(manifest.convertedSha256),
              manifest.convertedSha256 == release.convertedSHA256 else {
            throw ModelHealthFailure.manifestMismatch("pinned model provenance does not match")
        }
        let actualDigest: String
        do {
            actualDigest = try sha256(
                of: convertedURL,
                expectedBytes: release.convertedBytes
            )
        }
        catch { throw ModelHealthFailure.manifestMismatch("converted weights are unavailable") }
        guard actualDigest == manifest.convertedSha256 else {
            throw ModelHealthFailure.manifestMismatch("converted weights checksum does not match")
        }
        return ModelInstallation(
            directory: directory.standardizedFileURL,
            sourceRevision: manifest.modelRevision,
            sourceSHA256: manifest.sourceSHA256,
            convertedSHA256: manifest.convertedSha256,
            engineVersion: manifest.engineVersion
        )
    }

    static func verifySource(
        _ url: URL,
        release: LingbotModelRelease,
        progress: @Sendable (Int64) -> Void = { _ in }
    ) throws {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw SecureModelFileError.unavailable }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG else {
            throw SecureModelFileError.notRegular
        }
        guard status.st_size == release.sourceBytes else {
            throw SecureModelFileError.sizeMismatch
        }
        var hasher = SHA256()
        var readBytes: Int64 = 0
        while true {
            let data = try handle.read(upToCount: 8 * 1_048_576) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
            readBytes += Int64(data.count)
            progress(readBytes)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == release.sourceSHA256 else {
            throw SecureModelFileError.sizeMismatch
        }
    }

    static func sha256(of url: URL, expectedBytes: Int64) throws -> String {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw SecureModelFileError.unavailable }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG else {
            throw SecureModelFileError.notRegular
        }
        guard status.st_size == expectedBytes else {
            throw SecureModelFileError.sizeMismatch
        }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 8 * 1_048_576) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func readRegularFile(_ url: URL, maximumBytes: Int) throws -> Data {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw SecureModelFileError.unavailable }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG else {
            throw SecureModelFileError.notRegular
        }
        guard status.st_size >= 0, status.st_size <= maximumBytes else {
            throw SecureModelFileError.tooLarge
        }
        return try handle.readToEnd() ?? Data()
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { $0.isNumber || ("a"..."f").contains(String($0)) }
    }
}

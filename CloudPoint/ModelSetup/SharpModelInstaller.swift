import CryptoKit
import Darwin
import Foundation

struct SharpModelInstallation: Sendable, Equatable {
    let directory: URL
    let checkpoint: URL
    let checkpointSHA256: String
    let sourceCommit: String
}

enum SharpModelSetupEvent: Sendable, Equatable {
    case downloading(received: Int64, expected: Int64)
    case verifying(bytesRead: Int64, expected: Int64)
    case publishing
    case ready(SharpModelInstallation)
}

enum SharpModelHealth: Sendable, Equatable {
    case absent
    case preparing(SharpModelSetupEvent)
    case ready(SharpModelInstallation)
    case invalid
}

enum SharpModelInstallerError: Error, LocalizedError, Sendable, Equatable {
    case alreadyPreparing
    case licenseAcceptanceRequired
    case insufficientDiskSpace(required: Int64, available: Int64)
    case checkpointMismatch
    case invalidInstallation
    case unsafeInstallPath

    var errorDescription: String? {
        switch self {
        case .alreadyPreparing: "SHARP model setup is already running."
        case .licenseAcceptanceRequired:
            "Accept Apple's research-model license before downloading SHARP."
        case let .insufficientDiskSpace(required, available):
            "SHARP setup needs \(Self.gib(required)) GiB free; \(Self.gib(available)) GiB is available."
        case .checkpointMismatch:
            "The SHARP checkpoint did not match the pinned Apple release."
        case .invalidInstallation:
            "The installed SHARP model or its provenance is invalid."
        case .unsafeInstallPath: "The SHARP model location is unsafe."
        }
    }

    private static func gib(_ bytes: Int64) -> String {
        (Double(bytes) / 1_073_741_824).formatted(.number.precision(.fractionLength(1)))
    }
}

struct SharpModelAcceptance: Codable, Sendable, Equatable {
    let licenseKind: SharpModelLicenseKind
    let licenseSHA256: String
    let acceptedAt: Date
}

struct SharpModelProvenance: Codable, Sendable, Equatable {
    let schemaVersion: Int
    let identifier: String
    let sourceCommit: String
    let checkpointFilename: String
    let checkpointBytes: Int64
    let checkpointSHA256: String
    let downloadURL: URL
    let licenseKind: SharpModelLicenseKind
    let licenseSHA256: String
    let acceptedAt: Date
}

protocol SharpModelInstalling: Sendable {
    func health() async -> SharpModelHealth
    func prepare(acceptingLicense: Bool) async -> AsyncThrowingStream<SharpModelSetupEvent, Error>
    func cancel() async
}

actor SharpModelInstaller: SharpModelInstalling {
    let release: SharpModelRelease
    let directories: SharpModelDirectories

    private let downloader: any ModelDownloading
    private let fileManager: FileManager
    private let now: @Sendable () -> Date
    private let availableCapacity: (@Sendable () throws -> Int64)?
    private let licenseText: String
    private var operation: Task<Void, Never>?
    private var cachedHealth: SharpModelHealth?

    init(
        release: SharpModelRelease = .v1,
        directories: SharpModelDirectories,
        downloader: any ModelDownloading,
        fileManager: FileManager = .default,
        licenseText: String = "Fixture Apple Machine Learning Research Model License Agreement",
        now: @escaping @Sendable () -> Date = Date.init,
        availableCapacity: (@Sendable () throws -> Int64)? = nil
    ) {
        self.release = release
        self.directories = directories
        self.downloader = downloader
        self.fileManager = fileManager
        self.licenseText = licenseText
        self.now = now
        self.availableCapacity = availableCapacity
    }

    func health() -> SharpModelHealth {
        if let cachedHealth { return cachedHealth }
        let result: SharpModelHealth
        do {
            if !fileManager.fileExists(atPath: directories.installation.path) {
                result = .absent
            } else {
                result = .ready(try inspectInstallation())
            }
        } catch {
            result = .invalid
        }
        cachedHealth = result
        return result
    }

    func prepare(
        acceptingLicense: Bool
    ) -> AsyncThrowingStream<SharpModelSetupEvent, Error> {
        guard operation == nil else {
            return AsyncThrowingStream {
                $0.finish(throwing: SharpModelInstallerError.alreadyPreparing)
            }
        }
        let pair = AsyncThrowingStream.makeStream(
            of: SharpModelSetupEvent.self,
            bufferingPolicy: .bufferingNewest(32)
        )
        operation = Task { [weak self] in
            guard let self else { return }
            await self.runPreparation(
                acceptingLicense: acceptingLicense,
                continuation: pair.continuation
            )
        }
        pair.continuation.onTermination = { [weak self] termination in
            guard case .cancelled = termination else { return }
            Task { await self?.cancel() }
        }
        return pair.stream
    }

    func cancel() async {
        operation?.cancel()
        if let resume = await downloader.cancel(), !resume.isEmpty {
            try? persistResumeData(resume)
        }
    }

    private func runPreparation(
        acceptingLicense: Bool,
        continuation: AsyncThrowingStream<SharpModelSetupEvent, Error>.Continuation
    ) async {
        defer { operation = nil }
        do {
            try validateInstallRoot()
            cachedHealth = nil
            if case let .ready(installation) = health() {
                continuation.yield(.ready(installation))
                continuation.finish()
                return
            }
            guard acceptingLicense else {
                throw SharpModelInstallerError.licenseAcceptanceRequired
            }
            try fileManager.createDirectory(at: directories.root, withIntermediateDirectories: true)
            try validateInstallRoot()
            let capacity = try measuredAvailableCapacity()
            guard capacity >= release.requiredFreeBytes else {
                throw SharpModelInstallerError.insufficientDiskSpace(
                    required: release.requiredFreeBytes,
                    available: capacity
                )
            }
            try fileManager.createDirectory(at: directories.releaseRoot, withIntermediateDirectories: true)
            try validateInstallRoot()
            let downloaded = try await obtainCheckpoint(continuation)
            try Task.checkCancellation()
            continuation.yield(.publishing)
            let installation = try publish(downloadedCheckpoint: downloaded)
            cachedHealth = .ready(installation)
            continuation.yield(.ready(installation))
            continuation.finish()
        } catch {
            cachedHealth = nil
            try? removeGeneratedItemIfPresent(directories.partial)
            try? removeStagingDirectories()
            if Self.isCancellation(error) {
                continuation.finish(throwing: CancellationError())
            } else {
                continuation.finish(throwing: error)
            }
        }
    }

    private func obtainCheckpoint(
        _ continuation: AsyncThrowingStream<SharpModelSetupEvent, Error>.Continuation
    ) async throws -> URL {
        let resume = try readResumeDataIfPresent()
        var request = URLRequest(url: release.downloadURL)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 86_400
        let report: @Sendable (Int64, Int64) -> Void = { received, expected in
            continuation.yield(.downloading(received: received, expected: expected))
        }
        let temporary: URL
        do {
            temporary = try await downloader.download(
                request: request,
                resumeData: resume,
                progress: report
            )
        } catch {
            if Self.isCancellation(error) { throw CancellationError() }
            guard resume != nil else { throw error }
            try removeGeneratedItemIfPresent(directories.resumeData)
            temporary = try await downloader.download(
                request: request,
                resumeData: nil,
                progress: report
            )
        }
        defer { removeOwnedDownloadTemporaryIfPresent(temporary) }
        try Task.checkCancellation()
        try removeGeneratedItemIfPresent(directories.resumeData)
        try removeGeneratedItemIfPresent(directories.partial)
        do {
            try fileManager.moveItem(at: temporary, to: directories.partial)
        } catch {
            try fileManager.copyItem(at: temporary, to: directories.partial)
        }
        do {
            let digest = try secureSHA256(
                directories.partial,
                expectedBytes: release.checkpointBytes
            ) { bytes in
                continuation.yield(.verifying(
                    bytesRead: bytes,
                    expected: release.checkpointBytes
                ))
            }
            guard digest == release.checkpointSHA256 else {
                throw SharpModelInstallerError.checkpointMismatch
            }
        } catch let error as SharpModelInstallerError {
            throw error
        } catch {
            throw SharpModelInstallerError.checkpointMismatch
        }
        return directories.partial
    }

    private func publish(downloadedCheckpoint: URL) throws -> SharpModelInstallation {
        let staging = directories.releaseRoot.appending(
            path: ".verified.partial.\(UUID().uuidString.lowercased())",
            directoryHint: .isDirectory
        )
        let backup = directories.releaseRoot.appending(
            path: ".verified.backup.\(UUID().uuidString.lowercased())",
            directoryHint: .isDirectory
        )
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: false)
        let stagedCheckpoint = staging.appending(path: release.filename)
        try fileManager.moveItem(at: downloadedCheckpoint, to: stagedCheckpoint)

        let licenseData = Data(licenseText.utf8)
        let licenseDigest = Self.sha256(licenseData)
        let acceptedAt = now()
        let acceptance = SharpModelAcceptance(
            licenseKind: release.licenseKind,
            licenseSHA256: licenseDigest,
            acceptedAt: acceptedAt
        )
        let provenance = SharpModelProvenance(
            schemaVersion: 1,
            identifier: release.identifier,
            sourceCommit: release.sourceCommit,
            checkpointFilename: release.filename,
            checkpointBytes: release.checkpointBytes,
            checkpointSHA256: release.checkpointSHA256,
            downloadURL: release.downloadURL,
            licenseKind: release.licenseKind,
            licenseSHA256: licenseDigest,
            acceptedAt: acceptedAt
        )
        try licenseData.write(to: staging.appending(path: "LICENSE_MODEL.txt"), options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(acceptance).write(
            to: staging.appending(path: "license-acceptance.json"),
            options: .atomic
        )
        try encoder.encode(provenance).write(
            to: staging.appending(path: "model-provenance.json"),
            options: .atomic
        )

        let hadExisting = fileManager.fileExists(atPath: directories.installation.path)
        if hadExisting { try fileManager.moveItem(at: directories.installation, to: backup) }
        do {
            try fileManager.moveItem(at: staging, to: directories.installation)
            if hadExisting { try? fileManager.removeItem(at: backup) }
            try? removeGeneratedItemIfPresent(directories.resumeData)
        } catch {
            if hadExisting,
               !fileManager.fileExists(atPath: directories.installation.path),
               fileManager.fileExists(atPath: backup.path) {
                try? fileManager.moveItem(at: backup, to: directories.installation)
            }
            throw error
        }
        return try inspectInstallation()
    }

    private func inspectInstallation() throws -> SharpModelInstallation {
        try validateInstallRoot()
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let acceptance = try decoder.decode(
            SharpModelAcceptance.self,
            from: try readRegularFile(directories.acceptance, maximumBytes: 64 * 1_024)
        )
        let provenance = try decoder.decode(
            SharpModelProvenance.self,
            from: try readRegularFile(directories.provenance, maximumBytes: 64 * 1_024)
        )
        let licenseData = try readRegularFile(directories.license, maximumBytes: 256 * 1_024)
        let licenseDigest = Self.sha256(licenseData)
        guard acceptance.licenseKind == release.licenseKind,
              acceptance.licenseSHA256 == licenseDigest,
              provenance.schemaVersion == 1,
              provenance.identifier == release.identifier,
              provenance.sourceCommit == release.sourceCommit,
              provenance.checkpointFilename == release.filename,
              provenance.checkpointBytes == release.checkpointBytes,
              provenance.checkpointSHA256 == release.checkpointSHA256,
              provenance.downloadURL == release.downloadURL,
              provenance.licenseKind == release.licenseKind,
              provenance.licenseSHA256 == licenseDigest,
              provenance.acceptedAt == acceptance.acceptedAt else {
            throw SharpModelInstallerError.invalidInstallation
        }
        let digest = try secureSHA256(
            directories.checkpoint,
            expectedBytes: release.checkpointBytes
        )
        guard digest == release.checkpointSHA256 else {
            throw SharpModelInstallerError.checkpointMismatch
        }
        return SharpModelInstallation(
            directory: directories.installation,
            checkpoint: directories.checkpoint,
            checkpointSHA256: digest,
            sourceCommit: release.sourceCommit
        )
    }

    private func measuredAvailableCapacity() throws -> Int64 {
        if let availableCapacity { return try availableCapacity() }
        let values = try directories.root.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        )
        return values.volumeAvailableCapacityForImportantUsage ?? 0
    }

    private func secureSHA256(
        _ url: URL,
        expectedBytes: Int64,
        progress: @Sendable (Int64) -> Void = { _ in }
    ) throws -> String {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw SharpModelInstallerError.invalidInstallation }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG,
              status.st_size == expectedBytes else {
            throw SharpModelInstallerError.checkpointMismatch
        }
        var hasher = SHA256()
        var read: Int64 = 0
        while true {
            let data = try handle.read(upToCount: 8 * 1_048_576) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
            read += Int64(data.count)
            progress(read)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func persistResumeData(_ data: Data) throws {
        try validateInstallRoot()
        try fileManager.createDirectory(at: directories.releaseRoot, withIntermediateDirectories: true)
        try validateNoSymlinkAncestors(of: directories.resumeData)
        try data.write(to: directories.resumeData, options: .atomic)
    }

    private func readResumeDataIfPresent() throws -> Data? {
        guard fileManager.fileExists(atPath: directories.resumeData.path) else { return nil }
        return try readRegularFile(directories.resumeData, maximumBytes: 64 * 1_048_576)
    }

    private func readRegularFile(_ url: URL, maximumBytes: Int) throws -> Data {
        try validateNoSymlinkAncestors(of: url)
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw SharpModelInstallerError.invalidInstallation }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG,
              status.st_size >= 0,
              status.st_size <= maximumBytes else {
            throw SharpModelInstallerError.invalidInstallation
        }
        return try handle.readToEnd() ?? Data()
    }

    private func validateInstallRoot() throws {
        let root = directories.root.standardizedFileURL.path
        let release = directories.releaseRoot.standardizedFileURL.path
        guard directories.root.isFileURL,
              root.hasPrefix("/"),
              release.hasPrefix(root + "/") else {
            throw SharpModelInstallerError.unsafeInstallPath
        }
        try validateNoSymlinkAncestors(of: directories.root)
        try validateNoSymlinkAncestors(of: directories.releaseRoot)
    }

    private func validateNoSymlinkAncestors(of url: URL) throws {
        guard url.isFileURL, url.path.hasPrefix("/") else {
            throw SharpModelInstallerError.unsafeInstallPath
        }
        var current = URL(filePath: "/", directoryHint: .isDirectory)
        for component in url.standardizedFileURL.pathComponents.dropFirst() {
            current.append(path: component)
            var status = stat()
            if Darwin.lstat(current.path, &status) != 0 {
                if errno == ENOENT { break }
                throw SharpModelInstallerError.unsafeInstallPath
            }
            if status.st_mode & S_IFMT == S_IFLNK {
                throw SharpModelInstallerError.unsafeInstallPath
            }
        }
    }

    private func removeGeneratedItemIfPresent(_ url: URL) throws {
        try validateInstallRoot()
        guard url.standardizedFileURL.path.hasPrefix(
            directories.releaseRoot.standardizedFileURL.path + "/"
        ) else { throw SharpModelInstallerError.unsafeInstallPath }
        var status = stat()
        guard Darwin.lstat(url.path, &status) == 0 else {
            if errno == ENOENT { return }
            throw SharpModelInstallerError.unsafeInstallPath
        }
        guard status.st_mode & S_IFMT != S_IFLNK else {
            throw SharpModelInstallerError.unsafeInstallPath
        }
        try fileManager.removeItem(at: url)
    }

    private func removeStagingDirectories() throws {
        guard fileManager.fileExists(atPath: directories.releaseRoot.path) else { return }
        for url in try fileManager.contentsOfDirectory(
            at: directories.releaseRoot,
            includingPropertiesForKeys: nil
        ) where url.lastPathComponent.hasPrefix(".verified.partial.") {
            try removeGeneratedItemIfPresent(url)
        }
    }

    private func removeOwnedDownloadTemporaryIfPresent(_ url: URL) {
        let standardized = url.standardizedFileURL
        let temporary = fileManager.temporaryDirectory.standardizedFileURL
        guard standardized.deletingLastPathComponent() == temporary,
              standardized.lastPathComponent.hasPrefix("cloudpoint-model-download-") else {
            return
        }
        try? fileManager.removeItem(at: standardized)
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? URLError)?.code == .cancelled
            || (error as NSError).domain == NSURLErrorDomain
                && (error as NSError).code == URLError.cancelled.rawValue
    }
}

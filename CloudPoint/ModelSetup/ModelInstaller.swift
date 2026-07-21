import Darwin
import Foundation

enum ModelConversionPhase: String, Sendable, Equatable {
    case verifying
    case restrictedLoading
    case trustedArtifactLoading
    case converting
    case validating
    case ready
}

enum ModelSetupEvent: Sendable, Equatable {
    case downloading(received: Int64, expected: Int64)
    case verifying(bytesRead: Int64, expected: Int64)
    case converting(ModelConversionPhase)
    case validating
    case ready(ModelInstallation)
}

enum ModelInstallerError: Error, Sendable, Equatable {
    case alreadyPreparing
    case sourceChecksumMismatch
    case convertedArtifactsInvalid
    case converterFailed(String)
    case unsafeInstallPath
}

protocol ModelDownloading: Sendable {
    func download(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (_ received: Int64, _ expected: Int64) -> Void
    ) async throws -> URL
    func cancel() async -> Data?
}

protocol ModelConverting: Sendable {
    func convert(
        checkpoint: URL,
        destination: URL,
        progress: @escaping @Sendable (ModelConversionPhase) -> Void
    ) async throws
}

protocol ModelInstalling: Sendable {
    func health() async -> ModelHealth
    func prepare() async -> AsyncThrowingStream<ModelSetupEvent, Error>
    func cancel() async
}

actor ModelInstaller: ModelInstalling {
    let release: LingbotModelRelease
    let directories: ModelDirectories
    private let downloader: any ModelDownloading
    private let converter: any ModelConverting
    private let fileManager: FileManager
    private var operation: Task<Void, Never>?
    private var cachedHealth: ModelHealth?

    init(
        release: LingbotModelRelease = .v1,
        directories: ModelDirectories,
        downloader: any ModelDownloading,
        converter: any ModelConverting,
        fileManager: FileManager = .default
    ) {
        self.release = release
        self.directories = directories
        self.downloader = downloader
        self.converter = converter
        self.fileManager = fileManager
    }

    func health() -> ModelHealth {
        if let cachedHealth { return cachedHealth }
        let result: ModelHealth
        do {
            guard let installation = try ModelHealthInspector.inspect(
                release: release,
                directory: directories.converted
            ) else {
                cachedHealth = .absent
                return .absent
            }
            result = .ready(installation)
        } catch let failure as ModelHealthFailure {
            result = .invalid(failure)
        } catch {
            result = .invalid(.operationFailed(String(describing: error)))
        }
        cachedHealth = result
        return result
    }

    func prepare() -> AsyncThrowingStream<ModelSetupEvent, Error> {
        guard operation == nil else {
            return AsyncThrowingStream { $0.finish(throwing: ModelInstallerError.alreadyPreparing) }
        }
        let pair = AsyncThrowingStream.makeStream(
            of: ModelSetupEvent.self,
            bufferingPolicy: .bufferingNewest(32)
        )
        operation = Task { [weak self] in
            guard let self else { return }
            await self.runPreparation(pair.continuation)
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
        _ continuation: AsyncThrowingStream<ModelSetupEvent, Error>.Continuation
    ) async {
        defer { operation = nil }
        var installRootValidated = false
        do {
            cachedHealth = nil
            try validateInstallRoot()
            installRootValidated = true
            if case let .ready(installation) = health() {
                continuation.yield(.ready(installation))
                continuation.finish()
                return
            }
            try fileManager.createDirectory(
                at: directories.releaseRoot,
                withIntermediateDirectories: true
            )
            try validateInstallRoot()
            let checkpoint = try await obtainCheckpoint(continuation)
            try Task.checkCancellation()
            try removeGeneratedItemIfPresent(directories.convertedPartial)
            try fileManager.createDirectory(
                at: directories.convertedPartial,
                withIntermediateDirectories: false
            )
            continuation.yield(.converting(.verifying))
            try await converter.convert(
                checkpoint: checkpoint,
                destination: directories.convertedPartial
            ) { phase in
                continuation.yield(.converting(phase))
            }
            try Task.checkCancellation()
            continuation.yield(.validating)
            guard let partialInstallation = try ModelHealthInspector.inspect(
                release: release,
                directory: directories.convertedPartial
            ) else { throw ModelInstallerError.convertedArtifactsInvalid }
            try publish(checkpoint: checkpoint)
            let installation = ModelInstallation(
                directory: directories.converted,
                sourceRevision: partialInstallation.sourceRevision,
                sourceSHA256: partialInstallation.sourceSHA256,
                convertedSHA256: partialInstallation.convertedSHA256,
                engineVersion: partialInstallation.engineVersion
            )
            cachedHealth = .ready(installation)
            continuation.yield(.ready(installation))
            continuation.finish()
        } catch {
            if installRootValidated { cleanupGeneratedPartials() }
            cachedHealth = nil
            if Self.isCancellation(error) {
                continuation.finish(throwing: CancellationError())
                return
            }
            if error is SecureModelFileError {
                continuation.finish(throwing: ModelInstallerError.sourceChecksumMismatch)
                return
            }
            continuation.finish(throwing: error)
        }
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? URLError)?.code == .cancelled
            || (error as NSError).domain == NSURLErrorDomain
                && (error as NSError).code == URLError.cancelled.rawValue
    }

    private func cleanupGeneratedPartials() {
        try? removeGeneratedItemIfPresent(directories.sourcePartial)
        try? removeGeneratedItemIfPresent(directories.convertedPartial)
    }

    private func obtainCheckpoint(
        _ continuation: AsyncThrowingStream<ModelSetupEvent, Error>.Continuation
    ) async throws -> URL {
        if fileManager.fileExists(atPath: directories.sourceCheckpoint.path) {
            do {
                try ModelHealthInspector.verifySource(
                    directories.sourceCheckpoint,
                    release: release
                ) { bytes in
                    continuation.yield(.verifying(bytesRead: bytes, expected: release.sourceBytes))
                }
                return directories.sourceCheckpoint
            } catch {
                try removeGeneratedItemIfPresent(directories.sourceCheckpoint)
            }
        }

        let resume = try readResumeDataIfPresent()
        var request = URLRequest(url: release.downloadURL)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 86_400
        let progress: @Sendable (Int64, Int64) -> Void = { received, expected in
            continuation.yield(.downloading(received: received, expected: expected))
        }
        let temporary: URL
        do {
            temporary = try await downloader.download(
                request: request,
                resumeData: resume,
                progress: progress
            )
        } catch {
            if Self.isCancellation(error) { throw CancellationError() }
            guard resume != nil else { throw error }
            try removeGeneratedItemIfPresent(directories.resumeData)
            try Task.checkCancellation()
            temporary = try await downloader.download(
                request: request,
                resumeData: nil,
                progress: progress
            )
        }
        defer { removeOwnedDownloadTemporaryIfPresent(temporary) }
        try Task.checkCancellation()
        try removeGeneratedItemIfPresent(directories.resumeData)
        try removeGeneratedItemIfPresent(directories.sourcePartial)
        do { try fileManager.moveItem(at: temporary, to: directories.sourcePartial) }
        catch {
            try fileManager.copyItem(at: temporary, to: directories.sourcePartial)
            try? fileManager.removeItem(at: temporary)
        }
        do {
            try ModelHealthInspector.verifySource(
                directories.sourcePartial,
                release: release
            ) { bytes in
                continuation.yield(.verifying(bytesRead: bytes, expected: release.sourceBytes))
            }
        } catch {
            try? removeGeneratedItemIfPresent(directories.sourcePartial)
            throw ModelInstallerError.sourceChecksumMismatch
        }
        try fileManager.createDirectory(at: directories.source, withIntermediateDirectories: true)
        try validateInstallRoot()
        try removeGeneratedItemIfPresent(directories.sourceCheckpoint)
        try fileManager.moveItem(at: directories.sourcePartial, to: directories.sourceCheckpoint)
        return directories.sourceCheckpoint
    }

    private func publish(checkpoint: URL) throws {
        _ = checkpoint
        let backup = directories.releaseRoot.appending(
            path: ".converted.backup.\(UUID().uuidString.lowercased())",
            directoryHint: .isDirectory
        )
        let hadExisting = fileManager.fileExists(atPath: directories.converted.path)
        if hadExisting {
            try validateNoSymlinkAncestors(of: directories.converted)
            try fileManager.moveItem(at: directories.converted, to: backup)
        }
        do {
            try validateNoSymlinkAncestors(of: directories.convertedPartial)
            try fileManager.moveItem(at: directories.convertedPartial, to: directories.converted)
            if hadExisting { try? removeGeneratedItemIfPresent(backup) }
            try? removeGeneratedItemIfPresent(directories.resumeData)
        } catch {
            if hadExisting,
               !fileManager.fileExists(atPath: directories.converted.path),
               fileManager.fileExists(atPath: backup.path) {
                try? fileManager.moveItem(at: backup, to: directories.converted)
            }
            throw error
        }
    }

    private func persistResumeData(_ data: Data) throws {
        try validateInstallRoot()
        try fileManager.createDirectory(
            at: directories.releaseRoot,
            withIntermediateDirectories: true
        )
        try validateInstallRoot()
        try validateNoSymlinkAncestors(of: directories.resumeData)
        try data.write(to: directories.resumeData, options: .atomic)
    }

    private func validateInstallRoot() throws {
        let root = directories.root.standardizedFileURL.path
        let releasePath = directories.releaseRoot.standardizedFileURL.path
        guard directories.root.isFileURL,
              root.hasPrefix("/"),
              releasePath.hasPrefix(root + "/") else {
            throw ModelInstallerError.unsafeInstallPath
        }
        try validateNoSymlinkAncestors(of: directories.root)
        try validateNoSymlinkAncestors(of: directories.releaseRoot)
    }

    private func removeGeneratedItemIfPresent(_ url: URL) throws {
        try validateInstallRoot()
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(directories.releaseRoot.standardizedFileURL.path + "/") else {
            throw ModelInstallerError.unsafeInstallPath
        }
        try validateNoSymlinkAncestors(of: url)
        var status = stat()
        guard Darwin.lstat(path, &status) == 0 else {
            if errno == ENOENT { return }
            throw ModelInstallerError.unsafeInstallPath
        }
        guard status.st_mode & S_IFMT != S_IFLNK else {
            throw ModelInstallerError.unsafeInstallPath
        }
        try fileManager.removeItem(at: url)
    }

    private func readResumeDataIfPresent() throws -> Data? {
        let path = directories.resumeData.path
        var status = stat()
        guard Darwin.lstat(path, &status) == 0 else {
            if errno == ENOENT { return nil }
            throw ModelInstallerError.unsafeInstallPath
        }
        try validateInstallRoot()
        try validateNoSymlinkAncestors(of: directories.resumeData)
        guard status.st_mode & S_IFMT == S_IFREG else {
            throw ModelInstallerError.unsafeInstallPath
        }
        let maximumResumeBytes: Int64 = 64 * 1_048_576
        guard status.st_size > 0, status.st_size <= maximumResumeBytes else {
            try removeGeneratedItemIfPresent(directories.resumeData)
            return nil
        }
        let descriptor = Darwin.open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw ModelInstallerError.unsafeInstallPath }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        return try handle.readToEnd()
    }

    private func validateNoSymlinkAncestors(of url: URL) throws {
        guard url.isFileURL, url.path.hasPrefix("/") else {
            throw ModelInstallerError.unsafeInstallPath
        }
        var current = URL(filePath: "/", directoryHint: .isDirectory)
        for component in url.standardizedFileURL.pathComponents.dropFirst() {
            current.append(path: component)
            var status = stat()
            if Darwin.lstat(current.path, &status) != 0 {
                if errno == ENOENT { break }
                throw ModelInstallerError.unsafeInstallPath
            }
            if status.st_mode & S_IFMT == S_IFLNK {
                throw ModelInstallerError.unsafeInstallPath
            }
        }
    }

    private func removeOwnedDownloadTemporaryIfPresent(_ url: URL) {
        let standardized = url.standardizedFileURL
        let temporaryRoot = fileManager.temporaryDirectory.standardizedFileURL
        guard standardized.deletingLastPathComponent() == temporaryRoot,
              standardized.lastPathComponent.hasPrefix("cloudpoint-model-download-") else {
            return
        }
        var status = stat()
        guard Darwin.lstat(standardized.path, &status) == 0,
              status.st_mode & S_IFMT != S_IFLNK else { return }
        try? fileManager.removeItem(at: standardized)
    }
}

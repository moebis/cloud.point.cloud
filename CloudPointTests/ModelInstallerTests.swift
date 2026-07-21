import Darwin
import Foundation
import Security
import XCTest
@testable import CloudPoint

final class ModelInstallerTests: XCTestCase {
    func testAppSandboxAllowsPinnedModelDownload() throws {
        let task = try XCTUnwrap(SecTaskCreateFromSelf(nil))
        let sandboxed = SecTaskCopyValueForEntitlement(
            task,
            "com.apple.security.app-sandbox" as CFString,
            nil
        ) as? Bool
        guard sandboxed == true else {
            throw XCTSkip("Requires the signed CloudPoint app test host")
        }
        let value = SecTaskCopyValueForEntitlement(
            task,
            "com.apple.security.network.client" as CFString,
            nil
        )

        XCTAssertEqual(value as? Bool, true)
    }

    func testOfficialReleaseIsFullyPinned() throws {
        let release = LingbotModelRelease.v1

        XCTAssertEqual(release.repository, "robbyant/lingbot-map")
        XCTAssertEqual(
            release.revision,
            "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
        )
        XCTAssertEqual(release.sourceBytes, 4_632_303_465)
        XCTAssertEqual(
            release.sourceSHA256,
            "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
        )
        XCTAssertEqual(release.convertedBytes, 2_316_040_080)
        XCTAssertEqual(
            release.convertedSHA256,
            "eb966484923b5a205677b3ce7316d079c46fc6503bc9b6ac256b6e11560ea2e5"
        )
        XCTAssertEqual(
            release.downloadURL.absoluteString,
            "https://huggingface.co/robbyant/lingbot-map/resolve/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/lingbot-map-long.pt?download=true"
        )
    }

    func testBadDigestNeverRunsConverter() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: String(repeating: "0", count: 64)
        )
        let downloader = ModelInstallerFakeDownloader(data: Data("wrong".utf8))
        let converter = ModelInstallerFakeConverter()
        let installer = ModelInstaller(
            release: release,
            directories: ModelDirectories(root: temporary.url, release: release),
            downloader: downloader,
            converter: converter
        )

        let stream = await installer.prepare()
        do {
            for try await _ in stream {}
            XCTFail("Expected checksum validation to fail")
        } catch let error as ModelInstallerError {
            XCTAssertEqual(error, .sourceChecksumMismatch)
        }
        let badDigestConversionCount = await converter.invocationCount
        XCTAssertEqual(badDigestConversionCount, 0)
    }

    func testCompletedInstallRequiresEveryConvertedArtifact() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        try FileManager.default.createDirectory(
            at: directories.converted,
            withIntermediateDirectories: true
        )
        try Data("[]".utf8).write(to: directories.weightsManifest)
        try Data("{}".utf8).write(to: directories.modelManifest)
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: ModelInstallerFakeDownloader(data: Data()),
            converter: ModelInstallerFakeConverter()
        )

        let missingArtifactHealth = await installer.health()
        XCTAssertEqual(
            missingArtifactHealth,
            .invalid(.missingConvertedArtifact("lingbot-map-long-f16.safetensors"))
        )
    }

    func testValidPreparedArtifactsReportReadyInstallation() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        let converter = ModelInstallerFakeConverter()
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: ModelInstallerFakeDownloader(data: Data("hello".utf8)),
            converter: converter
        )

        let stream = await installer.prepare()
        var events: [ModelSetupEvent] = []
        for try await event in stream { events.append(event) }

        guard case let .ready(installation) = await installer.health() else {
            return XCTFail("Expected a ready installation")
        }
        XCTAssertEqual(installation.directory, directories.converted)
        let conversionCount = await converter.invocationCount
        XCTAssertEqual(conversionCount, 1)
        XCTAssertTrue(events.contains { if case .converting = $0 { true } else { false } })
        XCTAssertEqual(events.last, .ready(installation))
    }

    func testRepeatedHealthUsesLaunchLocalVerifiedResultWithoutRehashingWeights() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: ModelInstallerFakeDownloader(data: Data("hello".utf8)),
            converter: ModelInstallerFakeConverter()
        )
        let stream = await installer.prepare()
        for try await _ in stream {}
        let first = await installer.health()
        try FileManager.default.removeItem(at: directories.convertedWeights)

        let second = await installer.health()

        XCTAssertEqual(second, first)
    }

    func testPrepareInvalidatesCachedHealthBeforeRepairingArtifacts() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        let converter = ModelInstallerFakeConverter()
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: ModelInstallerFakeDownloader(data: Data("hello".utf8)),
            converter: converter
        )
        var stream = await installer.prepare()
        for try await _ in stream {}
        _ = await installer.health()
        try FileManager.default.removeItem(at: directories.convertedWeights)

        stream = await installer.prepare()
        for try await _ in stream {}

        let conversionCount = await converter.invocationCount
        XCTAssertEqual(conversionCount, 2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directories.convertedWeights.path))
    }

    func testRepeatedInvalidHealthUsesLaunchLocalInspectionResult() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        let builder = ModelInstaller(
            release: release,
            directories: directories,
            downloader: ModelInstallerFakeDownloader(data: Data("hello".utf8)),
            converter: ModelInstallerFakeConverter()
        )
        let stream = await builder.prepare()
        for try await _ in stream {}
        try FileManager.default.removeItem(at: directories.convertedWeights)
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: ModelInstallerFakeDownloader(data: Data()),
            converter: ModelInstallerFakeConverter()
        )
        let first = await installer.health()
        try Data("converted".utf8).write(to: directories.convertedWeights)

        let second = await installer.health()

        XCTAssertEqual(second, first)
        XCTAssertEqual(
            second,
            .invalid(.missingConvertedArtifact(ModelHealthInspector.convertedFilename))
        )
    }

    func testPoisonedResumeDataFallsBackOnceToFreshPinnedRequest() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        try FileManager.default.createDirectory(
            at: directories.releaseRoot,
            withIntermediateDirectories: true
        )
        let poison = Data("stale-resume-data".utf8)
        try poison.write(to: directories.resumeData)
        let downloader = SequencedModelDownloader([
            .failure(URLError(.cannotDecodeRawData)),
            .success(Data("hello".utf8)),
        ])
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: downloader,
            converter: ModelInstallerFakeConverter()
        )

        let stream = await installer.prepare()
        for try await _ in stream {}

        let attempts = await downloader.attempts
        XCTAssertEqual(attempts.map(\.resumeData), [poison, nil])
        XCTAssertEqual(attempts.map(\.url), [release.downloadURL, release.downloadURL])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directories.resumeData.path))
    }

    func testPoisonedResumeDataRetriesFreshOnlyOnceAndClearsBlob() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        try FileManager.default.createDirectory(
            at: directories.releaseRoot,
            withIntermediateDirectories: true
        )
        try Data("stale-resume-data".utf8).write(to: directories.resumeData)
        let downloader = SequencedModelDownloader([
            .failure(URLError(.cannotDecodeRawData)),
            .failure(URLError(.notConnectedToInternet)),
            .success(Data("must-not-be-used".utf8)),
        ])
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: downloader,
            converter: ModelInstallerFakeConverter()
        )

        let stream = await installer.prepare()
        do {
            for try await _ in stream {}
            XCTFail("Expected the fresh attempt to fail")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .notConnectedToInternet)
        }

        let attempts = await downloader.attempts
        XCTAssertEqual(attempts.count, 2)
        XCTAssertEqual(
            attempts.map { $0.resumeData?.count ?? 0 },
            ["stale-resume-data".utf8.count, 0]
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: directories.resumeData.path))
    }

    func testURLSessionCancellationIsReportedAsTaskCancellationAndCleansPartials() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        try FileManager.default.createDirectory(
            at: directories.convertedPartial,
            withIntermediateDirectories: true
        )
        try Data("old partial".utf8).write(to: directories.sourcePartial)
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: SequencedModelDownloader([.failure(URLError(.cancelled))]),
            converter: ModelInstallerFakeConverter()
        )

        let stream = await installer.prepare()
        do {
            for try await _ in stream {}
            XCTFail("Expected cancellation")
        } catch {
            XCTAssertTrue(error is CancellationError, "Unexpected error: \(error)")
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: directories.sourcePartial.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directories.convertedPartial.path))
    }

    func testConverterFailureRemovesGeneratedPartialsButKeepsVerifiedCheckpoint() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: temporary.url, release: release)
        let downloader = SequencedModelDownloader([.success(Data("hello".utf8))])
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: downloader,
            converter: FailingPartialModelConverter()
        )

        let stream = await installer.prepare()
        do {
            for try await _ in stream {}
            XCTFail("Expected conversion failure")
        } catch let error as ModelInstallerError {
            XCTAssertEqual(error, .converterFailed("fixture conversion failed"))
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: directories.sourcePartial.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directories.convertedPartial.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directories.sourceCheckpoint.path))
        let temporaryDownloads = await downloader.createdTemporaryFiles
        XCTAssertTrue(temporaryDownloads.allSatisfy {
            !FileManager.default.fileExists(atPath: $0.path)
        })
    }

    func testSymlinkedInstallAncestorFailsClosedWithoutDeletingOutsideBoundary() async throws {
        let temporary = try ModelInstallerTemporaryDirectory()
        let outside = temporary.url.appending(path: "outside", directoryHint: .isDirectory)
        let linkedRoot = temporary.url.appending(path: "linked-model-root", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: linkedRoot, withDestinationURL: outside)
        let release = LingbotModelRelease.fixture(
            sourceBytes: 5,
            sourceSHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        let directories = ModelDirectories(root: linkedRoot, release: release)
        try FileManager.default.createDirectory(
            at: directories.convertedPartial,
            withIntermediateDirectories: true
        )
        let outsideMarker = directories.convertedPartial.appending(path: "outside-marker")
        try Data("preserve".utf8).write(to: outsideMarker)
        let downloader = SequencedModelDownloader([.success(Data("hello".utf8))])
        let installer = ModelInstaller(
            release: release,
            directories: directories,
            downloader: downloader,
            converter: ModelInstallerFakeConverter()
        )

        let stream = await installer.prepare()
        do {
            for try await _ in stream {}
            XCTFail("Expected an unsafe install path failure")
        } catch let error as ModelInstallerError {
            XCTAssertEqual(error, .unsafeInstallPath)
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: outsideMarker.path))
        let attempts = await downloader.attempts
        XCTAssertTrue(attempts.isEmpty)
    }

    func testConverterStreamsProgressWhileConcurrentlyDrainingLargeDiagnostics() async throws {
        try skipIfSandboxedFixtureExecution()
        let fixture = try ProcessModelConverterFixture(
            script: """
            #!/bin/sh
            printf '{"phase":"restrictedLoading"}\\n'
            /usr/bin/yes 'bounded diagnostic output' | /usr/bin/head -c 2097152 >&2
            printf '{"phase":"trustedArtifactLoading"}\\n'
            """
        )
        let destination = fixture.root.appending(path: "converted", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let firstProgress = expectation(description: "progress streamed before process exit")
        let conversion = Task {
            try await ProcessModelConverter(runtime: fixture.runtime).convert(
                checkpoint: fixture.checkpoint,
                destination: destination
            ) { phase in
                if phase == .restrictedLoading { firstProgress.fulfill() }
            }
        }
        let watchdog = Task {
            try? await Task.sleep(for: .seconds(3))
            conversion.cancel()
        }

        await fulfillment(of: [firstProgress], timeout: 1)
        try await conversion.value
        watchdog.cancel()
    }

    func testConverterCancellationTerminatesEntireProcessGroupAfterGracePeriod() async throws {
        try skipIfSandboxedFixtureExecution()
        let fixture = try ProcessModelConverterFixture(
            script: """
            #!/bin/sh
            trap '' TERM
            (
              trap '' TERM
              while :; do /bin/sleep 1; done
            ) &
            child=$!
            printf '%s' "$$" > "$5/parent.pid"
            printf '%s' "$child" > "$5/child.pid"
            printf '{"phase":"converting"}\\n'
            while :; do /bin/sleep 1; done
            """
        )
        let destination = fixture.root.appending(path: "converted", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let conversion = Task {
            try await ProcessModelConverter(runtime: fixture.runtime).convert(
                checkpoint: fixture.checkpoint,
                destination: destination,
                progress: { _ in }
            )
        }
        let parentPIDURL = destination.appending(path: "parent.pid")
        let childPIDURL = destination.appending(path: "child.pid")
        let parentPID = try await readProcessID(at: parentPIDURL)
        let childPID = try await readProcessID(at: childPIDURL)
        let emergencyCleanup = Task {
            try? await Task.sleep(for: .seconds(4))
            Darwin.kill(parentPID, SIGKILL)
            Darwin.kill(childPID, SIGKILL)
        }
        defer {
            emergencyCleanup.cancel()
            Darwin.kill(parentPID, SIGKILL)
            Darwin.kill(childPID, SIGKILL)
        }

        XCTAssertEqual(Darwin.getpgid(parentPID), parentPID)
        XCTAssertEqual(Darwin.getpgid(childPID), parentPID)

        let completed = expectation(description: "cancelled converter exits after escalation")
        Task {
            _ = await conversion.result
            completed.fulfill()
        }
        conversion.cancel()
        await fulfillment(of: [completed], timeout: 3)
        switch await conversion.result {
        case .success: XCTFail("Expected cancellation")
        case let .failure(error): XCTAssertTrue(error is CancellationError)
        }
        let childExited = await processHasExited(childPID)
        XCTAssertTrue(childExited)
    }

    func testModelDownloadAllowlistAcceptsCurrentPinnedHostsOnlyOverHTTPS() {
        XCTAssertTrue(URLSessionModelDownloader.isAllowed(
            URL(string: "https://huggingface.co/robbyant/lingbot-map")
        ))
        XCTAssertTrue(URLSessionModelDownloader.isAllowed(
            URL(string: "https://us.aws.cdn.hf.co/xet-bridge-us/model")
        ))
        XCTAssertFalse(URLSessionModelDownloader.isAllowed(
            URL(string: "http://huggingface.co/robbyant/lingbot-map")
        ))
        XCTAssertFalse(URLSessionModelDownloader.isAllowed(
            URL(string: "https://huggingface.co.evil.example/model")
        ))
        XCTAssertFalse(URLSessionModelDownloader.isAllowed(
            URL(string: "https://example.com/model")
        ))
    }

    func testPreparedRealModelMatchesPinnedConvertedDigest() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["CLOUDPOINT_REAL_MODEL_DIR"],
              modelPath.hasPrefix("/"),
              FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("Set CLOUDPOINT_REAL_MODEL_DIR for the prepared-model health check")
        }
        let installation = try XCTUnwrap(ModelHealthInspector.inspect(
            release: .v1,
            directory: URL(filePath: modelPath, directoryHint: .isDirectory)
        ))
        XCTAssertEqual(installation.convertedSHA256, LingbotModelRelease.v1.convertedSHA256)
    }
}

private final class ModelInstallerTemporaryDirectory: @unchecked Sendable {
    let url: URL

    init() throws {
        url = modelTestScratchDirectory()
            .appending(
                path: "cloudpoint-model-tests-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: url) }
}

private actor ModelInstallerFakeDownloader: ModelDownloading {
    let data: Data

    init(data: Data) { self.data = data }

    func download(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> URL {
        _ = request
        _ = resumeData
        progress(Int64(data.count), Int64(data.count))
        let url = FileManager.default.temporaryDirectory
            .appending(path: "cloudpoint-download-\(UUID().uuidString)")
        try data.write(to: url, options: .withoutOverwriting)
        return url
    }

    func cancel() async -> Data? { nil }
}

private actor ModelInstallerFakeConverter: ModelConverting {
    private(set) var invocationCount = 0

    func convert(
        checkpoint: URL,
        destination: URL,
        progress: @escaping @Sendable (ModelConversionPhase) -> Void
    ) async throws {
        _ = checkpoint
        invocationCount += 1
        progress(.converting)
        let weights = Data("converted".utf8)
        try weights.write(
            to: destination.appending(path: "lingbot-map-long-f16.safetensors")
        )
        try Data("[{}]".utf8).write(
            to: destination.appending(path: "weights-manifest.json")
        )
        try Data(
            """
            {"schemaVersion":1,"modelIdentifier":"fixture/model","modelRevision":"fixture-revision","sourceSHA256":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824","convertedSha256":"ba451247dcf0d65bb50c654ae2ebfb3e3173ec730bd5174a4b9eef5b3dc7c6da","tensorCount":1,"mlxVersion":"fixture-mlx","engineVersion":"fixture","sourceCommit":"fixture-source","modelFilename":"fixture.pt","modelSize":5,"conversionUTC":"2026-07-21T00:00:00Z"}
            """.utf8
        ).write(to: destination.appending(path: "model-manifest.json"))
        progress(.validating)
    }
}

private actor SequencedModelDownloader: ModelDownloading {
    enum Step: @unchecked Sendable {
        case success(Data)
        case failure(Error)
    }

    struct Attempt: Sendable {
        let url: URL
        let resumeData: Data?
    }

    private var steps: [Step]
    private(set) var attempts: [Attempt] = []
    private(set) var createdTemporaryFiles: [URL] = []

    init(_ steps: [Step]) { self.steps = steps }

    func download(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> URL {
        attempts.append(Attempt(url: request.url!, resumeData: resumeData))
        guard !steps.isEmpty else { throw URLError(.unknown) }
        switch steps.removeFirst() {
        case let .failure(error): throw error
        case let .success(data):
            progress(Int64(data.count), Int64(data.count))
            let url = FileManager.default.temporaryDirectory
                .appending(path: "cloudpoint-model-download-\(UUID().uuidString)")
            try data.write(to: url, options: .withoutOverwriting)
            createdTemporaryFiles.append(url)
            return url
        }
    }

    func cancel() async -> Data? { nil }
}

private struct FailingPartialModelConverter: ModelConverting {
    func convert(
        checkpoint: URL,
        destination: URL,
        progress: @escaping @Sendable (ModelConversionPhase) -> Void
    ) async throws {
        _ = checkpoint
        progress(.converting)
        try Data("partial output".utf8).write(to: destination.appending(path: "partial.bin"))
        throw ModelInstallerError.converterFailed("fixture conversion failed")
    }
}

private final class ProcessModelConverterFixture: @unchecked Sendable {
    let root: URL
    let runtime: WorkerRuntime
    let checkpoint: URL

    init(script: String) throws {
        root = modelTestScratchDirectory()
            .appending(
                path: "cloudpoint-converter-tests-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
        let bin = root.appending(path: "bin", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let executable = bin.appending(path: "cloudpoint-model")
        try Data(script.utf8).write(to: executable, options: .withoutOverwriting)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: executable.path
        )
        checkpoint = root.appending(path: "checkpoint.pt")
        try Data("checkpoint".utf8).write(to: checkpoint)
        runtime = .unchecked(root: root)
    }

    deinit { try? FileManager.default.removeItem(at: root) }
}

private func readProcessID(at url: URL) async throws -> pid_t {
    for _ in 0..<200 {
        if let data = try? Data(contentsOf: url),
           let value = Int32(String(decoding: data, as: UTF8.self)) {
            return value
        }
        try await Task.sleep(for: .milliseconds(10))
    }
    throw ModelInstallerError.converterFailed("The fixture process did not publish its PID.")
}

private func processHasExited(_ pid: pid_t) async -> Bool {
    for _ in 0..<100 {
        if Darwin.kill(pid, 0) != 0, errno == ESRCH { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return false
}

private func modelTestScratchDirectory() -> URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appending(path: ".CloudPointTests", directoryHint: .isDirectory)
}

private func skipIfSandboxedFixtureExecution() throws {
    guard let task = SecTaskCreateFromSelf(nil) else { return }
    let sandboxed = SecTaskCopyValueForEntitlement(
        task,
        "com.apple.security.app-sandbox" as CFString,
        nil
    ) as? Bool
    if sandboxed == true {
        throw XCTSkip("Process fixtures run in the isolated unsandboxed converter test pass")
    }
}

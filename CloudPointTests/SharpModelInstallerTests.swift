import CryptoKit
import Foundation
import XCTest
@testable import CloudPoint

final class SharpModelInstallerTests: XCTestCase {
    func testOfficialReleaseUsesOnlyPinnedAppleOriginAndResearchLicense() {
        let release = SharpModelRelease.v1

        XCTAssertEqual(release.sourceCommit, "1eaa046834b81852261262b41b0919f5c1efdd2e")
        XCTAssertEqual(release.filename, "sharp_2572gikvuh.pt")
        XCTAssertEqual(release.checkpointBytes, 2_809_738_232)
        XCTAssertEqual(release.downloadURL.scheme, "https")
        XCTAssertEqual(release.downloadURL.host, "ml-site.cdn-apple.com")
        XCTAssertEqual(release.licenseKind, .appleMachineLearningResearchModel)
        XCTAssertEqual(
            release.checkpointSHA256,
            "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
        )
    }

    func testBundledResearchLicenseMatchesPinnedDigest() throws {
        let text = try SharpModelLicenseAgreement.load()
        let digest = SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }
            .joined()

        XCTAssertEqual(digest, SharpModelLicenseAgreement.sha256)
        XCTAssertTrue(text.contains("Research Purposes"))
        XCTAssertTrue(text.contains("Copyright (C) 2025 Apple Inc."))
    }

    func testPrepareRequiresExplicitLicenseAcceptanceBeforeNetwork() async throws {
        let fixture = try SharpInstallerFixture()

        let stream = await fixture.installer.prepare(acceptingLicense: false)
        do {
            for try await _ in stream {}
            XCTFail("Expected research-license acceptance to be required")
        } catch let error as SharpModelInstallerError {
            XCTAssertEqual(error, .licenseAcceptanceRequired)
        }

        let requestCount = await fixture.downloader.requestCount
        XCTAssertEqual(requestCount, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.directories.acceptance.path))
    }

    func testAcceptedDownloadIsVerifiedAndPublishedWithProvenanceAtomically() async throws {
        let fixture = try SharpInstallerFixture()

        let stream = await fixture.installer.prepare(acceptingLicense: true)
        var events: [SharpModelSetupEvent] = []
        for try await event in stream { events.append(event) }

        guard case let .ready(installation) = await fixture.installer.health() else {
            return XCTFail("Expected verified SHARP installation")
        }
        XCTAssertEqual(installation.checkpoint, fixture.directories.checkpoint)
        XCTAssertEqual(installation.checkpointSHA256, fixture.release.checkpointSHA256)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.directories.acceptance.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.directories.provenance.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.directories.partial.path))
        XCTAssertEqual(events.last, .ready(installation))

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let provenance = try decoder.decode(
            SharpModelProvenance.self,
            from: Data(contentsOf: fixture.directories.provenance)
        )
        XCTAssertEqual(provenance.sourceCommit, fixture.release.sourceCommit)
        XCTAssertEqual(provenance.checkpointSHA256, fixture.release.checkpointSHA256)
        XCTAssertEqual(provenance.licenseKind, fixture.release.licenseKind)
    }

    func testOrdinaryReopenInspectsLocalArtifactsWithoutNetwork() async throws {
        let fixture = try SharpInstallerFixture()
        var stream = await fixture.installer.prepare(acceptingLicense: true)
        for try await _ in stream {}

        let reopenDownloader = SharpFakeDownloader(data: Data("must not download".utf8))
        let reopened = SharpModelInstaller(
            release: fixture.release,
            directories: fixture.directories,
            downloader: reopenDownloader,
            availableCapacity: { Int64.max }
        )

        guard case .ready = await reopened.health() else {
            return XCTFail("Expected ordinary reopen to be ready")
        }
        stream = await reopened.prepare(acceptingLicense: false)
        for try await _ in stream {}
        let reopenRequestCount = await reopenDownloader.requestCount
        XCTAssertEqual(reopenRequestCount, 0)
    }

    func testDiskPreflightRejectsBeforeDownload() async throws {
        let fixture = try SharpInstallerFixture(availableCapacity: 3)

        let stream = await fixture.installer.prepare(acceptingLicense: true)
        do {
            for try await _ in stream {}
            XCTFail("Expected disk-space preflight failure")
        } catch let error as SharpModelInstallerError {
            XCTAssertEqual(
                error,
                .insufficientDiskSpace(required: fixture.release.requiredFreeBytes, available: 3)
            )
        }
        let requestCount = await fixture.downloader.requestCount
        XCTAssertEqual(requestCount, 0)
    }

    func testDigestMismatchNeverPublishesCheckpoint() async throws {
        let fixture = try SharpInstallerFixture(
            release: .fixture(data: Data("expected".utf8)),
            downloadData: Data("tampered".utf8)
        )

        let stream = await fixture.installer.prepare(acceptingLicense: true)
        do {
            for try await _ in stream {}
            XCTFail("Expected checksum failure")
        } catch let error as SharpModelInstallerError {
            XCTAssertEqual(error, .checkpointMismatch)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.directories.checkpoint.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.directories.partial.path))
    }

    func testDownloaderAllowsExactAppleModelHostButRejectsLookalikes() {
        XCTAssertTrue(URLSessionModelDownloader.isAllowed(
            URL(string: "https://ml-site.cdn-apple.com/models/sharp/checkpoint.pt")
        ))
        XCTAssertFalse(URLSessionModelDownloader.isAllowed(
            URL(string: "https://ml-site.cdn-apple.com.example.test/checkpoint.pt")
        ))
        XCTAssertFalse(URLSessionModelDownloader.isAllowed(
            URL(string: "http://ml-site.cdn-apple.com/models/sharp/checkpoint.pt")
        ))
    }
}

private struct SharpInstallerFixture {
    let temporary: SharpTemporaryDirectory
    let release: SharpModelRelease
    let directories: SharpModelDirectories
    let downloader: SharpFakeDownloader
    let installer: SharpModelInstaller

    init(
        release: SharpModelRelease = .fixture(data: Data("sharp fixture".utf8)),
        downloadData: Data? = nil,
        availableCapacity: Int64 = .max
    ) throws {
        temporary = try SharpTemporaryDirectory()
        self.release = release
        directories = SharpModelDirectories(root: temporary.url, release: release)
        downloader = SharpFakeDownloader(data: downloadData ?? Data("sharp fixture".utf8))
        installer = SharpModelInstaller(
            release: release,
            directories: directories,
            downloader: downloader,
            now: { Date(timeIntervalSince1970: 1_750_000_000) },
            availableCapacity: { availableCapacity }
        )
    }
}

private final class SharpTemporaryDirectory: @unchecked Sendable {
    let url: URL

    init() throws {
        url = FileManager.default.temporaryDirectory.appending(
            path: "CloudPointSharpModelTests-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: url) }
}

private actor SharpFakeDownloader: ModelDownloading {
    private let data: Data
    private(set) var requestCount = 0

    init(data: Data) { self.data = data }

    func download(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> URL {
        _ = request
        _ = resumeData
        requestCount += 1
        progress(Int64(data.count), Int64(data.count))
        let url = FileManager.default.temporaryDirectory.appending(
            path: "cloudpoint-model-download-\(UUID().uuidString)"
        )
        try data.write(to: url)
        return url
    }

    func cancel() async -> Data? { nil }
}

private extension SharpModelRelease {
    static func fixture(data: Data) -> SharpModelRelease {
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return SharpModelRelease(
            identifier: "apple-sharp-fixture",
            sourceCommit: "fixture-source",
            filename: "fixture.pt",
            checkpointBytes: Int64(data.count),
            checkpointSHA256: digest,
            requiredFreeBytes: Int64(data.count * 3),
            downloadURL: URL(string: "https://ml-site.cdn-apple.com/models/sharp/fixture.pt")!,
            licenseKind: .appleMachineLearningResearchModel
        )
    }
}

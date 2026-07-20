import XCTest
@testable import CloudPoint

final class ProjectManifestTests: XCTestCase {
    func testLoadIgnoresPartialArtifacts() throws {
        let package = try TemporaryProjectPackage.make()
        var manifest = ProjectManifest.fixture()
        manifest.completedWindows = [.fixture(index: 0)]

        try manifest.writeAtomically(to: package.url)
        FileManager.default.createFile(
            atPath: package.url.appending(path: "Points/window-1.cpc.partial").path,
            contents: Data("partial".utf8)
        )

        let loaded = try ProjectManifest.load(from: package.url)

        XCTAssertEqual(loaded.completedWindows.map(\.index), [0])
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: package.url.appending(path: "Points/window-1.cpc.partial").path
            )
        )
    }

    func testAtomicWriteReplacesAnExistingManifest() throws {
        let package = try TemporaryProjectPackage.make()
        var manifest = ProjectManifest.fixture()
        try manifest.writeAtomically(to: package.url)

        manifest.frames = [PersistedFrame(index: 3, sourceTimestamp: 0.6, relativePath: "Frames/00000003.jpg")]
        try manifest.writeAtomically(to: package.url)

        let loaded = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(loaded.frames, manifest.frames)
        XCTAssertTrue(FileManager.default.fileExists(atPath: package.url.appending(path: "Manifest.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.url.appending(path: "Manifest.json.partial").path))
    }

    @MainActor
    func testDocumentDeclaresTheCloudPointProjectType() {
        XCTAssertEqual(CloudPointDocument.readableContentTypes, [.cloudPointProject])
    }

    @MainActor
    func testDocumentPackageWrapperCreatesTheRequiredDirectories() throws {
        let wrapper = try CloudPointDocument.packageWrapper(for: .fixture())

        XCTAssertEqual(
            Set(wrapper.fileWrappers?.keys.map { $0 } ?? []),
            ["Manifest.json", "Frames", "Predictions", "Points", "Logs"]
        )
    }

    @MainActor
    func testDocumentSaveRetainsExistingPayloadFiles() throws {
        let package = try TemporaryProjectPackage.make()
        let original = ProjectManifest.fixture()
        try original.writeAtomically(to: package.url)
        let payloads = [
            "Frames/existing.jpg": Data("frame".utf8),
            "Predictions/existing.depth-f16": Data("depth".utf8),
            "Points/window-0.cpc": Data("points".utf8),
            "Logs/worker.log": Data("log".utf8),
        ]

        for (relativePath, contents) in payloads {
            XCTAssertTrue(
                FileManager.default.createFile(
                    atPath: package.url.appending(path: relativePath).path,
                    contents: contents
                )
            )
        }

        var updated = original
        updated.frames = [PersistedFrame(index: 1, sourceTimestamp: 0.2, relativePath: "Frames/existing.jpg")]
        let existingPackage = try FileWrapper(url: package.url, options: .immediate)
        let wrapper = try CloudPointDocument.packageWrapper(for: updated, preserving: existingPackage)
        try wrapper.write(
            to: package.url,
            options: FileWrapper.WritingOptions.atomic,
            originalContentsURL: package.url
        )

        XCTAssertEqual(try ProjectManifest.load(from: package.url).frames, updated.frames)
        for (relativePath, contents) in payloads {
            XCTAssertEqual(try Data(contentsOf: package.url.appending(path: relativePath)), contents)
        }
    }

    func testLoadRejectsAnUnsupportedFormatVersion() throws {
        let package = try TemporaryProjectPackage.make()
        var unsupported = ProjectManifest.fixture()
        unsupported.formatVersion = 2
        try ProjectManifest.encode(unsupported).write(to: package.url.appending(path: "Manifest.json"))

        XCTAssertThrowsError(try ProjectManifest.load(from: package.url)) { error in
            XCTAssertEqual(error as? ProjectManifestError, .unsupportedFormatVersion(2))
        }
    }

    @MainActor
    func testDocumentReadRejectsAnUnsupportedFormatVersion() throws {
        var unsupported = ProjectManifest.fixture()
        unsupported.formatVersion = 2
        let packageWrapper = FileWrapper(directoryWithFileWrappers: [
            "Manifest.json": FileWrapper(regularFileWithContents: try ProjectManifest.encode(unsupported)),
        ])

        XCTAssertThrowsError(try CloudPointDocument.loadManifest(from: packageWrapper)) { error in
            XCTAssertEqual(error as? ProjectManifestError, .unsupportedFormatVersion(2))
        }
    }

    func testAtomicWriteRejectsAnUnsupportedFormatVersion() throws {
        let package = try TemporaryProjectPackage.make()
        var unsupported = ProjectManifest.fixture()
        unsupported.formatVersion = 2

        XCTAssertThrowsError(try unsupported.writeAtomically(to: package.url)) { error in
            XCTAssertEqual(error as? ProjectManifestError, .unsupportedFormatVersion(2))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.url.appending(path: "Manifest.json").path))
    }

    func testApplicationExportsTheCloudPointPackageType() {
        let declarations = Bundle.main.object(forInfoDictionaryKey: "UTExportedTypeDeclarations") as? [[String: Any]]
        let identifiers = declarations?.compactMap { $0["UTTypeIdentifier"] as? String } ?? []

        XCTAssertTrue(identifiers.contains("cloud.point.cloud.project"))
    }
}

extension ProjectManifest {
    static func fixture() -> ProjectManifest {
        ProjectManifest(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            createdAt: Date(timeIntervalSinceReferenceDate: 1_000),
            updatedAt: Date(timeIntervalSinceReferenceDate: 2_000),
            frames: [],
            completedWindows: [],
            sessionState: .empty
        )
    }
}

extension CompletedWindow {
    static func fixture(index: Int) -> CompletedWindow {
        CompletedWindow(
            index: index,
            frameStart: index * 32,
            frameEnd: (index * 32) + 31,
            pointChunkRelativePath: "Points/window-\(index).cpc",
            alignmentRowMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ]
        )
    }
}

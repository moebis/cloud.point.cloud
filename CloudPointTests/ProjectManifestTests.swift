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

    func testDocumentPackageWrapperCreatesTheRequiredDirectories() throws {
        let wrapper = try CloudPointDocument.packageWrapper(for: .fixture())

        XCTAssertEqual(
            Set(wrapper.fileWrappers?.keys.map { $0 } ?? []),
            ["Manifest.json", "Frames", "Predictions", "Points", "Logs"]
        )
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

import XCTest
@testable import CloudPoint
import UniformTypeIdentifiers

@MainActor
final class WorkspaceViewModelTests: XCTestCase {
    func testManagedWorkspaceHasNoSavePrerequisite() throws {
        let package = try TemporaryProjectPackage.make()
        let manifest = ProjectManifest()
        try manifest.writeAtomically(to: package.url)
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: manifest),
            packageURL: package.url,
            arguments: []
        )

        XCTAssertNil(viewModel.snapshot.setupText)
    }

    func testMockEngineRequiresExactDebugFlag() {
#if DEBUG
        XCTAssertTrue(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint", "--mock-engine"]))
#else
        XCTAssertFalse(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint", "--mock-engine"]))
#endif
        XCTAssertFalse(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint"]))
        XCTAssertFalse(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint", "--mock-engine-extra"]))
    }

}

import XCTest
@testable import CloudPoint

@MainActor
final class WorkspaceViewModelTests: XCTestCase {
    func testUntitledDocumentDisablesInputAndExplainsSaveRequirement() {
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(),
            packageURL: nil,
            arguments: []
        )

        XCTAssertEqual(viewModel.snapshot.setupText, "Save this project first")
        XCTAssertFalse(viewModel.snapshot.capabilities.canImportRecording)
        XCTAssertFalse(viewModel.snapshot.capabilities.canUseCamera)
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

import XCTest
@testable import CloudPoint
import UniformTypeIdentifiers

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

    func testRecordingPickerAcceptsMovMp4AndM4v() throws {
        for filenameExtension in ["mov", "mp4", "m4v"] {
            let fileType = try XCTUnwrap(UTType(filenameExtension: filenameExtension))
            XCTAssertTrue(
                WorkspaceViewModel.recordingContentTypes.contains {
                    fileType.conforms(to: $0)
                },
                "Expected .\(filenameExtension) to be accepted by the recording picker"
            )
        }
    }
}

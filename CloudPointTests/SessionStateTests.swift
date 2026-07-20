import XCTest
@testable import CloudPoint

final class SessionStateTests: XCTestCase {
    func testCaptureCanTrailIntoProcessing() throws {
        var state = SessionState.empty
        state = try state.applying(.prepare)
        state = try state.applying(.ready)
        state = try state.applying(.startCapture)
        state = try state.applying(.enqueueFrame)
        state = try state.applying(.stopCapture)

        XCTAssertEqual(state.phase, .processing)
        XCTAssertFalse(state.isCapturing)
        XCTAssertEqual(state.capturedCount, 1)
    }

    func testCompletedCannotRestartCapture() throws {
        let completed = SessionState(phase: .completed)

        XCTAssertThrowsError(try completed.applying(.startCapture))
    }
}

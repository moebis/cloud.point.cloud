import XCTest
@testable import CloudPoint

final class SessionStateTests: XCTestCase {
    func testDurableCommitAndAdmissionAreDistinctCheckedTransitions() throws {
        var state = try ready().applying(.startImport)

        state = try state.applying(.durableFrameCommitted)
        XCTAssertEqual(state.capturedCount, 1)
        XCTAssertEqual(state.queuedCount, 0)

        state = try state.applying(.frameAdmitted)
        XCTAssertEqual(state.capturedCount, 1)
        XCTAssertEqual(state.queuedCount, 1)
    }

    func testPausedCaptureContinuesDurableCommitAndAdmission() throws {
        var state = try ready().applying(.startCapture)
        state = try state.applying(.pause)
        XCTAssertEqual(state.phase, .paused)
        XCTAssertTrue(state.isCapturing)

        state = try state.applying(.durableFrameCommitted)
        state = try state.applying(.frameAdmitted)

        XCTAssertEqual(state.capturedCount, 1)
        XCTAssertEqual(state.queuedCount, 1)
        XCTAssertEqual(state.phase, .paused)
    }

    func testStopAndFinishAreLegalWhilePausedAndResumeProcessing() throws {
        var state = try ready().applying(.startCapture)
        state = try state.applying(.pause)
        state = try state.applying(.stopCapture)
        XCTAssertEqual(state.phase, .paused)
        XCTAssertFalse(state.isCapturing)

        state = try state.applying(.finishInput)
        XCTAssertEqual(state.phase, .paused)
        state = try state.applying(.resume)
        XCTAssertEqual(state.phase, .processing)
    }

    func testProcessedCountChangesOnlyOnAtomicWindowCommit() throws {
        var state = try ready().applying(.startImport)
        for _ in 0..<2 {
            state = try state.applying(.durableFrameCommitted)
            state = try state.applying(.frameAdmitted)
        }

        state = try state.applying(.frameStarted(windowIndex: 7))
        XCTAssertEqual(state.processedCount, 0)
        XCTAssertEqual(state.currentWindow, 7)

        state = try state.applying(.windowCommitted(windowIndex: 7, processedFrames: 2))
        XCTAssertEqual(state.processedCount, 2)
        XCTAssertNil(state.currentWindow)
    }

    func testWindowCommitRejectsCounterRegressionAndWindowMismatch() throws {
        let state = SessionState(
            phase: .processing,
            capturedCount: 2,
            queuedCount: 2,
            processedCount: 1,
            currentWindow: 3
        )

        XCTAssertThrowsError(try state.applying(.windowCommitted(windowIndex: 3, processedFrames: 0)))
        XCTAssertThrowsError(try state.applying(.windowCommitted(windowIndex: 4, processedFrames: 1)))
        XCTAssertThrowsError(try state.applying(.windowCommitted(windowIndex: 3, processedFrames: 2)))
    }

    func testEveryCounterUsesCheckedArithmetic() throws {
        XCTAssertThrowsError(
            try SessionState(phase: .importing, capturedCount: .max, queuedCount: .max)
                .applying(.durableFrameCommitted)
        ) { XCTAssertEqual($0 as? SessionTransitionError, .counterOverflow) }

        XCTAssertThrowsError(
            try SessionState(phase: .importing, capturedCount: .max, queuedCount: .max)
                .applying(.frameAdmitted)
        ) { XCTAssertEqual($0 as? SessionTransitionError, .counterOverflow) }

        XCTAssertThrowsError(
            try SessionState(
                phase: .processing,
                capturedCount: .max,
                queuedCount: .max,
                processedCount: .max,
                currentWindow: 0
            ).applying(.windowCommitted(windowIndex: 0, processedFrames: 1))
        ) { XCTAssertEqual($0 as? SessionTransitionError, .counterOverflow) }

        XCTAssertThrowsError(
            try SessionState(phase: .processing, failedCount: .max).applying(.fail)
        ) { XCTAssertEqual($0 as? SessionTransitionError, .counterOverflow) }
    }

    func testInvalidCounterInvariantIsRejectedBeforeTransition() {
        let invalid = SessionState(phase: .processing, capturedCount: 1, queuedCount: 2)

        XCTAssertThrowsError(try invalid.applying(.pause)) {
            XCTAssertEqual($0 as? SessionTransitionError, .invalidCounters)
        }
    }

    func testCompletedCannotRestartCapture() {
        XCTAssertThrowsError(try SessionState(phase: .completed).applying(.startCapture))
    }

    private func ready() throws -> SessionState {
        try SessionState.empty.applying(.prepare).applying(.ready)
    }
}

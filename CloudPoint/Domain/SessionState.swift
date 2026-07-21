import Foundation

enum SessionPhase: String, Codable, Sendable {
    case empty
    case preparing
    case ready
    case importing
    case capturing
    case processing
    case paused
    case finalizing
    case completed
    case cancelled
    case failed
}

enum SessionEvent: Sendable, Equatable {
    case prepare
    case ready
    case startImport
    case startCapture
    case durableFrameCommitted
    case frameAdmitted
    case frameStarted(windowIndex: UInt32)
    case windowCommitted(windowIndex: UInt32, processedFrames: UInt64)
    case gaussianCommitted
    case stopCapture
    case finishInput
    case pause
    case resume
    case beginFinalizing
    case complete
    case cancel
    case fail
}

enum SessionTransitionError: Error, Equatable, Sendable {
    case illegal(from: SessionPhase, event: SessionEvent)
    case counterOverflow
    case invalidCounters
    case windowMismatch(expected: UInt32?, actual: UInt32)
}

struct SessionState: Codable, Sendable, Equatable {
    var phase: SessionPhase
    var isCapturing: Bool
    var capturedCount: UInt64
    var queuedCount: UInt64
    var processedCount: UInt64
    var failedCount: UInt64
    var currentWindow: UInt32?

    static let empty = SessionState(phase: .empty)

    init(
        phase: SessionPhase,
        isCapturing: Bool = false,
        capturedCount: UInt64 = 0,
        queuedCount: UInt64 = 0,
        processedCount: UInt64 = 0,
        failedCount: UInt64 = 0,
        currentWindow: UInt32? = nil
    ) {
        self.phase = phase
        self.isCapturing = isCapturing
        self.capturedCount = capturedCount
        self.queuedCount = queuedCount
        self.processedCount = processedCount
        self.failedCount = failedCount
        self.currentWindow = currentWindow
    }

    var isProcessing: Bool {
        switch phase {
        case .importing, .capturing, .processing, .paused, .finalizing:
            true
        case .empty, .preparing, .ready, .completed, .cancelled, .failed:
            false
        }
    }

    var backlogCount: UInt64 {
        queuedCount - processedCount
    }

    func validated() throws -> SessionState {
        guard processedCount <= queuedCount, queuedCount <= capturedCount else {
            throw SessionTransitionError.invalidCounters
        }
        return self
    }

    func applying(_ event: SessionEvent) throws -> SessionState {
        _ = try validated()
        var next = self

        switch (phase, event) {
        case (.empty, .prepare):
            next.phase = .preparing
        case (.preparing, .ready):
            next.phase = .ready
        case (.ready, .startImport):
            next.phase = .importing
        case (.ready, .startCapture):
            next.phase = .capturing
            next.isCapturing = true

        case (.importing, .durableFrameCommitted),
             (.capturing, .durableFrameCommitted),
             (.processing, .durableFrameCommitted),
             (.paused, .durableFrameCommitted):
            next.capturedCount = try Self.increment(next.capturedCount)
        case (.preparing, .frameAdmitted),
             (.importing, .frameAdmitted),
             (.capturing, .frameAdmitted),
             (.processing, .frameAdmitted),
             (.paused, .frameAdmitted):
            next.queuedCount = try Self.increment(next.queuedCount)

        case (.importing, let .frameStarted(windowIndex)),
             (.capturing, let .frameStarted(windowIndex)),
             (.processing, let .frameStarted(windowIndex)),
             (.paused, let .frameStarted(windowIndex)),
             (.finalizing, let .frameStarted(windowIndex)):
            guard next.currentWindow == nil || next.currentWindow == windowIndex else {
                throw SessionTransitionError.windowMismatch(
                    expected: next.currentWindow,
                    actual: windowIndex
                )
            }
            next.currentWindow = windowIndex

        case (.importing, let .windowCommitted(windowIndex, count)),
             (.capturing, let .windowCommitted(windowIndex, count)),
             (.processing, let .windowCommitted(windowIndex, count)),
             (.paused, let .windowCommitted(windowIndex, count)),
             (.finalizing, let .windowCommitted(windowIndex, count)):
            guard next.currentWindow == windowIndex else {
                throw SessionTransitionError.windowMismatch(
                    expected: next.currentWindow,
                    actual: windowIndex
                )
            }
            guard count > 0 else { throw SessionTransitionError.invalidCounters }
            next.processedCount = try Self.add(next.processedCount, count)
            next.currentWindow = nil

        case (.processing, .gaussianCommitted):
            guard next.processedCount == 0,
                  next.queuedCount == 1,
                  next.capturedCount == 1 else {
                throw SessionTransitionError.invalidCounters
            }
            next.processedCount = 1

        case (.capturing, .stopCapture):
            next.phase = .processing
            next.isCapturing = false
        case (.paused, .stopCapture) where isCapturing:
            next.isCapturing = false

        case (.importing, .finishInput), (.processing, .finishInput):
            next.phase = .processing
            next.isCapturing = false
        case (.capturing, .finishInput):
            next.phase = .processing
            next.isCapturing = false
        case (.paused, .finishInput):
            next.isCapturing = false

        case (.importing, .pause), (.capturing, .pause), (.processing, .pause):
            next.phase = .paused
        case (.paused, .resume):
            next.phase = next.isCapturing ? .capturing : .processing
        case (.processing, .beginFinalizing):
            next.phase = .finalizing
        case (.finalizing, .complete):
            next.phase = .completed
            next.currentWindow = nil

        case (.preparing, .cancel), (.ready, .cancel), (.importing, .cancel),
             (.capturing, .cancel), (.processing, .cancel), (.paused, .cancel),
             (.finalizing, .cancel):
            next.phase = .cancelled
            next.isCapturing = false
            next.currentWindow = nil

        case (.preparing, .fail), (.ready, .fail), (.importing, .fail),
             (.capturing, .fail), (.processing, .fail), (.paused, .fail),
             (.finalizing, .fail):
            next.failedCount = try Self.increment(next.failedCount)
            next.phase = .failed
            next.isCapturing = false
            next.currentWindow = nil
        default:
            throw SessionTransitionError.illegal(from: phase, event: event)
        }

        return try next.validated()
    }

    private static func increment(_ value: UInt64) throws -> UInt64 {
        try add(value, 1)
    }

    private static func add(_ value: UInt64, _ increment: UInt64) throws -> UInt64 {
        let (result, overflow) = value.addingReportingOverflow(increment)
        guard !overflow else { throw SessionTransitionError.counterOverflow }
        return result
    }
}

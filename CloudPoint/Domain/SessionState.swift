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

enum SessionEvent: String, Codable, Sendable {
    case prepare
    case ready
    case startImport
    case startCapture
    case enqueueFrame
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
}

struct SessionState: Codable, Sendable, Equatable {
    var phase: SessionPhase
    var isCapturing: Bool
    var capturedCount: Int
    var queuedCount: Int
    var processedCount: Int
    var failedCount: Int
    var currentWindow: Int?

    static let empty = SessionState(phase: .empty)

    init(
        phase: SessionPhase,
        isCapturing: Bool = false,
        capturedCount: Int = 0,
        queuedCount: Int = 0,
        processedCount: Int = 0,
        failedCount: Int = 0,
        currentWindow: Int? = nil
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

    var backlogCount: Int {
        max(0, queuedCount - processedCount)
    }

    func applying(_ event: SessionEvent) throws -> SessionState {
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
        case (.importing, .enqueueFrame), (.capturing, .enqueueFrame):
            next.capturedCount += 1
            next.queuedCount += 1
        case (.capturing, .stopCapture):
            next.phase = .processing
            next.isCapturing = false
        case (.importing, .finishInput), (.processing, .finishInput):
            next.phase = .processing
        case (.capturing, .finishInput):
            next.phase = .processing
            next.isCapturing = false
        case (.importing, .pause), (.capturing, .pause), (.processing, .pause):
            next.phase = .paused
        case (.paused, .resume):
            next.phase = next.isCapturing ? .capturing : .processing
        case (.processing, .beginFinalizing):
            next.phase = .finalizing
        case (.finalizing, .complete):
            next.phase = .completed
        case (.preparing, .cancel), (.ready, .cancel), (.importing, .cancel),
             (.capturing, .cancel), (.processing, .cancel), (.paused, .cancel),
             (.finalizing, .cancel):
            next.phase = .cancelled
            next.isCapturing = false
        case (.preparing, .fail), (.ready, .fail), (.importing, .fail),
             (.capturing, .fail), (.processing, .fail), (.paused, .fail),
             (.finalizing, .fail):
            next.phase = .failed
            next.isCapturing = false
            next.failedCount += 1
        default:
            throw SessionTransitionError.illegal(from: phase, event: event)
        }

        return next
    }
}

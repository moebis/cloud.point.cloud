@preconcurrency import AVFoundation
import Combine
import Foundation

enum SharpCameraSnapshotError: Error, LocalizedError, Sendable, Equatable {
    case authorizationDenied
    case deviceNotFound
    case configurationFailed
    case noFrameAvailable

    var errorDescription: String? {
        switch self {
        case .authorizationDenied:
            "CloudPoint needs camera access to capture a SHARP source image."
        case .deviceNotFound:
            "The selected camera is no longer available."
        case .configurationFailed:
            "CloudPoint could not start the selected camera."
        case .noFrameAvailable:
            "The camera has not produced a usable frame yet."
        }
    }
}

private final class SharpCameraSnapshotAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var latestCandidate: VideoKeyFrameCandidate?
    private var receivedFrameCount = 0

    func receive(_ frame: CapturedFrame) {
        receivedFrameCount += 1
        guard receivedFrameCount == 1 || receivedFrameCount.isMultiple(of: 10) else { return }
        guard let candidate = try? VideoKeyFrameSelector.candidate(from: frame) else { return }
        lock.lock()
        latestCandidate = candidate
        lock.unlock()
    }

    func latest() -> VideoKeyFrameCandidate? {
        lock.lock()
        defer { lock.unlock() }
        return latestCandidate
    }
}

actor SharpCameraSnapshotSource {
    nonisolated let previewSession: AVCaptureSession

    private let authority: any CameraAuthorizing
    private let captureSession: any CameraCaptureSession
    private let accumulator = SharpCameraSnapshotAccumulator()
    private var started = false

    init(
        authority: any CameraAuthorizing = SystemCameraAuthority(),
        captureSession: any CameraCaptureSession = AVCameraCaptureSession()
    ) {
        self.authority = authority
        self.captureSession = captureSession
        previewSession = captureSession.previewSession
    }

    func start(deviceID: String) async throws {
        guard !started else { return }
        guard await authority.requestAccess() == .authorized else {
            throw SharpCameraSnapshotError.authorizationDenied
        }
        do {
            try await captureSession.connect(
                deviceID: deviceID,
                frameHandler: { [accumulator] frame in accumulator.receive(frame) },
                disconnectHandler: {}
            )
            try await captureSession.startRunning()
            started = true
        } catch CameraCaptureSessionError.deviceNotFound {
            throw SharpCameraSnapshotError.deviceNotFound
        } catch {
            throw SharpCameraSnapshotError.configurationFailed
        }
    }

    func hasSnapshot() -> Bool { accumulator.latest() != nil }

    func snapshot() throws -> VideoKeyFrameCandidate {
        guard let candidate = accumulator.latest() else {
            throw SharpCameraSnapshotError.noFrameAvailable
        }
        return candidate
    }

    func stop() async {
        guard started else { return }
        started = false
        await captureSession.shutdown()
    }
}

@MainActor
final class SharpCameraSnapshotViewModel: ObservableObject {
    @Published private(set) var isStarting = false
    @Published private(set) var isReady = false
    @Published private(set) var errorMessage: String?

    let previewSession: AVCaptureSession
    private let deviceID: String?
    private let source: SharpCameraSnapshotSource

    init(deviceID: String?) {
        self.deviceID = deviceID
        let source = SharpCameraSnapshotSource()
        self.source = source
        previewSession = source.previewSession
    }

    func start() async {
        guard let deviceID, !isStarting, !isReady else { return }
        isStarting = true
        errorMessage = nil
        do {
            try await source.start(deviceID: deviceID)
            let clock = ContinuousClock()
            let deadline = clock.now + .seconds(5)
            while clock.now < deadline, !Task.isCancelled {
                if await source.hasSnapshot() {
                    isReady = true
                    isStarting = false
                    return
                }
                try await clock.sleep(for: .milliseconds(50))
            }
            if !Task.isCancelled { throw SharpCameraSnapshotError.noFrameAvailable }
        } catch is CancellationError {
        } catch {
            errorMessage = error.localizedDescription
        }
        isStarting = false
    }

    func capture() async throws -> VideoKeyFrameCandidate {
        let candidate = try await source.snapshot()
        await source.stop()
        isReady = false
        return candidate
    }

    func stop() async {
        await source.stop()
        isReady = false
        isStarting = false
    }
}

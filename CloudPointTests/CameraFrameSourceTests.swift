@preconcurrency import AVFoundation
import CoreVideo
import XCTest
@testable import CloudPoint

final class CameraSampleGateTests: XCTestCase {
    func testFiveFPSSelectsExactBoundariesFromThirtyFPSInput() throws {
        var gate = try CameraSampleGate(rate: 5)

        let selected = (0..<30).compactMap { frame -> CMTime? in
            let timestamp = CMTime(value: CMTimeValue(frame), timescale: 30)
            return gate.accepts(timestamp) ? timestamp : nil
        }

        XCTAssertEqual(
            selected,
            [0, 6, 12, 18, 24].map { CMTime(value: CMTimeValue($0), timescale: 30) }
        )
    }

    func testBoundaryImmediatelyBeforeTargetIsRejected() throws {
        var gate = try CameraSampleGate(rate: 5)

        XCTAssertTrue(gate.accepts(.zero))
        XCTAssertFalse(gate.accepts(CMTime(value: 199, timescale: 1_000)))
        XCTAssertTrue(gate.accepts(CMTime(value: 200, timescale: 1_000)))
    }

    func testResetStartsA_newMediaTimeline() throws {
        var gate = try CameraSampleGate(rate: 5)
        XCTAssertTrue(gate.accepts(CMTime(seconds: 12, preferredTimescale: 600)))
        XCTAssertFalse(gate.accepts(CMTime(seconds: 12.1, preferredTimescale: 600)))

        gate.reset()

        XCTAssertTrue(gate.accepts(.zero))
        XCTAssertTrue(gate.accepts(CMTime(seconds: 0.2, preferredTimescale: 600)))
    }

    func testInvalidNonfiniteNegativeAndNonmonotonicTimesAreRejected() throws {
        var gate = try CameraSampleGate(rate: 5)

        XCTAssertFalse(gate.accepts(.invalid))
        XCTAssertFalse(gate.accepts(.indefinite))
        XCTAssertFalse(gate.accepts(.positiveInfinity))
        XCTAssertFalse(gate.accepts(.negativeInfinity))
        XCTAssertFalse(gate.accepts(CMTime(value: -1, timescale: 30)))
        XCTAssertTrue(gate.accepts(.zero))
        XCTAssertFalse(gate.accepts(.zero))
        XCTAssertFalse(gate.accepts(CMTime(value: -1, timescale: 30)))
        XCTAssertFalse(gate.accepts(CMTime(value: 5, timescale: 30)))
        XCTAssertTrue(gate.accepts(CMTime(value: 6, timescale: 30)))
    }

    func testRateUsesTheSharedSamplingValidation() {
        XCTAssertThrowsError(try CameraSampleGate(rate: 0)) {
            XCTAssertEqual($0 as? FrameSamplingError, .invalidRate(0))
        }
        XCTAssertThrowsError(try CameraSampleGate(rate: 11)) {
            XCTAssertEqual($0 as? FrameSamplingError, .invalidRate(11))
        }
    }
}

final class CameraFrameSourceTests: XCTestCase {
    func testThirtyPreviewFramesProduceFivePersistedFrames() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let collector = PersistedFrameCollector(stream: source.events())
        defer { collector.cancel() }

        await source.start(deviceID: "camera-a", sampleRate: 5)
        for sequence in 0..<30 {
            session.emit(try makeFrame(sequence: sequence, timestamp: CMTime(value: CMTimeValue(sequence), timescale: 30)))
            if sequence.isMultiple(of: 6) {
                let expectedCount = sequence / 6 + 1
                try await eventually { await persistence.callCount == expectedCount }
            }
        }

        try await eventually {
            let callCount = await persistence.callCount
            let persistedCount = await collector.persisted.count
            return callCount == 5 && persistedCount == 5
        }
        let metrics = await source.metrics
        XCTAssertEqual(metrics.previewFrameCount, 30)
        XCTAssertEqual(metrics.acceptedPersistenceCount, 5)
        XCTAssertEqual(metrics.droppedPersistenceCount, 0)
        let timestamps = await persistence.timestamps
        let persisted = await collector.persisted
        XCTAssertEqual(timestamps, [0, 6, 12, 18, 24].map {
            CMTime(value: CMTimeValue($0), timescale: 30)
        })
        XCTAssertEqual(persisted.map(\.index), Array(0..<5))
    }

    func testBusyPersistenceHasOneBufferedFrameAndReportsNewestDrop() async throws {
        let persistence = PersistenceProbe(suspended: true)
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let collector = PersistedFrameCollector(stream: source.events())
        defer { collector.cancel() }
        await source.start(deviceID: "camera-a", sampleRate: 5)

        session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { await persistence.callCount == 1 }
        session.emit(try makeFrame(sequence: 1, timestamp: CMTime(value: 1, timescale: 5)))
        session.emit(try makeFrame(sequence: 2, timestamp: CMTime(value: 2, timescale: 5)))

        try await eventually {
            let droppedCount = await source.metrics.droppedPersistenceCount
            let reportedCount = await collector.drops.count
            return droppedCount == 1 && reportedCount == 1
        }
        let busyCallCount = await persistence.callCount
        let drops = await collector.drops
        XCTAssertEqual(busyCallCount, 1)
        XCTAssertEqual(
            drops,
            [CameraDroppedFrame(timestamp: CMTime(value: 2, timescale: 5), reason: .persistenceBusy)]
        )

        await persistence.resumeOne()
        try await eventually { await persistence.callCount == 2 }
        await persistence.resumeOne()
        try await eventually { await collector.persisted.count == 2 }
        let persistedTimestamps = await persistence.timestamps
        XCTAssertEqual(persistedTimestamps, [.zero, CMTime(value: 1, timescale: 5)])
    }

    func testPersistenceFailureTransitionsToFailedWithoutEmittingPersistedFrame() async throws {
        let persistence = PersistenceProbe(error: ProbeError.writeFailed)
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let collector = PersistedFrameCollector(stream: source.events())
        defer { collector.cancel() }
        await source.start(deviceID: "camera-a", sampleRate: 5)

        session.emit(try makeFrame(sequence: 0, timestamp: .zero))

        try await eventually { await source.state == .failed(.persistenceFailed) }
        let persisted = await collector.persisted
        XCTAssertEqual(persisted, [])
    }

    func testAuthorizationDenialFailsWithoutConnectingHardware() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.denied),
            captureSession: session
        )

        await source.start(deviceID: "camera-a", sampleRate: 5)

        let state = await source.state
        XCTAssertEqual(state, .failed(.authorizationDenied))
        XCTAssertFalse(session.isConnected)
    }

    func testSelectedDeviceIsReflectedByRunningSession() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a", "camera-b"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )

        await source.start(deviceID: "camera-b", sampleRate: 5)

        let state = await source.state
        XCTAssertEqual(state, .running(deviceID: "camera-b"))
        XCTAssertEqual(session.connectedDeviceID, "camera-b")
        XCTAssertTrue(session.isRunning)
    }

    func testMissingDeviceFailsWithoutStartingSession() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )

        await source.start(deviceID: "missing", sampleRate: 5)

        let state = await source.state
        XCTAssertEqual(state, .failed(.deviceNotFound))
        XCTAssertFalse(session.isRunning)
    }

    func testStartAndStopAreIdempotent() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )

        await source.start(deviceID: "camera-a", sampleRate: 5)
        await source.start(deviceID: "camera-a", sampleRate: 5)
        XCTAssertEqual(session.startCount, 1)

        await source.stop()
        await source.stop()
        let state = await source.state
        XCTAssertEqual(state, .stopped)
        XCTAssertEqual(session.stopCount, 1)
    }

    func testDisconnectPreservesPersistedFileAndSuppressesLaterEmissions() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let persistence = FileMarkerPersistence(directory: directory)
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let collector = PersistedFrameCollector(stream: source.events())
        defer { collector.cancel() }
        await source.start(deviceID: "camera-a", sampleRate: 5)
        session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { await collector.persisted.count == 1 }
        let persistedURL = directory.appending(path: "00000000.jpg")
        XCTAssertTrue(FileManager.default.fileExists(atPath: persistedURL.path))

        session.disconnect()
        try await eventually { await source.state == .failed(.cameraDisconnected) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: persistedURL.path))
        session.emit(try makeFrame(sequence: 1, timestamp: CMTime(value: 1, timescale: 5)))
        for _ in 0..<20 { await Task.yield() }
        let persistedCount = await collector.persisted.count
        XCTAssertEqual(persistedCount, 1)
    }

    func testStopRemovesDisconnectObservation() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)

        await source.stop()
        session.disconnect()
        for _ in 0..<20 { await Task.yield() }

        let state = await source.state
        XCTAssertEqual(state, .stopped)
        XCTAssertFalse(session.hasDisconnectObserver)
    }

    func testStopWinsAgainstDelayedConfigurationFailure() async throws {
        let session = DelayedFailureCameraCaptureSession()
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let start = Task { await source.start(deviceID: "camera-a", sampleRate: 5) }
        try await eventually { await session.connectIsWaiting }

        await source.stop()
        await session.failConnect()
        await start.value

        let state = await source.state
        XCTAssertEqual(state, .stopped)
        XCTAssertFalse(session.isRunning)
    }

    func testConcurrentDisconnectAndStopCannotRestoreCapture() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)

        session.disconnect()
        await source.stop()
        for _ in 0..<20 { await Task.yield() }

        let state = await source.state
        XCTAssertEqual(state, .stopped)
        XCTAssertFalse(session.isRunning)
        XCTAssertFalse(session.hasDisconnectObserver)
    }

    @MainActor
    func testPreviewLayerTracksSessionResizesAndTearsDown() {
        let first = AVCaptureSession()
        let second = AVCaptureSession()
        let view = CameraPreviewNSView(session: first)
        view.frame = NSRect(x: 0, y: 0, width: 640, height: 360)
        view.layout()

        XCTAssertTrue(view.previewLayer.session === first)
        XCTAssertEqual(view.previewLayer.videoGravity, .resizeAspect)
        XCTAssertEqual(view.previewLayer.frame, view.bounds)

        view.setSession(second)
        XCTAssertTrue(view.previewLayer.session === second)
        view.tearDown()
        XCTAssertNil(view.previewLayer.session)
        XCTAssertNil(view.layer)
    }
}

final class CameraManualTests: XCTestCase {
    func testPhysicalCameraCaptureWhenExplicitlyEnabled() async throws {
        guard ProcessInfo.processInfo.environment["CLOUDPOINT_RUN_CAMERA_MANUAL_TESTS"] == "1" else {
            throw XCTSkip("Set CLOUDPOINT_RUN_CAMERA_MANUAL_TESTS=1 to exercise physical capture")
        }
        guard let device = await CameraCatalog.devices().first else {
            throw XCTSkip("No camera is connected")
        }
        let package = try TemporaryProjectPackage.make()
        defer { try? FileManager.default.removeItem(at: package.url) }
        let source = CameraFrameSource(persistence: try JPEGFramePersistence(packageURL: package.url))

        await source.start(deviceID: device.id, sampleRate: 1)
        guard case .running = await source.state else {
            XCTFail("Physical camera did not enter the running state: \(await source.state)")
            return
        }
        await source.stop()
        let stoppedState = await source.state
        XCTAssertEqual(stoppedState, .stopped)
    }
}

private enum ProbeError: Error {
    case writeFailed
}

private struct FixedCameraAuthority: CameraAuthorizing {
    let result: CameraAuthorization

    init(_ result: CameraAuthorization) {
        self.result = result
    }

    func requestAccess() async -> CameraAuthorization { result }
}

private actor PersistenceProbe: FramePersistence {
    private(set) var frames: [CapturedFrame] = []
    private var continuations: [CheckedContinuation<Void, Never>] = []
    private let suspended: Bool
    private let error: Error?

    init(suspended: Bool = false, error: Error? = nil) {
        self.suspended = suspended
        self.error = error
    }

    var callCount: Int { frames.count }
    var timestamps: [CMTime] { frames.map(\.presentationTimestamp) }

    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame {
        frames.append(frame)
        if suspended {
            await withCheckedContinuation { continuation in
                continuations.append(continuation)
            }
        }
        try Task.checkCancellation()
        if let error { throw error }
        return PersistedFrame(
            index: frame.index,
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: String(format: "Frames/%08d.jpg", frame.index)
        )
    }

    func resumeOne() {
        guard !continuations.isEmpty else { return }
        continuations.removeFirst().resume()
    }
}

private actor FileMarkerPersistence: FramePersistence {
    let directory: URL

    init(directory: URL) {
        self.directory = directory
    }

    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame {
        let filename = String(format: "%08d.jpg", frame.index)
        try Data([0xFF, 0xD8, 0xFF, 0xD9]).write(to: directory.appending(path: filename))
        return PersistedFrame(
            index: frame.index,
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: "Frames/\(filename)"
        )
    }
}

private final class SyntheticCameraCaptureSession: CameraCaptureSession, @unchecked Sendable {
    let previewSession = AVCaptureSession()
    private let lock = NSLock()
    private let deviceIDs: Set<String>
    private var frameHandler: (@Sendable (CapturedFrame) -> Void)?
    private var disconnectHandler: (@Sendable () -> Void)?
    private var _connectedDeviceID: String?
    private var _isRunning = false
    private var _startCount = 0
    private var _stopCount = 0

    init(deviceIDs: Set<String>) {
        self.deviceIDs = deviceIDs
    }

    var connectedDeviceID: String? { lock.withLock { _connectedDeviceID } }
    var isConnected: Bool { connectedDeviceID != nil }
    var isRunning: Bool { lock.withLock { _isRunning } }
    var startCount: Int { lock.withLock { _startCount } }
    var stopCount: Int { lock.withLock { _stopCount } }
    var hasDisconnectObserver: Bool { lock.withLock { disconnectHandler != nil } }

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws {
        guard deviceIDs.contains(deviceID) else { throw CameraCaptureSessionError.deviceNotFound }
        lock.withLock {
            _connectedDeviceID = deviceID
            self.frameHandler = frameHandler
            self.disconnectHandler = disconnectHandler
        }
    }

    func startRunning() async throws {
        lock.withLock {
            _isRunning = true
            _startCount += 1
        }
    }

    func stopRunning() async {
        lock.withLock {
            if _isRunning || _connectedDeviceID != nil { _stopCount += 1 }
            _isRunning = false
            _connectedDeviceID = nil
            frameHandler = nil
            disconnectHandler = nil
        }
    }

    func emit(_ frame: CapturedFrame) {
        let handler = lock.withLock { _isRunning ? frameHandler : nil }
        handler?(frame)
    }

    func disconnect() {
        let handler = lock.withLock { disconnectHandler }
        handler?()
    }
}

private final class DelayedFailureCameraCaptureSession: CameraCaptureSession, @unchecked Sendable {
    let previewSession = AVCaptureSession()
    private let gate = DelayedConnectGate()

    var connectIsWaiting: Bool {
        get async { await gate.isWaiting }
    }

    var isRunning: Bool { false }

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws {
        await gate.wait()
        throw CameraCaptureSessionError.deviceNotFound
    }

    func startRunning() async throws {}
    func stopRunning() async {}

    func failConnect() async {
        await gate.resume()
    }
}

private actor DelayedConnectGate {
    private(set) var isWaiting = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        isWaiting = true
        await withCheckedContinuation { continuation = $0 }
    }

    func resume() {
        continuation?.resume()
        continuation = nil
    }
}

private final class PersistedFrameCollector: @unchecked Sendable {
    private let storage = EventStorage()
    private let task: Task<Void, Never>

    init(stream: AsyncStream<CameraFrameSourceEvent>) {
        let storage = self.storage
        task = Task {
            for await event in stream {
                await storage.append(event)
            }
        }
    }

    var persisted: [PersistedFrame] {
        get async { await storage.persisted }
    }

    var drops: [CameraDroppedFrame] {
        get async { await storage.drops }
    }

    func cancel() { task.cancel() }
}

private actor EventStorage {
    private(set) var persisted: [PersistedFrame] = []
    private(set) var drops: [CameraDroppedFrame] = []

    func append(_ event: CameraFrameSourceEvent) {
        switch event {
        case let .persisted(frame): persisted.append(frame)
        case let .dropped(frame): drops.append(frame)
        }
    }
}

private func makeFrame(sequence: Int, timestamp: CMTime) throws -> CapturedFrame {
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        2,
        2,
        kCVPixelFormatType_32BGRA,
        nil,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pixelBuffer else {
        throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
    }
    return CapturedFrame(
        index: sequence,
        presentationTimestamp: timestamp,
        pixelBuffer: pixelBuffer,
        orientation: .identity,
        sourceSampleSequence: sequence
    )
}

private func eventually(
    attempts: Int = 2_000,
    _ condition: @escaping @Sendable () async -> Bool
) async throws {
    for _ in 0..<attempts {
        if await condition() { return }
        await Task.yield()
    }
    XCTFail("Condition did not become true")
}

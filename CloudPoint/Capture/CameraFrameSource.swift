@preconcurrency import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

struct CameraDescriptor: Sendable, Equatable, Identifiable {
    enum Kind: String, Sendable {
        case builtIn
        case continuity
        case external
        case other
    }

    let id: String
    let name: String
    let kind: Kind
}

enum CameraCatalog {
    static func devices() async -> [CameraDescriptor] {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        var seen = Set<String>()
        return discovery.devices
            .compactMap { device -> CameraDescriptor? in
                guard seen.insert(device.uniqueID).inserted else { return nil }
                let kind: CameraDescriptor.Kind
                switch device.deviceType {
                case .builtInWideAngleCamera: kind = .builtIn
                case .continuityCamera: kind = .continuity
                case .external: kind = .external
                default: kind = .other
                }
                return CameraDescriptor(id: device.uniqueID, name: device.localizedName, kind: kind)
            }
            .sorted {
                if $0.name == $1.name { return $0.id < $1.id }
                return $0.name.localizedStandardCompare($1.name) == .orderedAscending
            }
    }
}

struct CameraSampleGate: Sendable {
    private let interval: CMTime
    private var lastTimestamp: CMTime?
    private var nextTarget: CMTime?

    init(rate: Int) throws {
        _ = try FrameSamplingPlan(
            duration: CMTime(value: 1, timescale: 1),
            framesPerSecond: rate
        )
        interval = CMTime(
            value: CMTimeValue(FrameSamplingPlan.timescale) / CMTimeValue(rate),
            timescale: FrameSamplingPlan.timescale
        )
    }

    mutating func accepts(_ timestamp: CMTime) -> Bool {
        guard timestamp.isNumeric,
              CMTimeCompare(timestamp, .zero) >= 0 else {
            return false
        }
        if let lastTimestamp,
           CMTimeCompare(timestamp, lastTimestamp) <= 0 {
            return false
        }
        self.lastTimestamp = timestamp

        guard var target = nextTarget else {
            nextTarget = CMTimeAdd(timestamp, interval)
            return true
        }
        guard CMTimeCompare(timestamp, target) >= 0 else { return false }

        repeat {
            target = CMTimeAdd(target, interval)
        } while CMTimeCompare(target, timestamp) <= 0
        nextTarget = target
        return true
    }

    mutating func reset() {
        lastTimestamp = nil
        nextTarget = nil
    }
}

enum CameraAuthorization: Sendable {
    case authorized
    case denied
}

protocol CameraAuthorizing: Sendable {
    func requestAccess() async -> CameraAuthorization
}

struct SystemCameraAuthority: CameraAuthorizing {
    func requestAccess() async -> CameraAuthorization {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return .authorized
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video) ? .authorized : .denied
        case .denied, .restricted:
            return .denied
        @unknown default:
            return .denied
        }
    }
}

enum CameraCaptureSessionError: Error, Sendable {
    case deviceNotFound
    case cannotCreateInput
    case cannotAddInput
    case cannotAddOutput
}

protocol CameraCaptureSession: Sendable {
    var previewSession: AVCaptureSession { get }

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws
    func startRunning() async throws
    func stopRunning() async
}

enum CameraFrameSourceFailure: Error, Sendable, Equatable {
    case authorizationDenied
    case invalidSampleRate
    case deviceNotFound
    case configurationFailed
    case cameraDisconnected
    case persistenceFailed
}

enum CameraFrameSourceState: Sendable, Equatable {
    case idle
    case starting
    case running(deviceID: String)
    case stopped
    case failed(CameraFrameSourceFailure)
}

enum CameraFrameDropReason: Sendable, Equatable {
    case persistenceBusy
}

struct CameraDroppedFrame: Sendable, Equatable {
    let timestamp: CMTime
    let reason: CameraFrameDropReason
}

enum CameraFrameSourceEvent: Sendable, Equatable {
    case persisted(PersistedFrame)
    case dropped(CameraDroppedFrame)
}

struct CameraFrameSourceMetrics: Sendable, Equatable {
    var previewFrameCount = 0
    var acceptedPersistenceCount = 0
    var droppedPersistenceCount = 0
}

actor CameraFrameSource {
    nonisolated let previewSession: AVCaptureSession

    private(set) var state: CameraFrameSourceState = .idle
    private let persistence: any FramePersistence
    private let authority: any CameraAuthorizing
    private let captureSession: any CameraCaptureSession
    private nonisolated let eventStream: AsyncStream<CameraFrameSourceEvent>
    private nonisolated let eventContinuation: AsyncStream<CameraFrameSourceEvent>.Continuation

    private var ingress: CameraFrameIngress?
    private var persistenceTask: Task<Void, Never>?
    private var lastMetrics = CameraFrameSourceMetrics()
    private var lifecycleID = 0

    init(
        persistence: any FramePersistence,
        authority: any CameraAuthorizing = SystemCameraAuthority(),
        captureSession: any CameraCaptureSession = AVCameraCaptureSession()
    ) {
        self.persistence = persistence
        self.authority = authority
        self.captureSession = captureSession
        previewSession = captureSession.previewSession
        let stream = AsyncStream.makeStream(
            of: CameraFrameSourceEvent.self,
            bufferingPolicy: .unbounded
        )
        eventStream = stream.stream
        eventContinuation = stream.continuation
    }

    nonisolated func events() -> AsyncStream<CameraFrameSourceEvent> {
        eventStream
    }

    var metrics: CameraFrameSourceMetrics {
        ingress?.metrics ?? lastMetrics
    }

    func start(deviceID: String, sampleRate: Int) async {
        if case .running = state { return }

        let gate: CameraSampleGate
        do {
            gate = try CameraSampleGate(rate: sampleRate)
        } catch {
            state = .failed(.invalidSampleRate)
            return
        }

        lifecycleID += 1
        let currentLifecycle = lifecycleID
        state = .starting
        guard await authority.requestAccess() == .authorized else {
            guard currentLifecycle == lifecycleID else { return }
            state = .failed(.authorizationDenied)
            return
        }
        guard currentLifecycle == lifecycleID else { return }

        let stream = AsyncStream.makeStream(
            of: CapturedFrame.self,
            bufferingPolicy: .bufferingOldest(1)
        )
        let ingress = CameraFrameIngress(
            gate: gate,
            continuation: stream.continuation,
            eventContinuation: eventContinuation
        )
        self.ingress = ingress
        lastMetrics = CameraFrameSourceMetrics()
        persistenceTask = makePersistenceTask(
            stream: stream.stream,
            lifecycleID: currentLifecycle
        )

        do {
            try await captureSession.connect(
                deviceID: deviceID,
                frameHandler: { frame in ingress.receive(frame) },
                disconnectHandler: { [weak self] in
                    Task { await self?.cameraDisconnected(lifecycleID: currentLifecycle) }
                }
            )
            guard currentLifecycle == lifecycleID else {
                await captureSession.stopRunning()
                return
            }
            try await captureSession.startRunning()
            guard currentLifecycle == lifecycleID else {
                await captureSession.stopRunning()
                return
            }
            state = .running(deviceID: deviceID)
        } catch let error as CameraCaptureSessionError {
            guard currentLifecycle == lifecycleID else { return }
            await endCapture(lifecycleID: currentLifecycle)
            state = .failed(error == .deviceNotFound ? .deviceNotFound : .configurationFailed)
        } catch {
            guard currentLifecycle == lifecycleID else { return }
            await endCapture(lifecycleID: currentLifecycle)
            state = .failed(.configurationFailed)
        }
    }

    func stop() async {
        switch state {
        case .stopped:
            return
        case .idle:
            state = .stopped
            return
        default:
            break
        }
        let endingLifecycle = lifecycleID
        lifecycleID += 1
        await endCapture(lifecycleID: endingLifecycle)
        state = .stopped
    }

    private func makePersistenceTask(
        stream: AsyncStream<CapturedFrame>,
        lifecycleID: Int
    ) -> Task<Void, Never> {
        let persistence = self.persistence
        return Task { [weak self] in
            do {
                for await frame in stream {
                    try Task.checkCancellation()
                    let persisted = try await persistence.persist(frame)
                    try Task.checkCancellation()
                    await self?.didPersist(persisted, lifecycleID: lifecycleID)
                }
            } catch is CancellationError {
                return
            } catch {
                await self?.persistenceFailed(lifecycleID: lifecycleID)
            }
        }
    }

    private func didPersist(_ frame: PersistedFrame, lifecycleID: Int) {
        guard lifecycleID == self.lifecycleID else { return }
        eventContinuation.yield(.persisted(frame))
    }

    private func persistenceFailed(lifecycleID: Int) async {
        guard lifecycleID == self.lifecycleID else { return }
        self.lifecycleID += 1
        await endCapture(lifecycleID: lifecycleID)
        state = .failed(.persistenceFailed)
    }

    private func cameraDisconnected(lifecycleID: Int) async {
        guard lifecycleID == self.lifecycleID else { return }
        self.lifecycleID += 1
        await endCapture(lifecycleID: lifecycleID)
        state = .failed(.cameraDisconnected)
    }

    private func endCapture(lifecycleID: Int) async {
        guard lifecycleID <= self.lifecycleID else { return }
        if let ingress {
            lastMetrics = ingress.metrics
            ingress.finish()
            self.ingress = nil
        }
        persistenceTask?.cancel()
        persistenceTask = nil
        await captureSession.stopRunning()
    }
}

private final class CameraFrameIngress: @unchecked Sendable {
    private let lock = NSLock()
    private var gate: CameraSampleGate
    private var continuation: AsyncStream<CapturedFrame>.Continuation?
    private let eventContinuation: AsyncStream<CameraFrameSourceEvent>.Continuation
    private var nextFrameIndex = 0
    private var currentMetrics = CameraFrameSourceMetrics()

    init(
        gate: CameraSampleGate,
        continuation: AsyncStream<CapturedFrame>.Continuation,
        eventContinuation: AsyncStream<CameraFrameSourceEvent>.Continuation
    ) {
        self.gate = gate
        self.continuation = continuation
        self.eventContinuation = eventContinuation
    }

    var metrics: CameraFrameSourceMetrics {
        lock.withLock { currentMetrics }
    }

    func receive(_ frame: CapturedFrame) {
        lock.withLock {
            guard let continuation else { return }
            currentMetrics.previewFrameCount += 1
            guard gate.accepts(frame.presentationTimestamp) else { return }
            let sampledFrame = CapturedFrame(
                index: nextFrameIndex,
                presentationTimestamp: frame.presentationTimestamp,
                pixelBuffer: frame.pixelBuffer,
                orientation: frame.orientation,
                sourceSampleSequence: frame.sourceSampleSequence
            )
            switch continuation.yield(sampledFrame) {
            case .enqueued:
                nextFrameIndex += 1
                currentMetrics.acceptedPersistenceCount += 1
            case .dropped:
                currentMetrics.droppedPersistenceCount += 1
                eventContinuation.yield(
                    .dropped(
                        CameraDroppedFrame(
                            timestamp: frame.presentationTimestamp,
                            reason: .persistenceBusy
                        )
                    )
                )
            case .terminated:
                self.continuation = nil
            @unknown default:
                self.continuation = nil
            }
        }
    }

    func finish() {
        lock.withLock {
            continuation?.finish()
            continuation = nil
        }
    }
}

final class AVCameraCaptureSession: NSObject, CameraCaptureSession, @unchecked Sendable {
    let previewSession = AVCaptureSession()

    private let configurationQueue = DispatchQueue(label: "cloud.point.cloud.camera.session")
    private let delegateQueue = DispatchQueue(label: "cloud.point.cloud.camera.frames")
    private var output: AVCaptureVideoDataOutput?
    private var outputDelegate: CameraVideoOutputDelegate?
    private var notificationTokens: [NSObjectProtocol] = []

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            configurationQueue.async { [self] in
                do {
                    try configure(
                        deviceID: deviceID,
                        frameHandler: frameHandler,
                        disconnectHandler: disconnectHandler
                    )
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func startRunning() async throws {
        await withCheckedContinuation { continuation in
            configurationQueue.async { [self] in
                if !previewSession.isRunning { previewSession.startRunning() }
                continuation.resume()
            }
        }
    }

    func stopRunning() async {
        await withCheckedContinuation { continuation in
            configurationQueue.async { [self] in
                tearDown()
                continuation.resume()
            }
        }
    }

    deinit {
        output?.setSampleBufferDelegate(nil, queue: nil)
        for token in notificationTokens {
            NotificationCenter.default.removeObserver(token)
        }
    }

    private func configure(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) throws {
        tearDown()
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        guard let device = discovery.devices.first(where: { $0.uniqueID == deviceID }) else {
            throw CameraCaptureSessionError.deviceNotFound
        }
        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            throw CameraCaptureSessionError.cannotCreateInput
        }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        let outputDelegate = CameraVideoOutputDelegate(handler: frameHandler)

        previewSession.beginConfiguration()
        defer { previewSession.commitConfiguration() }
        guard previewSession.canAddInput(input) else {
            throw CameraCaptureSessionError.cannotAddInput
        }
        previewSession.addInput(input)
        guard previewSession.canAddOutput(output) else {
            previewSession.removeInput(input)
            throw CameraCaptureSessionError.cannotAddOutput
        }
        previewSession.addOutput(output)
        output.setSampleBufferDelegate(outputDelegate, queue: delegateQueue)
        probeIntrinsicMatrixDelivery(on: output.connection(with: .video))

        let token = NotificationCenter.default.addObserver(
            forName: AVCaptureDevice.wasDisconnectedNotification,
            object: device,
            queue: nil
        ) { _ in
            disconnectHandler()
        }
        notificationTokens = [token]
        self.output = output
        self.outputDelegate = outputDelegate
    }

    private func tearDown() {
        output?.setSampleBufferDelegate(nil, queue: nil)
        for token in notificationTokens {
            NotificationCenter.default.removeObserver(token)
        }
        notificationTokens.removeAll()
        if previewSession.isRunning { previewSession.stopRunning() }
        previewSession.beginConfiguration()
        for output in previewSession.outputs { previewSession.removeOutput(output) }
        for input in previewSession.inputs { previewSession.removeInput(input) }
        previewSession.commitConfiguration()
        output = nil
        outputDelegate = nil
    }

    private func probeIntrinsicMatrixDelivery(on connection: AVCaptureConnection?) {
        guard let connection else { return }
        let supportedSelector = NSSelectorFromString("isCameraIntrinsicMatrixDeliverySupported")
        let enabledSelector = NSSelectorFromString("setCameraIntrinsicMatrixDeliveryEnabled:")
        guard connection.responds(to: supportedSelector),
              connection.responds(to: enabledSelector),
              (connection.value(forKey: "cameraIntrinsicMatrixDeliverySupported") as? Bool) == true else {
            return
        }
        connection.setValue(true, forKey: "cameraIntrinsicMatrixDeliveryEnabled")
    }
}

private final class CameraVideoOutputDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
    @unchecked Sendable
{
    private let handler: @Sendable (CapturedFrame) -> Void
    private var sequence = 0

    init(handler: @escaping @Sendable (CapturedFrame) -> Void) {
        self.handler = handler
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let frame = CapturedFrame(
            index: sequence,
            presentationTimestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
            pixelBuffer: pixelBuffer,
            orientation: .identity,
            sourceSampleSequence: sequence
        )
        sequence += 1
        handler(frame)
    }
}

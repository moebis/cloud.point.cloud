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
    private let rate: CMTimeScale
    private let interval: CMTime
    private var anchorTimestamp: CMTime?
    private var lastTimestamp: CMTime?
    private var nextTarget: CMTime?

    init(rate: Int) throws {
        _ = try FrameSamplingPlan(
            duration: CMTime(value: 1, timescale: 1),
            framesPerSecond: rate
        )
        self.rate = CMTimeScale(rate)
        interval = CMTime(value: 1, timescale: CMTimeScale(rate))
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

        guard let anchorTimestamp, let target = nextTarget else {
            anchorTimestamp = timestamp
            nextTarget = CMTimeAdd(timestamp, interval)
            return true
        }
        guard CMTimeCompare(timestamp, target) >= 0 else { return false }

        let elapsed = CMTimeSubtract(timestamp, anchorTimestamp)
        let elapsedIntervals = CMTimeConvertScale(
            CMTimeMultiply(elapsed, multiplier: rate),
            timescale: 1,
            method: .roundTowardNegativeInfinity
        )
        guard elapsedIntervals.isNumeric,
              elapsedIntervals.value < CMTimeValue.max else {
            nextTarget = .positiveInfinity
            return true
        }
        nextTarget = CMTimeAdd(
            anchorTimestamp,
            CMTime(value: elapsedIntervals.value + 1, timescale: rate)
        )
        return true
    }

    mutating func reset() {
        anchorTimestamp = nil
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
    func shutdown() async
}

extension CameraCaptureSession {
    func shutdown() async {
        await stopRunning()
    }
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

struct CameraPersistedFrame: Sendable, Equatable {
    let lifecycleID: UInt64
    let sequence: UInt64
    let frame: PersistedFrame
}

struct CameraFrameSourceFailureEvent: Sendable, Equatable {
    let lifecycleID: UInt64
    let sequence: UInt64
    let failure: CameraFrameSourceFailure
}

struct CameraLifecycleCompletion: Sendable, Equatable {
    let lifecycleID: UInt64
    let durablePersistedEventCount: UInt64
    let terminalFailure: CameraFrameSourceFailure?
    let durablePersistedEvents: [CameraPersistedFrame]

    init(
        lifecycleID: UInt64,
        durablePersistedEventCount: UInt64,
        terminalFailure: CameraFrameSourceFailure?,
        durablePersistedEvents: [CameraPersistedFrame] = []
    ) {
        self.lifecycleID = lifecycleID
        self.durablePersistedEventCount = durablePersistedEventCount
        self.terminalFailure = terminalFailure
        self.durablePersistedEvents = durablePersistedEvents
    }
}

enum CameraFrameSourceEvent: Sendable, Equatable {
    case persisted(CameraPersistedFrame)
    case dropped(CameraDroppedFrame)
    case failed(CameraFrameSourceFailureEvent)
}

struct CameraFrameSourceMetrics: Sendable, Equatable {
    var previewFrameCount = 0
    var acceptedPersistenceCount = 0
    var droppedPersistenceCount = 0
}

protocol CameraPreviewObserving: Sendable {
    /// Called synchronously from capture ingress; implementations must return promptly.
    func observe(timestamp: CMTime)
}

private struct NoopCameraPreviewObserver: CameraPreviewObserving {
    func observe(timestamp: CMTime) {}
}

actor CameraFrameSource {
    private enum ShutdownState {
        case active
        case shuttingDown
        case shutDown
    }

    private struct ActiveConfiguration: Equatable {
        let deviceID: String
        let sampleRate: Int
    }

    private struct DetachedCapture: Sendable {
        let ingress: CameraFrameIngress?
        let lifecycleID: UInt64?
        let persistenceTask: Task<CameraLifecycleCompletion, Never>?
        let terminalFailure: CameraFrameSourceFailure?

        var needsSessionStop: Bool {
            ingress != nil || persistenceTask != nil
        }

        var fallbackCompletion: CameraLifecycleCompletion? {
            lifecycleID.map {
                CameraLifecycleCompletion(
                    lifecycleID: $0,
                    durablePersistedEventCount: 0,
                    terminalFailure: terminalFailure
                )
            }
        }
    }

    nonisolated let previewSession: AVCaptureSession

    private(set) var state: CameraFrameSourceState = .idle
    private let persistence: any FramePersistence
    private let authority: any CameraAuthorizing
    private let captureSession: any CameraCaptureSession
    private let previewObserver: any CameraPreviewObserving
    private let startingFrameIndex: Int
    private nonisolated let eventChannel = CameraEventChannel(capacity: 16)

    private var ingress: CameraFrameIngress?
    private var persistenceTask: Task<CameraLifecycleCompletion, Never>?
    private var lastMetrics = CameraFrameSourceMetrics()
    private var lifecycleID: UInt64 = 0
    private var activeLifecycleID: UInt64?
    private var lastCompletion: CameraLifecycleCompletion?
    private var activeConfiguration: ActiveConfiguration?
    private var shutdownState = ShutdownState.active
    private var shutdownWaiters: [CheckedContinuation<Void, Never>] = []
    private var retirementBarrier: Task<Void, Never>?

    init(
        persistence: any FramePersistence,
        authority: any CameraAuthorizing = SystemCameraAuthority(),
        captureSession: any CameraCaptureSession = AVCameraCaptureSession(),
        previewObserver: any CameraPreviewObserving = NoopCameraPreviewObserver(),
        startingFrameIndex: UInt32 = 0
    ) {
        self.persistence = persistence
        self.authority = authority
        self.captureSession = captureSession
        self.previewObserver = previewObserver
        self.startingFrameIndex = Int(startingFrameIndex)
        previewSession = captureSession.previewSession
    }

    /// Returns the sole authoritative event sequence. A second active sequence
    /// completes immediately so persisted notifications cannot be load-balanced.
    nonisolated func events() -> CameraFrameSourceEvents {
        eventChannel.stream()
    }

    var metrics: CameraFrameSourceMetrics {
        ingress?.metrics ?? lastMetrics
    }

    func start(deviceID: String, sampleRate: Int) async {
        guard case .active = shutdownState else { return }
        let requestedConfiguration = ActiveConfiguration(
            deviceID: deviceID,
            sampleRate: sampleRate
        )
        if case .running = state,
           activeConfiguration == requestedConfiguration {
            return
        }

        lifecycleID += 1
        let currentLifecycle = lifecycleID
        activeConfiguration = nil
        let previousCapture = detachCapture()
        activeLifecycleID = currentLifecycle
        state = .starting

        let gate: CameraSampleGate
        do {
            gate = try CameraSampleGate(rate: sampleRate)
        } catch {
            recordCompletion(await retire(previousCapture))
            guard currentLifecycle == lifecycleID else { return }
            recordEarlyFailure(.invalidSampleRate, lifecycleID: currentLifecycle)
            state = .failed(.invalidSampleRate)
            return
        }

        recordCompletion(await retire(previousCapture))
        guard currentLifecycle == lifecycleID else { return }

        guard await authority.requestAccess() == .authorized else {
            guard currentLifecycle == lifecycleID else { return }
            recordEarlyFailure(.authorizationDenied, lifecycleID: currentLifecycle)
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
            eventChannel: eventChannel,
            previewObserver: previewObserver,
            startingFrameIndex: startingFrameIndex
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
            guard currentLifecycle == lifecycleID else { return }
            try await captureSession.startRunning()
            guard currentLifecycle == lifecycleID else { return }
            activeConfiguration = requestedConfiguration
            state = .running(deviceID: deviceID)
        } catch let error as CameraCaptureSessionError {
            guard currentLifecycle == lifecycleID else { return }
            await fail(
                error == .deviceNotFound ? .deviceNotFound : .configurationFailed,
                lifecycleID: currentLifecycle
            )
        } catch {
            guard currentLifecycle == lifecycleID else { return }
            await fail(.configurationFailed, lifecycleID: currentLifecycle)
        }
    }

    func stop() async -> CameraLifecycleCompletion {
        switch shutdownState {
        case .shuttingDown:
            await waitForShutdown()
            return lastCompletion ?? emptyCompletion()
        case .shutDown:
            return lastCompletion ?? emptyCompletion()
        case .active:
            break
        }
        if case .stopped = state { return lastCompletion ?? emptyCompletion() }
        lifecycleID += 1
        let stoppedLifecycle = lifecycleID
        activeConfiguration = nil
        let capture = detachCapture()
        recordCompletion(await retire(capture))
        guard stoppedLifecycle == lifecycleID else {
            return lastCompletion ?? emptyCompletion()
        }
        state = .stopped
        return lastCompletion ?? emptyCompletion()
    }

    func shutdown() async {
        switch shutdownState {
        case .shuttingDown:
            await waitForShutdown()
            return
        case .shutDown:
            return
        case .active:
            shutdownState = .shuttingDown
        }

        lifecycleID += 1
        activeConfiguration = nil
        let capture = detachCapture()
        let previousRetirement = retirementBarrier
        let captureSession = self.captureSession
        let shutdownTask = Task {
            await previousRetirement?.value
            await captureSession.shutdown()
            _ = await capture.persistenceTask?.value
        }
        retirementBarrier = shutdownTask
        await shutdownTask.value
        recordCompletion(await completion(for: capture))
        eventChannel.finish()
        state = .stopped
        shutdownState = .shutDown
        let waiters = shutdownWaiters
        shutdownWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    private func makePersistenceTask(
        stream: AsyncStream<CapturedFrame>,
        lifecycleID: UInt64
    ) -> Task<CameraLifecycleCompletion, Never> {
        let persistence = self.persistence
        let eventProducer = eventChannel.makeProducer()
        return Task { [weak self] in
            defer { eventProducer.finish() }
            var durablePersistedEventCount: UInt64 = 0
            var durablePersistedEvents: [CameraPersistedFrame] = []
            do {
                for await frame in stream {
                    try Task.checkCancellation()
                    guard await eventProducer.prepareForPersistence() else {
                        return CameraLifecycleCompletion(
                            lifecycleID: lifecycleID,
                            durablePersistedEventCount: durablePersistedEventCount,
                            terminalFailure: nil,
                            durablePersistedEvents: durablePersistedEvents
                        )
                    }
                    do {
                        let persisted = try await persistence.persist(frame)
                        let event = CameraPersistedFrame(
                            lifecycleID: lifecycleID,
                            sequence: durablePersistedEventCount,
                            frame: persisted
                        )
                        await eventProducer.send(.persisted(event))
                        durablePersistedEvents.append(event)
                        durablePersistedEventCount += 1
                    } catch {
                        eventProducer.cancelPersistencePermit()
                        throw error
                    }
                }
            } catch is CancellationError {
                return CameraLifecycleCompletion(
                    lifecycleID: lifecycleID,
                    durablePersistedEventCount: durablePersistedEventCount,
                    terminalFailure: nil,
                    durablePersistedEvents: durablePersistedEvents
                )
            } catch {
                Task { [weak self] in
                    await self?.persistenceFailed(lifecycleID: lifecycleID)
                }
                return CameraLifecycleCompletion(
                    lifecycleID: lifecycleID,
                    durablePersistedEventCount: durablePersistedEventCount,
                    terminalFailure: .persistenceFailed,
                    durablePersistedEvents: durablePersistedEvents
                )
            }
            return CameraLifecycleCompletion(
                lifecycleID: lifecycleID,
                durablePersistedEventCount: durablePersistedEventCount,
                terminalFailure: nil,
                durablePersistedEvents: durablePersistedEvents
            )
        }
    }

    private func persistenceFailed(lifecycleID: UInt64) async {
        guard lifecycleID == self.lifecycleID else { return }
        self.lifecycleID += 1
        let failedLifecycle = self.lifecycleID
        activeConfiguration = nil
        let capture = detachCapture(terminalFailure: .persistenceFailed)
        let completion = await retire(capture)
        recordCompletion(completion)
        guard failedLifecycle == self.lifecycleID else { return }
        publishRuntimeFailure(.persistenceFailed, completion: completion)
        state = .failed(.persistenceFailed)
    }

    private func cameraDisconnected(lifecycleID: UInt64) async {
        guard lifecycleID == self.lifecycleID else { return }
        self.lifecycleID += 1
        let failedLifecycle = self.lifecycleID
        activeConfiguration = nil
        let capture = detachCapture(terminalFailure: .cameraDisconnected)
        let completion = await retire(capture)
        recordCompletion(completion)
        guard failedLifecycle == self.lifecycleID else { return }
        publishRuntimeFailure(.cameraDisconnected, completion: completion)
        state = .failed(.cameraDisconnected)
    }

    private func publishRuntimeFailure(
        _ failure: CameraFrameSourceFailure,
        completion: CameraLifecycleCompletion?
    ) {
        guard let completion else { return }
        eventChannel.offerDurableTerminal(
            .failed(
                CameraFrameSourceFailureEvent(
                    lifecycleID: completion.lifecycleID,
                    sequence: completion.durablePersistedEventCount,
                    failure: failure
                )
            )
        )
    }

    private func fail(_ failure: CameraFrameSourceFailure, lifecycleID: UInt64) async {
        guard lifecycleID == self.lifecycleID else { return }
        self.lifecycleID += 1
        let failedLifecycle = self.lifecycleID
        activeConfiguration = nil
        let capture = detachCapture(terminalFailure: failure)
        recordCompletion(await retire(capture))
        guard failedLifecycle == self.lifecycleID else { return }
        state = .failed(failure)
    }

    private func detachCapture(
        terminalFailure: CameraFrameSourceFailure? = nil
    ) -> DetachedCapture {
        let currentIngress = ingress
        if let currentIngress {
            lastMetrics = currentIngress.metrics
            currentIngress.finish()
            self.ingress = nil
        }
        let capture = DetachedCapture(
            ingress: currentIngress,
            lifecycleID: activeLifecycleID,
            persistenceTask: persistenceTask,
            terminalFailure: terminalFailure
        )
        activeLifecycleID = nil
        persistenceTask = nil
        capture.persistenceTask?.cancel()
        return capture
    }

    private func retire(_ capture: DetachedCapture) async -> CameraLifecycleCompletion? {
        let previousRetirement = retirementBarrier
        guard capture.needsSessionStop else {
            await previousRetirement?.value
            return capture.fallbackCompletion
        }
        let captureSession = self.captureSession
        let retirement = Task {
            await previousRetirement?.value
            await captureSession.stopRunning()
            _ = await capture.persistenceTask?.value
        }
        retirementBarrier = retirement
        await retirement.value
        return await completion(for: capture)
    }

    private func completion(
        for capture: DetachedCapture
    ) async -> CameraLifecycleCompletion? {
        let base = await capture.persistenceTask?.value ?? capture.fallbackCompletion
        guard let base, let terminalFailure = capture.terminalFailure else { return base }
        return CameraLifecycleCompletion(
            lifecycleID: base.lifecycleID,
            durablePersistedEventCount: base.durablePersistedEventCount,
            terminalFailure: terminalFailure,
            durablePersistedEvents: base.durablePersistedEvents
        )
    }

    private func recordCompletion(_ completion: CameraLifecycleCompletion?) {
        guard let completion else { return }
        lastCompletion = completion
    }

    private func recordEarlyFailure(
        _ failure: CameraFrameSourceFailure,
        lifecycleID: UInt64
    ) {
        activeLifecycleID = nil
        lastCompletion = CameraLifecycleCompletion(
            lifecycleID: lifecycleID,
            durablePersistedEventCount: 0,
            terminalFailure: failure
        )
    }

    private func emptyCompletion() -> CameraLifecycleCompletion {
        CameraLifecycleCompletion(
            lifecycleID: lifecycleID,
            durablePersistedEventCount: 0,
            terminalFailure: nil
        )
    }

    private func waitForShutdown() async {
        guard case .shuttingDown = shutdownState else { return }
        await withCheckedContinuation { shutdownWaiters.append($0) }
    }

    deinit {
        ingress?.finish()
        persistenceTask?.cancel()
        eventChannel.finish()
    }
}

private final class CameraFrameIngress: @unchecked Sendable {
    private let lock = NSLock()
    private var gate: CameraSampleGate
    private var continuation: AsyncStream<CapturedFrame>.Continuation?
    private let eventChannel: CameraEventChannel
    private let previewObserver: any CameraPreviewObserving
    private var nextFrameIndex: Int
    private var currentMetrics = CameraFrameSourceMetrics()

    init(
        gate: CameraSampleGate,
        continuation: AsyncStream<CapturedFrame>.Continuation,
        eventChannel: CameraEventChannel,
        previewObserver: any CameraPreviewObserving,
        startingFrameIndex: Int = 0
    ) {
        self.gate = gate
        self.continuation = continuation
        self.eventChannel = eventChannel
        self.previewObserver = previewObserver
        nextFrameIndex = startingFrameIndex
    }

    var metrics: CameraFrameSourceMetrics {
        lock.withLock { currentMetrics }
    }

    func receive(_ frame: CapturedFrame) {
        var shouldObservePreview = false
        lock.withLock {
            guard let continuation else { return }
            shouldObservePreview = true
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
                eventChannel.offerTelemetry(
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
        if shouldObservePreview {
            previewObserver.observe(timestamp: frame.presentationTimestamp)
        }
    }

    func finish() {
        lock.withLock {
            continuation?.finish()
            continuation = nil
        }
    }
}

struct CameraFrameSourceEvents: AsyncSequence, Sendable {
    typealias Element = CameraFrameSourceEvent

    struct AsyncIterator: AsyncIteratorProtocol {
        private let subscription: CameraEventSubscription

        fileprivate init(subscription: CameraEventSubscription) {
            self.subscription = subscription
        }

        mutating func next() async -> CameraFrameSourceEvent? {
            await subscription.next()
        }
    }

    private let subscription: CameraEventSubscription

    fileprivate init(subscription: CameraEventSubscription) {
        self.subscription = subscription
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(subscription: subscription)
    }
}

final class CameraEventChannel: @unchecked Sendable {
    // The bounded buffer may have one channel-owned overflow event and one
    // lifecycle-terminal event behind it. A persistence permit prevents a
    // replacement lifecycle from overtaking either durable tail event.
    private struct PendingSend {
        let id: UUID
        let event: CameraFrameSourceEvent
        var continuation: CheckedContinuation<Void, Never>?
    }

    private struct WaitingPersistencePermit {
        let id: UUID
        let producerID: UUID
        let continuation: CheckedContinuation<Bool, Never>
    }

    private struct WaitingConsumer {
        let subscriptionID: UUID
        let continuation: CheckedContinuation<CameraFrameSourceEvent?, Never>
    }

    private let lock = NSLock()
    private let capacity: Int
    private var buffer: [CameraFrameSourceEvent] = []
    private var pendingSend: PendingSend?
    private var durableTerminal: CameraFrameSourceEvent?
    private var waitingConsumer: WaitingConsumer?
    private var activeSubscriptionID: UUID?
    private var activeProducerIDs = Set<UUID>()
    private var persistencePermitProducerID: UUID?
    private var waitingPersistencePermit: WaitingPersistencePermit?
    private var finishRequested = false

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
    }

    func stream() -> CameraFrameSourceEvents {
        let subscriptionID = lock.withLock { () -> UUID? in
            guard activeSubscriptionID == nil else { return nil }
            let id = UUID()
            activeSubscriptionID = id
            return id
        }
        return CameraFrameSourceEvents(
            subscription: CameraEventSubscription(channel: self, id: subscriptionID)
        )
    }

    func makeProducer() -> CameraEventProducer {
        let producerID = lock.withLock { () -> UUID? in
            guard !finishRequested else { return nil }
            let id = UUID()
            activeProducerIDs.insert(id)
            return id
        }
        return CameraEventProducer(channel: self, id: producerID)
    }

    /// Drop events are telemetry. They are retained only while the sole
    /// consumer has spare queue capacity and never displace persisted events.
    func offerTelemetry(_ event: CameraFrameSourceEvent) {
        var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
        lock.withLock {
            guard !finishRequested,
                  durableTerminal == nil,
                  activeSubscriptionID != nil else { return }
            if let waitingConsumer {
                self.waitingConsumer = nil
                consumer = waitingConsumer.continuation
            } else if buffer.count < capacity {
                buffer.append(event)
            }
        }
        consumer?.resume(returning: event)
    }

    /// Retains the one terminal event for a failed lifecycle after every
    /// buffered and overflow durable event, including across channel finish.
    func offerDurableTerminal(_ event: CameraFrameSourceEvent) {
        guard case .failed = event else {
            preconditionFailure("Only camera failures may occupy the terminal slot")
        }
        var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
        lock.withLock {
            guard !finishRequested else { return }
            precondition(
                durableTerminal == nil,
                "A camera lifecycle may publish exactly one terminal failure"
            )
            if buffer.isEmpty, pendingSend == nil, let waitingConsumer {
                self.waitingConsumer = nil
                consumer = waitingConsumer.continuation
            } else {
                durableTerminal = event
            }
        }
        consumer?.resume(returning: event)
    }

    func finish() {
        var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
        var sender: CheckedContinuation<Void, Never>?
        var permitWaiter: CheckedContinuation<Bool, Never>?
        lock.withLock {
            guard !finishRequested else { return }
            finishRequested = true
            if var pendingSend {
                sender = pendingSend.continuation
                pendingSend.continuation = nil
                self.pendingSend = pendingSend
            }
            permitWaiter = waitingPersistencePermit?.continuation
            waitingPersistencePermit = nil
            if isTerminalLocked {
                consumer = waitingConsumer?.continuation
                waitingConsumer = nil
            }
        }
        sender?.resume()
        permitWaiter?.resume(returning: false)
        consumer?.resume(returning: nil)
    }

    fileprivate func next(subscriptionID: UUID) async -> CameraFrameSourceEvent? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                var result: CameraFrameSourceEvent??
                var sender: CheckedContinuation<Void, Never>?
                var permitWaiter: CheckedContinuation<Bool, Never>?
                lock.withLock {
                    guard activeSubscriptionID == subscriptionID, !Task.isCancelled else {
                        result = .some(nil)
                        return
                    }
                    if !buffer.isEmpty {
                        result = .some(buffer.removeFirst())
                        if let pending = pendingSend {
                            pendingSend = nil
                            buffer.append(pending.event)
                            sender = pending.continuation
                            permitWaiter = promotePersistencePermitLocked()
                        }
                    } else if let pending = pendingSend {
                        pendingSend = nil
                        result = .some(pending.event)
                        sender = pending.continuation
                        permitWaiter = promotePersistencePermitLocked()
                    } else if let terminal = durableTerminal {
                        durableTerminal = nil
                        result = .some(terminal)
                        permitWaiter = promotePersistencePermitLocked()
                    } else if isTerminalLocked {
                        result = .some(nil)
                    } else if waitingConsumer == nil {
                        waitingConsumer = WaitingConsumer(
                            subscriptionID: subscriptionID,
                            continuation: continuation
                        )
                    } else {
                        // A single subscription may only have one outstanding next call.
                        result = .some(nil)
                    }
                }
                sender?.resume()
                permitWaiter?.resume(returning: true)
                if let result { continuation.resume(returning: result) }
            }
        } onCancel: { [weak self] in
            self?.cancelNext(subscriptionID: subscriptionID)
        }
    }

    fileprivate func cancelSubscription(id: UUID?) {
        guard let id else { return }
        var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
        lock.withLock {
            guard activeSubscriptionID == id else { return }
            activeSubscriptionID = nil
            if waitingConsumer?.subscriptionID == id {
                consumer = waitingConsumer?.continuation
                waitingConsumer = nil
            }
        }
        consumer?.resume(returning: nil)
    }

    private func cancelSend(id: UUID) {
        var sender: CheckedContinuation<Void, Never>?
        lock.withLock {
            guard var pendingSend, pendingSend.id == id else { return }
            sender = pendingSend.continuation
            pendingSend.continuation = nil
            self.pendingSend = pendingSend
        }
        sender?.resume()
    }

    private func cancelNext(subscriptionID: UUID) {
        var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
        lock.withLock {
            guard waitingConsumer?.subscriptionID == subscriptionID else { return }
            consumer = waitingConsumer?.continuation
            waitingConsumer = nil
        }
        consumer?.resume(returning: nil)
    }

    fileprivate func prepareForPersistence(producerID: UUID) async -> Bool {
        let permitID = UUID()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let result = lock.withLock { () -> Bool? in
                    guard activeProducerIDs.contains(producerID),
                          !finishRequested,
                          !Task.isCancelled else {
                        return false
                    }
                    guard pendingSend == nil,
                          durableTerminal == nil,
                          persistencePermitProducerID == nil else {
                        precondition(
                            waitingPersistencePermit == nil,
                            "CameraEventChannel supports one persistence producer"
                        )
                        waitingPersistencePermit = WaitingPersistencePermit(
                            id: permitID,
                            producerID: producerID,
                            continuation: continuation
                        )
                        return nil
                    }
                    persistencePermitProducerID = producerID
                    return true
                }
                if let result { continuation.resume(returning: result) }
            }
        } onCancel: { [weak self] in
            self?.cancelPersistencePermitWait(id: permitID)
        }
    }

    fileprivate func send(
        _ event: CameraFrameSourceEvent,
        producerID: UUID
    ) async {
        let sendID = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
                var permitWaiter: CheckedContinuation<Bool, Never>?
                let shouldResumeSender = lock.withLock { () -> Bool in
                    guard activeProducerIDs.contains(producerID),
                          persistencePermitProducerID == producerID else {
                        return true
                    }
                    persistencePermitProducerID = nil
                    if let waitingConsumer {
                        self.waitingConsumer = nil
                        consumer = waitingConsumer.continuation
                        permitWaiter = promotePersistencePermitLocked()
                        return true
                    }
                    guard buffer.count >= capacity else {
                        buffer.append(event)
                        permitWaiter = promotePersistencePermitLocked()
                        return true
                    }
                    precondition(
                        pendingSend == nil,
                        "Prepared persistence must own the sole overflow slot"
                    )
                    let shouldReleaseProducer = Task.isCancelled || finishRequested
                    pendingSend = PendingSend(
                        id: sendID,
                        event: event,
                        continuation: shouldReleaseProducer ? nil : continuation
                    )
                    return shouldReleaseProducer
                }
                consumer?.resume(returning: event)
                permitWaiter?.resume(returning: true)
                if shouldResumeSender { continuation.resume() }
            }
        } onCancel: { [weak self] in
            self?.cancelSend(id: sendID)
        }
    }

    fileprivate func cancelPersistencePermit(producerID: UUID) {
        var permitWaiter: CheckedContinuation<Bool, Never>?
        lock.withLock {
            guard persistencePermitProducerID == producerID else { return }
            persistencePermitProducerID = nil
            permitWaiter = promotePersistencePermitLocked()
        }
        permitWaiter?.resume(returning: true)
    }

    fileprivate func finishProducer(id: UUID) {
        var consumer: CheckedContinuation<CameraFrameSourceEvent?, Never>?
        var permitWaiter: (CheckedContinuation<Bool, Never>, Bool)?
        lock.withLock {
            guard activeProducerIDs.remove(id) != nil else { return }
            if persistencePermitProducerID == id {
                persistencePermitProducerID = nil
                if let promoted = promotePersistencePermitLocked() {
                    permitWaiter = (promoted, true)
                }
            }
            if waitingPersistencePermit?.producerID == id {
                if let continuation = waitingPersistencePermit?.continuation {
                    permitWaiter = (continuation, false)
                }
                waitingPersistencePermit = nil
            }
            if isTerminalLocked {
                consumer = waitingConsumer?.continuation
                waitingConsumer = nil
            }
        }
        if let (continuation, result) = permitWaiter {
            continuation.resume(returning: result)
        }
        consumer?.resume(returning: nil)
    }

    private func cancelPersistencePermitWait(id: UUID) {
        var permitWaiter: CheckedContinuation<Bool, Never>?
        lock.withLock {
            guard waitingPersistencePermit?.id == id else { return }
            permitWaiter = waitingPersistencePermit?.continuation
            waitingPersistencePermit = nil
        }
        permitWaiter?.resume(returning: false)
    }

    private func promotePersistencePermitLocked() -> CheckedContinuation<Bool, Never>? {
        guard !finishRequested,
              pendingSend == nil,
              durableTerminal == nil,
              persistencePermitProducerID == nil,
              let waitingPersistencePermit,
              activeProducerIDs.contains(waitingPersistencePermit.producerID) else {
            return nil
        }
        self.waitingPersistencePermit = nil
        persistencePermitProducerID = waitingPersistencePermit.producerID
        return waitingPersistencePermit.continuation
    }

    private var isTerminalLocked: Bool {
        finishRequested
            && activeProducerIDs.isEmpty
            && buffer.isEmpty
            && pendingSend == nil
            && durableTerminal == nil
    }
}

final class CameraEventProducer: @unchecked Sendable {
    private let lock = NSLock()
    private let channel: CameraEventChannel
    private let id: UUID?
    private var operationCount = 0
    private var hasPersistencePermit = false
    private var finishRequested = false
    private var didFinishChannel = false

    init(channel: CameraEventChannel, id: UUID?) {
        self.channel = channel
        self.id = id
    }

    func prepareForPersistence() async -> Bool {
        guard let id else { return false }
        let mayPrepare = lock.withLock { () -> Bool in
            guard !finishRequested,
                  !hasPersistencePermit,
                  operationCount == 0 else {
                return false
            }
            operationCount += 1
            return true
        }
        guard mayPrepare else { return false }
        let granted = await channel.prepareForPersistence(producerID: id)
        completeOperation(grantedPermit: granted)
        return granted
    }

    func send(_ event: CameraFrameSourceEvent) async {
        guard let id else { return }
        let maySend = lock.withLock { () -> Bool in
            guard hasPersistencePermit, operationCount == 0 else { return false }
            hasPersistencePermit = false
            operationCount += 1
            return true
        }
        guard maySend else { return }
        await channel.send(event, producerID: id)
        completeOperation()
    }

    func cancelPersistencePermit() {
        guard let id else { return }
        let result = lock.withLock { () -> (cancel: Bool, finish: Bool) in
            guard hasPersistencePermit, operationCount == 0 else { return (false, false) }
            hasPersistencePermit = false
            return (true, shouldFinishChannelLocked())
        }
        guard result.cancel else { return }
        channel.cancelPersistencePermit(producerID: id)
        if result.finish { channel.finishProducer(id: id) }
    }

    func finish() {
        guard let id else { return }
        let shouldFinish = lock.withLock { () -> Bool in
            finishRequested = true
            return shouldFinishChannelLocked()
        }
        if shouldFinish { channel.finishProducer(id: id) }
    }

    deinit {
        abandon()
    }

    private func completeOperation(grantedPermit: Bool = false) {
        guard let id else { return }
        let shouldFinish = lock.withLock { () -> Bool in
            precondition(operationCount == 1)
            operationCount = 0
            if grantedPermit { hasPersistencePermit = true }
            return shouldFinishChannelLocked()
        }
        if shouldFinish { channel.finishProducer(id: id) }
    }

    private func shouldFinishChannelLocked() -> Bool {
        guard finishRequested,
              operationCount == 0,
              !hasPersistencePermit,
              !didFinishChannel else {
            return false
        }
        didFinishChannel = true
        return true
    }

    private func abandon() {
        guard let id else { return }
        let action = lock.withLock { () -> (cancelPermit: Bool, finish: Bool) in
            precondition(operationCount == 0)
            finishRequested = true
            let cancelPermit = hasPersistencePermit
            hasPersistencePermit = false
            return (cancelPermit, shouldFinishChannelLocked())
        }
        if action.cancelPermit { channel.cancelPersistencePermit(producerID: id) }
        if action.finish { channel.finishProducer(id: id) }
    }
}

private final class CameraEventSubscription: @unchecked Sendable {
    private let channel: CameraEventChannel
    private let id: UUID?

    init(channel: CameraEventChannel, id: UUID?) {
        self.channel = channel
        self.id = id
    }

    func next() async -> CameraFrameSourceEvent? {
        guard let id else { return nil }
        return await channel.next(subscriptionID: id)
    }

    deinit {
        channel.cancelSubscription(id: id)
    }
}

final class AVCameraCaptureSession: NSObject, CameraCaptureSession, @unchecked Sendable {
    let previewSession = AVCaptureSession()

    private let configurationQueue = DispatchQueue(label: "cloud.point.cloud.camera.session")
    private let configurationQueueKey = DispatchSpecificKey<UInt8>()
    private let delegateQueue = DispatchQueue(label: "cloud.point.cloud.camera.frames")
    private var output: AVCaptureVideoDataOutput?
    private var outputDelegate: CameraVideoOutputDelegate?
    private var notificationTokens: [NSObjectProtocol] = []

    override init() {
        super.init()
        configurationQueue.setSpecific(key: configurationQueueKey, value: 1)
    }

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

    func shutdown() async {
        await stopRunning()
    }

    deinit {
        let session = previewSession
        let output = output
        let tokens = notificationTokens
        let tearDown = {
            Self.tearDownResources(
                session: session,
                output: output,
                notificationTokens: tokens
            )
        }
        if DispatchQueue.getSpecific(key: configurationQueueKey) != nil {
            tearDown()
        } else {
            configurationQueue.sync(execute: tearDown)
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
        let videoConnection = output.connection(with: .video)
        CameraDisplayPolicy(mirrorDisplay: false).configure(videoConnection, for: .frameOutput)
        probeIntrinsicMatrixDelivery(on: videoConnection)

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
        Self.tearDownResources(
            session: previewSession,
            output: output,
            notificationTokens: notificationTokens
        )
        notificationTokens.removeAll()
        output = nil
        outputDelegate = nil
    }

    private static func tearDownResources(
        session: AVCaptureSession,
        output: AVCaptureVideoDataOutput?,
        notificationTokens: [NSObjectProtocol]
    ) {
        output?.setSampleBufferDelegate(nil, queue: nil)
        for token in notificationTokens {
            NotificationCenter.default.removeObserver(token)
        }
        if session.isRunning { session.stopRunning() }
        session.beginConfiguration()
        for output in session.outputs { session.removeOutput(output) }
        for input in session.inputs { session.removeInput(input) }
        session.commitConfiguration()
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

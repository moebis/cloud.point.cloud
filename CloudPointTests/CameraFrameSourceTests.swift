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

    func testSevenAndNineFPSUseExactRationalBoundaries() throws {
        for rate in [7, 9] {
            var gate = try CameraSampleGate(rate: rate)
            XCTAssertTrue(gate.accepts(.zero))
            XCTAssertFalse(
                gate.accepts(CMTime(value: 999_999, timescale: CMTimeScale(rate * 1_000_000))),
                "\(rate) FPS accepted a timestamp before its exact 1/rate boundary"
            )
            XCTAssertTrue(gate.accepts(CMTime(value: 1, timescale: CMTimeScale(rate))))
        }
    }

    func testLargeTimestampJumpAdvancesInBoundedTimeAndKeepsExactGrid() throws {
        var gate = try CameraSampleGate(rate: 7)
        XCTAssertTrue(gate.accepts(.zero))
        let clock = ContinuousClock()
        let start = clock.now

        XCTAssertTrue(gate.accepts(CMTime(value: 7_000_000, timescale: 7)))

        XCTAssertLessThan(clock.now - start, .milliseconds(500))
        XCTAssertFalse(gate.accepts(CMTime(value: 7_000_000_999_999, timescale: 7_000_000)))
        XCTAssertTrue(gate.accepts(CMTime(value: 7_000_001, timescale: 7)))
    }
}

final class CameraEventChannelTests: XCTestCase {
    func testAuthoritativePersistedEventIsRetainedUntilConsumerAttaches() async {
        let channel = CameraEventChannel(capacity: 16)
        let producer = channel.makeProducer()
        await sendPersisted(index: 7, through: producer)
        producer.finish()

        var iterator = channel.stream().makeAsyncIterator()
        guard case let .persisted(frame)? = await iterator.next() else {
            return XCTFail("Expected the retained persisted event")
        }
        XCTAssertEqual(frame.frame.index, 7)
        channel.finish()
    }

    func testAuthoritativePersistedEventsBackpressureWithoutLosingIndexesBeyondCapacity() async {
        let channel = CameraEventChannel(capacity: 16)
        let stream = channel.stream()
        let producer = channel.makeProducer()
        let senderCompletion = CompletionProbe()
        let sender = Task {
            defer { producer.finish() }
            for index in 0..<40 {
                guard await producer.prepareForPersistence() else { return }
                await producer.send(persistedEvent(index: index))
            }
            senderCompletion.markComplete()
        }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertFalse(senderCompletion.isComplete)

        var iterator = stream.makeAsyncIterator()
        var indexes: [UInt32] = []
        for _ in 0..<40 {
            guard case let .persisted(frame)? = await iterator.next() else {
                return XCTFail("Authoritative persisted event was lost")
            }
            indexes.append(frame.frame.index)
        }
        await sender.value
        channel.finish()

        XCTAssertEqual(indexes, (0..<40).map(UInt32.init))
    }

    func testCancelingPendingSendReleasesProducerButRetainsEventForFIFODrain() async throws {
        let channel = CameraEventChannel(capacity: 1)
        let stream = channel.stream()
        let producer = channel.makeProducer()
        await sendPersisted(index: 0, through: producer)
        let hasPermit = await producer.prepareForPersistence()
        XCTAssertTrue(hasPermit)
        let senderStarted = CompletionProbe()
        let senderCompleted = CompletionProbe()
        let sender = Task {
            senderStarted.markComplete()
            await producer.send(persistedEvent(index: 1))
            senderCompleted.markComplete()
        }
        try await eventually { senderStarted.isComplete }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertFalse(senderCompleted.isComplete)

        sender.cancel()
        await sender.value
        channel.finish()
        producer.finish()

        XCTAssertTrue(senderCompleted.isComplete)
        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, [0, 1])
    }

    func testSendEnteredAlreadyCanceledStillTransfersEventToChannel() async {
        let channel = CameraEventChannel(capacity: 1)
        let stream = channel.stream()
        let producer = channel.makeProducer()
        await sendPersisted(index: 0, through: producer)
        let hasPermit = await producer.prepareForPersistence()
        XCTAssertTrue(hasPermit)
        let startGate = AsyncGate()
        let sender = Task {
            await startGate.wait()
            await producer.send(persistedEvent(index: 1))
        }
        sender.cancel()

        await startGate.open()
        await sender.value
        channel.finish()
        producer.finish()

        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, [0, 1])
    }

    func testFinishReleasesPendingProducerAndDrainsBufferedAndPendingEventsBeforeNil() async throws {
        let channel = CameraEventChannel(capacity: 2)
        let stream = channel.stream()
        let producer = channel.makeProducer()
        await sendPersisted(index: 0, through: producer)
        await sendPersisted(index: 1, through: producer)
        let hasPermit = await producer.prepareForPersistence()
        XCTAssertTrue(hasPermit)
        let senderStarted = CompletionProbe()
        let senderCompleted = CompletionProbe()
        let sender = Task {
            senderStarted.markComplete()
            await producer.send(persistedEvent(index: 2))
            senderCompleted.markComplete()
        }
        try await eventually { senderStarted.isComplete }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertFalse(senderCompleted.isComplete)

        channel.finish()
        await sender.value
        producer.finish()

        XCTAssertTrue(senderCompleted.isComplete)
        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, [0, 1, 2])
    }

    func testFinishAfterPersistencePermitBeforeSendMakesLateOverflowNonblocking() async throws {
        let channel = CameraEventChannel(capacity: 1)
        let stream = channel.stream()
        let producer = channel.makeProducer()
        await sendPersisted(index: 0, through: producer)
        let hasPermit = await producer.prepareForPersistence()
        XCTAssertTrue(hasPermit)
        channel.finish()

        let senderStarted = CompletionProbe()
        let senderCompleted = CompletionProbe()
        let sender = Task {
            senderStarted.markComplete()
            await producer.send(persistedEvent(index: 1))
            senderCompleted.markComplete()
        }
        try await eventually { senderStarted.isComplete }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertTrue(senderCompleted.isComplete)
        if !senderCompleted.isComplete { sender.cancel() }
        await sender.value
        producer.finish()

        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, [0, 1])
    }

    func testProducerFinishAfterGrantedPermitDefersUnregistrationUntilFinalSend() async {
        let channel = CameraEventChannel(capacity: 1)
        let stream = channel.stream()
        let producer = channel.makeProducer()
        let hasPermit = await producer.prepareForPersistence()
        XCTAssertTrue(hasPermit)

        producer.finish()
        await producer.send(persistedEvent(index: 0))
        channel.finish()

        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, [0])
    }

    func testDroppingProducerWithUnusedPermitReleasesRegistrationAndFinishesStream() async {
        let channel = CameraEventChannel(capacity: 1)
        let stream = channel.stream()
        var producer: CameraEventProducer? = channel.makeProducer()
        let hasPermit = await producer!.prepareForPersistence()
        XCTAssertTrue(hasPermit)

        producer = nil
        channel.finish()

        var iterator = stream.makeAsyncIterator()
        let nextCompleted = CompletionProbe()
        let next = Task {
            let event = await iterator.next()
            nextCompleted.markComplete()
            return event
        }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertTrue(nextCompleted.isComplete)
        if !nextCompleted.isComplete { next.cancel() }
        let event = await next.value
        XCTAssertNil(event)
    }
}

final class CameraFrameSourceTests: XCTestCase {
    func testThirtyPreviewFramesProduceFivePersistedFrames() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let previewObserver = PreviewObservationProbe()
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session,
            previewObserver: previewObserver
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
        XCTAssertEqual(
            previewObserver.timestamps,
            (0..<30).map { CMTime(value: CMTimeValue($0), timescale: 30) }
        )
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

    func testSecondAuthoritativeEventConsumerIsRejected() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let first = PersistedFrameCollector(stream: source.events())
        var rejectedIterator = source.events().makeAsyncIterator()
        let rejectedNext = Task { await rejectedIterator.next() }
        defer { first.cancel() }
        await source.start(deviceID: "camera-a", sampleRate: 5)

        for sequence in 0..<30 {
            session.emit(
                try makeFrame(
                    sequence: sequence,
                    timestamp: CMTime(value: CMTimeValue(sequence), timescale: 30)
                )
            )
            if sequence.isMultiple(of: 6) {
                let expectedCount = sequence / 6 + 1
                try await eventually { await persistence.callCount == expectedCount }
            }
        }

        try await eventually { await first.persisted.count == 5 }
        let firstFrames = await first.persisted
        XCTAssertEqual(firstFrames.map(\.index), Array(0..<5))
        let rejectedEvent = await rejectedNext.value
        XCTAssertNil(rejectedEvent)
    }

    func testSlowEventSubscriberRetainsBoundedOldestTelemetryWindow() async throws {
        let persistence = PersistenceProbe(suspended: true)
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 10)
        session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { await persistence.callCount == 1 }
        session.emit(try makeFrame(sequence: 1, timestamp: CMTime(value: 1, timescale: 10)))
        for sequence in 2..<42 {
            session.emit(
                try makeFrame(
                    sequence: sequence,
                    timestamp: CMTime(value: CMTimeValue(sequence), timescale: 10)
                )
            )
        }

        var iterator = stream.makeAsyncIterator()
        var drops: [CameraDroppedFrame] = []
        for _ in 0..<16 {
            guard case let .dropped(drop)? = await iterator.next() else {
                return XCTFail("Expected a buffered drop event")
            }
            drops.append(drop)
        }
        XCTAssertEqual(
            drops.map(\.timestamp),
            (2..<18).map { CMTime(value: CMTimeValue($0), timescale: 10) }
        )
        await persistence.resumeOne()
    }

    func testEventsPublishedWithoutSubscribersAreNotReplayed() async throws {
        let persistence = PersistenceProbe(suspended: true)
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 10)
        session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { await persistence.callCount == 1 }
        session.emit(try makeFrame(sequence: 1, timestamp: CMTime(value: 1, timescale: 10)))
        for sequence in 2..<12 {
            session.emit(
                try makeFrame(
                    sequence: sequence,
                    timestamp: CMTime(value: CMTimeValue(sequence), timescale: 10)
                )
            )
        }

        let stream = source.events()
        session.emit(try makeFrame(sequence: 12, timestamp: CMTime(value: 12, timescale: 10)))
        var iterator = stream.makeAsyncIterator()
        guard case let .dropped(drop)? = await iterator.next() else {
            return XCTFail("Expected the post-subscription drop event")
        }
        XCTAssertEqual(drop.timestamp, CMTime(value: 12, timescale: 10))
        await persistence.resumeOne()
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

    func testPersistenceFailureEmitsOrderedLifecycleFailureAfterDurableFrames() async throws {
        let persistence = FailAfterPersistenceProbe(successfulCallCount: 1)
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)

        session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { await persistence.callCount == 1 }
        session.emit(try makeFrame(sequence: 1, timestamp: CMTime(value: 1, timescale: 5)))

        try await eventually { await source.state == .failed(.persistenceFailed) }
        await source.shutdown()

        var iterator = stream.makeAsyncIterator()
        guard case let .persisted(persisted)? = await iterator.next() else {
            return XCTFail("Expected the durable frame before the failure event")
        }
        XCTAssertEqual(persisted.lifecycleID, 1)
        XCTAssertEqual(persisted.sequence, 0)
        XCTAssertEqual(persisted.frame.index, 0)
        guard case let .failed(failure)? = await iterator.next() else {
            return XCTFail("Expected the persistence failure event")
        }
        XCTAssertEqual(
            failure,
            CameraFrameSourceFailureEvent(
                lifecycleID: 1,
                sequence: 1,
                failure: .persistenceFailed
            )
        )
        let terminal = await iterator.next()
        XCTAssertNil(terminal)
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

    func testDifferentDeviceOrRateRestartsRunningCapture() async {
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a", "camera-b"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )

        await source.start(deviceID: "camera-a", sampleRate: 5)
        await source.start(deviceID: "camera-b", sampleRate: 7)
        await source.start(deviceID: "camera-b", sampleRate: 9)

        XCTAssertEqual(session.startCount, 3)
        XCTAssertEqual(session.stopCount, 2)
        XCTAssertEqual(session.connectedDeviceID, "camera-b")
        let finalState = await source.state
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
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

    func testDisconnectFailureIsRetainedAfterFullBufferAndDurableOverflow() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source,
            session: session,
            persistence: persistence
        )

        session.disconnect()
        try await eventually { await source.state == .failed(.cameraDisconnected) }
        await source.shutdown()

        var iterator = stream.makeAsyncIterator()
        for expectedIndex in 0..<17 {
            guard case let .persisted(persisted)? = await iterator.next() else {
                return XCTFail("Expected durable frame \(expectedIndex) before disconnect")
            }
            XCTAssertEqual(persisted.lifecycleID, 1)
            XCTAssertEqual(persisted.sequence, UInt64(expectedIndex))
            XCTAssertEqual(persisted.frame.index, UInt32(expectedIndex))
        }
        guard case let .failed(failure)? = await iterator.next() else {
            return XCTFail("Expected disconnect after every durable frame event")
        }
        XCTAssertEqual(
            failure,
            CameraFrameSourceFailureEvent(
                lifecycleID: 1,
                sequence: 17,
                failure: .cameraDisconnected
            )
        )
        let terminal = await iterator.next()
        XCTAssertNil(terminal)
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

    func testDelayedConnectFromStaleStartCannotStopConcurrentRestart() async throws {
        let delayedConnect = AsyncGate()
        let session = ScriptedCameraCaptureSession(
            deviceIDs: ["camera-a", "camera-b"],
            connectGates: [delayedConnect, nil]
        )
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let firstStart = Task { await source.start(deviceID: "camera-a", sampleRate: 5) }
        try await eventually { await delayedConnect.isWaiting }

        let restart = Task { await source.start(deviceID: "camera-b", sampleRate: 5) }
        try await eventually { await source.state == .running(deviceID: "camera-b") }
        await delayedConnect.open()
        await firstStart.value
        await restart.value

        let finalState = await source.state
        let connectedDeviceID = await session.connectedDeviceID
        let isRunning = await session.isRunning
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
        XCTAssertEqual(connectedDeviceID, "camera-b")
        XCTAssertTrue(isRunning)
    }

    func testDelayedStartRunningFromStaleStartCannotStopConcurrentRestart() async throws {
        let delayedStart = AsyncGate()
        let session = ScriptedCameraCaptureSession(
            deviceIDs: ["camera-a", "camera-b"],
            startGates: [delayedStart, nil]
        )
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let firstStart = Task { await source.start(deviceID: "camera-a", sampleRate: 5) }
        try await eventually { await delayedStart.isWaiting }

        let restart = Task { await source.start(deviceID: "camera-b", sampleRate: 5) }
        try await eventually { await source.state == .running(deviceID: "camera-b") }
        await delayedStart.open()
        await firstStart.value
        await restart.value

        let finalState = await source.state
        let connectedDeviceID = await session.connectedDeviceID
        let isRunning = await session.isRunning
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
        XCTAssertEqual(connectedDeviceID, "camera-b")
        XCTAssertTrue(isRunning)
    }

    func testDelayedTeardownCannotStopOrOverwriteConcurrentRestart() async throws {
        let delayedStop = AsyncGate()
        let session = ScriptedCameraCaptureSession(
            deviceIDs: ["camera-a", "camera-b"],
            stopGates: [delayedStop, nil]
        )
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)
        let stop = Task { await source.stop() }
        try await eventually { await delayedStop.isWaiting }

        let restart = Task { await source.start(deviceID: "camera-b", sampleRate: 5) }
        await delayedStop.open()
        await stop.value
        await restart.value

        let finalState = await source.state
        let connectedDeviceID = await session.connectedDeviceID
        let isRunning = await session.isRunning
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
        XCTAssertEqual(connectedDeviceID, "camera-b")
        XCTAssertTrue(isRunning)
    }

    func testInvalidRateInvalidatesSuspendedPriorStart() async throws {
        let delayedAuthorization = AsyncGate()
        let authority = DelayedCameraAuthority(gate: delayedAuthorization, result: .authorized)
        let session = ScriptedCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: authority,
            captureSession: session
        )
        let firstStart = Task { await source.start(deviceID: "camera-a", sampleRate: 5) }
        try await eventually { await delayedAuthorization.isWaiting }

        await source.start(deviceID: "camera-a", sampleRate: 11)
        await delayedAuthorization.open()
        await firstStart.value

        let finalState = await source.state
        let connectCount = await session.connectCount
        let isRunning = await session.isRunning
        XCTAssertEqual(finalState, .failed(.invalidSampleRate))
        XCTAssertEqual(connectCount, 0)
        XCTAssertFalse(isRunning)
    }

    func testDisconnectTeardownCannotOverwriteRestart() async throws {
        let delayedStop = AsyncGate()
        let session = ScriptedCameraCaptureSession(
            deviceIDs: ["camera-a", "camera-b"],
            stopGates: [delayedStop, nil]
        )
        let source = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)
        await session.disconnect()
        try await eventually { await delayedStop.isWaiting }

        let restart = Task { await source.start(deviceID: "camera-b", sampleRate: 5) }
        await delayedStop.open()
        await restart.value

        let finalState = await source.state
        let connectedDeviceID = await session.connectedDeviceID
        let isRunning = await session.isRunning
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
        XCTAssertEqual(connectedDeviceID, "camera-b")
        XCTAssertTrue(isRunning)
    }

    func testPersistenceFailureTeardownCannotOverwriteRestart() async throws {
        let delayedStop = AsyncGate()
        let session = ScriptedCameraCaptureSession(
            deviceIDs: ["camera-a", "camera-b"],
            stopGates: [delayedStop, nil]
        )
        let source = CameraFrameSource(
            persistence: PersistenceProbe(error: ProbeError.writeFailed),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)
        await session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { await delayedStop.isWaiting }

        let restart = Task { await source.start(deviceID: "camera-b", sampleRate: 5) }
        await delayedStop.open()
        await restart.value

        let finalState = await source.state
        let connectedDeviceID = await session.connectedDeviceID
        let isRunning = await session.isRunning
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
        XCTAssertEqual(connectedDeviceID, "camera-b")
        XCTAssertTrue(isRunning)
    }

    func testRestartWaitsForCancellationInsensitiveOldPersistenceWorker() async throws {
        let persistence = CancellationInsensitivePersistenceProbe()
        let session = ScriptedCameraCaptureSession(deviceIDs: ["camera-a", "camera-b"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)
        await session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { persistence.activeCallCount == 1 }

        let restart = Task {
            await source.stop()
            await source.start(deviceID: "camera-b", sampleRate: 5)
        }
        for _ in 0..<100 { await Task.yield() }
        await session.emit(try makeFrame(sequence: 1, timestamp: .zero))
        for _ in 0..<100 { await Task.yield() }
        XCTAssertEqual(persistence.callCount, 1)
        XCTAssertEqual(persistence.maximumConcurrentCallCount, 1)

        persistence.resumeFirst()
        await restart.value
        await session.emit(try makeFrame(sequence: 2, timestamp: .zero))
        try await eventually { persistence.callCount == 2 }
        XCTAssertEqual(persistence.maximumConcurrentCallCount, 1)
        let finalState = await source.state
        XCTAssertEqual(finalState, .running(deviceID: "camera-b"))
    }

    func testConcurrentRestartsShareTheOldPersistenceRetirementBarrier() async throws {
        let persistence = CancellationInsensitivePersistenceProbe()
        let session = ScriptedCameraCaptureSession(
            deviceIDs: ["camera-a", "camera-b", "camera-c"]
        )
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        await source.start(deviceID: "camera-a", sampleRate: 5)
        await session.emit(try makeFrame(sequence: 0, timestamp: .zero))
        try await eventually { persistence.activeCallCount == 1 }

        let firstRestart = Task {
            await source.start(deviceID: "camera-b", sampleRate: 5)
        }
        for _ in 0..<100 { await Task.yield() }
        let secondRestart = Task {
            await source.start(deviceID: "camera-c", sampleRate: 5)
        }
        for _ in 0..<500 { await Task.yield() }
        await session.emit(try makeFrame(sequence: 1, timestamp: .zero))
        for _ in 0..<100 { await Task.yield() }

        XCTAssertEqual(persistence.callCount, 1)
        XCTAssertEqual(persistence.maximumConcurrentCallCount, 1)

        persistence.resumeFirst()
        await firstRestart.value
        await secondRestart.value
        await session.emit(try makeFrame(sequence: 2, timestamp: .zero))
        try await eventually { persistence.callCount == 2 }
        XCTAssertEqual(persistence.maximumConcurrentCallCount, 1)
        let finalState = await source.state
        XCTAssertEqual(finalState, .running(deviceID: "camera-c"))
    }

    func testStopRetainsDurableEventBlockedBehindFullEventChannel() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source,
            session: session,
            persistence: persistence
        )

        await source.stop()
        let stoppedState = await source.state
        XCTAssertEqual(stoppedState, .stopped)
        await source.shutdown()

        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, Array(0..<17))
    }

    func testStopReturnsExactLifecycleDrainForBufferedPlusOverflowWithoutConsumer() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source,
            session: session,
            persistence: persistence
        )

        let didStop = CompletionProbe()
        let stop = Task {
            let completion = await source.stop()
            didStop.markComplete()
            return completion
        }
        defer { stop.cancel() }
        try await eventually { didStop.isComplete }
        let completion = await stop.value

        XCTAssertEqual(completion.lifecycleID, 1)
        XCTAssertEqual(completion.durablePersistedEventCount, 17)
        XCTAssertNil(completion.terminalFailure)
        XCTAssertEqual(
            completion.durablePersistedEvents.map(\.sequence),
            (0..<17).map(UInt64.init)
        )
        await source.shutdown()
        var iterator = stream.makeAsyncIterator()
        var events: [CameraPersistedFrame] = []
        while let event = await iterator.next() {
            guard case let .persisted(frame) = event else { continue }
            events.append(frame)
        }
        XCTAssertEqual(events.map(\.lifecycleID), Array(repeating: 1, count: 17))
        XCTAssertEqual(events.map(\.sequence), (0..<17).map(UInt64.init))
        XCTAssertEqual(events.map(\.frame.index), (0..<17).map(UInt32.init))
        XCTAssertEqual(completion.durablePersistedEvents, events)
    }

    func testRestartRetainsDurableEventBlockedBehindFullEventChannel() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a", "camera-b"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source,
            session: session,
            persistence: persistence
        )

        await source.start(deviceID: "camera-b", sampleRate: 5)
        let restartedState = await source.state
        XCTAssertEqual(restartedState, .running(deviceID: "camera-b"))
        await source.shutdown()

        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, Array(0..<17))
    }

    func testReplacementLifecycleWaitsForOverflowDrainBeforePersistingNextFrame() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a", "camera-b"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source,
            session: session,
            persistence: persistence
        )
        await source.start(deviceID: "camera-b", sampleRate: 5)

        session.emit(try makeFrame(sequence: 17, timestamp: .zero))
        for _ in 0..<100 { await Task.yield() }
        let durableCountBeforeDrain = await persistence.durableFrames.count
        XCTAssertEqual(durableCountBeforeDrain, 17)

        var iterator = stream.makeAsyncIterator()
        guard case let .persisted(firstFrame)? = await iterator.next() else {
            return XCTFail("Expected the first buffered authoritative event")
        }
        try await eventually { await persistence.durableFrames.count == 18 }
        await source.shutdown()

        var indexes = [firstFrame.frame.index]
        while let event = await iterator.next() {
            guard case let .persisted(frame) = event else { continue }
            indexes.append(frame.frame.index)
        }
        XCTAssertEqual(indexes, Array(0..<17) + [0])
    }

    func testShutdownRetainsDurableEventBlockedBehindFullEventChannel() async throws {
        let persistence = PersistenceProbe()
        let session = SyntheticCameraCaptureSession(deviceIDs: ["camera-a"])
        let source = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session
        )
        let stream = source.events()
        await source.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source,
            session: session,
            persistence: persistence
        )

        await source.shutdown()

        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, Array(0..<17))
    }

    @MainActor
    func testDroppingSourceWithFullEventChannelRunsSessionFallbackBeforeConsumerDrains() async throws {
        let persistence = PersistenceProbe()
        var session: EmittingAVFallbackCaptureSession? = EmittingAVFallbackCaptureSession(
            deviceIDs: ["camera-a"]
        )
        weak let weakSession = session
        var source: CameraFrameSource? = CameraFrameSource(
            persistence: persistence,
            authority: FixedCameraAuthority(.authorized),
            captureSession: session!
        )
        weak let weakSource = source
        let stream = source!.events()
        let retainedPreviewSession = source!.previewSession
        let output = AVCaptureVideoDataOutput()
        retainedPreviewSession.beginConfiguration()
        XCTAssertTrue(retainedPreviewSession.canAddOutput(output))
        retainedPreviewSession.addOutput(output)
        retainedPreviewSession.commitConfiguration()
        await source!.start(deviceID: "camera-a", sampleRate: 5)
        try await fillAuthoritativeEventCapacity(
            source: source!,
            session: session!,
            persistence: persistence
        )

        session = nil
        source = nil

        XCTAssertNil(weakSource)
        XCTAssertNil(weakSession)
        XCTAssertTrue(retainedPreviewSession.outputs.isEmpty)
        let indexes = await persistedIndexesUntilFinished(from: stream)
        XCTAssertEqual(indexes, Array(0..<17))
    }

    func testShutdownTearsDownCaptureFinishesEventsAndReleasesInjectedSession() async throws {
        var session: SyntheticCameraCaptureSession? = SyntheticCameraCaptureSession(
            deviceIDs: ["camera-a"]
        )
        weak let weakSession = session
        var source: CameraFrameSource? = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: session!
        )
        let retainedPreviewSession = source!.previewSession
        var eventIterator = source!.events().makeAsyncIterator()
        await source!.start(deviceID: "camera-a", sampleRate: 5)

        await source!.shutdown()

        XCTAssertEqual(session!.shutdownCount, 1)
        XCTAssertFalse(session!.isRunning)
        XCTAssertFalse(session!.hasDisconnectObserver)
        let eventAfterShutdown = await eventIterator.next()
        XCTAssertNil(eventAfterShutdown)
        await source!.start(deviceID: "camera-a", sampleRate: 5)
        XCTAssertEqual(session!.startCount, 1)
        XCTAssertFalse(session!.isRunning)
        let state = await source!.state
        XCTAssertEqual(state, .stopped)

        source = nil
        session = nil
        XCTAssertNil(weakSession)
        withExtendedLifetime(retainedPreviewSession) {}
    }

    func testDroppingSourceWithoutShutdownRunsAVFallbackOnRetainedPreviewSession() {
        var wrapper: AVCameraCaptureSession? = AVCameraCaptureSession()
        weak let weakWrapper = wrapper
        var source: CameraFrameSource? = CameraFrameSource(
            persistence: PersistenceProbe(),
            authority: FixedCameraAuthority(.authorized),
            captureSession: wrapper!
        )
        let retainedPreviewSession = source!.previewSession
        wrapper = nil
        let output = AVCaptureVideoDataOutput()
        retainedPreviewSession.beginConfiguration()
        XCTAssertTrue(retainedPreviewSession.canAddOutput(output))
        retainedPreviewSession.addOutput(output)
        retainedPreviewSession.commitConfiguration()
        XCTAssertEqual(retainedPreviewSession.outputs.count, 1)

        source = nil

        XCTAssertNil(weakWrapper)
        XCTAssertTrue(retainedPreviewSession.outputs.isEmpty)
    }

    @MainActor
    func testPreviewLayerTracksSessionResizesAndTearsDown() {
        let first = AVCaptureSession()
        let second = AVCaptureSession()
        let view = CameraPreviewNSView(session: first, mirrorDisplay: true)
        view.frame = NSRect(x: 0, y: 0, width: 640, height: 360)
        view.layout()

        XCTAssertTrue(view.previewLayer.session === first)
        XCTAssertEqual(view.previewLayer.videoGravity, .resizeAspect)
        XCTAssertEqual(view.previewLayer.frame, view.bounds)
        XCTAssertTrue(view.mirrorDisplay)

        view.setSession(second)
        XCTAssertTrue(view.previewLayer.session === second)
        view.setMirrorDisplay(false)
        XCTAssertFalse(view.mirrorDisplay)
        view.tearDown()
        XCTAssertNil(view.previewLayer.session)
        XCTAssertNil(view.layer)
    }

    func testCameraDisplayPolicyMirrorsPreviewButNeverPersistedFrames() {
        let mirrored = CameraDisplayPolicy(mirrorDisplay: true)
        let unmirrored = CameraDisplayPolicy(mirrorDisplay: false)

        XCTAssertTrue(mirrored.shouldMirrorVideo(for: .preview))
        XCTAssertFalse(mirrored.shouldMirrorVideo(for: .frameOutput))
        XCTAssertFalse(unmirrored.shouldMirrorVideo(for: .preview))
        XCTAssertFalse(unmirrored.shouldMirrorVideo(for: .frameOutput))
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

private final class CompletionProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var complete = false

    var isComplete: Bool { lock.withLock { complete } }

    func markComplete() {
        lock.withLock { complete = true }
    }
}

private final class PreviewObservationProbe: CameraPreviewObserving, @unchecked Sendable {
    private let lock = NSLock()
    private var observedTimestamps: [CMTime] = []

    var timestamps: [CMTime] { lock.withLock { observedTimestamps } }

    func observe(timestamp: CMTime) {
        lock.withLock { observedTimestamps.append(timestamp) }
    }
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
    private(set) var durableFrames: [PersistedFrame] = []
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
        let persisted = PersistedFrame(
            index: UInt32(frame.index),
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: String(format: "Frames/%08d.jpg", frame.index)
        )
        durableFrames.append(persisted)
        return persisted
    }

    func resumeOne() {
        guard !continuations.isEmpty else { return }
        continuations.removeFirst().resume()
    }
}

private actor FailAfterPersistenceProbe: FramePersistence {
    private let successfulCallCount: Int
    private(set) var callCount = 0

    init(successfulCallCount: Int) {
        self.successfulCallCount = successfulCallCount
    }

    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame {
        callCount += 1
        guard callCount <= successfulCallCount else { throw ProbeError.writeFailed }
        return PersistedFrame(
            index: UInt32(frame.index),
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: String(format: "Frames/%08d.jpg", frame.index)
        )
    }
}

private final class EmittingAVFallbackCaptureSession: FrameEmittingCameraCaptureSession,
    @unchecked Sendable
{
    private let wrappedSession = AVCameraCaptureSession()
    private let lock = NSLock()
    private let deviceIDs: Set<String>
    private var frameHandler: (@Sendable (CapturedFrame) -> Void)?
    private var disconnectHandler: (@Sendable () -> Void)?
    private var isRunning = false

    var previewSession: AVCaptureSession { wrappedSession.previewSession }

    init(deviceIDs: Set<String>) {
        self.deviceIDs = deviceIDs
    }

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws {
        guard deviceIDs.contains(deviceID) else { throw CameraCaptureSessionError.deviceNotFound }
        lock.withLock {
            self.frameHandler = frameHandler
            self.disconnectHandler = disconnectHandler
        }
    }

    func startRunning() async {
        lock.withLock { isRunning = true }
    }

    func stopRunning() async {
        lock.withLock {
            isRunning = false
            frameHandler = nil
            disconnectHandler = nil
        }
    }

    func shutdown() async {
        await stopRunning()
        await wrappedSession.shutdown()
    }

    func emit(_ frame: CapturedFrame) {
        let handler = lock.withLock { isRunning ? frameHandler : nil }
        handler?(frame)
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
            index: UInt32(frame.index),
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: "Frames/\(filename)"
        )
    }
}

private final class SyntheticCameraCaptureSession: FrameEmittingCameraCaptureSession,
    @unchecked Sendable
{
    let previewSession = AVCaptureSession()
    private let lock = NSLock()
    private let deviceIDs: Set<String>
    private var frameHandler: (@Sendable (CapturedFrame) -> Void)?
    private var disconnectHandler: (@Sendable () -> Void)?
    private var _connectedDeviceID: String?
    private var _isRunning = false
    private var _startCount = 0
    private var _stopCount = 0
    private var _shutdownCount = 0

    init(deviceIDs: Set<String>) {
        self.deviceIDs = deviceIDs
    }

    var connectedDeviceID: String? { lock.withLock { _connectedDeviceID } }
    var isConnected: Bool { connectedDeviceID != nil }
    var isRunning: Bool { lock.withLock { _isRunning } }
    var startCount: Int { lock.withLock { _startCount } }
    var stopCount: Int { lock.withLock { _stopCount } }
    var shutdownCount: Int { lock.withLock { _shutdownCount } }
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

    func shutdown() async {
        lock.withLock {
            _shutdownCount += 1
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

    init(stream: CameraFrameSourceEvents) {
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

private actor AsyncGate {
    private(set) var isWaiting = false
    private var isOpen = false
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        isWaiting = true
        await withCheckedContinuation { continuations.append($0) }
    }

    func open() {
        guard !isOpen else { return }
        isOpen = true
        isWaiting = false
        let current = continuations
        continuations.removeAll()
        for continuation in current { continuation.resume() }
    }
}

private struct DelayedCameraAuthority: CameraAuthorizing {
    let gate: AsyncGate
    let result: CameraAuthorization

    func requestAccess() async -> CameraAuthorization {
        await gate.wait()
        return result
    }
}

private final class ScriptedCameraCaptureSession: CameraCaptureSession, @unchecked Sendable {
    let previewSession = AVCaptureSession()
    private let backend: ScriptedCameraCaptureBackend

    init(
        deviceIDs: Set<String>,
        connectGates: [AsyncGate?] = [],
        startGates: [AsyncGate?] = [],
        stopGates: [AsyncGate?] = []
    ) {
        backend = ScriptedCameraCaptureBackend(
            deviceIDs: deviceIDs,
            connectGates: connectGates,
            startGates: startGates,
            stopGates: stopGates
        )
    }

    var connectCount: Int { get async { await backend.connectCount } }
    var connectedDeviceID: String? { get async { await backend.connectedDeviceID } }
    var isRunning: Bool { get async { await backend.isRunning } }

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws {
        try await backend.connect(
            deviceID: deviceID,
            frameHandler: frameHandler,
            disconnectHandler: disconnectHandler
        )
    }

    func startRunning() async throws {
        await backend.startRunning()
    }

    func stopRunning() async {
        await backend.stopRunning()
    }

    func emit(_ frame: CapturedFrame) async {
        await backend.emit(frame)
    }

    func disconnect() async {
        await backend.disconnect()
    }
}

private actor ScriptedCameraCaptureBackend {
    let deviceIDs: Set<String>
    let connectGates: [AsyncGate?]
    let startGates: [AsyncGate?]
    let stopGates: [AsyncGate?]
    private(set) var connectCount = 0
    private(set) var connectedDeviceID: String?
    private(set) var isRunning = false
    private var startCount = 0
    private var stopCount = 0
    private var frameHandler: (@Sendable (CapturedFrame) -> Void)?
    private var disconnectHandler: (@Sendable () -> Void)?

    init(
        deviceIDs: Set<String>,
        connectGates: [AsyncGate?],
        startGates: [AsyncGate?],
        stopGates: [AsyncGate?]
    ) {
        self.deviceIDs = deviceIDs
        self.connectGates = connectGates
        self.startGates = startGates
        self.stopGates = stopGates
    }

    func connect(
        deviceID: String,
        frameHandler: @escaping @Sendable (CapturedFrame) -> Void,
        disconnectHandler: @escaping @Sendable () -> Void
    ) async throws {
        guard deviceIDs.contains(deviceID) else { throw CameraCaptureSessionError.deviceNotFound }
        let call = connectCount
        connectCount += 1
        connectedDeviceID = deviceID
        self.frameHandler = frameHandler
        self.disconnectHandler = disconnectHandler
        if call < connectGates.count, let gate = connectGates[call] {
            await gate.wait()
        }
    }

    func startRunning() async {
        let call = startCount
        startCount += 1
        isRunning = true
        if call < startGates.count, let gate = startGates[call] {
            await gate.wait()
        }
    }

    func stopRunning() async {
        let call = stopCount
        stopCount += 1
        isRunning = false
        connectedDeviceID = nil
        frameHandler = nil
        disconnectHandler = nil
        if call < stopGates.count, let gate = stopGates[call] {
            await gate.wait()
        }
    }

    func emit(_ frame: CapturedFrame) {
        guard isRunning else { return }
        frameHandler?(frame)
    }

    func disconnect() {
        disconnectHandler?()
    }
}

private final class CancellationInsensitivePersistenceProbe: FramePersistence, @unchecked Sendable {
    private let lock = NSLock()
    private var firstContinuation: CheckedContinuation<Void, Never>?
    private var _callCount = 0
    private var _activeCallCount = 0
    private var _maximumConcurrentCallCount = 0

    var callCount: Int { lock.withLock { _callCount } }
    var activeCallCount: Int { lock.withLock { _activeCallCount } }
    var maximumConcurrentCallCount: Int { lock.withLock { _maximumConcurrentCallCount } }

    func persist(_ frame: CapturedFrame) async throws -> PersistedFrame {
        let shouldSuspend = lock.withLock { () -> Bool in
            _callCount += 1
            _activeCallCount += 1
            _maximumConcurrentCallCount = max(_maximumConcurrentCallCount, _activeCallCount)
            return _callCount == 1
        }
        if shouldSuspend {
            await withCheckedContinuation { continuation in
                lock.withLock { firstContinuation = continuation }
            }
        }
        lock.withLock { _activeCallCount -= 1 }
        return PersistedFrame(
            index: UInt32(frame.index),
            sourceTimestamp: frame.presentationTimestamp.seconds,
            relativePath: String(format: "Frames/%08d.jpg", frame.index)
        )
    }

    func resumeFirst() {
        let continuation = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            defer { firstContinuation = nil }
            return firstContinuation
        }
        continuation?.resume()
    }
}

private actor EventStorage {
    private(set) var persisted: [PersistedFrame] = []
    private(set) var drops: [CameraDroppedFrame] = []

    func append(_ event: CameraFrameSourceEvent) {
        switch event {
        case let .persisted(frame): persisted.append(frame.frame)
        case let .dropped(frame): drops.append(frame)
        case .failed: break
        }
    }
}

private protocol FrameEmittingCameraCaptureSession: CameraCaptureSession, AnyObject {
    func emit(_ frame: CapturedFrame)
}

private func persistedEvent(index: Int) -> CameraFrameSourceEvent {
    .persisted(
        CameraPersistedFrame(
            lifecycleID: 0,
            sequence: UInt64(index),
            frame: PersistedFrame(
                index: UInt32(index),
                sourceTimestamp: Double(index),
                relativePath: String(format: "Frames/%08d.jpg", index)
            )
        )
    )
}

private func sendPersisted(index: Int, through producer: CameraEventProducer) async {
    guard await producer.prepareForPersistence() else {
        XCTFail("Authoritative event producer did not receive a persistence permit")
        return
    }
    await producer.send(persistedEvent(index: index))
}

private func persistedIndexesUntilFinished(from stream: CameraFrameSourceEvents) async -> [UInt32] {
    var iterator = stream.makeAsyncIterator()
    var indexes: [UInt32] = []
    while let event = await iterator.next() {
        guard case let .persisted(frame) = event else { continue }
        indexes.append(frame.frame.index)
    }
    return indexes
}

private func fillAuthoritativeEventCapacity<Session: FrameEmittingCameraCaptureSession>(
    source: CameraFrameSource,
    session: Session,
    persistence: PersistenceProbe
) async throws {
    for index in 0..<17 {
        let frame = try makeFrame(
            sequence: index,
            timestamp: CMTime(value: CMTimeValue(index), timescale: 5)
        )
        session.emit(frame)
        try await eventually { await persistence.durableFrames.count == index + 1 }
    }
    let acceptedCount = await source.metrics.acceptedPersistenceCount
    XCTAssertEqual(acceptedCount, 17)
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

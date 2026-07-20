import Darwin
import Foundation
import XCTest
@testable import CloudPoint

final class WorkerProcessTests: XCTestCase {
    private let projectID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!

    func testHeartbeatWorkerBecomesReadyAndTerminationEmitsOneExitAndClosesStreams() async throws {
        let baselineFDs = try openFileDescriptorCount()
        let worker = try await startWorker(mode: "heartbeat")
        var events = worker.events().makeAsyncIterator()

        XCTAssertReady(try await nextEnvelope(from: &events))
        await worker.terminate()

        var exits: [Int32] = []
        while let event = try await events.next() {
            if case let .processExited(status) = event { exits.append(status) }
        }
        XCTAssertEqual(exits.count, 1)
        await assertSendFails(worker, expected: .notRunning)
        XCTAssertLessThanOrEqual(try openFileDescriptorCount(), baselineFDs + 1)
    }

    func testNormalWorkerShutsDownCleanlyAndAcknowledgesCommand() async throws {
        let worker = try await startWorker(mode: "normal")
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        let commandID = UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!
        try await worker.send(.command(.shutdown, id: commandID, projectId: projectID))

        var sawAck = false
        var exits: [Int32] = []
        while let event = try await events.next() {
            switch event {
            case let .envelope(envelope):
                if case let .ack(acknowledgedID, command)? = envelope.event {
                    sawAck = acknowledgedID == commandID && command == "shutdown"
                }
            case let .processExited(status): exits.append(status)
            }
        }
        XCTAssertTrue(sawAck)
        XCTAssertEqual(exits, [0])
    }

    func testCrashAfterReadyEmitsExitExactlyOnce() async throws {
        let worker = try await startWorker(mode: "crash-after-ready")
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        var exits: [Int32] = []
        while let event = try await events.next() {
            if case let .processExited(status) = event { exits.append(status) }
        }
        XCTAssertEqual(exits, [23])
    }

    func testCrashAfterReadyCleansInheritedOutputDescendantAndTerminatesPromptly() async throws {
        let worker = try await startWorker(
            mode: "crash-after-ready",
            environment: ["CLOUDPOINT_MOCK_SPAWN_CHILD": "1"]
        )
        var diagnostics = worker.diagnostics().makeAsyncIterator()
        var childPID: pid_t?
        while let line = await diagnostics.next() {
            if line.hasPrefix("child-pid:"), let value = Int32(line.dropFirst("child-pid:".count)) {
                childPID = value
                break
            }
        }
        let pid = try XCTUnwrap(childPID)
        let terminalReached = expectation(description: "terminal event arrives despite inherited stdout")
        let collection = Task { () -> [WorkerProcessEvent] in
            var received: [WorkerProcessEvent] = []
            do {
                for try await event in worker.events() { received.append(event) }
            } catch {}
            terminalReached.fulfill()
            return received
        }

        await fulfillment(of: [terminalReached], timeout: 1)
        if kill(pid, 0) == 0 { _ = kill(pid, SIGKILL) }
        let received = await collection.value
        let exits: [Int32] = received.compactMap {
            if case let .processExited(status) = $0 { status } else { nil }
        }
        XCTAssertEqual(exits, [23])
        XCTAssertEqual(kill(pid, 0), -1)
        XCTAssertEqual(errno, ESRCH)
    }

    func testStderrDiagnosticsNeverEnterProtocolDecoder() async throws {
        let worker = try await startWorker(mode: "heartbeat")
        var events = worker.events().makeAsyncIterator()
        var diagnostics = worker.diagnostics().makeAsyncIterator()

        XCTAssertReady(try await nextEnvelope(from: &events))
        let diagnostic = await diagnostics.next()
        XCTAssertTrue(try XCTUnwrap(diagnostic).contains("mode=heartbeat"))
        await worker.terminate()
        while try await events.next() != nil {}
        while await diagnostics.next() != nil {}
    }

    func testThreeMissedHeartbeatTicksFailAndCleanUpSilentWorker() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "silent", clock: clock)
        let collection = Task { () -> Error? in
            do {
                for try await _ in worker.events() {}
                return nil
            } catch { return error }
        }

        clock.advance(); clock.advance(); clock.advance()
        let error = await collection.value
        XCTAssertEqual(error as? WorkerProcessError, .unresponsive)
        await assertSendFails(worker, expected: .notRunning)
    }

    func testTerminationKillsEntireWorkerProcessGroup() async throws {
        let worker = try await startWorker(mode: "heartbeat", environment: ["CLOUDPOINT_MOCK_SPAWN_CHILD": "1"])
        var events = worker.events().makeAsyncIterator()
        var diagnostics = worker.diagnostics().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        var childPID: pid_t?
        while let line = await diagnostics.next() {
            if line.hasPrefix("child-pid:"), let value = Int32(line.dropFirst("child-pid:".count)) {
                childPID = value
                break
            }
        }
        let pid = try XCTUnwrap(childPID)
        let processGroupID = await worker.processGroupIdentifier
        let workerPID = await worker.processIdentifier
        XCTAssertEqual(processGroupID, workerPID)
        await worker.terminate()
        while try await events.next() != nil {}

        XCTAssertEqual(kill(pid, 0), -1)
        XCTAssertEqual(errno, ESRCH)
    }

    func testFinalFragmentedFrameIsDeliveredBeforeExactlyOneExit() async throws {
        let worker = try await startWorker(mode: "fragmented-final")
        var received: [WorkerProcessEvent] = []
        for try await event in worker.events() { received.append(event) }

        let envelopes = received.compactMap { event -> WorkerEnvelope? in
            guard case let .envelope(envelope) = event else { return nil }
            return envelope
        }
        XCTAssertEqual(envelopes.count, 2)
        guard case let .warning(payload)? = envelopes.last?.event else {
            return XCTFail("Expected final warning envelope")
        }
        XCTAssertEqual(payload.code, "finalFragment")
        let exits: [Int32] = received.compactMap { if case let .processExited(status) = $0 { status } else { nil } }
        XCTAssertEqual(exits, [0])
    }

    func testTERMToKILLEscalationUsesInjectedClock() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "ignore-term", clock: clock)
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        await worker.terminate()
        clock.advance(interval: .seconds(2))

        var exitStatus: Int32?
        while let event = try await events.next() {
            if case let .processExited(status) = event { exitStatus = status }
        }
        XCTAssertEqual(exitStatus, SIGKILL)
    }

    func testTerminateBeforeThirdHeartbeatCannotBecomeUnresponsive() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "ignore-term", clock: clock)
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        clock.advance(); clock.advance()
        await worker.terminate()
        clock.advance()
        clock.advance(interval: .seconds(2))

        var exitStatus: Int32?
        while let event = try await events.next() {
            if case let .processExited(status) = event { exitStatus = status }
        }
        XCTAssertEqual(exitStatus, SIGKILL)
    }

    func testRepeatedTerminateKeepsOriginalEscalationDeadline() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "ignore-term", clock: clock)
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        await worker.terminate()
        await worker.terminate()
        XCTAssertEqual(clock.requestCount(for: .seconds(2)), 1)
        clock.advance(interval: .seconds(2))
        while try await events.next() != nil {}
    }

    func testSaturatedInputQueueDoesNotBlockTerminationAndPendingWritesClose() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "silent", clock: clock)
        let payload = String(repeating: "x", count: 900_000)
        let projectID = self.projectID
        let queueRejected = expectation(description: "bounded queue rejects excess input")
        queueRejected.assertForOverFulfill = false

        let sends = (0..<8).map { _ in
            Task { () -> WorkerProcessError? in
                do {
                    try await worker.send(.command(.hello(clientVersion: payload, supportedProtocolVersions: [1]), projectId: projectID))
                    return nil
                } catch let error as WorkerProcessError {
                    if error == .inputQueueFull { queueRejected.fulfill() }
                    return error
                } catch { return nil }
            }
        }
        await fulfillment(of: [queueRejected], timeout: 2)
        await worker.terminate()
        _ = await sends.asyncMap { await $0.value }

        await assertSendFails(worker, expected: .notRunning)
    }

    func testProtocolEventOverflowTerminatesInsteadOfDroppingSilently() async throws {
        let worker = try await startWorker(
            mode: "normal",
            environment: ["CLOUDPOINT_MOCK_EVENT_COUNT": "10000"]
        )
        do {
            _ = try await worker.waitForTermination()
            XCTFail("Expected overflow termination")
        } catch let error as WorkerProcessError {
            XCTAssertEqual(error, .eventBufferOverflow)
        }
        var observedError: WorkerProcessError?
        do {
            for try await _ in worker.events() {}
        } catch let error as WorkerProcessError {
            observedError = error
        }
        XCTAssertEqual(observedError, .eventBufferOverflow)
    }

    func testLongUnterminatedStderrIsEmittedInBoundedChunks() async throws {
        let worker = try await startWorker(
            mode: "heartbeat",
            environment: ["CLOUDPOINT_MOCK_STDERR_BYTES": "10000"]
        )
        var diagnostics = worker.diagnostics().makeAsyncIterator()
        var payloadBytes = 0
        while let chunk = await diagnostics.next() {
            XCTAssertLessThanOrEqual(chunk.utf8.count, 4_096)
            if chunk.allSatisfy({ $0 == "d" }) { payloadBytes += chunk.utf8.count }
            if payloadBytes == 10_000 { break }
        }
        XCTAssertEqual(payloadBytes, 10_000)
        await worker.terminate()
        for try await _ in worker.events() {}
    }

    func testHeartbeatTimerRequestsExactFiveSecondInterval() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "heartbeat", clock: clock)
        XCTAssertTrue(clock.didRequest(.seconds(5)))
        await worker.terminate()
        for try await _ in worker.events() {}
    }

    func testMissingLauncherFailsWithoutLeakingDescriptors() async throws {
        let baseline = try openFileDescriptorCount()
        do {
            _ = try await WorkerProcess.start(
                executable: try mockWorkerURL(),
                launcherExecutable: URL(fileURLWithPath: "/definitely/missing/CloudPointWorkerLauncher")
            )
            XCTFail("Expected launch failure")
        } catch let error as WorkerProcessError {
            guard case .executableNotFound = error else { return XCTFail("Unexpected error: \(error)") }
        }
        XCTAssertLessThanOrEqual(try openFileDescriptorCount(), baseline)
    }

    func testLauncherRunFailureCleansUpAllocatedDescriptors() async throws {
        let baseline = try openFileDescriptorCount()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let invalidLauncher = directory.appendingPathComponent("invalid-launcher")
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: invalidLauncher)
        XCTAssertEqual(chmod(invalidLauncher.path, 0o700), 0)

        do {
            _ = try await WorkerProcess.start(
                executable: mockWorkerURL(),
                launcherExecutable: invalidLauncher
            )
            XCTFail("Expected launcher run failure")
        } catch let error as WorkerProcessError {
            guard case .launchFailed = error else { return XCTFail("Unexpected error: \(error)") }
        }
        XCTAssertLessThanOrEqual(try openFileDescriptorCount(), baseline)
    }

    func testDefaultStartUsesLauncherBundledWithApplication() async throws {
        let worker = try await WorkerProcess.start(
            executable: mockWorkerURL(),
            arguments: ["--mode", "heartbeat"]
        )
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))
        await worker.terminate()
        while try await events.next() != nil {}
    }

    private func nextEnvelope(
        from iterator: inout AsyncThrowingStream<WorkerProcessEvent, Error>.Iterator
    ) async throws -> WorkerEnvelope {
        while let event = try await iterator.next() {
            if case let .envelope(envelope) = event { return envelope }
        }
        throw WorkerProcessError.notRunning
    }

    private func XCTAssertReady(_ envelope: WorkerEnvelope, file: StaticString = #filePath, line: UInt = #line) {
        guard case .ready? = envelope.event else {
            XCTFail("Expected ready, got \(envelope)", file: file, line: line)
            return
        }
    }

    private func mockWorkerURL() throws -> URL {
        let bundle = Bundle(for: Self.self)
        return URL(fileURLWithPath: try XCTUnwrap(
            bundle.object(forInfoDictionaryKey: "CloudPointMockWorkerExecutable") as? String,
            "CloudPointMockWorkerExecutable must be injected by the test target build settings"
        ))
    }

    private func launcherURL() throws -> URL {
        URL(fileURLWithPath: try XCTUnwrap(
            Bundle(for: Self.self).object(forInfoDictionaryKey: "CloudPointWorkerLauncherExecutable") as? String,
            "CloudPointWorkerLauncherExecutable must be injected by build settings"
        ))
    }

    private func startWorker(
        mode: String,
        environment: [String: String] = [:],
        clock: any WorkerProcessClock = ContinuousWorkerProcessClock()
    ) async throws -> WorkerProcess {
        try await WorkerProcess.start(
            executable: mockWorkerURL(),
            arguments: ["--mode", mode],
            environment: environment,
            launcherExecutable: launcherURL(),
            clock: clock
        )
    }

    private func openFileDescriptorCount() throws -> Int {
        try FileManager.default.contentsOfDirectory(atPath: "/dev/fd").count
    }

    private func assertSendFails(_ worker: WorkerProcess, expected: WorkerProcessError) async {
        do {
            try await worker.send(.command(.pause, projectId: projectID))
            XCTFail("Expected send to fail")
        } catch let error as WorkerProcessError {
            XCTAssertEqual(error, expected)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}

private final class ManualWorkerProcessClock: WorkerProcessClock, @unchecked Sendable {
    private let lock = NSLock()
    private var continuations: [Duration: AsyncStream<Void>.Continuation] = [:]
    private var requestedIntervals: [Duration] = []

    func timer(interval: Duration) -> AsyncStream<Void> {
        AsyncStream { continuation in
            lock.withLock {
                requestedIntervals.append(interval)
                continuations[interval] = continuation
            }
        }
    }

    func advance(interval: Duration = .seconds(5)) {
        _ = lock.withLock { continuations[interval]?.yield(()) }
    }

    func didRequest(_ interval: Duration) -> Bool { lock.withLock { requestedIntervals.contains(interval) } }
    func requestCount(for interval: Duration) -> Int {
        lock.withLock { requestedIntervals.filter { $0 == interval }.count }
    }
}

private extension Array {
    func asyncMap<T: Sendable>(_ transform: (Element) async -> T) async -> [T] {
        var result: [T] = []
        for element in self { result.append(await transform(element)) }
        return result
    }
}

import Darwin
import Foundation
import XCTest
@testable import CloudPoint

final class WorkerProcessTests: XCTestCase {
    private let projectID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!

    func testHeartbeatWorkerStaysAliveToAcknowledgeShutdownAndClosesStreams() async throws {
        let baselineFDs = try openFileDescriptorCount()
        let worker = try await startWorker(mode: "heartbeat")
        var events = worker.events().makeAsyncIterator()

        XCTAssertReady(try await nextEnvelope(from: &events))
        let commandID = UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!
        try await worker.send(.command(.shutdown, id: commandID, projectId: projectID))

        var sawAcknowledgement = false
        var exits: [Int32] = []
        while let event = try await events.next() {
            switch event {
            case let .envelope(envelope):
                if case let .ack(acknowledgedID, command)? = envelope.event {
                    sawAcknowledgement = acknowledgedID == commandID && command == "shutdown"
                }
            case let .processExited(status):
                exits.append(status)
            }
        }
        XCTAssertTrue(sawAcknowledgement)
        XCTAssertEqual(exits, [0])
        await assertSendFails(worker, expected: .notRunning)
        XCTAssertLessThanOrEqual(try openFileDescriptorCount(), baselineFDs + 1)
    }

    func testMockWorkerUsesNegotiatedProjectIDForEveryPostHelloEvent() async throws {
        let negotiatedProjectID = UUID(uuidString: "22222222-2222-2222-2222-222222222222")!
        let worker = try await startWorker(mode: "heartbeat", performHello: false)
        let helloID = UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!
        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            id: helloID,
            projectId: negotiatedProjectID
        ))

        var events = worker.events().makeAsyncIterator()
        let acknowledgement = try await nextEnvelope(from: &events)
        let immediateHeartbeat = try await nextEnvelope(from: &events)
        let ready = try await nextEnvelope(from: &events)
        let scheduledHeartbeat = try await nextEnvelope(from: &events)
        XCTAssertEqual(
            [acknowledgement, immediateHeartbeat, ready, scheduledHeartbeat].map(\.projectId),
            Array(repeating: negotiatedProjectID, count: 4)
        )
        guard case .ack? = acknowledgement.event,
              case .heartbeat? = immediateHeartbeat.event,
              case .ready? = ready.event,
              case .heartbeat? = scheduledHeartbeat.event else {
            return XCTFail("Expected ACK, immediate heartbeat, ready, and heartbeat ordering")
        }

        let shutdownID = UUID(uuidString: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")!
        try await worker.send(.command(.shutdown, id: shutdownID, projectId: negotiatedProjectID))
        var sawShutdownAcknowledgement = false
        var exits: [Int32] = []
        while let event = try await events.next() {
            switch event {
            case let .envelope(envelope):
                if case let .ack(commandID, command)? = envelope.event {
                    sawShutdownAcknowledgement = commandID == shutdownID && command == "shutdown"
                }
            case let .processExited(status): exits.append(status)
            }
        }
        XCTAssertTrue(sawShutdownAcknowledgement)
        XCTAssertEqual(exits, [0])
    }

    func testNormalMockWorkerMaintainsFiveSecondIdleHeartbeatCadence() async throws {
        let worker = try await startWorker(mode: "normal", performHello: false)
        let heartbeatArrived = expectation(description: "normal worker emits its scheduled heartbeat")
        let observations = Task { () -> [WorkerEnvelope] in
            var envelopes: [WorkerEnvelope] = []
            do {
                for try await event in worker.events() {
                    guard case let .envelope(envelope) = event else { continue }
                    envelopes.append(envelope)
                    if envelopes.count == 4 {
                        heartbeatArrived.fulfill()
                        break
                    }
                }
            } catch {}
            return envelopes
        }

        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            projectId: projectID
        ))
        await fulfillment(of: [heartbeatArrived], timeout: 6)
        await worker.terminate()
        let envelopes = await observations.value

        XCTAssertEqual(envelopes.count, 4)
        if envelopes.count == 4 {
            guard case .ack? = envelopes[0].event,
                  case .heartbeat? = envelopes[1].event,
                  case .ready? = envelopes[2].event,
                  case .heartbeat? = envelopes[3].event else {
                return XCTFail("Expected ACK, immediate heartbeat, ready, and scheduled heartbeat")
            }
        }
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

    func testLauncherBlocksTargetUntilParentAcknowledgesVerifiedProcessGroup() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appendingPathComponent("target-started")
        let process = Process()
        let input = Pipe()
        let output = Pipe()
        let diagnostics = Pipe()
        process.executableURL = try launcherURL()
        process.arguments = [try mockWorkerURL().path, "--mode", "immediate-exit"]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "CLOUDPOINT_MOCK_START_MARKER": marker.path,
        ]) { _, supplied in supplied }
        process.standardInput = input
        process.standardOutput = output
        process.standardError = diagnostics

        try process.run()
        let pid = process.processIdentifier
        input.fileHandleForReading.closeFile()
        output.fileHandleForWriting.closeFile()
        diagnostics.fileHandleForWriting.closeFile()
        defer {
            _ = kill(-pid, SIGKILL)
            _ = kill(pid, SIGKILL)
            input.fileHandleForWriting.closeFile()
            output.fileHandleForReading.closeFile()
            diagnostics.fileHandleForReading.closeFile()
        }

        let ready = String(decoding: diagnostics.fileHandleForReading.availableData, as: UTF8.self)
        XCTAssertTrue(ready.contains("CLOUDPOINT_LAUNCHER_READY:\(pid)"))
        let acknowledgementDeadline = ContinuousClock.now + .milliseconds(200)
        while ContinuousClock.now < acknowledgementDeadline,
              !FileManager.default.fileExists(atPath: marker.path) {
            await Task.yield()
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))

        XCTAssertEqual(fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1), 0)
        var acknowledgement: UInt8 = 0x06
        XCTAssertEqual(Darwin.write(input.fileHandleForWriting.fileDescriptor, &acknowledgement, 1), 1)
        input.fileHandleForWriting.closeFile()

        let targetDeadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < targetDeadline,
              !FileManager.default.fileExists(atPath: marker.path) {
            await Task.yield()
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 24)
    }

    func testImmediateExitAfterHandshakeCleansInheritedOutputDescendantPromptly() async throws {
        let worker = try await startWorker(
            mode: "immediate-exit",
            environment: ["CLOUDPOINT_MOCK_SPAWN_CHILD": "1"]
        )
        var diagnostics = worker.diagnostics().makeAsyncIterator()
        let reportedChildPID = await childPID(from: &diagnostics)
        let childPID = try XCTUnwrap(reportedChildPID)
        let terminalReached = expectation(description: "immediate target exit reaches terminal state")
        let collection = Task { () -> [WorkerProcessEvent] in
            var received: [WorkerProcessEvent] = []
            do {
                for try await event in worker.events() { received.append(event) }
            } catch {}
            terminalReached.fulfill()
            return received
        }

        await fulfillment(of: [terminalReached], timeout: 1)
        if kill(childPID, 0) == 0 { _ = kill(childPID, SIGKILL) }
        let received = await collection.value
        let exits: [Int32] = received.compactMap {
            if case let .processExited(status) = $0 { status } else { nil }
        }
        XCTAssertEqual(exits, [24])
        XCTAssertEqual(kill(childPID, 0), -1)
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

        await worker.markProtocolReady()
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

    func testTruncatedStdoutFailsPromptlyAndKillsTheWorker() async throws {
        let worker = try await startWorker(mode: "truncated-hang")
        let pid = await worker.processIdentifier
        let terminalReached = expectation(description: "truncated stdout reaches terminal failure")
        let terminalError = Task { () -> WorkerProcessError? in
            defer { terminalReached.fulfill() }
            do {
                _ = try await worker.waitForTermination()
                return nil
            } catch let error as WorkerProcessError {
                return error
            } catch {
                return nil
            }
        }

        await fulfillment(of: [terminalReached], timeout: 1)
        await worker.terminate()

        let observedError = await terminalError.value
        XCTAssertEqual(observedError, .protocolFailure(.truncatedFrame))
        XCTAssertEqual(kill(pid, 0), -1)
        XCTAssertEqual(errno, ESRCH)
    }

    func testCleanStdoutEOFUsesExitGraceThenKillsAWorkerThatRemainsAlive() async throws {
        let worker = try await startWorker(mode: "clean-eof-hang")
        let pid = await worker.processIdentifier
        let terminalReached = expectation(description: "clean stdout EOF reaches terminal failure")
        let terminalError = Task { () -> WorkerProcessError? in
            defer { terminalReached.fulfill() }
            do {
                _ = try await worker.waitForTermination()
                return nil
            } catch let error as WorkerProcessError {
                return error
            } catch {
                return nil
            }
        }

        await fulfillment(of: [terminalReached], timeout: 1)
        await worker.terminate()
        let observedError = await terminalError.value

        XCTAssertEqual(observedError, .standardOutputClosed)
        XCTAssertEqual(kill(pid, 0), -1)
        XCTAssertEqual(errno, ESRCH)
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
        let recordedErrors = WorkerProcessErrorRecorder()

        let sends = (0..<8).map { _ in
            Task { () -> WorkerProcessError? in
                do {
                    try await worker.send(.command(.hello(clientVersion: payload, supportedProtocolVersions: [1]), projectId: projectID))
                    return nil
                } catch let error as WorkerProcessError {
                    recordedErrors.record(error)
                    return error
                } catch { return nil }
            }
        }
        let rejectionDeadline = ContinuousClock.now + .seconds(10)
        while ContinuousClock.now < rejectionDeadline,
              !recordedErrors.contains(.inputQueueFull) {
            try? await ContinuousClock().sleep(for: .milliseconds(10))
        }
        let rejectedBeforeTermination = recordedErrors.contains(.inputQueueFull)
        await worker.terminate()
        let results = await sends.asyncMap { await $0.value }

        XCTAssertTrue(rejectedBeforeTermination)
        XCTAssertTrue(results.contains(.inputQueueFull))
        XCTAssertTrue(results.contains(.notRunning))
        await assertSendFails(worker, expected: .notRunning)
    }

    func testProtocolEventOverflowTerminatesInsteadOfDroppingSilently() async throws {
        let worker = try await startWorker(
            mode: "normal",
            environment: ["CLOUDPOINT_MOCK_EVENT_COUNT": "10000"],
            performHello: false
        )
        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            projectId: projectID
        ))
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
        await worker.terminate()
        for try await _ in worker.events() {}

        var payloadBytes = 0
        for await chunk in worker.diagnostics() {
            XCTAssertLessThanOrEqual(chunk.utf8.count, 4_096)
            if chunk.allSatisfy({ $0 == "d" }) { payloadBytes += chunk.utf8.count }
        }
        XCTAssertEqual(payloadBytes, 10_000)
    }

    func testHeartbeatTimerRequestsExactFiveSecondInterval() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "heartbeat", clock: clock)
        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 1)
        await worker.terminate()
        for try await _ in worker.events() {}
    }

    func testHeartbeatSupervisionIsUnarmedUntilProtocolReadyAndDuplicateArmingIsIdempotent() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "silent", clock: clock)
        let collection = Task { () -> WorkerProcessError? in
            do {
                for try await _ in worker.events() {}
                return nil
            } catch let error as WorkerProcessError {
                return error
            } catch {
                return nil
            }
        }

        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 0)
        await worker.markProtocolReady()
        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 1)
        clock.advance()
        await assertHeartbeatState(worker, missed: 1, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)
        await worker.markProtocolReady()
        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 1)
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 3, observedSinceLastTick: false)

        let terminalError = await collection.value
        XCTAssertEqual(terminalError, .unresponsive)
    }

    func testHeartbeatNearTimerBoundaryStartsThreeFreshMissedIntervals() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "normal", clock: clock, performHello: false)
        await worker.markProtocolReady()
        clock.advance()
        await assertHeartbeatState(worker, missed: 1, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)

        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            projectId: projectID
        ))
        var events = worker.events().makeAsyncIterator()
        guard case .ack? = try await nextEnvelope(from: &events).event else {
            return XCTFail("Expected hello ACK")
        }
        guard case .heartbeat? = try await nextEnvelope(from: &events).event else {
            return XCTFail("Expected immediate heartbeat")
        }
        await assertHeartbeatState(worker, missed: 0, observedSinceLastTick: true)

        clock.advance()
        await assertHeartbeatState(worker, missed: 0, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 1, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 3, observedSinceLastTick: false)

        do {
            _ = try await worker.waitForTermination()
            XCTFail("Expected heartbeat timeout after three fresh missed intervals")
        } catch let error as WorkerProcessError {
            XCTAssertEqual(error, .unresponsive)
        }
    }

    func testProtocolReadyAfterProcessExitCannotArmOrCreateAnotherTerminalOutcome() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "immediate-exit", clock: clock)
        var exits: [Int32] = []
        for try await event in worker.events() {
            if case let .processExited(status) = event { exits.append(status) }
        }

        await worker.markProtocolReady()
        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 0)
        XCTAssertEqual(exits, [24])
        let terminalStatus = try await worker.waitForTermination()
        XCTAssertEqual(terminalStatus, 24)
    }

    func testFirstHeartbeatArmsSupervisionWithoutExplicitProtocolReady() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "normal", clock: clock, performHello: false)
        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 0)

        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            projectId: projectID
        ))
        var events = worker.events().makeAsyncIterator()
        guard case .ack? = try await nextEnvelope(from: &events).event else {
            return XCTFail("Expected hello ACK")
        }
        guard case .heartbeat? = try await nextEnvelope(from: &events).event else {
            return XCTFail("Expected immediate heartbeat")
        }
        XCTAssertEqual(clock.requestCount(for: .seconds(5)), 1)

        clock.advance()
        await assertHeartbeatState(worker, missed: 1, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)
        clock.advance()
        do {
            _ = try await worker.waitForTermination()
            XCTFail("Expected heartbeat timeout")
        } catch let error as WorkerProcessError {
            XCTAssertEqual(error, .unresponsive)
        }
    }

    func testHeartbeatResetsMissedIntervalsBeforeReady() async throws {
        let clock = ManualWorkerProcessClock()
        let worker = try await startWorker(mode: "normal", clock: clock, performHello: false)
        await worker.markProtocolReady()
        clock.advance()
        await assertHeartbeatState(worker, missed: 1, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)

        let helloID = UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!
        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            id: helloID,
            projectId: projectID
        ))
        var events = worker.events().makeAsyncIterator()
        guard case let .ack(commandID, command)? = try await nextEnvelope(from: &events).event else {
            return XCTFail("Expected hello ACK")
        }
        XCTAssertEqual(commandID, helloID)
        XCTAssertEqual(command, "hello")
        guard case .heartbeat? = try await nextEnvelope(from: &events).event else {
            return XCTFail("Expected immediate heartbeat")
        }
        await assertHeartbeatState(worker, missed: 0, observedSinceLastTick: true)
        XCTAssertReady(try await nextEnvelope(from: &events))

        clock.advance()
        await assertHeartbeatState(worker, missed: 0, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 1, observedSinceLastTick: false)
        let processGroupIdentifier = await worker.processGroupIdentifier
        XCTAssertNotNil(processGroupIdentifier)
        clock.advance()
        await assertHeartbeatState(worker, missed: 2, observedSinceLastTick: false)
        clock.advance()
        await assertHeartbeatState(worker, missed: 3, observedSinceLastTick: false)
        do {
            _ = try await worker.waitForTermination()
            XCTFail("Expected heartbeat timeout")
        } catch let error as WorkerProcessError {
            XCTAssertEqual(error, .unresponsive)
        }
    }

    func testChildEnvironmentIsExactReplacementAndDoesNotInheritSentinelSecret() async throws {
        let sentinelKey = "CLOUDPOINT_INHERITED_SENTINEL"
        XCTAssertEqual(setenv(sentinelKey, "must-not-leak", 1), 0)
        defer { unsetenv(sentinelKey) }
        let worker = try await startWorker(
            mode: "normal",
            environment: ["CLOUDPOINT_MOCK_REPORT_ENV_KEY": sentinelKey]
        )
        var diagnostics = worker.diagnostics().makeAsyncIterator()
        var report: String?
        while let line = await diagnostics.next() {
            if line.hasPrefix("env:\(sentinelKey)=") {
                report = line
                break
            }
        }
        XCTAssertEqual(report, "env:\(sentinelKey)=<absent>")

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

    func testLauncherEOFBeforeReadyFailsPromptlyAndClosesDescriptors() async throws {
        let baseline = try openFileDescriptorCount()
        do {
            _ = try await WorkerProcess.start(
                executable: mockWorkerURL(),
                launcherExecutable: URL(fileURLWithPath: "/usr/bin/true")
            )
            XCTFail("Expected launcher EOF failure")
        } catch let error as WorkerProcessError {
            guard case let .launchFailed(reason) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(reason.contains("launcher exited with status 0"))
        }
        XCTAssertLessThanOrEqual(try openFileDescriptorCount(), baseline + 1)
    }

    func testLauncherHandshakeTimeoutKillsUnverifiedGroupAndDescendant() async throws {
        let baseline = try openFileDescriptorCount()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let childPIDFile = directory.appendingPathComponent("child-pid")

        do {
            _ = try await WorkerProcess.start(
                executable: mockWorkerURL(),
                arguments: ["--mode", "silent"],
                environment: [
                    "CLOUDPOINT_MOCK_CHILD_PID_FILE": childPIDFile.path,
                    "CLOUDPOINT_MOCK_SET_PROCESS_GROUP": "1",
                    "CLOUDPOINT_MOCK_SPAWN_CHILD": "1",
                ],
                launcherExecutable: mockWorkerURL(),
                launcherHandshakeTimeout: .seconds(1)
            )
            XCTFail("Expected launcher handshake timeout")
        } catch let error as WorkerProcessError {
            guard case let .launchFailed(reason) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(reason.contains("handshake timed out"))
        }

        let childPIDText = try String(contentsOf: childPIDFile, encoding: .utf8)
        let childPID = try XCTUnwrap(pid_t(childPIDText))
        let disappearanceDeadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < disappearanceDeadline, kill(childPID, 0) == 0 {
            await Task.yield()
        }
        if kill(childPID, 0) == 0 { _ = kill(childPID, SIGKILL) }
        XCTAssertEqual(kill(childPID, 0), -1)
        XCTAssertEqual(errno, ESRCH)
        XCTAssertLessThanOrEqual(try openFileDescriptorCount(), baseline + 1)
    }

    func testDefaultStartUsesLauncherBundledWithApplication() async throws {
        let worker = try await WorkerProcess.start(
            executable: mockWorkerURL(),
            arguments: ["--mode", "heartbeat"],
            environment: [
                "HOME": NSHomeDirectory(),
                "TMPDIR": NSTemporaryDirectory(),
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
                "LC_ALL": "C",
                "LANG": "C",
            ]
        )
        try await performHelloHandshake(worker)
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

    private func childPID(
        from diagnostics: inout AsyncStream<String>.Iterator
    ) async -> pid_t? {
        while let line = await diagnostics.next() {
            if line.hasPrefix("child-pid:"), let value = Int32(line.dropFirst("child-pid:".count)) {
                return value
            }
        }
        return nil
    }

    private func startWorker(
        mode: String,
        environment: [String: String] = [:],
        clock: any WorkerProcessClock = ContinuousWorkerProcessClock(),
        performHello: Bool? = nil
    ) async throws -> WorkerProcess {
        let baseEnvironment = [
            "HOME": NSHomeDirectory(),
            "TMPDIR": NSTemporaryDirectory(),
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
            "LANG": "C",
        ]
        let worker = try await WorkerProcess.start(
            executable: mockWorkerURL(),
            arguments: ["--mode", mode],
            environment: baseEnvironment.merging(environment) { _, supplied in supplied },
            launcherExecutable: launcherURL(),
            launcherHandshakeTimeout: .seconds(1),
            clock: clock
        )
        let shouldPerformHello = performHello ?? !["silent", "immediate-exit"].contains(mode)
        if shouldPerformHello { try await performHelloHandshake(worker) }
        return worker
    }

    private func performHelloHandshake(_ worker: WorkerProcess) async throws {
        let commandID = UUID()
        var events = worker.events().makeAsyncIterator()
        try await worker.send(.command(
            .hello(clientVersion: "tests", supportedProtocolVersions: [1]),
            id: commandID,
            projectId: projectID
        ))
        let acknowledgement = try await nextEnvelope(from: &events)
        guard case let .ack(acknowledgedID, command)? = acknowledgement.event,
              acknowledgedID == commandID,
              command == "hello" else {
            throw WorkerProcessError.protocolFailure(.invalidPayload("hello ACK"))
        }
        await worker.markProtocolReady()
        let heartbeat = try await nextEnvelope(from: &events)
        guard case .heartbeat? = heartbeat.event else {
            throw WorkerProcessError.protocolFailure(.invalidPayload("immediate heartbeat"))
        }
    }

    private func assertHeartbeatState(
        _ worker: WorkerProcess,
        missed: Int,
        observedSinceLastTick: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        let deadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < deadline {
            let actualMissed = await worker.missedHeartbeatCountForTesting
            let actualObserved = await worker.heartbeatObservedSinceLastTickForTesting
            if actualMissed == missed, actualObserved == observedSinceLastTick { return }
            await Task.yield()
        }
        let actualMissed = await worker.missedHeartbeatCountForTesting
        let actualObserved = await worker.heartbeatObservedSinceLastTickForTesting
        XCTFail(
            "Heartbeat state was missed=\(actualMissed), observed=\(actualObserved); expected missed=\(missed), observed=\(observedSinceLastTick)",
            file: file,
            line: line
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

private final class WorkerProcessErrorRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var errors: [WorkerProcessError] = []

    func record(_ error: WorkerProcessError) {
        lock.withLock { errors.append(error) }
    }

    func contains(_ error: WorkerProcessError) -> Bool {
        lock.withLock { errors.contains(error) }
    }
}

private extension Array {
    func asyncMap<T: Sendable>(_ transform: (Element) async -> T) async -> [T] {
        var result: [T] = []
        for element in self { result.append(await transform(element)) }
        return result
    }
}

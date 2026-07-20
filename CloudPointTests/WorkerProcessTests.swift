import Darwin
import Foundation
import XCTest
@testable import CloudPoint

final class WorkerProcessTests: XCTestCase {
    private let projectID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!

    func testHeartbeatWorkerBecomesReadyAndTerminationEmitsOneExitAndClosesStreams() async throws {
        let baselineFDs = try openFileDescriptorCount()
        let worker = try await WorkerProcess.start(executable: try mockWorkerURL(), arguments: ["--mode", "heartbeat"])
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
        let worker = try await WorkerProcess.start(executable: try mockWorkerURL(), arguments: ["--mode", "normal"])
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
        let worker = try await WorkerProcess.start(executable: try mockWorkerURL(), arguments: ["--mode", "crash-after-ready"])
        var events = worker.events().makeAsyncIterator()
        XCTAssertReady(try await nextEnvelope(from: &events))

        var exits: [Int32] = []
        while let event = try await events.next() {
            if case let .processExited(status) = event { exits.append(status) }
        }
        XCTAssertEqual(exits, [23])
    }

    func testStderrDiagnosticsNeverEnterProtocolDecoder() async throws {
        let worker = try await WorkerProcess.start(executable: try mockWorkerURL(), arguments: ["--mode", "heartbeat"])
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
        let worker = try await WorkerProcess.start(
            executable: try mockWorkerURL(),
            arguments: ["--mode", "silent"],
            clock: clock
        )
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
        let worker = try await WorkerProcess.start(
            executable: try mockWorkerURL(),
            arguments: ["--mode", "heartbeat"],
            environment: ["CLOUDPOINT_MOCK_SPAWN_CHILD": "1"]
        )
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
        await worker.terminate()
        while try await events.next() != nil {}

        XCTAssertEqual(kill(pid, 0), -1)
        XCTAssertEqual(errno, ESRCH)
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
    private var continuation: AsyncStream<Void>.Continuation?

    func timer(interval: Duration) -> AsyncStream<Void> {
        AsyncStream { continuation in
            lock.withLock { self.continuation = continuation }
        }
    }

    func advance() {
        lock.withLock { continuation?.yield(()) }
    }
}

import Darwin
import Foundation

enum MockWorkerMode: String {
    case normal
    case heartbeat
    case crashAfterReady = "crash-after-ready"
    case fragmentedFinal = "fragmented-final"
    case immediateExit = "immediate-exit"
    case ignoreTerm = "ignore-term"
    case silent
}

let arguments = CommandLine.arguments
let modeIndex = arguments.firstIndex(of: "--mode")
let modeName = modeIndex.flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil } ?? "normal"
guard let mode = MockWorkerMode(rawValue: modeName) else {
    FileHandle.standardError.write(Data("unknown mode: \(modeName)\n".utf8))
    exit(64)
}

if ProcessInfo.processInfo.environment["CLOUDPOINT_MOCK_SET_PROCESS_GROUP"] == "1",
   setpgid(0, 0) != 0,
   getpgrp() != getpid() {
    FileHandle.standardError.write(Data("mock setpgid failed: \(errno)\n".utf8))
    exit(71)
}

if let markerPath = ProcessInfo.processInfo.environment["CLOUDPOINT_MOCK_START_MARKER"] {
    guard FileManager.default.createFile(atPath: markerPath, contents: Data()) else {
        FileHandle.standardError.write(Data("start marker creation failed\n".utf8))
        exit(73)
    }
}

if mode == .ignoreTerm { Darwin.signal(SIGTERM, SIG_IGN) }
let projectID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!
let outputQueue = DispatchQueue(label: "cloud.point.cloud.mock-worker.stdout")

func writeEnvelope(_ envelope: WorkerEnvelope) {
    do {
        let frame = try LengthPrefixedJSONCodec.encode(envelope)
        outputQueue.sync { FileHandle.standardOutput.write(frame) }
    } catch {
        FileHandle.standardError.write(Data("encode failed: \(error)\n".utf8))
        exit(70)
    }
}

FileHandle.standardError.write(Data("mode=\(mode.rawValue)\n".utf8))
if let count = ProcessInfo.processInfo.environment["CLOUDPOINT_MOCK_STDERR_BYTES"].flatMap(Int.init), count > 0 {
    FileHandle.standardError.write(Data(repeating: 0x64, count: count))
}

if ProcessInfo.processInfo.environment["CLOUDPOINT_MOCK_SPAWN_CHILD"] == "1" {
    var pid: pid_t = 0
    let executable = strdup("/bin/sleep")!
    let argumentZero = strdup("sleep")!
    let argumentOne = strdup("60")!
    var childArguments: [UnsafeMutablePointer<CChar>?] = [argumentZero, argumentOne, nil]
    var childEnvironment: [UnsafeMutablePointer<CChar>?] = ProcessInfo.processInfo.environment.map {
        strdup("\($0.key)=\($0.value)")
    }
    childEnvironment.append(nil)
    let spawnResult = childArguments.withUnsafeMutableBufferPointer { argumentsBuffer in
        childEnvironment.withUnsafeMutableBufferPointer { environmentBuffer in
            posix_spawn(&pid, executable, nil, nil, argumentsBuffer.baseAddress!, environmentBuffer.baseAddress!)
        }
    }
    free(executable)
    free(argumentZero)
    free(argumentOne)
    childEnvironment.dropLast().forEach { free($0) }
    guard spawnResult == 0 else {
        FileHandle.standardError.write(Data("child spawn failed: \(spawnResult)\n".utf8))
        exit(71)
    }
    FileHandle.standardError.write(Data("child-pid:\(pid)\n".utf8))
    if let childPIDPath = ProcessInfo.processInfo.environment["CLOUDPOINT_MOCK_CHILD_PID_FILE"] {
        do { try Data("\(pid)".utf8).write(to: URL(fileURLWithPath: childPIDPath)) }
        catch {
            FileHandle.standardError.write(Data("child pid file failed: \(error)\n".utf8))
            exit(73)
        }
    }
}

if mode == .immediateExit { exit(24) }

if mode != .silent {
    writeEnvelope(.event(.ready(
        engineVersion: "mock-1.0",
        modelIdentifier: "mock-depth",
        modelRevision: "test",
        convertedWeightsSHA256: String(repeating: "0", count: 64)
    ), projectId: projectID))
}

if mode == .crashAfterReady { exit(23) }
if mode == .fragmentedFinal {
    let payload = WorkerErrorPayload(code: "finalFragment", message: "final frame", recoverable: false, details: [:])
    let frame = try LengthPrefixedJSONCodec.encode(.event(.warning(payload), projectId: projectID))
    FileHandle.standardOutput.write(frame.prefix(3))
    FileHandle.standardOutput.write(frame.dropFirst(3))
    exit(0)
}

if let count = ProcessInfo.processInfo.environment["CLOUDPOINT_MOCK_EVENT_COUNT"].flatMap(Int.init), count > 0 {
    for index in 0..<count {
        writeEnvelope(.event(.heartbeat(
            busy: false,
            monotonicSeconds: Double(index),
            queuedFrames: 0,
            processedFrames: index,
            currentWindow: nil
        ), projectId: projectID))
    }
    FileHandle.standardError.write(Data("flood-complete\n".utf8))
}

if mode == .silent || mode == .ignoreTerm { dispatchMain() }

var heartbeatTimer: DispatchSourceTimer?
if mode == .heartbeat {
    func heartbeat() {
        writeEnvelope(.event(.heartbeat(
            busy: false,
            monotonicSeconds: ProcessInfo.processInfo.systemUptime,
            queuedFrames: 0,
            processedFrames: 0,
            currentWindow: nil
        ), projectId: projectID))
    }
    heartbeat()
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
    timer.schedule(deadline: .now() + 5, repeating: 5)
    timer.setEventHandler(handler: heartbeat)
    timer.resume()
    heartbeatTimer = timer
}

var decoder = LengthPrefixedJSONCodec.Decoder()
while true {
    let data = FileHandle.standardInput.availableData
    guard !data.isEmpty else { break }
    do {
        for envelope in try decoder.append(data) {
            guard let command = envelope.command else { continue }
            let commandName: String
            switch command {
            case .hello: commandName = "hello"
            case .configure: commandName = "configure"
            case .beginSession: commandName = "beginSession"
            case .enqueueFrame: commandName = "enqueueFrame"
            case .finishInput: commandName = "finishInput"
            case .pause: commandName = "pause"
            case .resume: commandName = "resume"
            case .cancel: commandName = "cancel"
            case .shutdown: commandName = "shutdown"
            }
            writeEnvelope(.event(.ack(commandId: envelope.id, command: commandName), projectId: envelope.projectId))
            if case .shutdown = command { exit(0) }
        }
    } catch {
        FileHandle.standardError.write(Data("decode failed: \(error)\n".utf8))
        exit(65)
    }
}

heartbeatTimer?.cancel()

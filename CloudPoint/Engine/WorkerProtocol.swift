import Foundation

enum WorkerProtocolError: Error, Sendable, Equatable {
    case zeroLengthPayload
    case payloadTooLarge(Int)
    case truncatedFrame
    case malformedEnvelope
    case unsupportedProtocolVersion(Int)
    case unknownMessageType(String)
    case duplicateCommandID(UUID)
    case invalidPayload(String)
}

enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { throw WorkerProtocolError.invalidPayload("invalid JSON value") }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case let .bool(value): try container.encode(value)
        case let .number(value):
            guard value.isFinite else { throw WorkerProtocolError.invalidPayload("nonfinite JSON number") }
            try container.encode(value)
        case let .string(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        }
    }
}

struct WorkerErrorPayload: Codable, Sendable, Equatable {
    var code: String
    var message: String
    var recoverable: Bool
    var details: [String: JSONValue]
}

enum WorkerModelProgressPhase: String, Codable, Sendable, Equatable {
    case validating
    case loading
}

enum WorkerCommand: Sendable, Equatable {
    case hello(clientVersion: String, supportedProtocolVersions: [Int])
    case configure(
        scaleFrames: Int,
        windowSize: Int,
        windowOverlap: Int,
        keyframeInterval: Int,
        cameraRefinementIterations: Int,
        confidenceThreshold: Double
    )
    case beginSession(resumeAfterFrameIndex: UInt32?)
    case enqueueFrame(frameIndex: UInt32, sourceTimestamp: Double, relativePath: String)
    case finishInput
    case pause
    case resume
    case cancel
    case shutdown
}

enum WorkerEvent: Sendable, Equatable {
    case ack(commandId: UUID, command: String)
    case error(commandId: UUID?, WorkerErrorPayload)
    case ready(engineVersion: String, modelIdentifier: String, modelRevision: String, convertedWeightsSHA256: String)
    case modelProgress(phase: WorkerModelProgressPhase, completed: Int, total: Int)
    case frameStarted(frameIndex: UInt32, windowIndex: Int)
    case frameCompleted(
        frameIndex: UInt32,
        windowIndex: Int,
        depthPath: String,
        confidencePath: String,
        geometryPath: String,
        pointChunkPath: String,
        durationSeconds: Double
    )
    case windowCompleted(
        windowIndex: Int,
        frameStart: UInt32,
        frameEnd: UInt32,
        pointChunkPath: String,
        alignmentTransform: [Double],
        lastProcessedFrameIndex: UInt32,
        inlierCount: Int,
        durationSeconds: Double
    )
    case sessionCompleted(processedFrames: Int, windowCount: Int, durationSeconds: Double)
    case paused(queuedFrames: Int, processedFrames: Int)
    case cancelled(lastCompletedWindowIndex: Int?)
    case warning(WorkerErrorPayload)
    case heartbeat(busy: Bool, monotonicSeconds: Double, queuedFrames: Int, processedFrames: Int, currentWindow: Int?)
}

struct WorkerEnvelope: Codable, Sendable, Equatable {
    static let currentProtocolVersion = 1

    let protocolVersion: Int
    let id: UUID
    let projectId: UUID
    let message: Message

    enum Message: Sendable, Equatable {
        case command(WorkerCommand)
        case event(WorkerEvent)
    }

    private init(protocolVersion: Int, id: UUID, projectId: UUID, message: Message) {
        self.protocolVersion = protocolVersion
        self.id = id
        self.projectId = projectId
        self.message = message
    }

    static func command(
        _ command: WorkerCommand,
        id: UUID = UUID(),
        projectId: UUID = UUID()
    ) -> WorkerEnvelope {
        WorkerEnvelope(protocolVersion: currentProtocolVersion, id: id, projectId: projectId, message: .command(command))
    }

    static func event(
        _ event: WorkerEvent,
        id: UUID = UUID(),
        projectId: UUID = UUID()
    ) -> WorkerEnvelope {
        WorkerEnvelope(protocolVersion: currentProtocolVersion, id: id, projectId: projectId, message: .event(event))
    }

    var command: WorkerCommand? {
        guard case let .command(command) = message else { return nil }
        return command
    }

    var event: WorkerEvent? {
        guard case let .event(event) = message else { return nil }
        return event
    }

    private enum CodingKeys: String, CodingKey {
        case protocolVersion, id, projectId, type, commandId, payload
    }

    private struct EmptyPayload: Codable, Equatable {}
    private struct HelloPayload: Codable { let clientVersion: String; let supportedProtocolVersions: [Int] }
    private struct ConfigurePayload: Codable {
        let scaleFrames: Int; let windowSize: Int; let windowOverlap: Int; let keyframeInterval: Int
        let cameraRefinementIterations: Int; let confidenceThreshold: Double
    }
    private struct BeginSessionPayload: Codable {
        let resumeAfterFrameIndex: UInt32?
        private enum CodingKeys: String, CodingKey { case resumeAfterFrameIndex }
        init(resumeAfterFrameIndex: UInt32?) { self.resumeAfterFrameIndex = resumeAfterFrameIndex }
        init(from decoder: Decoder) throws {
            resumeAfterFrameIndex = try decoder.container(keyedBy: CodingKeys.self).decode(UInt32?.self, forKey: .resumeAfterFrameIndex)
        }
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(resumeAfterFrameIndex, forKey: .resumeAfterFrameIndex)
        }
    }
    private struct EnqueueFramePayload: Codable { let frameIndex: UInt32; let sourceTimestamp: Double; let relativePath: String }
    private struct AckPayload: Codable { let command: String }
    private struct ReadyPayload: Codable { let engineVersion: String; let modelIdentifier: String; let modelRevision: String; let convertedWeightsSHA256: String }
    private struct ModelProgressPayload: Codable { let phase: WorkerModelProgressPhase; let completed: Int; let total: Int }
    private struct FrameStartedPayload: Codable { let frameIndex: UInt32; let windowIndex: Int }
    private struct FrameCompletedPayload: Codable {
        let frameIndex: UInt32; let windowIndex: Int; let depthPath: String; let confidencePath: String
        let geometryPath: String; let pointChunkPath: String; let durationSeconds: Double
    }
    private struct WindowCompletedPayload: Codable {
        let windowIndex: Int; let frameStart: UInt32; let frameEnd: UInt32; let pointChunkPath: String
        let alignmentTransform: [Double]; let lastProcessedFrameIndex: UInt32; let inlierCount: Int; let durationSeconds: Double
    }
    private struct SessionCompletedPayload: Codable { let processedFrames: Int; let windowCount: Int; let durationSeconds: Double }
    private struct PausedPayload: Codable { let queuedFrames: Int; let processedFrames: Int }
    private struct CancelledPayload: Codable {
        let lastCompletedWindowIndex: Int?
        private enum CodingKeys: String, CodingKey { case lastCompletedWindowIndex }
        init(lastCompletedWindowIndex: Int?) { self.lastCompletedWindowIndex = lastCompletedWindowIndex }
        init(from decoder: Decoder) throws {
            lastCompletedWindowIndex = try decoder.container(keyedBy: CodingKeys.self).decode(Int?.self, forKey: .lastCompletedWindowIndex)
        }
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(lastCompletedWindowIndex, forKey: .lastCompletedWindowIndex)
        }
    }
    private struct HeartbeatPayload: Codable {
        let busy: Bool; let monotonicSeconds: Double; let queuedFrames: Int; let processedFrames: Int; let currentWindow: Int?
        private enum CodingKeys: String, CodingKey { case busy, monotonicSeconds, queuedFrames, processedFrames, currentWindow }
        init(busy: Bool, monotonicSeconds: Double, queuedFrames: Int, processedFrames: Int, currentWindow: Int?) {
            self.busy = busy; self.monotonicSeconds = monotonicSeconds; self.queuedFrames = queuedFrames
            self.processedFrames = processedFrames; self.currentWindow = currentWindow
        }
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            busy = try container.decode(Bool.self, forKey: .busy)
            monotonicSeconds = try container.decode(Double.self, forKey: .monotonicSeconds)
            queuedFrames = try container.decode(Int.self, forKey: .queuedFrames)
            processedFrames = try container.decode(Int.self, forKey: .processedFrames)
            currentWindow = try container.decode(Int?.self, forKey: .currentWindow)
        }
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(busy, forKey: .busy)
            try container.encode(monotonicSeconds, forKey: .monotonicSeconds)
            try container.encode(queuedFrames, forKey: .queuedFrames)
            try container.encode(processedFrames, forKey: .processedFrames)
            try container.encode(currentWindow, forKey: .currentWindow)
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let version = try container.decode(Int.self, forKey: .protocolVersion)
        guard version == Self.currentProtocolVersion else { throw WorkerProtocolError.unsupportedProtocolVersion(version) }
        protocolVersion = version
        id = try container.decode(UUID.self, forKey: .id)
        projectId = try container.decode(UUID.self, forKey: .projectId)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "hello":
            let payload = try container.decode(HelloPayload.self, forKey: .payload)
            message = .command(.hello(clientVersion: payload.clientVersion, supportedProtocolVersions: payload.supportedProtocolVersions))
        case "configure":
            let p = try container.decode(ConfigurePayload.self, forKey: .payload)
            message = .command(.configure(scaleFrames: p.scaleFrames, windowSize: p.windowSize, windowOverlap: p.windowOverlap, keyframeInterval: p.keyframeInterval, cameraRefinementIterations: p.cameraRefinementIterations, confidenceThreshold: p.confidenceThreshold))
        case "beginSession":
            let p = try container.decode(BeginSessionPayload.self, forKey: .payload)
            message = .command(.beginSession(resumeAfterFrameIndex: p.resumeAfterFrameIndex))
        case "enqueueFrame":
            let p = try container.decode(EnqueueFramePayload.self, forKey: .payload)
            message = .command(.enqueueFrame(frameIndex: p.frameIndex, sourceTimestamp: p.sourceTimestamp, relativePath: p.relativePath))
        case "finishInput": message = try Self.decodeEmpty(.finishInput, from: container)
        case "pause": message = try Self.decodeEmpty(.pause, from: container)
        case "resume": message = try Self.decodeEmpty(.resume, from: container)
        case "cancel": message = try Self.decodeEmpty(.cancel, from: container)
        case "shutdown": message = try Self.decodeEmpty(.shutdown, from: container)
        case "ack":
            let p = try container.decode(AckPayload.self, forKey: .payload)
            message = .event(.ack(commandId: try container.decode(UUID.self, forKey: .commandId), command: p.command))
        case "error":
            let p = try container.decode(WorkerErrorPayload.self, forKey: .payload)
            message = .event(.error(commandId: try container.decode(UUID?.self, forKey: .commandId), p))
        case "ready":
            let p = try container.decode(ReadyPayload.self, forKey: .payload)
            message = .event(.ready(engineVersion: p.engineVersion, modelIdentifier: p.modelIdentifier, modelRevision: p.modelRevision, convertedWeightsSHA256: p.convertedWeightsSHA256))
        case "modelProgress":
            let p = try container.decode(ModelProgressPayload.self, forKey: .payload)
            message = .event(.modelProgress(phase: p.phase, completed: p.completed, total: p.total))
        case "frameStarted":
            let p = try container.decode(FrameStartedPayload.self, forKey: .payload)
            message = .event(.frameStarted(frameIndex: p.frameIndex, windowIndex: p.windowIndex))
        case "frameCompleted":
            let p = try container.decode(FrameCompletedPayload.self, forKey: .payload)
            message = .event(.frameCompleted(frameIndex: p.frameIndex, windowIndex: p.windowIndex, depthPath: p.depthPath, confidencePath: p.confidencePath, geometryPath: p.geometryPath, pointChunkPath: p.pointChunkPath, durationSeconds: p.durationSeconds))
        case "windowCompleted":
            let p = try container.decode(WindowCompletedPayload.self, forKey: .payload)
            message = .event(.windowCompleted(windowIndex: p.windowIndex, frameStart: p.frameStart, frameEnd: p.frameEnd, pointChunkPath: p.pointChunkPath, alignmentTransform: p.alignmentTransform, lastProcessedFrameIndex: p.lastProcessedFrameIndex, inlierCount: p.inlierCount, durationSeconds: p.durationSeconds))
        case "sessionCompleted":
            let p = try container.decode(SessionCompletedPayload.self, forKey: .payload)
            message = .event(.sessionCompleted(processedFrames: p.processedFrames, windowCount: p.windowCount, durationSeconds: p.durationSeconds))
        case "paused":
            let p = try container.decode(PausedPayload.self, forKey: .payload)
            message = .event(.paused(queuedFrames: p.queuedFrames, processedFrames: p.processedFrames))
        case "cancelled":
            let p = try container.decode(CancelledPayload.self, forKey: .payload)
            message = .event(.cancelled(lastCompletedWindowIndex: p.lastCompletedWindowIndex))
        case "warning": message = .event(.warning(try container.decode(WorkerErrorPayload.self, forKey: .payload)))
        case "heartbeat":
            let p = try container.decode(HeartbeatPayload.self, forKey: .payload)
            message = .event(.heartbeat(busy: p.busy, monotonicSeconds: p.monotonicSeconds, queuedFrames: p.queuedFrames, processedFrames: p.processedFrames, currentWindow: p.currentWindow))
        default: throw WorkerProtocolError.unknownMessageType(type)
        }
        try validate()
    }

    func encode(to encoder: Encoder) throws {
        guard protocolVersion == Self.currentProtocolVersion else { throw WorkerProtocolError.unsupportedProtocolVersion(protocolVersion) }
        try validate()
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(protocolVersion, forKey: .protocolVersion)
        try container.encode(id, forKey: .id)
        try container.encode(projectId, forKey: .projectId)

        switch message {
        case let .command(command): try encode(command, into: &container)
        case let .event(event): try encode(event, into: &container)
        }
    }

    private static func decodeEmpty(_ command: WorkerCommand, from container: KeyedDecodingContainer<CodingKeys>) throws -> Message {
        _ = try container.decode(EmptyPayload.self, forKey: .payload)
        return .command(command)
    }

    private func encode(_ command: WorkerCommand, into c: inout KeyedEncodingContainer<CodingKeys>) throws {
        switch command {
        case let .hello(clientVersion, versions):
            try c.encode("hello", forKey: .type); try c.encode(HelloPayload(clientVersion: clientVersion, supportedProtocolVersions: versions), forKey: .payload)
        case let .configure(scaleFrames, windowSize, overlap, keyframe, iterations, confidence):
            try c.encode("configure", forKey: .type); try c.encode(ConfigurePayload(scaleFrames: scaleFrames, windowSize: windowSize, windowOverlap: overlap, keyframeInterval: keyframe, cameraRefinementIterations: iterations, confidenceThreshold: confidence), forKey: .payload)
        case let .beginSession(index):
            try c.encode("beginSession", forKey: .type); try c.encode(BeginSessionPayload(resumeAfterFrameIndex: index), forKey: .payload)
        case let .enqueueFrame(index, timestamp, path):
            try c.encode("enqueueFrame", forKey: .type); try c.encode(EnqueueFramePayload(frameIndex: index, sourceTimestamp: timestamp, relativePath: path), forKey: .payload)
        case .finishInput: try encodeEmpty("finishInput", into: &c)
        case .pause: try encodeEmpty("pause", into: &c)
        case .resume: try encodeEmpty("resume", into: &c)
        case .cancel: try encodeEmpty("cancel", into: &c)
        case .shutdown: try encodeEmpty("shutdown", into: &c)
        }
    }

    private func encode(_ event: WorkerEvent, into c: inout KeyedEncodingContainer<CodingKeys>) throws {
        switch event {
        case let .ack(commandId, command):
            try c.encode("ack", forKey: .type); try c.encode(commandId, forKey: .commandId); try c.encode(AckPayload(command: command), forKey: .payload)
        case let .error(commandId, payload):
            try c.encode("error", forKey: .type); try c.encode(commandId, forKey: .commandId); try c.encode(payload, forKey: .payload)
        case let .ready(engine, model, revision, sha):
            try c.encode("ready", forKey: .type); try c.encode(ReadyPayload(engineVersion: engine, modelIdentifier: model, modelRevision: revision, convertedWeightsSHA256: sha), forKey: .payload)
        case let .modelProgress(phase, completed, total):
            try c.encode("modelProgress", forKey: .type); try c.encode(ModelProgressPayload(phase: phase, completed: completed, total: total), forKey: .payload)
        case let .frameStarted(frame, window):
            try c.encode("frameStarted", forKey: .type); try c.encode(FrameStartedPayload(frameIndex: frame, windowIndex: window), forKey: .payload)
        case let .frameCompleted(frame, window, depth, confidence, geometry, points, duration):
            try c.encode("frameCompleted", forKey: .type); try c.encode(FrameCompletedPayload(frameIndex: frame, windowIndex: window, depthPath: depth, confidencePath: confidence, geometryPath: geometry, pointChunkPath: points, durationSeconds: duration), forKey: .payload)
        case let .windowCompleted(window, start, end, path, transform, last, inliers, duration):
            try c.encode("windowCompleted", forKey: .type); try c.encode(WindowCompletedPayload(windowIndex: window, frameStart: start, frameEnd: end, pointChunkPath: path, alignmentTransform: transform, lastProcessedFrameIndex: last, inlierCount: inliers, durationSeconds: duration), forKey: .payload)
        case let .sessionCompleted(frames, windows, duration):
            try c.encode("sessionCompleted", forKey: .type); try c.encode(SessionCompletedPayload(processedFrames: frames, windowCount: windows, durationSeconds: duration), forKey: .payload)
        case let .paused(queued, processed):
            try c.encode("paused", forKey: .type); try c.encode(PausedPayload(queuedFrames: queued, processedFrames: processed), forKey: .payload)
        case let .cancelled(last):
            try c.encode("cancelled", forKey: .type); try c.encode(CancelledPayload(lastCompletedWindowIndex: last), forKey: .payload)
        case let .warning(payload):
            try c.encode("warning", forKey: .type); try c.encode(payload, forKey: .payload)
        case let .heartbeat(busy, seconds, queued, processed, current):
            try c.encode("heartbeat", forKey: .type); try c.encode(HeartbeatPayload(busy: busy, monotonicSeconds: seconds, queuedFrames: queued, processedFrames: processed, currentWindow: current), forKey: .payload)
        }
    }

    private func encodeEmpty(_ type: String, into c: inout KeyedEncodingContainer<CodingKeys>) throws {
        try c.encode(type, forKey: .type)
        try c.encode(EmptyPayload(), forKey: .payload)
    }

    private func validate() throws {
        switch message {
        case let .command(command): try Self.validate(command)
        case let .event(event): try Self.validate(event)
        }
    }

    private static func validate(_ command: WorkerCommand) throws {
        switch command {
        case let .hello(clientVersion, versions):
            guard !clientVersion.isEmpty, versions.contains(currentProtocolVersion) else { throw WorkerProtocolError.invalidPayload("hello") }
        case let .configure(scale, window, overlap, keyframe, iterations, confidence):
            guard scale > 0, window > 0, overlap >= 0, overlap < window, keyframe > 0, iterations > 0, confidence.isFinite, confidence > 0 else { throw WorkerProtocolError.invalidPayload("configure") }
        case .beginSession: break
        case let .enqueueFrame(_, timestamp, path):
            guard timestamp.isFinite, timestamp >= 0, isSafeRelativePath(path) else { throw WorkerProtocolError.invalidPayload("enqueueFrame") }
        case .finishInput, .pause, .resume, .cancel, .shutdown: break
        }
    }

    private static func validate(_ event: WorkerEvent) throws {
        switch event {
        case let .ack(_, command): guard !command.isEmpty else { throw WorkerProtocolError.invalidPayload("ack") }
        case let .error(_, payload), let .warning(payload): try validate(payload)
        case let .ready(engine, model, revision, sha):
            guard !engine.isEmpty, !model.isEmpty, !revision.isEmpty, !sha.isEmpty else { throw WorkerProtocolError.invalidPayload("ready") }
        case let .modelProgress(_, completed, total):
            guard completed >= 0, total >= 0, completed <= total else { throw WorkerProtocolError.invalidPayload("modelProgress") }
        case let .frameStarted(_, window): guard window >= 0 else { throw WorkerProtocolError.invalidPayload("frameStarted") }
        case let .frameCompleted(_, window, depth, confidence, geometry, points, duration):
            guard window >= 0, [depth, confidence, geometry, points].allSatisfy(isSafeRelativePath), validDuration(duration) else { throw WorkerProtocolError.invalidPayload("frameCompleted") }
        case let .windowCompleted(window, start, end, path, transform, last, inliers, duration):
            guard window >= 0, start <= end, last >= end, isSafeRelativePath(path), transform.count == 16, transform.allSatisfy(\.isFinite), inliers >= 0, validDuration(duration) else { throw WorkerProtocolError.invalidPayload("windowCompleted") }
        case let .sessionCompleted(frames, windows, duration):
            guard frames >= 0, windows >= 0, validDuration(duration) else { throw WorkerProtocolError.invalidPayload("sessionCompleted") }
        case let .paused(queued, processed):
            guard queued >= 0, processed >= 0 else { throw WorkerProtocolError.invalidPayload("paused") }
        case let .cancelled(last): guard last == nil || last! >= 0 else { throw WorkerProtocolError.invalidPayload("cancelled") }
        case let .heartbeat(_, seconds, queued, processed, current):
            guard validDuration(seconds), queued >= 0, processed >= 0, current == nil || current! >= 0 else { throw WorkerProtocolError.invalidPayload("heartbeat") }
        }
    }

    private static func validate(_ payload: WorkerErrorPayload) throws {
        guard !payload.code.isEmpty, !payload.message.isEmpty else { throw WorkerProtocolError.invalidPayload("error") }
    }

    private static func validDuration(_ value: Double) -> Bool { value.isFinite && value >= 0 }

    private static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.hasPrefix("~"), !path.contains("\\"), !path.contains("\0") else { return false }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        return components.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

enum LengthPrefixedJSONCodec {
    static let maximumPayloadBytes = 1_048_576

    static func encode(_ envelope: WorkerEnvelope) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let payload = try encoder.encode(envelope)
        guard !payload.isEmpty else { throw WorkerProtocolError.zeroLengthPayload }
        guard payload.count <= maximumPayloadBytes else { throw WorkerProtocolError.payloadTooLarge(payload.count) }

        var frame = Data()
        var length = UInt32(payload.count).bigEndian
        Swift.withUnsafeBytes(of: &length) { frame.append(contentsOf: $0) }
        frame.append(payload)
        return frame
    }

    struct Decoder: Sendable {
        private let maxPayloadBytes: Int
        private var buffer = Data()
        private var recentCommandIDs: [UUID] = []
        private var recentCommandIDSet: Set<UUID> = []

        init(maxPayloadBytes: Int = LengthPrefixedJSONCodec.maximumPayloadBytes) {
            self.maxPayloadBytes = maxPayloadBytes
        }

        mutating func append<S: DataProtocol>(_ bytes: S) throws -> [WorkerEnvelope] {
            buffer.append(contentsOf: bytes)
            var envelopes: [WorkerEnvelope] = []

            while buffer.count >= 4 {
                let length = buffer.prefix(4).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
                guard length > 0 else { throw WorkerProtocolError.zeroLengthPayload }
                guard length <= maxPayloadBytes else { throw WorkerProtocolError.payloadTooLarge(Int(length)) }
                let frameLength = 4 + Int(length)
                guard buffer.count >= frameLength else { break }

                let payload = buffer.subdata(in: 4..<frameLength)
                let envelope: WorkerEnvelope
                do { envelope = try JSONDecoder().decode(WorkerEnvelope.self, from: payload) }
                catch let error as WorkerProtocolError { throw error }
                catch { throw WorkerProtocolError.malformedEnvelope }

                if envelope.command != nil {
                    guard recentCommandIDSet.insert(envelope.id).inserted else {
                        throw WorkerProtocolError.duplicateCommandID(envelope.id)
                    }
                    recentCommandIDs.append(envelope.id)
                    if recentCommandIDs.count > 4_096 {
                        recentCommandIDSet.remove(recentCommandIDs.removeFirst())
                    }
                }
                envelopes.append(envelope)
                buffer = Data(buffer.dropFirst(frameLength))
            }
            return envelopes
        }

        func finish() throws {
            guard buffer.isEmpty else { throw WorkerProtocolError.truncatedFrame }
        }
    }
}

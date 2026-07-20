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

private extension CodingUserInfoKey {
    static let exactJSONNumberEncoding = CodingUserInfoKey(rawValue: "cloud.point.cloud.exactJSONNumberEncoding")!
    static let exactJSONNumberDecodingPrefix = CodingUserInfoKey(rawValue: "cloud.point.cloud.exactJSONNumberDecodingPrefix")!
}

private final class ExactJSONNumberEncodingContext: @unchecked Sendable {
    let prefix = "__CLOUDPOINT_JSON_NUMBER_\(UUID().uuidString)__:"
    private(set) var replacements: [(placeholder: String, token: String)] = []

    func placeholder(for token: String) -> String {
        let placeholder = "\(prefix)\(replacements.count)"
        replacements.append((placeholder, token))
        return placeholder
    }
}

enum JSONNumber: Codable, Sendable, Equatable, ExpressibleByIntegerLiteral {
    case signed(Int64)
    case unsigned(UInt64)
    case decimal(Decimal)
    case raw(String)

    init(integerLiteral value: Int64) { self = .signed(value) }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let prefix = decoder.userInfo[.exactJSONNumberDecodingPrefix] as? String,
           let placeholder = try? container.decode(String.self),
           placeholder.hasPrefix(prefix) {
            let token = String(placeholder.dropFirst(prefix.count))
            guard Self.isValidToken(token) else { throw WorkerProtocolError.invalidPayload("invalid JSON number") }
            self = .raw(token)
            return
        }
        if let value = try? container.decode(Int64.self) { self = .signed(value) }
        else if let value = try? container.decode(UInt64.self) { self = .unsigned(value) }
        else { self = .decimal(try container.decode(Decimal.self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let context = encoder.userInfo[.exactJSONNumberEncoding] as? ExactJSONNumberEncodingContext {
            try container.encode(context.placeholder(for: try wireToken()))
            return
        }
        switch self {
        case let .signed(value): try container.encode(value)
        case let .unsigned(value): try container.encode(value)
        case let .decimal(value): try container.encode(value)
        case .raw: throw WorkerProtocolError.invalidPayload("exact JSON number requires protocol codec")
        }
    }

    static func == (lhs: JSONNumber, rhs: JSONNumber) -> Bool {
        guard let left = try? lhs.wireToken(), let right = try? rhs.wireToken() else { return false }
        return left == right
    }

    private func wireToken() throws -> String {
        let token: String
        switch self {
        case let .signed(value): token = String(value)
        case let .unsigned(value): token = String(value)
        case let .decimal(value): token = NSDecimalNumber(decimal: value).stringValue
        case let .raw(value): token = value
        }
        guard Self.isValidToken(token) else { throw WorkerProtocolError.invalidPayload("invalid JSON number") }
        return token
    }

    private static func isValidToken(_ token: String) -> Bool {
        token.range(
            of: #"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$"#,
            options: .regularExpression
        ) != nil
    }
}

enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(JSONNumber)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(JSONNumber.self) { self = .number(value) }
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
        case let .number(value): try container.encode(value)
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

    private struct AnyCodingKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
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
        let topLevelKeys = ["protocolVersion", "id", "projectId", "type", "payload"] + (["ack", "error"].contains(type) ? ["commandId"] : [])
        try Self.requireExactKeys(topLevelKeys, in: decoder, context: "envelope")

        switch type {
        case "hello":
            let payload = try Self.decodePayload(HelloPayload.self, keys: ["clientVersion", "supportedProtocolVersions"], from: container)
            message = .command(.hello(clientVersion: payload.clientVersion, supportedProtocolVersions: payload.supportedProtocolVersions))
        case "configure":
            let p = try Self.decodePayload(ConfigurePayload.self, keys: ["scaleFrames", "windowSize", "windowOverlap", "keyframeInterval", "cameraRefinementIterations", "confidenceThreshold"], from: container)
            message = .command(.configure(scaleFrames: p.scaleFrames, windowSize: p.windowSize, windowOverlap: p.windowOverlap, keyframeInterval: p.keyframeInterval, cameraRefinementIterations: p.cameraRefinementIterations, confidenceThreshold: p.confidenceThreshold))
        case "beginSession":
            let p = try Self.decodePayload(BeginSessionPayload.self, keys: ["resumeAfterFrameIndex"], from: container)
            message = .command(.beginSession(resumeAfterFrameIndex: p.resumeAfterFrameIndex))
        case "enqueueFrame":
            let p = try Self.decodePayload(EnqueueFramePayload.self, keys: ["frameIndex", "sourceTimestamp", "relativePath"], from: container)
            message = .command(.enqueueFrame(frameIndex: p.frameIndex, sourceTimestamp: p.sourceTimestamp, relativePath: p.relativePath))
        case "finishInput": message = try Self.decodeEmpty(.finishInput, from: container)
        case "pause": message = try Self.decodeEmpty(.pause, from: container)
        case "resume": message = try Self.decodeEmpty(.resume, from: container)
        case "cancel": message = try Self.decodeEmpty(.cancel, from: container)
        case "shutdown": message = try Self.decodeEmpty(.shutdown, from: container)
        case "ack":
            let p = try Self.decodePayload(AckPayload.self, keys: ["command"], from: container)
            message = .event(.ack(commandId: try container.decode(UUID.self, forKey: .commandId), command: p.command))
        case "error":
            let p = try Self.decodePayload(WorkerErrorPayload.self, keys: ["code", "message", "recoverable", "details"], from: container)
            message = .event(.error(commandId: try container.decode(UUID?.self, forKey: .commandId), p))
        case "ready":
            let p = try Self.decodePayload(ReadyPayload.self, keys: ["engineVersion", "modelIdentifier", "modelRevision", "convertedWeightsSHA256"], from: container)
            message = .event(.ready(engineVersion: p.engineVersion, modelIdentifier: p.modelIdentifier, modelRevision: p.modelRevision, convertedWeightsSHA256: p.convertedWeightsSHA256))
        case "modelProgress":
            let p = try Self.decodePayload(ModelProgressPayload.self, keys: ["phase", "completed", "total"], from: container)
            message = .event(.modelProgress(phase: p.phase, completed: p.completed, total: p.total))
        case "frameStarted":
            let p = try Self.decodePayload(FrameStartedPayload.self, keys: ["frameIndex", "windowIndex"], from: container)
            message = .event(.frameStarted(frameIndex: p.frameIndex, windowIndex: p.windowIndex))
        case "frameCompleted":
            let p = try Self.decodePayload(FrameCompletedPayload.self, keys: ["frameIndex", "windowIndex", "depthPath", "confidencePath", "geometryPath", "pointChunkPath", "durationSeconds"], from: container)
            message = .event(.frameCompleted(frameIndex: p.frameIndex, windowIndex: p.windowIndex, depthPath: p.depthPath, confidencePath: p.confidencePath, geometryPath: p.geometryPath, pointChunkPath: p.pointChunkPath, durationSeconds: p.durationSeconds))
        case "windowCompleted":
            let p = try Self.decodePayload(WindowCompletedPayload.self, keys: ["windowIndex", "frameStart", "frameEnd", "pointChunkPath", "alignmentTransform", "lastProcessedFrameIndex", "inlierCount", "durationSeconds"], from: container)
            message = .event(.windowCompleted(windowIndex: p.windowIndex, frameStart: p.frameStart, frameEnd: p.frameEnd, pointChunkPath: p.pointChunkPath, alignmentTransform: p.alignmentTransform, lastProcessedFrameIndex: p.lastProcessedFrameIndex, inlierCount: p.inlierCount, durationSeconds: p.durationSeconds))
        case "sessionCompleted":
            let p = try Self.decodePayload(SessionCompletedPayload.self, keys: ["processedFrames", "windowCount", "durationSeconds"], from: container)
            message = .event(.sessionCompleted(processedFrames: p.processedFrames, windowCount: p.windowCount, durationSeconds: p.durationSeconds))
        case "paused":
            let p = try Self.decodePayload(PausedPayload.self, keys: ["queuedFrames", "processedFrames"], from: container)
            message = .event(.paused(queuedFrames: p.queuedFrames, processedFrames: p.processedFrames))
        case "cancelled":
            let p = try Self.decodePayload(CancelledPayload.self, keys: ["lastCompletedWindowIndex"], from: container)
            message = .event(.cancelled(lastCompletedWindowIndex: p.lastCompletedWindowIndex))
        case "warning": message = .event(.warning(try Self.decodePayload(WorkerErrorPayload.self, keys: ["code", "message", "recoverable", "details"], from: container)))
        case "heartbeat":
            let p = try Self.decodePayload(HeartbeatPayload.self, keys: ["busy", "monotonicSeconds", "queuedFrames", "processedFrames", "currentWindow"], from: container)
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
        _ = try decodePayload(EmptyPayload.self, keys: [], from: container)
        return .command(command)
    }

    private static func decodePayload<T: Decodable>(
        _ type: T.Type,
        keys: [String],
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> T {
        let decoder = try container.superDecoder(forKey: .payload)
        try requireExactKeys(keys, in: decoder, context: "payload")
        return try T(from: decoder)
    }

    private static func requireExactKeys(_ expected: [String], in decoder: Decoder, context: String) throws {
        let container = try decoder.container(keyedBy: AnyCodingKey.self)
        let actual = Set(container.allKeys.map(\.stringValue))
        guard actual == Set(expected) else {
            throw WorkerProtocolError.invalidPayload("unexpected \(context) keys")
        }
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

private struct ExactJSONNumberDecodingRewriter {
    private static let maximumExpansionFactor = 32

    private let bytes: [UInt8]
    private let prefix: String
    private let maximumOutputBytes: Int
    private var index = 0
    private var output: [UInt8] = []
    private(set) var hasPlaceholderCollision = false

    init(payload: Data, prefix: String, maximumPayloadBytes: Int) {
        bytes = Array(payload)
        self.prefix = prefix
        maximumOutputBytes = maximumPayloadBytes * Self.maximumExpansionFactor
        output.reserveCapacity(min(payload.count * 2, maximumOutputBytes))
    }

    mutating func rewrite() throws -> Data {
        try copyWhitespace()
        try parseValue(preserveExactNumbers: false)
        try copyWhitespace()
        guard index == bytes.count else { throw WorkerProtocolError.malformedEnvelope }
        return Data(output)
    }

    private mutating func parseValue(preserveExactNumbers: Bool) throws {
        guard let byte = current else { throw WorkerProtocolError.malformedEnvelope }
        switch byte {
        case 0x7B: try parseObject(preserveExactNumbers: preserveExactNumbers)
        case 0x5B: try parseArray(preserveExactNumbers: preserveExactNumbers)
        case 0x22:
            let range = try parseString()
            if preserveExactNumbers,
               let decoded = try? JSONDecoder().decode(String.self, from: Data(bytes[range])),
               decoded.hasPrefix(prefix) {
                hasPlaceholderCollision = true
            }
        case 0x74: try copyLiteral("true")
        case 0x66: try copyLiteral("false")
        case 0x6E: try copyLiteral("null")
        case 0x2D, 0x30...0x39: try parseNumber(preserveExact: preserveExactNumbers)
        default: throw WorkerProtocolError.malformedEnvelope
        }
    }

    private mutating func parseObject(preserveExactNumbers: Bool) throws {
        try copyExpected(0x7B)
        try copyWhitespace()
        if current == 0x7D {
            try copyExpected(0x7D)
            return
        }
        while true {
            guard current == 0x22 else { throw WorkerProtocolError.malformedEnvelope }
            let keyRange = try parseString()
            let key: String
            do { key = try JSONDecoder().decode(String.self, from: Data(bytes[keyRange])) }
            catch { throw WorkerProtocolError.malformedEnvelope }
            try copyWhitespace()
            try copyExpected(0x3A)
            try copyWhitespace()
            try parseValue(preserveExactNumbers: preserveExactNumbers || key == "details")
            try copyWhitespace()
            if current == 0x7D {
                try copyExpected(0x7D)
                return
            }
            try copyExpected(0x2C)
            try copyWhitespace()
        }
    }

    private mutating func parseArray(preserveExactNumbers: Bool) throws {
        try copyExpected(0x5B)
        try copyWhitespace()
        if current == 0x5D {
            try copyExpected(0x5D)
            return
        }
        while true {
            try parseValue(preserveExactNumbers: preserveExactNumbers)
            try copyWhitespace()
            if current == 0x5D {
                try copyExpected(0x5D)
                return
            }
            try copyExpected(0x2C)
            try copyWhitespace()
        }
    }

    private mutating func parseString() throws -> Range<Int> {
        let start = index
        guard current == 0x22 else { throw WorkerProtocolError.malformedEnvelope }
        index += 1
        while let byte = current {
            if byte == 0x22 {
                index += 1
                let range = start..<index
                try append(bytes[range])
                return range
            }
            if byte == 0x5C {
                index += 1
                guard let escape = current else { throw WorkerProtocolError.malformedEnvelope }
                if escape == 0x75 {
                    index += 1
                    guard index + 4 <= bytes.count,
                          bytes[index..<(index + 4)].allSatisfy(Self.isHexDigit) else {
                        throw WorkerProtocolError.malformedEnvelope
                    }
                    index += 4
                    continue
                }
                guard [0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74].contains(escape) else {
                    throw WorkerProtocolError.malformedEnvelope
                }
                index += 1
                continue
            }
            guard byte >= 0x20 else { throw WorkerProtocolError.malformedEnvelope }
            index += 1
        }
        throw WorkerProtocolError.malformedEnvelope
    }

    private mutating func parseNumber(preserveExact: Bool) throws {
        let start = index
        if current == 0x2D { index += 1 }
        guard let firstDigit = current else { throw WorkerProtocolError.malformedEnvelope }
        if firstDigit == 0x30 {
            index += 1
            if let next = current, Self.isDigit(next) { throw WorkerProtocolError.malformedEnvelope }
        } else if (0x31...0x39).contains(firstDigit) {
            index += 1
            while let byte = current, Self.isDigit(byte) { index += 1 }
        } else {
            throw WorkerProtocolError.malformedEnvelope
        }
        if current == 0x2E {
            index += 1
            guard let digit = current, Self.isDigit(digit) else { throw WorkerProtocolError.malformedEnvelope }
            while let byte = current, Self.isDigit(byte) { index += 1 }
        }
        if current == 0x65 || current == 0x45 {
            index += 1
            if current == 0x2B || current == 0x2D { index += 1 }
            guard let digit = current, Self.isDigit(digit) else { throw WorkerProtocolError.malformedEnvelope }
            while let byte = current, Self.isDigit(byte) { index += 1 }
        }
        let range = start..<index
        if preserveExact {
            try append([0x22])
            try append(prefix.utf8)
            try append(bytes[range])
            try append([0x22])
        } else {
            try append(bytes[range])
        }
    }

    private mutating func copyLiteral(_ literal: StaticString) throws {
        let literalBytes = Array(String(describing: literal).utf8)
        guard index + literalBytes.count <= bytes.count,
              Array(bytes[index..<(index + literalBytes.count)]) == literalBytes else {
            throw WorkerProtocolError.malformedEnvelope
        }
        index += literalBytes.count
        try append(literalBytes)
    }

    private mutating func copyWhitespace() throws {
        let start = index
        while let byte = current, [0x20, 0x09, 0x0A, 0x0D].contains(byte) { index += 1 }
        try append(bytes[start..<index])
    }

    private mutating func copyExpected(_ expected: UInt8) throws {
        guard current == expected else { throw WorkerProtocolError.malformedEnvelope }
        index += 1
        try append([expected])
    }

    private mutating func append<S: Sequence>(_ newBytes: S) throws where S.Element == UInt8 {
        let collected = Array(newBytes)
        guard output.count + collected.count <= maximumOutputBytes else {
            throw WorkerProtocolError.payloadTooLarge(output.count + collected.count)
        }
        output.append(contentsOf: collected)
    }

    private var current: UInt8? { index < bytes.count ? bytes[index] : nil }
    private static func isDigit(_ byte: UInt8) -> Bool { (0x30...0x39).contains(byte) }
    private static func isHexDigit(_ byte: UInt8) -> Bool {
        isDigit(byte) || (0x41...0x46).contains(byte) || (0x61...0x66).contains(byte)
    }
}

enum LengthPrefixedJSONCodec {
    static let maximumPayloadBytes = 1_048_576

    static func encode(_ envelope: WorkerEnvelope) throws -> Data {
        let payload = try encodePayload(envelope)
        guard !payload.isEmpty else { throw WorkerProtocolError.zeroLengthPayload }
        guard payload.count <= maximumPayloadBytes else { throw WorkerProtocolError.payloadTooLarge(payload.count) }

        var frame = Data()
        var length = UInt32(payload.count).bigEndian
        Swift.withUnsafeBytes(of: &length) { frame.append(contentsOf: $0) }
        frame.append(payload)
        return frame
    }

    private static func encodePayload(_ envelope: WorkerEnvelope) throws -> Data {
        while true {
            let context = ExactJSONNumberEncodingContext()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            encoder.userInfo[.exactJSONNumberEncoding] = context
            let encoded = try encoder.encode(envelope)
            guard var json = String(data: encoded, encoding: .utf8) else {
                throw WorkerProtocolError.malformedEnvelope
            }
            let occurrenceCount = json.components(separatedBy: context.prefix).count - 1
            guard occurrenceCount == context.replacements.count else { continue }
            for replacement in context.replacements {
                let placeholder = "\"\(replacement.placeholder)\""
                guard json.range(of: placeholder) != nil else { throw WorkerProtocolError.malformedEnvelope }
                json = json.replacingOccurrences(of: placeholder, with: replacement.token)
            }
            return Data(json.utf8)
        }
    }

    struct Decoder: Sendable {
        private let maxPayloadBytes: Int
        private var header: [UInt8] = []
        private var expectedPayloadBytes: Int?
        private var payload = Data()
        private var recentCommandIDs: [UUID] = []
        private var recentCommandIDSet: Set<UUID> = []

        var bufferedByteCount: Int { header.count + payload.count }

        init(maxPayloadBytes: Int = LengthPrefixedJSONCodec.maximumPayloadBytes) {
            self.maxPayloadBytes = maxPayloadBytes
        }

        mutating func append<S: DataProtocol>(_ bytes: S) throws -> [WorkerEnvelope] {
            var envelopes: [WorkerEnvelope] = []

            for byte in bytes {
                if expectedPayloadBytes == nil {
                    header.append(byte)
                    guard header.count == 4 else { continue }
                    let length = header.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
                    if length == 0 {
                        resetFrame()
                        throw WorkerProtocolError.zeroLengthPayload
                    }
                    if length > maxPayloadBytes {
                        resetFrame()
                        throw WorkerProtocolError.payloadTooLarge(Int(length))
                    }
                    expectedPayloadBytes = Int(length)
                    payload.reserveCapacity(Int(length))
                    continue
                }

                payload.append(byte)
                guard payload.count == expectedPayloadBytes else { continue }
                let envelope = try decodePayload()
                envelopes.append(envelope)
                resetFrame()
            }
            return envelopes
        }

        func finish() throws {
            guard header.isEmpty, expectedPayloadBytes == nil, payload.isEmpty else {
                throw WorkerProtocolError.truncatedFrame
            }
        }

        private mutating func decodePayload() throws -> WorkerEnvelope {
            let envelope: WorkerEnvelope
            do {
                let rewritten = try Self.rewriteExactNumbers(in: payload, maximumPayloadBytes: maxPayloadBytes)
                let decoder = JSONDecoder()
                decoder.userInfo[.exactJSONNumberDecodingPrefix] = rewritten.prefix
                envelope = try decoder.decode(WorkerEnvelope.self, from: rewritten.payload)
            }
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
            return envelope
        }

        private static func rewriteExactNumbers(
            in payload: Data,
            maximumPayloadBytes: Int
        ) throws -> (payload: Data, prefix: String) {
            while true {
                let prefix = "__CLOUDPOINT_JSON_NUMBER_\(UUID().uuidString)__:"
                var rewriter = ExactJSONNumberDecodingRewriter(
                    payload: payload,
                    prefix: prefix,
                    maximumPayloadBytes: maximumPayloadBytes
                )
                let rewritten = try rewriter.rewrite()
                if !rewriter.hasPlaceholderCollision { return (rewritten, prefix) }
            }
        }

        private mutating func resetFrame() {
            header.removeAll(keepingCapacity: true)
            expectedPayloadBytes = nil
            payload.removeAll(keepingCapacity: true)
        }
    }
}

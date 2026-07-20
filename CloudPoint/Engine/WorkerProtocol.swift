import CoreFoundation
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

enum WorkerFrameDecodeOutcome: Sendable, Equatable {
    case envelope(WorkerEnvelope)
    case failure(WorkerProtocolError, jsonPayload: Data?)
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
    case hello(clientVersion: String, supportedProtocolVersions: [UInt32])
    case configure(EngineConfiguration)
    case beginSession(resumeCheckpoint: ResumeCheckpoint?)
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
    case modelProgress(phase: WorkerModelProgressPhase, completed: UInt64, total: UInt64)
    case frameStarted(frameIndex: UInt32, windowIndex: UInt32)
    case frameCompleted(FrameArtifacts)
    case windowCompleted(WindowResult)
    case sessionCompleted(processedFrames: UInt64, windowCount: UInt32, durationSeconds: Double)
    case paused(queuedFrames: UInt64, processedFrames: UInt64)
    case cancelled(lastCompletedWindowIndex: UInt32?)
    case warning(WorkerErrorPayload)
    case heartbeat(
        busy: Bool,
        monotonicSeconds: Double,
        queuedFrames: UInt64,
        processedFrames: UInt64,
        currentWindow: UInt32?
    )
}

extension WorkerEvent {
    func engineEvent() throws -> EngineEvent? {
        switch self {
        case .ack: return nil
        case let .error(commandID, payload):
            if commandID != nil { return nil }
            throw ReconstructionEngineError.workerFailure(
                code: payload.code,
                message: payload.message,
                recoverable: payload.recoverable,
                details: payload.details
            )
        case let .ready(engine, model, revision, sha):
            return .ready(
                engineVersion: engine,
                modelIdentifier: model,
                modelRevision: revision,
                convertedWeightsSHA256: sha
            )
        case let .modelProgress(phase, completed, total):
            return .modelProgress(phase: phase, completed: completed, total: total)
        case let .frameStarted(frame, window):
            return .frameStarted(frameIndex: frame, windowIndex: window)
        case let .frameCompleted(artifacts): return .frameCompleted(artifacts)
        case let .windowCompleted(result): return .windowCompleted(result)
        case let .sessionCompleted(frames, windows, duration):
            return .sessionCompleted(processedFrames: frames, windowCount: windows, durationSeconds: duration)
        case let .paused(queued, processed):
            return .paused(queuedFrames: queued, processedFrames: processed)
        case let .cancelled(last): return .cancelled(lastCompletedWindowIndex: last)
        case let .warning(payload):
            return .warning(
                code: payload.code,
                message: payload.message,
                recoverable: payload.recoverable,
                details: payload.details
            )
        case let .heartbeat(busy, seconds, queued, processed, current):
            return .heartbeat(
                busy: busy,
                monotonicSeconds: seconds,
                queuedFrames: queued,
                processedFrames: processed,
                currentWindow: current
            )
        }
    }
}

struct WorkerCommandHeader: Sendable, Equatable {
    var id: UUID
    var projectID: UUID
    var protocolVersion: Int
    var type: String

    static func recover(fromJSONPayload payload: Data) -> WorkerCommandHeader? {
        guard let object = try? JSONSerialization.jsonObject(with: payload),
              let dictionary = object as? [String: Any],
              let idText = dictionary["id"] as? String,
              let projectIDText = dictionary["projectId"] as? String,
              let versionNumber = dictionary["protocolVersion"] as? NSNumber,
              CFGetTypeID(versionNumber) != CFBooleanGetTypeID(),
              !["f", "d"].contains(String(cString: versionNumber.objCType)),
              let protocolVersion = Int(exactly: versionNumber.int64Value),
              let type = dictionary["type"] as? String,
              !type.isEmpty,
              idText == idText.lowercased(),
              projectIDText == projectIDText.lowercased(),
              let id = UUID(uuidString: idText),
              let projectID = UUID(uuidString: projectIDText),
              id.uuidString.lowercased() == idText,
              projectID.uuidString.lowercased() == projectIDText else {
            return nil
        }
        return WorkerCommandHeader(
            id: id,
            projectID: projectID,
            protocolVersion: protocolVersion,
            type: type
        )
    }
}

enum WorkerProtocolFailureDisposition: Sendable, Equatable {
    case closeWithoutResponse
    case asynchronousProtocolFaultThenClose
    case commandErrorThenContinue(WorkerCommandHeader)
    case commandErrorThenClose(WorkerCommandHeader)

    static func classify(
        _ error: WorkerProtocolError,
        recoverableHeader: WorkerCommandHeader?
    ) -> WorkerProtocolFailureDisposition {
        switch error {
        case .zeroLengthPayload, .payloadTooLarge, .truncatedFrame:
            return .closeWithoutResponse
        case .unsupportedProtocolVersion:
            if let recoverableHeader { return .commandErrorThenClose(recoverableHeader) }
            return .asynchronousProtocolFaultThenClose
        case .unknownMessageType, .duplicateCommandID, .invalidPayload:
            if let recoverableHeader { return .commandErrorThenContinue(recoverableHeader) }
            return .asynchronousProtocolFaultThenClose
        case .malformedEnvelope:
            return .asynchronousProtocolFaultThenClose
        }
    }

    static func classify(
        _ error: WorkerProtocolError,
        JSONPayload payload: Data?
    ) -> WorkerProtocolFailureDisposition {
        classify(
            error,
            recoverableHeader: payload.flatMap(WorkerCommandHeader.recover(fromJSONPayload:))
        )
    }
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
    private struct HelloPayload: Codable { let clientVersion: String; let supportedProtocolVersions: [UInt32] }
    private struct ConfigurePayload: Codable {
        let scaleFrames: UInt32
        let windowSize: UInt32
        let windowOverlap: UInt32
        let keyframeInterval: UInt32
        let cameraRefinementIterations: UInt32
        let confidenceThreshold: Double
        let voxelSize: Double

        init(_ configuration: EngineConfiguration) {
            scaleFrames = configuration.scaleFrames
            windowSize = configuration.windowSize
            windowOverlap = configuration.windowOverlap
            keyframeInterval = configuration.keyframeInterval
            cameraRefinementIterations = configuration.cameraRefinementIterations
            confidenceThreshold = configuration.confidenceThreshold
            voxelSize = configuration.voxelSize
        }

        var configuration: EngineConfiguration {
            EngineConfiguration(
                scaleFrames: scaleFrames,
                windowSize: windowSize,
                windowOverlap: windowOverlap,
                keyframeInterval: keyframeInterval,
                cameraRefinementIterations: cameraRefinementIterations,
                confidenceThreshold: confidenceThreshold,
                voxelSize: voxelSize
            )
        }
    }
    private struct ResumeCheckpointPayload: Codable {
        let lastCommittedFrameIndex: UInt32
        let replayFromFrameIndex: UInt32
        let nextWindowIndex: UInt32

        init(_ checkpoint: ResumeCheckpoint) {
            lastCommittedFrameIndex = checkpoint.lastCommittedFrameIndex
            replayFromFrameIndex = checkpoint.replayFromFrameIndex
            nextWindowIndex = checkpoint.nextWindowIndex
        }

        var checkpoint: ResumeCheckpoint {
            ResumeCheckpoint(
                lastCommittedFrameIndex: lastCommittedFrameIndex,
                replayFromFrameIndex: replayFromFrameIndex,
                nextWindowIndex: nextWindowIndex
            )
        }
    }
    private struct BeginSessionPayload: Codable {
        let resumeCheckpoint: ResumeCheckpointPayload?
        private enum CodingKeys: String, CodingKey { case resumeCheckpoint }
        init(resumeCheckpoint: ResumeCheckpoint?) {
            self.resumeCheckpoint = resumeCheckpoint.map(ResumeCheckpointPayload.init)
        }
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            resumeCheckpoint = try container.decode(ResumeCheckpointPayload?.self, forKey: .resumeCheckpoint)
            if resumeCheckpoint != nil {
                let nested = try container.superDecoder(forKey: .resumeCheckpoint)
                try WorkerEnvelope.requireExactKeys(
                    ["lastCommittedFrameIndex", "replayFromFrameIndex", "nextWindowIndex"],
                    in: nested,
                    context: "resumeCheckpoint"
                )
            }
        }
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(resumeCheckpoint, forKey: .resumeCheckpoint)
        }
    }
    private struct EnqueueFramePayload: Codable { let frameIndex: UInt32; let sourceTimestamp: Double; let relativePath: String }
    private struct AckPayload: Codable { let command: String }
    private struct ReadyPayload: Codable { let engineVersion: String; let modelIdentifier: String; let modelRevision: String; let convertedWeightsSHA256: String }
    private struct ModelProgressPayload: Codable { let phase: WorkerModelProgressPhase; let completed: UInt64; let total: UInt64 }
    private struct FrameStartedPayload: Codable { let frameIndex: UInt32; let windowIndex: UInt32 }
    private struct FrameCompletedPayload: Codable {
        let frameIndex: UInt32
        let windowIndex: UInt32
        let depthPath: String
        let confidencePath: String
        let geometryPath: String
        let durationSeconds: Double

        init(_ artifacts: FrameArtifacts) {
            frameIndex = artifacts.frameIndex
            windowIndex = artifacts.windowIndex
            depthPath = artifacts.depthRelativePath
            confidencePath = artifacts.confidenceRelativePath
            geometryPath = artifacts.geometryRelativePath
            durationSeconds = artifacts.durationSeconds
        }

        var artifacts: FrameArtifacts {
            FrameArtifacts(
                frameIndex: frameIndex,
                windowIndex: windowIndex,
                depthRelativePath: depthPath,
                confidenceRelativePath: confidencePath,
                geometryRelativePath: geometryPath,
                durationSeconds: durationSeconds
            )
        }
    }
    private struct WindowCompletedPayload: Codable {
        let windowIndex: UInt32
        let inferenceFrameStart: UInt32
        let frameStart: UInt32
        let frameEnd: UInt32
        let pointChunkPath: String
        let alignmentTransform: [Double]
        let lastProcessedFrameIndex: UInt32
        let inlierCount: UInt64
        let durationSeconds: Double

        init(_ result: WindowResult) {
            windowIndex = result.windowIndex
            inferenceFrameStart = result.inferenceFrameStart
            frameStart = result.frameStart
            frameEnd = result.frameEnd
            pointChunkPath = result.pointChunkRelativePath
            alignmentTransform = result.alignmentRowMajor
            lastProcessedFrameIndex = result.lastProcessedFrameIndex
            inlierCount = result.inlierCount
            durationSeconds = result.durationSeconds
        }

        var result: WindowResult {
            WindowResult(
                windowIndex: windowIndex,
                inferenceFrameStart: inferenceFrameStart,
                frameStart: frameStart,
                frameEnd: frameEnd,
                pointChunkRelativePath: pointChunkPath,
                alignmentRowMajor: alignmentTransform,
                lastProcessedFrameIndex: lastProcessedFrameIndex,
                inlierCount: inlierCount,
                durationSeconds: durationSeconds
            )
        }
    }
    private struct SessionCompletedPayload: Codable { let processedFrames: UInt64; let windowCount: UInt32; let durationSeconds: Double }
    private struct PausedPayload: Codable { let queuedFrames: UInt64; let processedFrames: UInt64 }
    private struct CancelledPayload: Codable {
        let lastCompletedWindowIndex: UInt32?
        private enum CodingKeys: String, CodingKey { case lastCompletedWindowIndex }
        init(lastCompletedWindowIndex: UInt32?) { self.lastCompletedWindowIndex = lastCompletedWindowIndex }
        init(from decoder: Decoder) throws {
            lastCompletedWindowIndex = try decoder.container(keyedBy: CodingKeys.self).decode(UInt32?.self, forKey: .lastCompletedWindowIndex)
        }
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(lastCompletedWindowIndex, forKey: .lastCompletedWindowIndex)
        }
    }
    private struct HeartbeatPayload: Codable {
        let busy: Bool
        let monotonicSeconds: Double
        let queuedFrames: UInt64
        let processedFrames: UInt64
        let currentWindow: UInt32?
        private enum CodingKeys: String, CodingKey { case busy, monotonicSeconds, queuedFrames, processedFrames, currentWindow }
        init(busy: Bool, monotonicSeconds: Double, queuedFrames: UInt64, processedFrames: UInt64, currentWindow: UInt32?) {
            self.busy = busy; self.monotonicSeconds = monotonicSeconds; self.queuedFrames = queuedFrames
            self.processedFrames = processedFrames; self.currentWindow = currentWindow
        }
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            busy = try container.decode(Bool.self, forKey: .busy)
            monotonicSeconds = try container.decode(Double.self, forKey: .monotonicSeconds)
            queuedFrames = try container.decode(UInt64.self, forKey: .queuedFrames)
            processedFrames = try container.decode(UInt64.self, forKey: .processedFrames)
            currentWindow = try container.decode(UInt32?.self, forKey: .currentWindow)
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
        let idText = try container.decode(String.self, forKey: .id)
        let projectIDText = try container.decode(String.self, forKey: .projectId)
        guard idText == idText.lowercased(),
              projectIDText == projectIDText.lowercased(),
              let decodedID = UUID(uuidString: idText),
              let decodedProjectID = UUID(uuidString: projectIDText),
              decodedID.uuidString.lowercased() == idText,
              decodedProjectID.uuidString.lowercased() == projectIDText else {
            throw WorkerProtocolError.malformedEnvelope
        }
        id = decodedID
        projectId = decodedProjectID
        let type = try container.decode(String.self, forKey: .type)
        let topLevelKeys = ["protocolVersion", "id", "projectId", "type", "payload"] + (["ack", "error"].contains(type) ? ["commandId"] : [])
        try Self.requireExactKeys(topLevelKeys, in: decoder, context: "envelope")

        switch type {
        case "hello":
            let payload = try Self.decodePayload(HelloPayload.self, keys: ["clientVersion", "supportedProtocolVersions"], from: container)
            message = .command(.hello(clientVersion: payload.clientVersion, supportedProtocolVersions: payload.supportedProtocolVersions))
        case "configure":
            let p = try Self.decodePayload(ConfigurePayload.self, keys: ["scaleFrames", "windowSize", "windowOverlap", "keyframeInterval", "cameraRefinementIterations", "confidenceThreshold", "voxelSize"], from: container)
            message = .command(.configure(p.configuration))
        case "beginSession":
            let p = try Self.decodePayload(BeginSessionPayload.self, keys: ["resumeCheckpoint"], from: container)
            message = .command(.beginSession(resumeCheckpoint: p.resumeCheckpoint?.checkpoint))
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
            let commandIDText = try container.decode(String.self, forKey: .commandId)
            guard commandIDText == commandIDText.lowercased(),
                  let commandID = UUID(uuidString: commandIDText),
                  commandID.uuidString.lowercased() == commandIDText else {
                throw WorkerProtocolError.malformedEnvelope
            }
            message = .event(.ack(commandId: commandID, command: p.command))
        case "error":
            let p = try Self.decodePayload(WorkerErrorPayload.self, keys: ["code", "message", "recoverable", "details"], from: container)
            let optionalCommandIDText = try container.decode(String?.self, forKey: .commandId)
            let commandID = try optionalCommandIDText.map {
                guard $0 == $0.lowercased(),
                      let id = UUID(uuidString: $0),
                      id.uuidString.lowercased() == $0 else {
                    throw WorkerProtocolError.malformedEnvelope
                }
                return id
            }
            message = .event(.error(commandId: commandID, p))
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
            let p = try Self.decodePayload(FrameCompletedPayload.self, keys: ["frameIndex", "windowIndex", "depthPath", "confidencePath", "geometryPath", "durationSeconds"], from: container)
            message = .event(.frameCompleted(p.artifacts))
        case "windowCompleted":
            let p = try Self.decodePayload(WindowCompletedPayload.self, keys: ["windowIndex", "inferenceFrameStart", "frameStart", "frameEnd", "pointChunkPath", "alignmentTransform", "lastProcessedFrameIndex", "inlierCount", "durationSeconds"], from: container)
            message = .event(.windowCompleted(p.result))
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
        try container.encode(id.uuidString.lowercased(), forKey: .id)
        try container.encode(projectId.uuidString.lowercased(), forKey: .projectId)

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
        case let .configure(configuration):
            try c.encode("configure", forKey: .type); try c.encode(ConfigurePayload(configuration), forKey: .payload)
        case let .beginSession(checkpoint):
            try c.encode("beginSession", forKey: .type); try c.encode(BeginSessionPayload(resumeCheckpoint: checkpoint), forKey: .payload)
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
            try c.encode("ack", forKey: .type); try c.encode(commandId.uuidString.lowercased(), forKey: .commandId); try c.encode(AckPayload(command: command), forKey: .payload)
        case let .error(commandId, payload):
            try c.encode("error", forKey: .type); try c.encode(commandId?.uuidString.lowercased(), forKey: .commandId); try c.encode(payload, forKey: .payload)
        case let .ready(engine, model, revision, sha):
            try c.encode("ready", forKey: .type); try c.encode(ReadyPayload(engineVersion: engine, modelIdentifier: model, modelRevision: revision, convertedWeightsSHA256: sha), forKey: .payload)
        case let .modelProgress(phase, completed, total):
            try c.encode("modelProgress", forKey: .type); try c.encode(ModelProgressPayload(phase: phase, completed: completed, total: total), forKey: .payload)
        case let .frameStarted(frame, window):
            try c.encode("frameStarted", forKey: .type); try c.encode(FrameStartedPayload(frameIndex: frame, windowIndex: window), forKey: .payload)
        case let .frameCompleted(artifacts):
            try c.encode("frameCompleted", forKey: .type); try c.encode(FrameCompletedPayload(artifacts), forKey: .payload)
        case let .windowCompleted(result):
            try c.encode("windowCompleted", forKey: .type); try c.encode(WindowCompletedPayload(result), forKey: .payload)
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
            guard !clientVersion.isEmpty,
                  versions.contains(UInt32(currentProtocolVersion)) else {
                throw WorkerProtocolError.invalidPayload("hello")
            }
        case let .configure(configuration):
            do { try configuration.validate() }
            catch { throw WorkerProtocolError.invalidPayload("configure") }
        case let .beginSession(checkpoint):
            guard checkpoint == nil || checkpoint!.replayFromFrameIndex <= checkpoint!.lastCommittedFrameIndex else {
                throw WorkerProtocolError.invalidPayload("beginSession")
            }
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
            guard completed <= total else { throw WorkerProtocolError.invalidPayload("modelProgress") }
        case .frameStarted: break
        case let .frameCompleted(artifacts):
            guard artifacts.depthRelativePath == WorkerArtifactPath.depth(frameIndex: artifacts.frameIndex),
                  artifacts.confidenceRelativePath == WorkerArtifactPath.confidence(frameIndex: artifacts.frameIndex),
                  artifacts.geometryRelativePath == WorkerArtifactPath.geometry(frameIndex: artifacts.frameIndex),
                  validDuration(artifacts.durationSeconds) else {
                throw WorkerProtocolError.invalidPayload("frameCompleted")
            }
        case let .windowCompleted(result):
            guard result.inferenceFrameStart <= result.frameStart,
                  result.frameStart <= result.frameEnd,
                  result.frameEnd <= result.lastProcessedFrameIndex,
                  result.pointChunkRelativePath == WorkerArtifactPath.points(windowIndex: result.windowIndex),
                  result.alignmentRowMajor.count == 16,
                  result.alignmentRowMajor.allSatisfy(\.isFinite),
                  validDuration(result.durationSeconds) else {
                throw WorkerProtocolError.invalidPayload("windowCompleted")
            }
        case let .sessionCompleted(_, _, duration):
            guard validDuration(duration) else { throw WorkerProtocolError.invalidPayload("sessionCompleted") }
        case let .paused(queued, processed):
            guard processed <= queued else { throw WorkerProtocolError.invalidPayload("paused") }
        case .cancelled: break
        case let .heartbeat(_, seconds, queued, processed, _):
            guard validDuration(seconds), processed <= queued else {
                throw WorkerProtocolError.invalidPayload("heartbeat")
            }
        }
    }

    private static func validate(_ payload: WorkerErrorPayload) throws {
        guard !payload.code.isEmpty, !payload.message.isEmpty else { throw WorkerProtocolError.invalidPayload("error") }
    }

    private static func validDuration(_ value: Double) -> Bool { value.isFinite && value >= 0 }

    private static func isSafeRelativePath(_ path: String) -> Bool { ProjectRelativePath.isSafe(path) }
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
            json = canonicalizeTypedNumbers(in: json)
            let occurrenceCount = json.components(separatedBy: context.prefix).count - 1
            guard occurrenceCount == context.replacements.count else { continue }
            guard let substituted = substituteExactJSONNumbers(in: json, context: context) else { continue }
            return substituted
        }
    }

    private static func substituteExactJSONNumbers(
        in json: String,
        context: ExactJSONNumberEncodingContext
    ) -> Data? {
        let source = Array(json.utf8)
        let marker = [UInt8(0x22)] + Array(context.prefix.utf8)
        var output: [UInt8] = []
        output.reserveCapacity(source.count)
        var replaced = Set<Int>()
        var index = 0

        while index < source.count {
            guard index + marker.count <= source.count,
                  source[index..<(index + marker.count)].elementsEqual(marker) else {
                output.append(source[index])
                index += 1
                continue
            }

            var cursor = index + marker.count
            var replacementIndex = 0
            var sawDigit = false
            while cursor < source.count, (0x30...0x39).contains(source[cursor]) {
                sawDigit = true
                let (scaled, multiplyOverflow) = replacementIndex.multipliedReportingOverflow(by: 10)
                let (advanced, addOverflow) = scaled.addingReportingOverflow(Int(source[cursor] - 0x30))
                guard !multiplyOverflow, !addOverflow else { return nil }
                replacementIndex = advanced
                cursor += 1
            }
            guard sawDigit,
                  cursor < source.count,
                  source[cursor] == 0x22,
                  context.replacements.indices.contains(replacementIndex),
                  replaced.insert(replacementIndex).inserted else {
                return nil
            }
            output.append(contentsOf: context.replacements[replacementIndex].token.utf8)
            index = cursor + 1
        }

        guard replaced.count == context.replacements.count else { return nil }
        return Data(output)
    }

    private static func canonicalizeTypedNumbers(in json: String) -> String {
        let bytes = Array(json.utf8)
        var output: [UInt8] = []
        output.reserveCapacity(bytes.count)
        var index = 0
        var inString = false
        var escaped = false

        while index < bytes.count {
            let byte = bytes[index]
            if inString {
                output.append(byte)
                if escaped {
                    escaped = false
                } else if byte == 0x5C {
                    escaped = true
                } else if byte == 0x22 {
                    inString = false
                }
                index += 1
                continue
            }
            if byte == 0x22 {
                inString = true
                output.append(byte)
                index += 1
                continue
            }
            if byte == 0x2D || (0x30...0x39).contains(byte) {
                let start = index
                index += 1
                while index < bytes.count,
                      (bytes[index] == 0x2B || bytes[index] == 0x2D || bytes[index] == 0x2E ||
                       bytes[index] == 0x45 || bytes[index] == 0x65 || (0x30...0x39).contains(bytes[index])) {
                    index += 1
                }
                let token = String(decoding: bytes[start..<index], as: UTF8.self)
                output.append(contentsOf: canonicalNumberToken(token).utf8)
                continue
            }
            output.append(byte)
            index += 1
        }
        return String(decoding: output, as: UTF8.self)
    }

    private static func canonicalNumberToken(_ token: String) -> String {
        let exponentSplit = token.firstIndex { $0 == "e" || $0 == "E" }
        var mantissa = exponentSplit.map { String(token[..<$0]) } ?? token
        let exponent = exponentSplit.map { String(token[token.index(after: $0)...]) }

        let mantissaDigits = mantissa.filter(\.isNumber)
        if !mantissaDigits.isEmpty, mantissaDigits.allSatisfy({ $0 == "0" }) {
            return "0"
        }
        if let decimal = mantissa.firstIndex(of: ".") {
            while mantissa.last == "0" { mantissa.removeLast() }
            if mantissa.lastIndex(of: ".") == decimal, mantissa.last == "." { mantissa.removeLast() }
        }
        guard var exponent else { return mantissa }
        var sign = ""
        if exponent.first == "+" { exponent.removeFirst() }
        else if exponent.first == "-" {
            sign = "-"
            exponent.removeFirst()
        }
        while exponent.count > 1, exponent.first == "0" { exponent.removeFirst() }
        if exponent.allSatisfy({ $0 == "0" }) { return mantissa }
        return "\(mantissa)e\(sign)\(exponent)"
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
            for outcome in appendOutcomes(bytes, continuingAfterRecoverableFailures: false) {
                switch outcome {
                case let .envelope(envelope): envelopes.append(envelope)
                case let .failure(error, _): throw error
                }
            }
            return envelopes
        }

        mutating func appendOutcomes<S: DataProtocol>(
            _ bytes: S,
            continuingAfterRecoverableFailures: Bool = true
        ) -> [WorkerFrameDecodeOutcome] {
            var outcomes: [WorkerFrameDecodeOutcome] = []
            for byte in bytes {
                if expectedPayloadBytes == nil {
                    header.append(byte)
                    guard header.count == 4 else { continue }
                    let length = header.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
                    if length == 0 {
                        resetFrame()
                        outcomes.append(.failure(.zeroLengthPayload, jsonPayload: nil))
                        break
                    }
                    if length > maxPayloadBytes {
                        resetFrame()
                        outcomes.append(.failure(.payloadTooLarge(Int(length)), jsonPayload: nil))
                        break
                    }
                    expectedPayloadBytes = Int(length)
                    payload.reserveCapacity(Int(length))
                    continue
                }

                payload.append(byte)
                guard payload.count == expectedPayloadBytes else { continue }
                let completedPayload = payload
                resetFrame()
                do {
                    outcomes.append(.envelope(try decodePayload(completedPayload)))
                } catch let error as WorkerProtocolError {
                    outcomes.append(.failure(error, jsonPayload: completedPayload))
                    switch WorkerProtocolFailureDisposition.classify(error, JSONPayload: completedPayload) {
                    case .commandErrorThenContinue where continuingAfterRecoverableFailures:
                        continue
                    case .closeWithoutResponse, .asynchronousProtocolFaultThenClose,
                         .commandErrorThenContinue, .commandErrorThenClose:
                        return outcomes
                    }
                } catch {
                    outcomes.append(.failure(.malformedEnvelope, jsonPayload: completedPayload))
                    return outcomes
                }
            }
            return outcomes
        }

        func finish() throws {
            guard header.isEmpty, expectedPayloadBytes == nil, payload.isEmpty else {
                throw WorkerProtocolError.truncatedFrame
            }
        }

        private mutating func decodePayload(_ payload: Data) throws -> WorkerEnvelope {
            let envelope: WorkerEnvelope
            do {
                let rewritten = try Self.rewriteExactNumbers(in: payload, maximumPayloadBytes: maxPayloadBytes)
                let decoder = JSONDecoder()
                decoder.userInfo[.exactJSONNumberDecodingPrefix] = rewritten.prefix
                envelope = try decoder.decode(WorkerEnvelope.self, from: rewritten.payload)
            }
            catch let error as WorkerProtocolError { throw error }
            catch {
                if let header = WorkerCommandHeader.recover(fromJSONPayload: payload) {
                    throw WorkerProtocolError.invalidPayload(header.type)
                }
                throw WorkerProtocolError.malformedEnvelope
            }

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
                let random = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
                let prefix = "~cpn\(random.prefix(8)):"
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

import Foundation

enum SharpProgressStage: String, Codable, Sendable, Equatable {
    case loading
    case inference
    case validating
    case committing
}

struct SharpWorkerCompletion: Codable, Sendable, Equatable {
    let sourceFrameIndex: UInt32
    let plyRelativePath: String
    let provenanceRelativePath: String
    let gaussianCount: UInt64
    let durationSeconds: Double
    let device: String
    let usedCPUFallback: Bool
}

enum SharpWorkerEvent: Sendable, Equatable {
    case progress(stage: SharpProgressStage, fraction: Double)
    case heartbeat(monotonicSeconds: Double)
    case warning(code: String, message: String, recoverable: Bool)
    case completed(SharpWorkerCompletion)
    case failed(code: String, message: String, recoverable: Bool)
    case cancelled
}

enum SharpWorkerProtocolError: Error, Sendable, Equatable {
    case oversizedLine
    case malformedJSON
    case unsupportedVersion(Int)
    case unknownType(String)
    case unexpectedFields
    case invalidPayload
}

enum SharpWorkerLineCodec {
    static let protocolVersion = 1
    static let maximumLineBytes = 1_048_576

    private struct Header: Decodable {
        let protocolVersion: Int
        let type: String
    }

    private struct Progress: Decodable {
        let protocolVersion: Int
        let type: String
        let stage: SharpProgressStage
        let fraction: Double
    }

    private struct Heartbeat: Decodable {
        let protocolVersion: Int
        let type: String
        let monotonicSeconds: Double
    }

    private struct Diagnostic: Decodable {
        let protocolVersion: Int
        let type: String
        let code: String
        let message: String
        let recoverable: Bool
    }

    private struct Completed: Decodable {
        let protocolVersion: Int
        let type: String
        let sourceFrameIndex: UInt32
        let plyRelativePath: String
        let provenanceRelativePath: String
        let gaussianCount: UInt64
        let durationSeconds: Double
        let device: String
        let usedCPUFallback: Bool

        var value: SharpWorkerCompletion {
            SharpWorkerCompletion(
                sourceFrameIndex: sourceFrameIndex,
                plyRelativePath: plyRelativePath,
                provenanceRelativePath: provenanceRelativePath,
                gaussianCount: gaussianCount,
                durationSeconds: durationSeconds,
                device: device,
                usedCPUFallback: usedCPUFallback
            )
        }
    }

    static func decode(_ data: Data) throws -> SharpWorkerEvent {
        guard data.count <= maximumLineBytes else {
            throw SharpWorkerProtocolError.oversizedLine
        }
        guard !data.isEmpty,
              !data.contains(0x0A),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any],
              JSONSerialization.isValidJSONObject(dictionary) else {
            throw SharpWorkerProtocolError.malformedJSON
        }
        let decoder = JSONDecoder()
        guard let header = try? decoder.decode(Header.self, from: data) else {
            throw SharpWorkerProtocolError.malformedJSON
        }
        guard header.protocolVersion == protocolVersion else {
            throw SharpWorkerProtocolError.unsupportedVersion(header.protocolVersion)
        }

        func require(_ keys: Set<String>) throws {
            guard Set(dictionary.keys) == keys else {
                throw SharpWorkerProtocolError.unexpectedFields
            }
            let text = String(decoding: data, as: UTF8.self)
            guard keys.allSatisfy({ key in
                text.components(separatedBy: "\"\(key)\"").count - 1 == 1
            }) else {
                throw SharpWorkerProtocolError.unexpectedFields
            }
        }

        switch header.type {
        case "progress":
            try require(["protocolVersion", "type", "stage", "fraction"])
            guard let payload = try? decoder.decode(Progress.self, from: data),
                  payload.protocolVersion == protocolVersion,
                  payload.type == header.type,
                  payload.fraction.isFinite,
                  (0...1).contains(payload.fraction) else {
                throw SharpWorkerProtocolError.invalidPayload
            }
            return .progress(stage: payload.stage, fraction: payload.fraction)

        case "heartbeat":
            try require(["protocolVersion", "type", "monotonicSeconds"])
            guard let payload = try? decoder.decode(Heartbeat.self, from: data),
                  payload.protocolVersion == protocolVersion,
                  payload.type == header.type,
                  payload.monotonicSeconds.isFinite,
                  payload.monotonicSeconds >= 0 else {
                throw SharpWorkerProtocolError.invalidPayload
            }
            return .heartbeat(monotonicSeconds: payload.monotonicSeconds)

        case "warning", "failed":
            try require(["protocolVersion", "type", "code", "message", "recoverable"])
            guard let payload = try? decoder.decode(Diagnostic.self, from: data),
                  payload.protocolVersion == protocolVersion,
                  payload.type == header.type,
                  !payload.code.isEmpty,
                  !payload.message.isEmpty else {
                throw SharpWorkerProtocolError.invalidPayload
            }
            if header.type == "warning" {
                return .warning(
                    code: payload.code,
                    message: payload.message,
                    recoverable: payload.recoverable
                )
            }
            return .failed(
                code: payload.code,
                message: payload.message,
                recoverable: payload.recoverable
            )

        case "completed":
            try require([
                "protocolVersion", "type", "sourceFrameIndex", "plyRelativePath",
                "provenanceRelativePath", "gaussianCount", "durationSeconds",
                "device", "usedCPUFallback",
            ])
            guard let payload = try? decoder.decode(Completed.self, from: data),
                  payload.protocolVersion == protocolVersion,
                  payload.type == header.type,
                  payload.gaussianCount > 0,
                  payload.durationSeconds.isFinite,
                  payload.durationSeconds >= 0,
                  ["mps", "cpu"].contains(payload.device),
                  isPLYPath(payload.plyRelativePath, frameIndex: payload.sourceFrameIndex),
                  isProvenancePath(
                    payload.provenanceRelativePath,
                    frameIndex: payload.sourceFrameIndex
                  ) else {
                throw SharpWorkerProtocolError.invalidPayload
            }
            return .completed(payload.value)

        case "cancelled":
            try require(["protocolVersion", "type"])
            return .cancelled

        default:
            throw SharpWorkerProtocolError.unknownType(header.type)
        }
    }

    private static func isPLYPath(_ path: String, frameIndex: UInt32) -> Bool {
        path == String(format: "Outputs/Gaussians/%08u.ply", frameIndex)
    }

    private static func isProvenancePath(_ path: String, frameIndex: UInt32) -> Bool {
        path == String(format: "Outputs/Gaussians/%08u.json", frameIndex)
    }
}

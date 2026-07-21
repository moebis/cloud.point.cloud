import Foundation

struct ReconstructionModeID: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let lingbotPointCloud = ReconstructionModeID(
        rawValue: "cloudpoint.lingbot.point-cloud.v1"
    )
    static let sharpGaussian = ReconstructionModeID(
        rawValue: "cloudpoint.apple.sharp.gaussian.v1"
    )
}

struct SharpReconstructionConfiguration: Codable, Sendable, Equatable {
    var inputFrameIndex: UInt32?
    var preferMPS: Bool

    init(inputFrameIndex: UInt32? = nil, preferMPS: Bool = true) {
        self.inputFrameIndex = inputFrameIndex
        self.preferMPS = preferMPS
    }
}

/// JSON value used only to preserve configuration owned by a newer CloudPoint
/// version. Unknown modes are read-only, but their data still round-trips if a
/// caller explicitly serializes the manifest.
enum ManifestValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Decimal)
    case string(String)
    case array([ManifestValue])
    case object([String: ManifestValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Decimal.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([ManifestValue].self) { self = .array(value) }
        else if let value = try? container.decode([String: ManifestValue].self) { self = .object(value) }
        else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported manifest JSON value"
            )
        }
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

enum ReconstructionPlan: Codable, Sendable, Equatable {
    case lingbot(EngineConfiguration)
    case sharp(SharpReconstructionConfiguration)
    case unavailable(modeID: ReconstructionModeID, configuration: ManifestValue)

    private enum CodingKeys: String, CodingKey {
        case modeID
        case configuration
    }

    private enum ConfigurationKeys: String, CodingKey {
        case type
        case settings
    }

    private enum ConfigurationType: String, Codable {
        case lingbotPointCloud
        case sharpGaussian
    }

    var modeID: ReconstructionModeID {
        switch self {
        case .lingbot: .lingbotPointCloud
        case .sharp: .sharpGaussian
        case let .unavailable(modeID, _): modeID
        }
    }

    var isRunnable: Bool {
        if case .unavailable = self { return false }
        return true
    }

    var lingbotConfiguration: EngineConfiguration? {
        guard case let .lingbot(configuration) = self else { return nil }
        return configuration
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let modeID = try container.decode(ReconstructionModeID.self, forKey: .modeID)
        switch modeID {
        case .lingbotPointCloud:
            let configuration = try container.nestedContainer(
                keyedBy: ConfigurationKeys.self,
                forKey: .configuration
            )
            guard try configuration.decode(ConfigurationType.self, forKey: .type)
                    == .lingbotPointCloud else {
                throw DecodingError.dataCorruptedError(
                    forKey: .configuration,
                    in: container,
                    debugDescription: "LingBot mode requires a LingBot configuration"
                )
            }
            self = .lingbot(
                try configuration.decode(EngineConfiguration.self, forKey: .settings)
            )
        case .sharpGaussian:
            let configuration = try container.nestedContainer(
                keyedBy: ConfigurationKeys.self,
                forKey: .configuration
            )
            guard try configuration.decode(ConfigurationType.self, forKey: .type)
                    == .sharpGaussian else {
                throw DecodingError.dataCorruptedError(
                    forKey: .configuration,
                    in: container,
                    debugDescription: "SHARP mode requires a SHARP configuration"
                )
            }
            self = .sharp(
                try configuration.decode(SharpReconstructionConfiguration.self, forKey: .settings)
            )
        default:
            self = .unavailable(
                modeID: modeID,
                configuration: try container.decode(ManifestValue.self, forKey: .configuration)
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modeID, forKey: .modeID)
        switch self {
        case let .lingbot(settings):
            var configuration = container.nestedContainer(
                keyedBy: ConfigurationKeys.self,
                forKey: .configuration
            )
            try configuration.encode(ConfigurationType.lingbotPointCloud, forKey: .type)
            try configuration.encode(settings, forKey: .settings)
        case let .sharp(settings):
            var configuration = container.nestedContainer(
                keyedBy: ConfigurationKeys.self,
                forKey: .configuration
            )
            try configuration.encode(ConfigurationType.sharpGaussian, forKey: .type)
            try configuration.encode(settings, forKey: .settings)
        case let .unavailable(_, configuration):
            try container.encode(configuration, forKey: .configuration)
        }
    }
}

struct GaussianSceneOutput: Codable, Sendable, Equatable {
    var sourceFrameIndex: UInt32
    var plyRelativePath: String
    var provenanceRelativePath: String
    var gaussianCount: UInt64
    var modelIdentifier: String
    var modelRevision: String
    var checkpointSHA256: String
    var device: String
    var usedCPUFallback: Bool
    var durationSeconds: Double
}

enum ReconstructionOutputState: Codable, Sendable, Equatable {
    case pointCloud
    case gaussian(GaussianSceneOutput?)
    case unavailable(type: String, payload: ManifestValue)

    private enum CodingKeys: String, CodingKey {
        case type
        case result
    }

    private enum OutputType: String, Codable {
        case pointCloud
        case gaussian
    }

    init(from decoder: Decoder) throws {
        let raw = try ManifestValue(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch OutputType(rawValue: type) {
        case .pointCloud:
            self = .pointCloud
        case .gaussian:
            self = .gaussian(
                try container.decodeIfPresent(GaussianSceneOutput.self, forKey: .result)
            )
        case nil:
            self = .unavailable(type: type, payload: raw)
        }
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .pointCloud:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(OutputType.pointCloud.rawValue, forKey: .type)
        case let .gaussian(result):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(OutputType.gaussian.rawValue, forKey: .type)
            try container.encodeIfPresent(result, forKey: .result)
        case let .unavailable(_, payload):
            try payload.encode(to: encoder)
        }
    }
}

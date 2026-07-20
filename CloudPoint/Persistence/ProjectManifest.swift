import Darwin
import Foundation

enum ProjectManifestError: Error, Equatable, Sendable {
    case unsupportedFormatVersion(Int)
    case invalidConfiguration
    case invalidFrame
    case invalidFrameOrder
    case invalidWindow
    case invalidWindowOrder
    case invalidArtifact
    case invalidArtifactOrder
    case invalidSessionState
    case checkpointWindowIndexOverflow
}

struct ProjectManifest: Codable, Sendable, Equatable {
    static let currentFormatVersion = 2

    var formatVersion: Int
    var projectID: UUID
    var createdAt: Date
    var updatedAt: Date
    var engineConfiguration: EngineConfiguration
    var frames: [PersistedFrame]
    var completedWindows: [CompletedWindow]
    var sessionState: SessionState

    init(
        formatVersion: Int = ProjectManifest.currentFormatVersion,
        projectID: UUID = UUID(),
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        engineConfiguration: EngineConfiguration = EngineConfiguration(),
        frames: [PersistedFrame] = [],
        completedWindows: [CompletedWindow] = [],
        sessionState: SessionState = .empty
    ) {
        self.formatVersion = formatVersion
        self.projectID = projectID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.engineConfiguration = engineConfiguration
        self.frames = frames
        self.completedWindows = completedWindows
        self.sessionState = sessionState
    }

    static func load(from packageURL: URL) throws -> ProjectManifest {
        let data = try Data(contentsOf: manifestURL(in: packageURL))
        return try decode(data)
    }

    func writeAtomically(to packageURL: URL, fileManager: FileManager = .default) throws {
        _ = try Self.validate(self)
        let manifestURL = Self.manifestURL(in: packageURL)
        let partialURL = packageURL.appending(
            path: ".Manifest.json.\(UUID().uuidString.lowercased()).partial"
        )
        let data = try Self.encode(self)
        defer { try? fileManager.removeItem(at: partialURL) }

        try data.write(to: partialURL, options: .withoutOverwriting)
        let partialHandle = try FileHandle(forWritingTo: partialURL)
        do {
            try partialHandle.synchronize()
            try partialHandle.close()
        } catch {
            try? partialHandle.close()
            throw error
        }

        let renameResult = partialURL.path.withCString { sourcePath in
            manifestURL.path.withCString { destinationPath in
                Darwin.rename(sourcePath, destinationPath)
            }
        }
        guard renameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    static func encode(_ manifest: ProjectManifest) throws -> Data {
        _ = try validate(manifest)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(try rfc3339String(for: date, codingPath: encoder.codingPath))
        }
        return try encoder.encode(manifest)
    }

    static func decode(_ data: Data) throws -> ProjectManifest {
        struct VersionProbe: Decodable { let formatVersion: Int }
        let probe = try JSONDecoder().decode(VersionProbe.self, from: data)
        guard probe.formatVersion == currentFormatVersion else {
            throw ProjectManifestError.unsupportedFormatVersion(probe.formatVersion)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            return try date(fromRFC3339: value, codingPath: decoder.codingPath)
        }
        return try validate(decoder.decode(ProjectManifest.self, from: data))
    }

    static func validate(_ manifest: ProjectManifest) throws -> ProjectManifest {
        guard manifest.formatVersion == currentFormatVersion else {
            throw ProjectManifestError.unsupportedFormatVersion(manifest.formatVersion)
        }
        do { try manifest.engineConfiguration.validate() }
        catch { throw ProjectManifestError.invalidConfiguration }
        guard manifest.sessionState.processedCount <= manifest.sessionState.queuedCount,
              manifest.sessionState.queuedCount <= manifest.sessionState.capturedCount,
              UInt64(exactly: manifest.frames.count) == manifest.sessionState.capturedCount else {
            throw ProjectManifestError.invalidSessionState
        }

        var previousFrameIndex: UInt32?
        var persistedFrameIndices = Set<UInt32>()
        for frame in manifest.frames {
            guard frame.sourceTimestamp.isFinite,
                  frame.sourceTimestamp >= 0,
                  ProjectRelativePath.isSafe(frame.relativePath) else {
                throw ProjectManifestError.invalidFrame
            }
            if let previousFrameIndex, frame.index <= previousFrameIndex {
                throw ProjectManifestError.invalidFrameOrder
            }
            previousFrameIndex = frame.index
            persistedFrameIndices.insert(frame.index)
        }

        var previousWindowIndex: UInt32?
        var previousOutputEnd: UInt32?
        var committedArtifactCount: UInt64 = 0
        for window in manifest.completedWindows {
            guard window.inferenceFrameStart <= window.frameStart,
                  window.frameStart <= window.frameEnd,
                  window.frameEnd <= window.lastProcessedFrameIndex,
                  window.pointChunkRelativePath == WorkerArtifactPath.points(windowIndex: window.index),
                  window.alignmentRowMajor.count == 16,
                  window.alignmentRowMajor.allSatisfy(\.isFinite),
                  validDuration(window.durationSeconds),
                  !window.frameArtifacts.isEmpty else {
                throw ProjectManifestError.invalidWindow
            }
            if let previousWindowIndex, window.index <= previousWindowIndex {
                throw ProjectManifestError.invalidWindowOrder
            }
            if let previousOutputEnd, window.frameStart <= previousOutputEnd {
                throw ProjectManifestError.invalidWindowOrder
            }

            var previousArtifactIndex: UInt32?
            for artifact in window.frameArtifacts {
                guard artifact.windowIndex == window.index,
                      persistedFrameIndices.contains(artifact.frameIndex),
                      artifact.frameIndex >= window.frameStart,
                      artifact.frameIndex <= window.frameEnd,
                      artifact.depthRelativePath == WorkerArtifactPath.depth(frameIndex: artifact.frameIndex),
                      artifact.confidenceRelativePath == WorkerArtifactPath.confidence(frameIndex: artifact.frameIndex),
                      artifact.geometryRelativePath == WorkerArtifactPath.geometry(frameIndex: artifact.frameIndex),
                      validDuration(artifact.durationSeconds) else {
                    throw ProjectManifestError.invalidArtifact
                }
                if let previousArtifactIndex, artifact.frameIndex <= previousArtifactIndex {
                    throw ProjectManifestError.invalidArtifactOrder
                }
                previousArtifactIndex = artifact.frameIndex
            }
            let (nextArtifactCount, artifactCountOverflow) = committedArtifactCount
                .addingReportingOverflow(UInt64(window.frameArtifacts.count))
            guard !artifactCountOverflow else {
                throw ProjectManifestError.invalidSessionState
            }
            committedArtifactCount = nextArtifactCount
            guard window.frameArtifacts.first?.frameIndex == window.frameStart,
                  window.frameArtifacts.last?.frameIndex == window.frameEnd,
                  Array(manifest.frames
                    .lazy
                    .filter { $0.index >= window.frameStart && $0.index <= window.frameEnd }
                    .map(\.index)) == window.frameArtifacts.map(\.frameIndex) else {
                throw ProjectManifestError.invalidArtifact
            }

            previousWindowIndex = window.index
            previousOutputEnd = window.frameEnd
        }

        guard committedArtifactCount == manifest.sessionState.processedCount else {
            throw ProjectManifestError.invalidSessionState
        }

        return manifest
    }

    func resumeCheckpoint() throws -> ResumeCheckpoint? {
        _ = try Self.validate(self)
        guard let finalWindow = completedWindows.last else { return nil }
        let (nextWindowIndex, overflow) = finalWindow.index.addingReportingOverflow(1)
        guard !overflow else { throw ProjectManifestError.checkpointWindowIndexOverflow }

        let replayCount = max(Int(engineConfiguration.windowOverlap), 1)
        let committedArtifacts = completedWindows.flatMap(\.frameArtifacts)
        guard let replayFrom = committedArtifacts.suffix(replayCount).first?.frameIndex else {
            throw ProjectManifestError.invalidArtifact
        }
        return ResumeCheckpoint(
            lastCommittedFrameIndex: finalWindow.frameEnd,
            replayFromFrameIndex: replayFrom,
            nextWindowIndex: nextWindowIndex
        )
    }

    private static func validDuration(_ value: Double) -> Bool {
        value.isFinite && value >= 0
    }

    private static func rfc3339String(for date: Date, codingPath: [CodingKey]) throws -> String {
        let seconds = date.timeIntervalSinceReferenceDate
        guard seconds.isFinite,
              let components = decimalInstantComponents(for: seconds) else {
            throw invalidEncodingDate(date, codingPath: codingPath)
        }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        let wholeSecondString = formatter.string(
            from: Date(timeIntervalSinceReferenceDate: Double(components.wholeSeconds))
        )
        guard wholeSecondString.hasSuffix("Z") else {
            throw invalidEncodingDate(date, codingPath: codingPath)
        }

        return wholeSecondString.dropLast()
            + "."
            + components.fractionalDigits
            + "Z"
    }

    private static func date(fromRFC3339 value: String, codingPath: [CodingKey]) throws -> Date {
        guard let timeSeparator = value.firstIndex(of: "T") else {
            throw invalidDate(value, codingPath: codingPath)
        }

        let fractionSeparator = value[timeSeparator...].firstIndex(of: ".")
        let wholeSecondString: String
        let fractionalDigits: String
        if let fractionSeparator {
            let digitsStart = value.index(after: fractionSeparator)
            let suffixStart = value[digitsStart...].firstIndex { !$0.isASCII || !$0.isNumber }
                ?? value.endIndex
            let digits = value[digitsStart..<suffixStart]
            guard !digits.isEmpty,
                  digits.allSatisfy({ $0.isASCII && $0.isNumber }),
                  suffixStart < value.endIndex else {
                throw invalidDate(value, codingPath: codingPath)
            }
            wholeSecondString = String(value[..<fractionSeparator]) + value[suffixStart...]
            fractionalDigits = String(digits)
        } else {
            wholeSecondString = value
            fractionalDigits = "0"
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        guard let wholeSecondDate = formatter.date(from: wholeSecondString) else {
            throw invalidDate(value, codingPath: codingPath)
        }
        guard let wholeSeconds = Int64(exactly: wholeSecondDate.timeIntervalSinceReferenceDate),
              let decimal = combinedDecimalSeconds(
                wholeSeconds: wholeSeconds,
                fractionalDigits: fractionalDigits
              ),
              let seconds = Double(decimal),
              seconds.isFinite else {
            throw invalidDate(value, codingPath: codingPath)
        }
        return Date(timeIntervalSinceReferenceDate: seconds)
    }

    private static func decimalInstantComponents(
        for seconds: Double
    ) -> (wholeSeconds: Int64, fractionalDigits: String)? {
        guard let decimal = normalizedDecimal(String(seconds)) else { return nil }
        guard let magnitude = Int64(decimal.integerDigits) else { return nil }
        let fractionIsZero = decimal.fractionalDigits.allSatisfy { $0 == "0" }

        if !decimal.isNegative || magnitude == 0 && fractionIsZero {
            return (magnitude, fractionIsZero ? "0" : decimal.fractionalDigits)
        }
        if fractionIsZero {
            return (-magnitude, "0")
        }

        let (nextMagnitude, overflow) = magnitude.addingReportingOverflow(1)
        guard !overflow else { return nil }
        return (-nextMagnitude, tenComplement(decimal.fractionalDigits))
    }

    private static func normalizedDecimal(
        _ representation: String
    ) -> (isNegative: Bool, integerDigits: String, fractionalDigits: String)? {
        var text = representation.lowercased()
        let isNegative = text.first == "-"
        if isNegative { text.removeFirst() }

        let exponent: Int
        if let separator = text.firstIndex(of: "e") {
            guard let parsedExponent = Int(text[text.index(after: separator)...]) else { return nil }
            exponent = parsedExponent
            text = String(text[..<separator])
        } else {
            exponent = 0
        }

        let coefficientParts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard (1...2).contains(coefficientParts.count),
              coefficientParts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }) else {
            return nil
        }
        let coefficientIntegerCount = coefficientParts[0].count
        let digits = coefficientParts.joined()
        let decimalPosition = coefficientIntegerCount + exponent

        let integerDigits: String
        let fractionalDigits: String
        if decimalPosition <= 0 {
            integerDigits = "0"
            fractionalDigits = String(repeating: "0", count: -decimalPosition) + digits
        } else if decimalPosition >= digits.count {
            integerDigits = digits + String(repeating: "0", count: decimalPosition - digits.count)
            fractionalDigits = ""
        } else {
            let split = digits.index(digits.startIndex, offsetBy: decimalPosition)
            integerDigits = String(digits[..<split])
            fractionalDigits = String(digits[split...])
        }

        let normalizedInteger = String(integerDigits.drop { $0 == "0" })
        let normalizedFraction = String(fractionalDigits.reversed().drop { $0 == "0" }.reversed())
        return (
            isNegative,
            normalizedInteger.isEmpty ? "0" : normalizedInteger,
            normalizedFraction.isEmpty ? "0" : normalizedFraction
        )
    }

    private static func tenComplement(_ digits: String) -> String {
        var complemented = digits.compactMap { $0.wholeNumberValue.map { 9 - $0 } }
        var carry = 1
        for index in complemented.indices.reversed() where carry != 0 {
            let sum = complemented[index] + carry
            complemented[index] = sum % 10
            carry = sum / 10
        }
        return complemented.map(String.init).joined()
    }

    private static func combinedDecimalSeconds(
        wholeSeconds: Int64,
        fractionalDigits: String
    ) -> String? {
        guard !fractionalDigits.isEmpty,
              fractionalDigits.allSatisfy(\.isNumber) else {
            return nil
        }
        if wholeSeconds >= 0 {
            return "\(wholeSeconds).\(fractionalDigits)"
        }
        if fractionalDigits.allSatisfy({ $0 == "0" }) {
            return String(wholeSeconds)
        }

        let magnitude = String(wholeSeconds).dropFirst()
        let scaledWhole = magnitude + String(repeating: "0", count: fractionalDigits.count)
        guard var scaledMagnitude = subtractDecimalDigits(
            String(scaledWhole),
            fractionalDigits
        ) else {
            return nil
        }
        if scaledMagnitude.count <= fractionalDigits.count {
            scaledMagnitude = String(
                repeating: "0",
                count: fractionalDigits.count + 1 - scaledMagnitude.count
            ) + scaledMagnitude
        }
        let split = scaledMagnitude.index(
            scaledMagnitude.endIndex,
            offsetBy: -fractionalDigits.count
        )
        return "-" + scaledMagnitude[..<split] + "." + scaledMagnitude[split...]
    }

    private static func subtractDecimalDigits(_ lhs: String, _ rhs: String) -> String? {
        let left = lhs.utf8.reversed().map { Int($0 - 0x30) }
        let right = rhs.utf8.reversed().map { Int($0 - 0x30) }
        guard left.allSatisfy({ (0...9).contains($0) }),
              right.allSatisfy({ (0...9).contains($0) }) else {
            return nil
        }

        var result: [Int] = []
        var borrow = 0
        for index in left.indices {
            var digit = left[index] - borrow - (right.indices.contains(index) ? right[index] : 0)
            if digit < 0 {
                digit += 10
                borrow = 1
            } else {
                borrow = 0
            }
            result.append(digit)
        }
        guard borrow == 0,
              !right.indices.contains(where: { $0 >= left.count && right[$0] != 0 }) else {
            return nil
        }
        while result.count > 1, result.last == 0 { result.removeLast() }
        return result.reversed().map(String.init).joined()
    }

    private static func invalidEncodingDate(
        _ date: Date,
        codingPath: [CodingKey]
    ) -> EncodingError {
        .invalidValue(
            date,
            EncodingError.Context(
                codingPath: codingPath,
                debugDescription: "Manifest dates must be finite UTC RFC3339 instants."
            )
        )
    }

    private static func invalidDate(_ value: String, codingPath: [CodingKey]) -> DecodingError {
        .dataCorrupted(
            DecodingError.Context(
                codingPath: codingPath,
                debugDescription: "Invalid manifest RFC3339 date: \(value)"
            )
        )
    }

    private static func manifestURL(in packageURL: URL) -> URL {
        packageURL.appending(path: "Manifest.json")
    }
}

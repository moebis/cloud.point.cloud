import XCTest
@testable import CloudPoint

final class WorkerProtocolTests: XCTestCase {
    private let projectID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!

    func testCodecRoundTripsFragmentedFrame() throws {
        let envelope = WorkerEnvelope.command(
            .hello(clientVersion: "1.0", supportedProtocolVersions: [1]),
            id: UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!,
            projectId: projectID
        )
        let bytes = try LengthPrefixedJSONCodec.encode(envelope)
        var decoder = LengthPrefixedJSONCodec.Decoder()

        XCTAssertEqual(try decoder.append(bytes.prefix(3)), [])
        XCTAssertEqual(try decoder.append(bytes.dropFirst(3)), [envelope])
        XCTAssertNoThrow(try decoder.finish())
    }

    func testCodecDecodesMultipleFramesFromOneChunk() throws {
        let first = WorkerEnvelope.command(.finishInput, projectId: projectID)
        let second = WorkerEnvelope.command(.shutdown, projectId: projectID)
        var bytes = try LengthPrefixedJSONCodec.encode(first)
        bytes.append(try LengthPrefixedJSONCodec.encode(second))

        var decoder = LengthPrefixedJSONCodec.Decoder()
        XCTAssertEqual(try decoder.append(bytes), [first, second])
    }

    func testOutcomeDecoderPreservesValidPrefixAndContinuesAfterRecoverableInvalidFrame() throws {
        let hello = WorkerEnvelope.command(
            .hello(clientVersion: "native", supportedProtocolVersions: [1]),
            id: UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!,
            projectId: projectID
        )
        let invalidJSON = #"{"id":"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb","payload":{"cameraRefinementIterations":4,"confidenceThreshold":1.5,"keyframeInterval":1,"scaleFrames":"8","voxelSize":0.01,"windowOverlap":8,"windowSize":32},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"configure"}"#
        let pause = WorkerEnvelope.command(
            .pause,
            id: UUID(uuidString: "cccccccc-cccc-cccc-cccc-cccccccccccc")!,
            projectId: projectID
        )
        var bytes = try LengthPrefixedJSONCodec.encode(hello)
        bytes.append(framed(Data(invalidJSON.utf8)))
        bytes.append(try LengthPrefixedJSONCodec.encode(pause))

        var decoder = LengthPrefixedJSONCodec.Decoder()
        let outcomes = decoder.appendOutcomes(bytes)
        XCTAssertEqual(outcomes.count, 3)
        guard case let .envelope(decodedHello) = outcomes[0],
              case let .failure(error, jsonPayload) = outcomes[1],
              case let .envelope(decodedPause) = outcomes[2] else {
            return XCTFail("Expected valid, recoverable failure, valid outcomes")
        }
        XCTAssertEqual(decodedHello, hello)
        XCTAssertEqual(decodedPause, pause)
        guard case .invalidPayload = error else {
            return XCTFail("Wrong typed payload must be a recoverable invalid payload, got \(error)")
        }
        let header = WorkerCommandHeader(
            id: UUID(uuidString: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")!,
            projectID: projectID,
            protocolVersion: 1,
            type: "configure"
        )
        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(error, JSONPayload: jsonPayload),
            .commandErrorThenContinue(header)
        )
    }

    func testOutcomeDecoderConsumesDuplicateFrameAndContinuesWithNextCommand() throws {
        let repeated = WorkerEnvelope.command(
            .pause,
            id: UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!,
            projectId: projectID
        )
        let shutdown = WorkerEnvelope.command(
            .shutdown,
            id: UUID(uuidString: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")!,
            projectId: projectID
        )
        var decoder = LengthPrefixedJSONCodec.Decoder()
        XCTAssertEqual(try decoder.append(LengthPrefixedJSONCodec.encode(repeated)), [repeated])
        var bytes = try LengthPrefixedJSONCodec.encode(repeated)
        bytes.append(try LengthPrefixedJSONCodec.encode(shutdown))

        let outcomes = decoder.appendOutcomes(bytes)
        XCTAssertEqual(outcomes.count, 2)
        guard case let .failure(error, _) = outcomes[0],
              case let .envelope(decodedShutdown) = outcomes[1] else {
            return XCTFail("Expected duplicate failure followed by decoded shutdown")
        }
        XCTAssertEqual(error, .duplicateCommandID(repeated.id))
        XCTAssertEqual(decodedShutdown, shutdown)
    }

    func testCodecRejectsZeroAndOversizedLengthsBeforeBody() {
        var zeroDecoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: 16)
        XCTAssertThrowsError(try zeroDecoder.append(Data([0, 0, 0, 0]))) {
            XCTAssertEqual($0 as? WorkerProtocolError, .zeroLengthPayload)
        }

        var oversizedDecoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: 16)
        XCTAssertThrowsError(try oversizedDecoder.append(Data([0, 0, 0, 17]))) {
            XCTAssertEqual($0 as? WorkerProtocolError, .payloadTooLarge(17))
        }
    }

    func testOversizedHeaderRejectsBeforeAdmittingFollowingBodyBytes() {
        var decoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: 16)
        var bytes = Data([0, 0, 0, 17])
        bytes.append(Data(repeating: 0x61, count: 3 * 1_048_576))

        XCTAssertThrowsError(try decoder.append(bytes)) {
            XCTAssertEqual($0 as? WorkerProtocolError, .payloadTooLarge(17))
        }
        XCTAssertLessThanOrEqual(decoder.bufferedByteCount, 4)
    }

    func testCodecRejectsTruncatedFrameOnFinish() throws {
        let frame = try LengthPrefixedJSONCodec.encode(
            WorkerEnvelope.command(.pause, projectId: projectID)
        )
        var decoder = LengthPrefixedJSONCodec.Decoder()
        _ = try decoder.append(frame.dropLast())

        XCTAssertThrowsError(try decoder.finish()) {
            XCTAssertEqual($0 as? WorkerProtocolError, .truncatedFrame)
        }
    }

    func testCodecRejectsMalformedAndTrailingJSON() {
        for body in [Data("{".utf8), Data("{} {}".utf8)] {
            var decoder = LengthPrefixedJSONCodec.Decoder()
            XCTAssertThrowsError(try decoder.append(framed(body)))
        }
    }

    func testEncodingIsCanonicalSortedCompactJSON() throws {
        let envelope = WorkerEnvelope.command(
            .hello(clientVersion: "native", supportedProtocolVersions: [1]),
            id: UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!,
            projectId: projectID
        )
        let frame = try LengthPrefixedJSONCodec.encode(envelope)
        let json = String(decoding: frame.dropFirst(4), as: UTF8.self)

        XCTAssertEqual(
            json,
            #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"clientVersion":"native","supportedProtocolVersions":[1]},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"hello"}"#
        )
    }

    func testNullableContractFieldsEncodeAsExplicitNull() throws {
        let error = WorkerErrorPayload(code: "worker", message: "failed", recoverable: false, details: [:])
        let envelopes: [(WorkerEnvelope, String)] = [
            (.command(.beginSession(resumeCheckpoint: nil), projectId: projectID), #""resumeCheckpoint":null"#),
            (.event(.error(commandId: nil, error), projectId: projectID), #""commandId":null"#),
            (.event(.cancelled(lastCompletedWindowIndex: nil), projectId: projectID), #""lastCompletedWindowIndex":null"#),
            (.event(.heartbeat(busy: false, monotonicSeconds: 1, queuedFrames: 0, processedFrames: 0, currentWindow: nil), projectId: projectID), #""currentWindow":null"#),
        ]

        for (envelope, expectedField) in envelopes {
            let frame = try LengthPrefixedJSONCodec.encode(envelope)
            XCTAssertTrue(String(decoding: frame.dropFirst(4), as: UTF8.self).contains(expectedField))
        }
    }

    func testJSONDetailsRoundTripIntegersBeyondDoublePrecisionLosslessly() throws {
        let value: UInt64 = 9_007_199_254_740_993
        let payload = WorkerErrorPayload(
            code: "largeInteger",
            message: "preserve exact JSON",
            recoverable: false,
            details: ["nested": .object(["value": .number(.unsigned(value))])]
        )
        let envelope = WorkerEnvelope.event(.warning(payload), projectId: projectID)
        let frame = try LengthPrefixedJSONCodec.encode(envelope)

        XCTAssertTrue(String(decoding: frame.dropFirst(4), as: UTF8.self).contains("9007199254740993"))
        XCTAssertEqual(try roundTrip(envelope), envelope)
    }

    func testJSONDetailsPreserveArbitraryPrecisionNumberTokensWithoutStringifying() throws {
        let positive = "12345678901234567890123456789012345678901234567890"
        let negative = "-98765432109876543210987654321098765432109876543210"
        let decimal = "0.12345678901234567890123456789012345678901234567890"
        let exponent = "6.02214076000000000000000000000000000000000000000000e+123"
        let json = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"code":"exact","details":{"nested":{"decimal":DECIMAL,"exponent":EXPONENT,"integers":{"negative":NEGATIVE,"positive":POSITIVE}}},"message":"numbers","recoverable":false},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"warning"}"#
            .replacingOccurrences(of: "POSITIVE", with: positive)
            .replacingOccurrences(of: "NEGATIVE", with: negative)
            .replacingOccurrences(of: "DECIMAL", with: decimal)
            .replacingOccurrences(of: "EXPONENT", with: exponent)

        let decoded = try decodeOne(json)
        let firstFrame = try LengthPrefixedJSONCodec.encode(decoded)
        let secondFrame = try LengthPrefixedJSONCodec.encode(decoded)
        let encoded = String(decoding: firstFrame.dropFirst(4), as: UTF8.self)

        XCTAssertEqual(firstFrame, secondFrame)
        for token in [positive, negative, decimal, exponent] {
            XCTAssertTrue(encoded.contains(token), "Missing exact numeric token \(token) in \(encoded)")
            XCTAssertFalse(encoded.contains("\"\(token)\""), "Numeric token was stringified")
        }
        XCTAssertEqual(encoded, json)
        XCTAssertEqual(try decodeOne(encoded), decoded)
    }

    func testRawDetailNumberRewritingStaysBoundedForDenseValidPayload() throws {
        let denseNumbers = Array(repeating: "0", count: 50_000).joined(separator: ",")
        let json = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"code":"dense","details":{"values":[NUMBERS]},"message":"numbers","recoverable":false},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"warning"}"#
            .replacingOccurrences(of: "NUMBERS", with: denseNumbers)
        let body = Data(json.utf8)
        var decoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: body.count)

        let outcomes = decoder.appendOutcomes(framed(body))
        XCTAssertEqual(outcomes.count, 1)
        guard case let .envelope(envelope) = outcomes[0] else {
            return XCTFail("Dense valid raw-number payload was rejected: \(outcomes[0])")
        }
        XCTAssertEqual(
            Data(try LengthPrefixedJSONCodec.encode(envelope).dropFirst(4)),
            body
        )
    }

    func testDecoderRejectsInvalidJSONNumberGrammar() {
        for invalid in ["+1", "01", "1.", ".1", "1e", "1e+", "--1", "NaN", "Infinity"] {
            let json = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"code":"invalid","details":{"value":NUMBER},"message":"number","recoverable":false},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"warning"}"#
                .replacingOccurrences(of: "NUMBER", with: invalid)
            XCTAssertThrowsError(try decodeOne(json), "Accepted invalid JSON number \(invalid)")
        }
    }

    func testDecoderRejectsUnknownEnvelopeAndPayloadKeys() {
        let unknownEnvelopeKey = #"{"extra":true,"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"pause"}"#
        let unknownEmptyPayloadKey = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"extra":true},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"pause"}"#
        let unknownTypedPayloadKey = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"clientVersion":"native","extra":true,"supportedProtocolVersions":[1]},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"hello"}"#

        for json in [unknownEnvelopeKey, unknownEmptyPayloadKey, unknownTypedPayloadKey] {
            XCTAssertThrowsError(try decodeOne(json)) {
                guard case .invalidPayload? = $0 as? WorkerProtocolError else {
                    return XCTFail("Unexpected error: \($0)")
                }
            }
        }
    }

    func testDecoderRejectsMissingNullableKeys() {
        let missingResumeIndex = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"beginSession"}"#
        let missingErrorCommandID = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"code":"worker","details":{},"message":"failed","recoverable":false},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"error"}"#
        let missingCurrentWindow = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"busy":false,"monotonicSeconds":1,"processedFrames":0,"queuedFrames":0},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"heartbeat"}"#

        for json in [missingResumeIndex, missingErrorCommandID, missingCurrentWindow] {
            XCTAssertThrowsError(try decodeOne(json))
        }
    }

    func testDuplicateCommandIDWindowEvictsOldestAfter4096Entries() throws {
        let oldestID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
        var decoder = LengthPrefixedJSONCodec.Decoder()
        _ = try decoder.append(LengthPrefixedJSONCodec.encode(.command(.pause, id: oldestID, projectId: projectID)))
        for value in 2...4_097 {
            let id = UUID(uuidString: String(format: "00000000-0000-0000-0000-%012X", value))!
            _ = try decoder.append(LengthPrefixedJSONCodec.encode(.command(.pause, id: id, projectId: projectID)))
        }

        XCTAssertNoThrow(try decoder.append(LengthPrefixedJSONCodec.encode(.command(.pause, id: oldestID, projectId: projectID))))
    }

    func testDecoderRejectsUnknownVersionAndType() {
        let unknownVersion = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":2,"type":"pause"}"#
        let unknownType = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"future"}"#

        XCTAssertThrowsError(try decodeOne(unknownVersion)) {
            XCTAssertEqual($0 as? WorkerProtocolError, .unsupportedProtocolVersion(2))
        }
        XCTAssertThrowsError(try decodeOne(unknownType)) {
            XCTAssertEqual($0 as? WorkerProtocolError, .unknownMessageType("future"))
        }
    }

    func testDecoderRejectsDuplicateCommandIDsButAllowsEventCommandReferences() throws {
        let commandID = UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!
        let command = WorkerEnvelope.command(.resume, id: commandID, projectId: projectID)
        var bytes = try LengthPrefixedJSONCodec.encode(command)
        bytes.append(try LengthPrefixedJSONCodec.encode(command))
        var decoder = LengthPrefixedJSONCodec.Decoder()

        XCTAssertThrowsError(try decoder.append(bytes)) {
            XCTAssertEqual($0 as? WorkerProtocolError, .duplicateCommandID(commandID))
        }

        let ack = WorkerEnvelope.event(.ack(commandId: commandID, command: "resume"), projectId: projectID)
        var eventDecoder = LengthPrefixedJSONCodec.Decoder()
        XCTAssertEqual(try eventDecoder.append(try LengthPrefixedJSONCodec.encode(ack)), [ack])
    }

    func testAllCommandPayloadsRoundTrip() throws {
        let commands: [WorkerCommand] = [
            .hello(clientVersion: "1.2.3", supportedProtocolVersions: [1]),
            .configure(EngineConfiguration(scaleFrames: 4, windowSize: 32, windowOverlap: 8, keyframeInterval: 3, cameraRefinementIterations: 12, confidenceThreshold: 1.5, voxelSize: 0.01)),
            .beginSession(resumeCheckpoint: nil),
            .beginSession(resumeCheckpoint: ResumeCheckpoint(lastCommittedFrameIndex: 42, replayFromFrameIndex: 35, nextWindowIndex: 2)),
            .enqueueFrame(frameIndex: 7, sourceTimestamp: 1.25, relativePath: "Frames/00000007.jpg"),
            .finishInput, .pause, .resume, .cancel, .shutdown,
        ]

        for command in commands {
            let envelope = WorkerEnvelope.command(command, projectId: projectID)
            XCTAssertEqual(try roundTrip(envelope), envelope)
        }
    }

    func testAllEventPayloadsRoundTrip() throws {
        let commandID = UUID(uuidString: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")!
        let error = WorkerErrorPayload(code: "badFrame", message: "Bad frame", recoverable: true, details: ["frameIndex": .number(4)])
        let events: [WorkerEvent] = [
            .ack(commandId: commandID, command: "hello"),
            .error(commandId: nil, error),
            .ready(engineVersion: "1.0", modelIdentifier: "depth", modelRevision: "r1", convertedWeightsSHA256: String(repeating: "a", count: 64)),
            .modelProgress(phase: .validating, completed: 1, total: 2),
            .modelProgress(phase: .loading, completed: 2, total: 2),
            .frameStarted(frameIndex: 1, windowIndex: 0),
            .frameCompleted(FrameArtifacts(frameIndex: 1, windowIndex: 0, depthRelativePath: "Predictions/00000001.depth-f16", confidenceRelativePath: "Predictions/00000001.confidence-f16", geometryRelativePath: "Predictions/00000001.geometry.json", durationSeconds: 0.25)),
            .windowCompleted(WindowResult(windowIndex: 0, inferenceFrameStart: 0, frameStart: 0, frameEnd: 3, pointChunkRelativePath: "Points/window-00000000.cpc", alignmentRowMajor: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], lastProcessedFrameIndex: 3, inlierCount: 99, durationSeconds: 1.5)),
            .sessionCompleted(processedFrames: 4, windowCount: 1, durationSeconds: 2.0),
            .paused(queuedFrames: 3, processedFrames: 3),
            .cancelled(lastCompletedWindowIndex: nil),
            .cancelled(lastCompletedWindowIndex: 4),
            .warning(error),
            .heartbeat(busy: true, monotonicSeconds: 10.5, queuedFrames: 8, processedFrames: 8, currentWindow: 1),
        ]

        for event in events {
            let envelope = WorkerEnvelope.event(event, projectId: projectID)
            XCTAssertEqual(try roundTrip(envelope), envelope)
        }
    }

    func testPayloadValidationRejectsInvalidValues() throws {
        let invalidCommands: [WorkerCommand] = [
            .hello(clientVersion: "native", supportedProtocolVersions: [2]),
            .configure(EngineConfiguration(scaleFrames: 0)),
            .configure(EngineConfiguration(scaleFrames: 4, windowSize: 8, windowOverlap: 8)),
            .configure(EngineConfiguration(scaleFrames: 4, windowSize: 8, windowOverlap: 4, confidenceThreshold: .infinity)),
            .enqueueFrame(frameIndex: 1, sourceTimestamp: -1, relativePath: "Frames/1.jpg"),
            .enqueueFrame(frameIndex: 1, sourceTimestamp: 0, relativePath: "/tmp/1.jpg"),
            .enqueueFrame(frameIndex: 1, sourceTimestamp: 0, relativePath: "Frames/../1.jpg"),
        ]
        for command in invalidCommands {
            XCTAssertThrowsError(try LengthPrefixedJSONCodec.encode(.command(command, projectId: projectID)))
        }

        let invalidEvents: [WorkerEvent] = [
            .modelProgress(phase: .loading, completed: 3, total: 2),
            .frameCompleted(FrameArtifacts(frameIndex: 1, windowIndex: 0, depthRelativePath: "/Depth/1", confidenceRelativePath: "Predictions/00000001.confidence-f16", geometryRelativePath: "Predictions/00000001.geometry.json", durationSeconds: 1)),
            .windowCompleted(WindowResult(windowIndex: 0, inferenceFrameStart: 0, frameStart: 2, frameEnd: 1, pointChunkRelativePath: "Points/window-00000000.cpc", alignmentRowMajor: [1], lastProcessedFrameIndex: 1, inlierCount: 1, durationSeconds: 1)),
            .heartbeat(busy: false, monotonicSeconds: .nan, queuedFrames: 0, processedFrames: 0, currentWindow: nil),
        ]
        for event in invalidEvents {
            XCTAssertThrowsError(try LengthPrefixedJSONCodec.encode(.event(event, projectId: projectID)))
        }
    }

    private func roundTrip(_ envelope: WorkerEnvelope) throws -> WorkerEnvelope {
        var decoder = LengthPrefixedJSONCodec.Decoder()
        return try XCTUnwrap(decoder.append(LengthPrefixedJSONCodec.encode(envelope)).only)
    }

    private func decodeOne(_ json: String) throws -> WorkerEnvelope {
        var decoder = LengthPrefixedJSONCodec.Decoder()
        let decoded = try decoder.append(framed(Data(json.utf8)))
        guard let envelope = decoded.only else { throw WorkerProtocolError.malformedEnvelope }
        return envelope
    }

    private func framed(_ body: Data) -> Data {
        var result = Data()
        let length = UInt32(body.count).bigEndian
        withUnsafeBytes(of: length) { result.append(contentsOf: $0) }
        result.append(body)
        return result
    }
}

private extension Array {
    var only: Element? { count == 1 ? first : nil }
}

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
            #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{"clientVersion":"native","supportedProtocolVersions":[1]},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"hello"}"#
        )
    }

    func testNullableContractFieldsEncodeAsExplicitNull() throws {
        let error = WorkerErrorPayload(code: "worker", message: "failed", recoverable: false, details: [:])
        let envelopes: [(WorkerEnvelope, String)] = [
            (.command(.beginSession(resumeAfterFrameIndex: nil), projectId: projectID), #""resumeAfterFrameIndex":null"#),
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

    func testDecoderRejectsUnknownEnvelopeAndPayloadKeys() {
        let unknownEnvelopeKey = #"{"extra":true,"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"pause"}"#
        let unknownEmptyPayloadKey = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{"extra":true},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"pause"}"#
        let unknownTypedPayloadKey = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{"clientVersion":"native","extra":true,"supportedProtocolVersions":[1]},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"hello"}"#

        for json in [unknownEnvelopeKey, unknownEmptyPayloadKey, unknownTypedPayloadKey] {
            XCTAssertThrowsError(try decodeOne(json)) {
                guard case .invalidPayload? = $0 as? WorkerProtocolError else {
                    return XCTFail("Unexpected error: \($0)")
                }
            }
        }
    }

    func testDecoderRejectsMissingNullableKeys() {
        let missingResumeIndex = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"beginSession"}"#
        let missingErrorCommandID = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{"code":"worker","details":{},"message":"failed","recoverable":false},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"error"}"#
        let missingCurrentWindow = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{"busy":false,"monotonicSeconds":1,"processedFrames":0,"queuedFrames":0},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"heartbeat"}"#

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
        let unknownVersion = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":2,"type":"pause"}"#
        let unknownType = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"future"}"#

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
            .configure(scaleFrames: 4, windowSize: 32, windowOverlap: 8, keyframeInterval: 3, cameraRefinementIterations: 12, confidenceThreshold: 1.5),
            .beginSession(resumeAfterFrameIndex: nil),
            .beginSession(resumeAfterFrameIndex: 42),
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
            .frameCompleted(frameIndex: 1, windowIndex: 0, depthPath: "Depth/1.bin", confidencePath: "Confidence/1.bin", geometryPath: "Geometry/1.bin", pointChunkPath: "Points/1.cpc", durationSeconds: 0.25),
            .windowCompleted(windowIndex: 0, frameStart: 0, frameEnd: 3, pointChunkPath: "Points/window-0.cpc", alignmentTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], lastProcessedFrameIndex: 3, inlierCount: 99, durationSeconds: 1.5),
            .sessionCompleted(processedFrames: 4, windowCount: 1, durationSeconds: 2.0),
            .paused(queuedFrames: 2, processedFrames: 3),
            .cancelled(lastCompletedWindowIndex: nil),
            .cancelled(lastCompletedWindowIndex: 4),
            .warning(error),
            .heartbeat(busy: true, monotonicSeconds: 10.5, queuedFrames: 2, processedFrames: 8, currentWindow: 1),
        ]

        for event in events {
            let envelope = WorkerEnvelope.event(event, projectId: projectID)
            XCTAssertEqual(try roundTrip(envelope), envelope)
        }
    }

    func testPayloadValidationRejectsInvalidValues() throws {
        let invalidCommands: [WorkerCommand] = [
            .hello(clientVersion: "native", supportedProtocolVersions: [2]),
            .configure(scaleFrames: 0, windowSize: 32, windowOverlap: 8, keyframeInterval: 3, cameraRefinementIterations: 12, confidenceThreshold: 1.5),
            .configure(scaleFrames: 4, windowSize: 8, windowOverlap: 8, keyframeInterval: 3, cameraRefinementIterations: 12, confidenceThreshold: 1.5),
            .configure(scaleFrames: 4, windowSize: 8, windowOverlap: 4, keyframeInterval: 3, cameraRefinementIterations: 12, confidenceThreshold: .infinity),
            .enqueueFrame(frameIndex: 1, sourceTimestamp: -1, relativePath: "Frames/1.jpg"),
            .enqueueFrame(frameIndex: 1, sourceTimestamp: 0, relativePath: "/tmp/1.jpg"),
            .enqueueFrame(frameIndex: 1, sourceTimestamp: 0, relativePath: "Frames/../1.jpg"),
        ]
        for command in invalidCommands {
            XCTAssertThrowsError(try LengthPrefixedJSONCodec.encode(.command(command, projectId: projectID)))
        }

        let invalidEvents: [WorkerEvent] = [
            .modelProgress(phase: .loading, completed: -1, total: 2),
            .frameCompleted(frameIndex: 1, windowIndex: 0, depthPath: "/Depth/1", confidencePath: "Confidence/1", geometryPath: "Geometry/1", pointChunkPath: "Points/1", durationSeconds: 1),
            .windowCompleted(windowIndex: 0, frameStart: 2, frameEnd: 1, pointChunkPath: "Points/1", alignmentTransform: [1], lastProcessedFrameIndex: 1, inlierCount: 1, durationSeconds: 1),
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

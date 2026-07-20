import Foundation
import XCTest
@testable import CloudPoint

final class ProtocolOwnershipContractTests: XCTestCase {
    private let projectID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!

    func testCompleteConfigurationAndCheckpointRoundTripExactly() throws {
        let configuration = EngineConfiguration(
            scaleFrames: 4,
            windowSize: 32,
            windowOverlap: 8,
            keyframeInterval: 3,
            cameraRefinementIterations: 12,
            confidenceThreshold: 1.5,
            voxelSize: 0.01
        )
        let checkpoint = ResumeCheckpoint(
            lastCommittedFrameIndex: 44,
            replayFromFrameIndex: 31,
            nextWindowIndex: 2
        )

        XCTAssertNoThrow(try configuration.validate())
        XCTAssertEqual(
            try roundTrip(.command(.configure(configuration), projectId: projectID)).command,
            .configure(configuration)
        )
        XCTAssertEqual(
            try roundTrip(.command(.beginSession(resumeCheckpoint: checkpoint), projectId: projectID)).command,
            .beginSession(resumeCheckpoint: checkpoint)
        )
    }

    func testCheckpointRejectsReplayAfterCommittedBoundary() throws {
        let invalid = ResumeCheckpoint(
            lastCommittedFrameIndex: 4,
            replayFromFrameIndex: 5,
            nextWindowIndex: 1
        )

        XCTAssertThrowsError(
            try LengthPrefixedJSONCodec.encode(
                .command(.beginSession(resumeCheckpoint: invalid), projectId: projectID)
            )
        )
    }

    func testProtocolCompletionOwnershipAndUnsignedCountersRoundTrip() throws {
        let artifacts = FrameArtifacts(
            frameIndex: .max,
            windowIndex: 7,
            depthRelativePath: "Predictions/4294967295.depth-f16",
            confidenceRelativePath: "Predictions/4294967295.confidence-f16",
            geometryRelativePath: "Predictions/4294967295.geometry.json",
            durationSeconds: 0.25
        )
        let result = WindowResult(
            windowIndex: 7,
            inferenceFrameStart: 40,
            frameStart: 48,
            frameEnd: .max,
            pointChunkRelativePath: "Points/window-00000007.cpc",
            alignmentRowMajor: Self.identity,
            lastProcessedFrameIndex: .max,
            inlierCount: .max,
            durationSeconds: 1.5
        )
        let events: [WorkerEvent] = [
            .frameCompleted(artifacts),
            .windowCompleted(result),
            .sessionCompleted(processedFrames: .max, windowCount: .max, durationSeconds: 2),
            .paused(queuedFrames: .max, processedFrames: .max),
            .heartbeat(
                busy: true,
                monotonicSeconds: 0,
                queuedFrames: .max,
                processedFrames: .max,
                currentWindow: .max
            ),
        ]

        for event in events {
            XCTAssertEqual(try roundTrip(.event(event, projectId: projectID)).event, event)
        }
    }

    func testAsynchronousErrorOwnershipAndWarningsPreserveExactDetails() throws {
        let details: [String: JSONValue] = [
            "attempt": .number(.unsigned(.max)),
            "measurement": .number(.raw("1.2300e+04")),
            "nested": .object(["retry": .bool(true)]),
        ]
        let payload = WorkerErrorPayload(
            code: "inferenceFailed",
            message: "The worker could not finish inference.",
            recoverable: false,
            details: details
        )

        XCTAssertThrowsError(try WorkerEvent.error(commandId: nil, payload).engineEvent()) { error in
            XCTAssertEqual(
                error as? ReconstructionEngineError,
                .workerFailure(
                    code: payload.code,
                    message: payload.message,
                    recoverable: payload.recoverable,
                    details: details
                )
            )
        }
        XCTAssertNil(try WorkerEvent.error(commandId: UUID(), payload).engineEvent())
        XCTAssertEqual(
            try WorkerEvent.warning(payload).engineEvent(),
            .warning(
                code: payload.code,
                message: payload.message,
                recoverable: payload.recoverable,
                details: details
            )
        )
    }

    func testTelemetryRejectsProcessedCountsBeyondQueuedCounts() {
        let invalid: [WorkerEvent] = [
            .paused(queuedFrames: 3, processedFrames: 4),
            .heartbeat(busy: false, monotonicSeconds: 0, queuedFrames: 3, processedFrames: 4, currentWindow: nil),
        ]
        for event in invalid {
            XCTAssertThrowsError(try LengthPrefixedJSONCodec.encode(.event(event, projectId: projectID)))
        }
    }

    func testFrameAndWindowFixturesAreCanonicalExactJSON() throws {
        let frame = WorkerEnvelope.event(
            .frameCompleted(FrameArtifacts(
                frameIndex: 7,
                windowIndex: 2,
                depthRelativePath: "Predictions/00000007.depth-f16",
                confidenceRelativePath: "Predictions/00000007.confidence-f16",
                geometryRelativePath: "Predictions/00000007.geometry.json",
                durationSeconds: 0.25
            )),
            id: UUID(uuidString: "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA")!,
            projectId: projectID
        )
        let json = String(decoding: try LengthPrefixedJSONCodec.encode(frame).dropFirst(4), as: UTF8.self)

        XCTAssertEqual(
            json,
            #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"confidencePath":"Predictions/00000007.confidence-f16","depthPath":"Predictions/00000007.depth-f16","durationSeconds":0.25,"frameIndex":7,"geometryPath":"Predictions/00000007.geometry.json","windowIndex":2},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"frameCompleted"}"#
        )

        let numeric = WorkerEnvelope.event(
            .heartbeat(busy: false, monotonicSeconds: -0.0, queuedFrames: 0, processedFrames: 0, currentWindow: nil),
            id: UUID(uuidString: "BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB")!,
            projectId: projectID
        )
        let numericJSON = String(decoding: try LengthPrefixedJSONCodec.encode(numeric).dropFirst(4), as: UTF8.self)
        XCTAssertTrue(numericJSON.contains(#""id":"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb""#))
        XCTAssertTrue(numericJSON.contains(#""monotonicSeconds":0"#))
        XCTAssertFalse(numericJSON.contains("-0"))
    }

    func testCheckpointConfigurationAndWindowHaveExactCrossLanguageBytes() throws {
        let id = UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!
        let configuration = EngineConfiguration(
            scaleFrames: 8,
            windowSize: 32,
            windowOverlap: 8,
            keyframeInterval: 1,
            cameraRefinementIterations: 4,
            confidenceThreshold: 1.5,
            voxelSize: 0.01
        )
        let checkpoint = ResumeCheckpoint(
            lastCommittedFrameIndex: 44,
            replayFromFrameIndex: 31,
            nextWindowIndex: 2
        )
        let window = WindowResult(
            windowIndex: 2,
            inferenceFrameStart: 31,
            frameStart: 40,
            frameEnd: 44,
            pointChunkRelativePath: "Points/window-00000002.cpc",
            alignmentRowMajor: Self.identity,
            lastProcessedFrameIndex: 44,
            inlierCount: 99,
            durationSeconds: 1
        )
        let fixtures: [(WorkerEnvelope, String)] = [
            (
                .command(.configure(configuration), id: id, projectId: projectID),
                #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"cameraRefinementIterations":4,"confidenceThreshold":1.5,"keyframeInterval":1,"scaleFrames":8,"voxelSize":0.01,"windowOverlap":8,"windowSize":32},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"configure"}"#
            ),
            (
                .command(.beginSession(resumeCheckpoint: nil), id: id, projectId: projectID),
                #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"resumeCheckpoint":null},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"beginSession"}"#
            ),
            (
                .command(.beginSession(resumeCheckpoint: checkpoint), id: id, projectId: projectID),
                #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"resumeCheckpoint":{"lastCommittedFrameIndex":44,"nextWindowIndex":2,"replayFromFrameIndex":31}},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"beginSession"}"#
            ),
            (
                .event(.windowCompleted(window), id: id, projectId: projectID),
                #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"alignmentTransform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"durationSeconds":1,"frameEnd":44,"frameStart":40,"inferenceFrameStart":31,"inlierCount":99,"lastProcessedFrameIndex":44,"pointChunkPath":"Points/window-00000002.cpc","windowIndex":2},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"windowCompleted"}"#
            ),
        ]

        for (envelope, expected) in fixtures {
            XCTAssertEqual(
                String(decoding: try LengthPrefixedJSONCodec.encode(envelope).dropFirst(4), as: UTF8.self),
                expected
            )
        }
    }

    func testTypedDoubleCanonicalizationCoversIntegralNonintegralAndExponentForms() throws {
        let exponent = WorkerEnvelope.command(
            .enqueueFrame(frameIndex: 1, sourceTimestamp: 1e100, relativePath: "Frames/00000001.jpg"),
            id: UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!,
            projectId: projectID
        )
        let json = String(decoding: try LengthPrefixedJSONCodec.encode(exponent).dropFirst(4), as: UTF8.self)
        XCTAssertTrue(json.contains(#""sourceTimestamp":1e100"#), json)
        XCTAssertFalse(json.contains("e+"), json)

        let integral = WorkerEnvelope.event(
            .sessionCompleted(processedFrames: 1, windowCount: 1, durationSeconds: 1.0),
            id: UUID(uuidString: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")!,
            projectId: projectID
        )
        let integralJSON = String(decoding: try LengthPrefixedJSONCodec.encode(integral).dropFirst(4), as: UTF8.self)
        XCTAssertTrue(integralJSON.contains(#""durationSeconds":1"#), integralJSON)
        XCTAssertFalse(integralJSON.contains("1.0"), integralJSON)
    }

    func testStrictNestedCheckpointAndConfigurationKeys() {
        let nestedCheckpointExtra = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"resumeCheckpoint":{"extra":0,"lastCommittedFrameIndex":4,"nextWindowIndex":1,"replayFromFrameIndex":3}},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"beginSession"}"#
        let nestedCheckpointMissing = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"resumeCheckpoint":{"lastCommittedFrameIndex":4,"nextWindowIndex":1}},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"beginSession"}"#
        let missingVoxel = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"cameraRefinementIterations":12,"confidenceThreshold":1.5,"keyframeInterval":3,"scaleFrames":4,"windowOverlap":8,"windowSize":32},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"configure"}"#
        let extraConfiguration = #"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"cameraRefinementIterations":12,"confidenceThreshold":1.5,"extra":0,"keyframeInterval":3,"scaleFrames":4,"voxelSize":0.01,"windowOverlap":8,"windowSize":32},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"configure"}"#

        XCTAssertThrowsError(try decodeOne(nestedCheckpointExtra))
        XCTAssertThrowsError(try decodeOne(nestedCheckpointMissing))
        XCTAssertThrowsError(try decodeOne(missingVoxel))
        XCTAssertThrowsError(try decodeOne(extraConfiguration))
    }

    func testDecoderRejectsNoncanonicalUUIDText() {
        let uppercase = #"{"id":"AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"pause"}"#
        let unhyphenated = #"{"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","payload":{},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"pause"}"#
        XCTAssertThrowsError(try decodeOne(uppercase))
        XCTAssertThrowsError(try decodeOne(unhyphenated))
    }

    func testPendingAccumulatorRequiresExactExpectedIDs() throws {
        var accumulator = PendingWindowAccumulator()
        try accumulator.add(.fixture(frame: 3, window: 1))
        try accumulator.add(.fixture(frame: 7, window: 1))
        let result = WindowResult.fixture(window: 1, inferenceStart: 2, start: 3, end: 7)

        let completed = try accumulator.finalize(result, expectedFrameIndices: [3, 7])
        XCTAssertEqual(completed.frameArtifacts.map(\.frameIndex), [3, 7])

        var missing = PendingWindowAccumulator()
        try missing.add(.fixture(frame: 3, window: 1))
        XCTAssertThrowsError(try missing.finalize(result, expectedFrameIndices: [3, 7]))

        var duplicate = PendingWindowAccumulator()
        try duplicate.add(.fixture(frame: 3, window: 1))
        XCTAssertThrowsError(try duplicate.add(.fixture(frame: 3, window: 1)))

        var outOfOrder = PendingWindowAccumulator()
        try outOfOrder.add(.fixture(frame: 7, window: 1))
        XCTAssertThrowsError(try outOfOrder.add(.fixture(frame: 3, window: 1)))

        var crossWindow = PendingWindowAccumulator()
        try crossWindow.add(.fixture(frame: 3, window: 1))
        XCTAssertThrowsError(try crossWindow.add(.fixture(frame: 7, window: 2)))
    }

    func testPendingAccumulatorRejectsSemanticallyInvalidArtifactsAndWindow() throws {
        var invalidArtifact = FrameArtifacts.fixture(frame: 3, window: 1)
        invalidArtifact.depthRelativePath = "Predictions/wrong.depth-f16"
        var artifactAccumulator = PendingWindowAccumulator()
        XCTAssertThrowsError(try artifactAccumulator.add(invalidArtifact))

        var windowAccumulator = PendingWindowAccumulator()
        try windowAccumulator.add(.fixture(frame: 3, window: 1))
        var invalidWindow = WindowResult.fixture(window: 1, inferenceStart: 3, start: 3, end: 3)
        invalidWindow.alignmentRowMajor = [1]
        XCTAssertThrowsError(try windowAccumulator.finalize(invalidWindow, expectedFrameIndices: [3]))
    }

    func testManifestV2CheckpointUsesActualGappedArtifactsAcrossWindowBoundary() throws {
        var manifest = ProjectManifest.fixtureV2(windowOverlap: 3)
        manifest.frames = [2, 6, 9, 14, 20].map {
            PersistedFrame(index: $0, sourceTimestamp: Double($0), relativePath: String(format: "Frames/%08u.jpg", $0))
        }
        manifest.completedWindows = [
            .fixture(index: 0, inferenceStart: 2, frames: [2, 6, 9]),
            .fixture(index: 1, inferenceStart: 6, frames: [14, 20]),
        ]
        manifest.synchronizeSessionCounts()

        let checkpoint = try XCTUnwrap(manifest.resumeCheckpoint())
        XCTAssertEqual(
            checkpoint,
            ResumeCheckpoint(lastCommittedFrameIndex: 20, replayFromFrameIndex: 9, nextWindowIndex: 2)
        )
        XCTAssertEqual(try ProjectManifest.decode(ProjectManifest.encode(manifest)), manifest)
    }

    func testManifestZeroOverlapStillReplaysCommittedBoundary() throws {
        var manifest = ProjectManifest.fixtureV2(windowOverlap: 0)
        manifest.frames = [11, 19].map {
            PersistedFrame(index: $0, sourceTimestamp: Double($0), relativePath: String(format: "Frames/%08u.jpg", $0))
        }
        manifest.completedWindows = [.fixture(index: 0, inferenceStart: 11, frames: [11, 19])]
        manifest.synchronizeSessionCounts()

        XCTAssertEqual(
            try manifest.resumeCheckpoint(),
            ResumeCheckpoint(lastCommittedFrameIndex: 19, replayFromFrameIndex: 19, nextWindowIndex: 1)
        )
    }

    func testManifestRequiresEveryPersistedFrameInsideUniqueBoundsAndCanonicalPaths() throws {
        var missingMiddle = ProjectManifest.fixtureV2(windowOverlap: 1)
        missingMiddle.frames = [1, 2, 3].map {
            PersistedFrame(index: $0, sourceTimestamp: Double($0), relativePath: String(format: "Frames/%08u.jpg", $0))
        }
        missingMiddle.completedWindows = [.fixture(index: 0, inferenceStart: 1, frames: [1, 3])]
        XCTAssertThrowsError(try ProjectManifest.validate(missingMiddle))

        var wrongPath = ProjectManifest.fixtureV2(windowOverlap: 1)
        wrongPath.frames = [PersistedFrame(index: 4, sourceTimestamp: 4, relativePath: "Frames/00000004.jpg")]
        wrongPath.completedWindows = [.fixture(index: 0, inferenceStart: 4, frames: [4])]
        wrongPath.completedWindows[0].pointChunkRelativePath = "Points/other.cpc"
        XCTAssertThrowsError(try ProjectManifest.validate(wrongPath))
    }

    func testManifestRejectsMalformedConfigurationFrameWindowAndArtifactMatrix() throws {
        var baseline = ProjectManifest.fixtureV2(windowOverlap: 1)
        baseline.frames = [1, 3, 5].map {
            PersistedFrame(index: $0, sourceTimestamp: Double($0), relativePath: String(format: "Frames/%08u.jpg", $0))
        }
        baseline.completedWindows = [
            .fixture(index: 0, inferenceStart: 1, frames: [1, 3]),
            .fixture(index: 1, inferenceStart: 3, frames: [5]),
        ]
        baseline.synchronizeSessionCounts()
        XCTAssertNoThrow(try ProjectManifest.validate(baseline))

        let mutations: [(String, (inout ProjectManifest) -> Void)] = [
            ("configuration", { $0.engineConfiguration.voxelSize = 0 }),
            ("frame timestamp", { $0.frames[0].sourceTimestamp = .nan }),
            ("frame path", { $0.frames[0].relativePath = "../outside.jpg" }),
            ("frame order", { $0.frames[1].index = $0.frames[0].index }),
            ("window transform count", { $0.completedWindows[0].alignmentRowMajor = [1] }),
            ("window transform finite", { $0.completedWindows[0].alignmentRowMajor[0] = .infinity }),
            ("window duration", { $0.completedWindows[0].durationSeconds = -1 }),
            ("window order", { $0.completedWindows[1].index = 0 }),
            ("window bounds", { $0.completedWindows[0].inferenceFrameStart = 2 }),
            ("artifact path", { $0.completedWindows[0].frameArtifacts[0].depthRelativePath = "Predictions/wrong.depth-f16" }),
            ("artifact duration", { $0.completedWindows[0].frameArtifacts[0].durationSeconds = -.infinity }),
            ("artifact order", { $0.completedWindows[0].frameArtifacts.swapAt(0, 1) }),
            ("artifact window", { $0.completedWindows[0].frameArtifacts[0].windowIndex = 9 }),
        ]

        for (name, mutate) in mutations {
            var candidate = baseline
            mutate(&candidate)
            XCTAssertThrowsError(try ProjectManifest.validate(candidate), "Accepted malformed \(name)")
        }
    }

    func testManifestRejectsProcessedCounterBeyondCumulativeQueuedAdmissions() {
        var manifest = ProjectManifest.fixtureV2(windowOverlap: 1)
        manifest.sessionState = SessionState(
            phase: .processing,
            queuedCount: 2,
            processedCount: 3
        )

        XCTAssertThrowsError(try ProjectManifest.validate(manifest)) {
            XCTAssertEqual($0 as? ProjectManifestError, .invalidSessionState)
        }
    }

    func testManifestRejectsLegacyV1AndNextWindowOverflow() throws {
        let legacy = Data(#"{"formatVersion":1}"#.utf8)
        XCTAssertThrowsError(try ProjectManifest.decode(legacy)) {
            XCTAssertEqual($0 as? ProjectManifestError, .unsupportedFormatVersion(1))
        }

        var manifest = ProjectManifest.fixtureV2(windowOverlap: 1)
        manifest.frames = [PersistedFrame(index: .max, sourceTimestamp: 1, relativePath: "Frames/4294967295.jpg")]
        manifest.completedWindows = [.fixture(index: .max, inferenceStart: .max, frames: [.max])]
        manifest.synchronizeSessionCounts()
        XCTAssertThrowsError(try manifest.resumeCheckpoint()) {
            XCTAssertEqual($0 as? ProjectManifestError, .checkpointWindowIndexOverflow)
        }
    }

    func testProtocolFailureDispositionFollowsRecoverableHeaderTable() {
        let header = WorkerCommandHeader(
            id: UUID(uuidString: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")!,
            projectID: projectID,
            protocolVersion: 1,
            type: "configure"
        )

        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(.zeroLengthPayload, recoverableHeader: nil),
            .closeWithoutResponse
        )
        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(.malformedEnvelope, recoverableHeader: nil),
            .asynchronousProtocolFaultThenClose
        )
        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(.invalidPayload("configure"), recoverableHeader: header),
            .commandErrorThenContinue(header)
        )

        let unsupported = WorkerCommandHeader(
            id: header.id,
            projectID: header.projectID,
            protocolVersion: 2,
            type: header.type
        )
        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(.unsupportedProtocolVersion(2), recoverableHeader: unsupported),
            .commandErrorThenClose(unsupported)
        )

        let invalidPayload = Data(#"{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","payload":{"bad":true},"projectId":"11111111-1111-1111-1111-111111111111","protocolVersion":1,"type":"configure"}"#.utf8)
        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(.invalidPayload("configure"), JSONPayload: invalidPayload),
            .commandErrorThenContinue(header)
        )
        XCTAssertEqual(
            WorkerProtocolFailureDisposition.classify(.malformedEnvelope, JSONPayload: Data("{".utf8)),
            .asynchronousProtocolFaultThenClose
        )
    }

    private func roundTrip(_ envelope: WorkerEnvelope) throws -> WorkerEnvelope {
        var decoder = LengthPrefixedJSONCodec.Decoder()
        return try XCTUnwrap(decoder.append(LengthPrefixedJSONCodec.encode(envelope)).only)
    }

    private func decodeOne(_ json: String) throws -> WorkerEnvelope {
        var framed = Data()
        let body = Data(json.utf8)
        var length = UInt32(body.count).bigEndian
        withUnsafeBytes(of: &length) { framed.append(contentsOf: $0) }
        framed.append(body)
        var decoder = LengthPrefixedJSONCodec.Decoder()
        guard let envelope = try decoder.append(framed).only else {
            throw WorkerProtocolError.malformedEnvelope
        }
        return envelope
    }

    private static let identity: [Double] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
}

private extension FrameArtifacts {
    static func fixture(frame: UInt32, window: UInt32) -> FrameArtifacts {
        FrameArtifacts(
            frameIndex: frame,
            windowIndex: window,
            depthRelativePath: String(format: "Predictions/%08u.depth-f16", frame),
            confidenceRelativePath: String(format: "Predictions/%08u.confidence-f16", frame),
            geometryRelativePath: String(format: "Predictions/%08u.geometry.json", frame),
            durationSeconds: 0.1
        )
    }
}

private extension WindowResult {
    static func fixture(
        window: UInt32,
        inferenceStart: UInt32,
        start: UInt32,
        end: UInt32
    ) -> WindowResult {
        WindowResult(
            windowIndex: window,
            inferenceFrameStart: inferenceStart,
            frameStart: start,
            frameEnd: end,
            pointChunkRelativePath: String(format: "Points/window-%08u.cpc", window),
            alignmentRowMajor: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            lastProcessedFrameIndex: end,
            inlierCount: 4,
            durationSeconds: 0.5
        )
    }
}

private extension ProjectManifest {
    static func fixtureV2(windowOverlap: UInt32) -> ProjectManifest {
        ProjectManifest(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            createdAt: Date(timeIntervalSinceReferenceDate: 1_000),
            updatedAt: Date(timeIntervalSinceReferenceDate: 2_000),
            engineConfiguration: EngineConfiguration(windowOverlap: windowOverlap)
        )
    }

    mutating func synchronizeSessionCounts() {
        let captured = UInt64(frames.count)
        let processed = UInt64(completedWindows.flatMap(\.frameArtifacts).count)
        sessionState = SessionState(
            phase: .processing,
            capturedCount: captured,
            queuedCount: max(processed, captured),
            processedCount: processed
        )
    }
}

private extension CompletedWindow {
    static func fixture(index: UInt32, inferenceStart: UInt32, frames: [UInt32]) -> CompletedWindow {
        CompletedWindow(
            index: index,
            inferenceFrameStart: inferenceStart,
            frameStart: frames.first!,
            frameEnd: frames.last!,
            pointChunkRelativePath: String(format: "Points/window-%08u.cpc", index),
            alignmentRowMajor: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            lastProcessedFrameIndex: frames.last!,
            inlierCount: 10,
            durationSeconds: 1,
            frameArtifacts: frames.map { .fixture(frame: $0, window: index) }
        )
    }
}

private extension Array {
    var only: Element? { count == 1 ? first : nil }
}

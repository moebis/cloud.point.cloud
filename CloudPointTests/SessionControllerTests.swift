import XCTest
@testable import CloudPoint

final class SessionControllerTests: XCTestCase {
    func testInterruptedRecordingResumesAtDurableSampleCursorWithoutFinishingPrefix() async throws {
        let package = try TemporaryProjectPackage.make()
        let first = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let second = try WorkspaceTestFiles.writeJPEG(frameIndex: 1, in: package.url)
        var manifest = ProjectManifest.fixture()
        manifest.recordingSource = RecordingSourceReference(
            bookmarkData: Data("recording".utf8),
            originalFilename: "recording.mov",
            fingerprint: RecordingSourceFingerprint(
                byteCount: 42,
                sha256: String(repeating: "b", count: 64)
            ),
            durationSeconds: 2,
            framesPerSecond: 1,
            expectedSampleCount: 2,
            nextSampleOrdinal: 1
        )
        manifest.frames = [first]
        manifest.sessionState = SessionState(
            phase: .importing,
            capturedCount: 1,
            queuedCount: 1
        )
        try manifest.writeAtomically(to: package.url)

        let importer = CursorRecordingImporter(frames: [second])
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: importer,
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        let finishCountBeforeResume = await engine.finishInputCount
        XCTAssertEqual(finishCountBeforeResume, 0)

        try await controller.importRecording(
            URL(filePath: "/recording.mov"),
            framesPerSecond: 1
        )
        _ = try await effects.next { $0.queuedCount == 2 }
        await controller.flush()

        let requestedOrdinals = await importer.requestedOrdinals()
        XCTAssertEqual(requestedOrdinals, [1])
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.recordingSource?.nextSampleOrdinal, 2)
        XCTAssertEqual(disk.frames.map(\.index), [0, 1])
        let finishCountAfterResume = await engine.finishInputCount
        XCTAssertEqual(finishCountAfterResume, 1)
    }

    func testInterruptedProjectRunsDurableTailToTerminalCompletion() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        var manifest = ProjectManifest.fixture()
        manifest.frames = [frame]
        manifest.sessionState = SessionState(
            phase: .processing,
            capturedCount: 1,
            queuedCount: 1
        )
        try manifest.writeAtomically(to: package.url)

        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { MockReconstructionEngine() },
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        let completed = try await effects.next { $0.phase == .completed }

        XCTAssertEqual(completed.capturedCount, 1)
        XCTAssertEqual(completed.queuedCount, 1)
        XCTAssertEqual(completed.processedCount, 1)
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.sessionState.phase, .completed)
        XCTAssertEqual(disk.completedWindows.count, 1)
        let appendedRanges = await effects.appendedRanges
        XCTAssertEqual(appendedRanges, [0...0])
    }

    func testFailedProjectRetriesDurableTailToTerminalCompletion() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        var manifest = ProjectManifest.fixture()
        manifest.frames = [frame]
        manifest.sessionState = SessionState(
            phase: .failed,
            capturedCount: 1,
            queuedCount: 1,
            failedCount: 1
        )
        try manifest.writeAtomically(to: package.url)

        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { MockReconstructionEngine() },
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        let completed = try await effects.next { $0.phase == .completed }

        XCTAssertEqual(completed.processedCount, 1)
        XCTAssertEqual(completed.failedCount, 1)
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.sessionState.phase, .completed)
        XCTAssertEqual(disk.completedWindows.count, 1)
    }

    func testImmediateMockEventsCommitWindowBeforeRendererAndReachCompletion() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { MockReconstructionEngine() },
                manifestStore: AtomicManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: [frame]),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                now: { Date(timeIntervalSinceReferenceDate: 9_000) },
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        let completed = try await effects.next { $0.phase == .completed }

        XCTAssertEqual(completed.processedCount, 1)
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.sessionState.phase, .completed)
        XCTAssertEqual(disk.completedWindows.count, 1)
        XCTAssertEqual(disk.completedWindows[0].frameArtifacts.map(\.frameIndex), [0])
        let log = await effects.eventLog
        let completedAdoption = try XCTUnwrap(log.firstIndex(of: "adopt-1"))
        let append = try XCTUnwrap(log.lastIndex(of: "append-0-0"))
        XCTAssertLessThan(completedAdoption, append)
    }

    func testPreparingSnapshotKeepsCancelAvailable() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()

        let snapshots = await effects.snapshots
        let preparing = try XCTUnwrap(snapshots.first { $0.phase == .preparing })
        XCTAssertTrue(preparing.capabilities.canCancel)
    }

    func testCancelPreemptsBlockedPreparationAndCommitsCancellation() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = BlockingPrepareEngine()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        let openTask = Task {
            do {
                try await controller.open()
                return false
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }
        await engine.waitUntilPrepareStarts()

        let cancelFinished = expectation(description: "cancel preempts engine preparation")
        Task {
            await controller.cancel()
            cancelFinished.fulfill()
        }
        let waitResult = await XCTWaiter().fulfillment(
            of: [cancelFinished],
            timeout: 1
        )
        if waitResult != .completed {
            await engine.forceReleasePreparation()
            await fulfillment(of: [cancelFinished], timeout: 1)
        }

        let openWasCancelled = await openTask.value
        let beginCount = await engine.beginCount
        let cancelCount = await engine.cancelCount
        XCTAssertEqual(waitResult, .completed)
        XCTAssertTrue(openWasCancelled)
        XCTAssertEqual(beginCount, 0)
        XCTAssertEqual(cancelCount, 1)
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.sessionState.phase, .cancelled)
        XCTAssertEqual(disk.sessionState.failedCount, 0)
        let cancelled = try await effects.next { $0.phase == .cancelled }
        XCTAssertNil(cancelled.errorText)
        await controller.close()
    }

    func testCancelCommitsBeforeSlowEngineControlCallReturns() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = BlockingPrepareEngine(holdCancelCompletion: true)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        let openTask = Task { try? await controller.open() }
        await engine.waitUntilPrepareStarts()
        let cancelTask = Task { await controller.cancel() }

        let cancelled: WorkspaceSnapshot
        do {
            cancelled = try await effects.next(timeout: .seconds(1)) {
                $0.phase == .cancelled
            }
        } catch {
            await engine.releaseCancelCompletion()
            await engine.forceReleasePreparation()
            await cancelTask.value
            _ = await openTask.result
            throw error
        }
        let controlCallIsStillWaiting = await engine.waitingForCancelCompletion

        XCTAssertEqual(cancelled.phase, .cancelled)
        XCTAssertTrue(controlCallIsStillWaiting)
        XCTAssertEqual(try ProjectManifest.load(from: package.url).sessionState.phase, .cancelled)

        await engine.releaseCancelCompletion()
        await cancelTask.value
        _ = await openTask.result
        await controller.close()
    }

    func testClosePreemptsBlockedPreparationWithoutFailingProject() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = BlockingPrepareEngine()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: [])
            )
        )
        let openTask = Task {
            do {
                try await controller.open()
                return false
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }
        await engine.waitUntilPrepareStarts()

        let closeFinished = expectation(description: "close preempts engine preparation")
        Task {
            await controller.close()
            closeFinished.fulfill()
        }
        let waitResult = await XCTWaiter().fulfillment(
            of: [closeFinished],
            timeout: 1
        )
        if waitResult != .completed {
            await engine.forceReleasePreparation()
            await fulfillment(of: [closeFinished], timeout: 1)
        }

        let openWasCancelled = await openTask.value
        let beginCount = await engine.beginCount
        let shutdownCount = await engine.shutdownCount
        XCTAssertEqual(waitResult, .completed)
        XCTAssertTrue(openWasCancelled)
        XCTAssertEqual(beginCount, 0)
        XCTAssertEqual(shutdownCount, 1)
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.sessionState.phase, .preparing)
        XCTAssertEqual(disk.sessionState.failedCount, 0)
        do {
            _ = try await controller.currentManifest()
            XCTFail("closed controllers must reject new commands")
        } catch {
            XCTAssertEqual(error as? SessionControllerError, .controllerClosed)
        }
    }

    func testFrameIsDurableBeforeEnqueueAndAdmissionIsSecondCommit() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let engine = HarnessEngine(packageURL: package.url)
        let factory = HarnessEngineFactory(engine: engine)
        let store = HarnessManifestStore()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: factory,
                store: store,
                importer: HarnessRecordingImporter(frames: [frame]),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        let snapshot = try await effects.next { $0.queuedCount == 1 }

        XCTAssertEqual(snapshot.capturedCount, 1)
        let enqueueSawDurableManifest = await engine.enqueueSawDurableManifest
        XCTAssertTrue(enqueueSawDurableManifest)
        let writes = await store.writes
        let capturedWrite = try XCTUnwrap(writes.first { $0.sessionState.capturedCount == 1 && $0.sessionState.queuedCount == 0 })
        let queuedWrite = try XCTUnwrap(writes.first { $0.sessionState.queuedCount == 1 })
        XCTAssertEqual(capturedWrite.frames, [frame])
        XCTAssertEqual(queuedWrite.frames, [frame])
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
    }

    func testEnqueueFailureRetainsDurableFrameAndLeavesQueuedCountUnchanged() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let engine = HarnessEngine(packageURL: package.url, failEnqueue: true)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: [frame]),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        let failed = try await effects.next { $0.phase == .failed }

        XCTAssertEqual(failed.capturedCount, 1)
        XCTAssertEqual(failed.queuedCount, 0)
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.phase, .failed)
        XCTAssertEqual(disk.sessionState.queuedCount, 0)
    }

    func testCompletedReopenRestoresChunksWithoutConstructingEngine() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let mock = MockReconstructionEngine()
        var manifest = ProjectManifest.fixture()
        manifest.frames = [frame]
        manifest.sessionState = SessionState(phase: .processing, capturedCount: 1, queuedCount: 1)
        try manifest.writeAtomically(to: package.url)
        try await mock.prepare(configuration: manifest.engineConfiguration)
        try await mock.begin(project: ProjectDescriptor(projectID: manifest.projectID, packageURL: package.url))
        try await mock.enqueue(frame)
        try await mock.finishInput()
        var artifacts: FrameArtifacts?
        var result: WindowResult?
        for try await event in mock.events() {
            if case let .frameCompleted(value) = event { artifacts = value }
            if case let .windowCompleted(value) = event { result = value }
        }
        let completedArtifacts = try XCTUnwrap(artifacts)
        let completedResult = try XCTUnwrap(result)
        manifest.completedWindows = [CompletedWindow(
            index: completedResult.windowIndex,
            inferenceFrameStart: completedResult.inferenceFrameStart,
            frameStart: completedResult.frameStart,
            frameEnd: completedResult.frameEnd,
            pointChunkRelativePath: completedResult.pointChunkRelativePath,
            alignmentRowMajor: completedResult.alignmentRowMajor,
            lastProcessedFrameIndex: completedResult.lastProcessedFrameIndex,
            inlierCount: completedResult.inlierCount,
            durationSeconds: completedResult.durationSeconds,
            frameArtifacts: [completedArtifacts]
        )]
        manifest.sessionState = SessionState(
            phase: .completed,
            capturedCount: 1,
            queuedCount: 1,
            processedCount: 1
        )
        try manifest.writeAtomically(to: package.url)

        let probeEngine = HarnessEngine(packageURL: package.url)
        let factory = HarnessEngineFactory(engine: probeEngine)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: factory,
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        let completed = try await effects.next { $0.phase == .completed }

        XCTAssertEqual(completed.processedCount, 1)
        XCTAssertEqual(factory.calls, 0)
        let appendedRanges = await effects.appendedRanges
        XCTAssertEqual(appendedRanges, [0...0])
    }

    func testUnknownReconstructionModeOpensReadOnlyWithoutConstructingLingBotEngine() async throws {
        let package = try TemporaryProjectPackage.make()
        let encoded = try ProjectManifest.encode(.fixture())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var plan = try XCTUnwrap(object["reconstructionPlan"] as? [String: Any])
        plan["modeID"] = "example.future.reconstruction.v9"
        plan["configuration"] = ["type": "future", "settings": ["quality": 9]]
        object["reconstructionPlan"] = plan
        let manifest = try ProjectManifest.decode(JSONSerialization.data(withJSONObject: object))
        try manifest.writeAtomically(to: package.url)

        let factory = HarnessEngineFactory(engine: HarnessEngine(packageURL: package.url))
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: factory,
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        let snapshot = try await effects.next { $0.setupText != nil }

        XCTAssertEqual(factory.calls, 0)
        XCTAssertEqual(snapshot.phase, .empty)
        XCTAssertEqual(
            snapshot.setupText,
            SessionControllerError.reconstructionModeUnavailable(
                "example.future.reconstruction.v9"
            ).localizedDescription
        )
    }

    func testRelocationCopiesGaussianArtifactsFromManifestEnumeration() async throws {
        let original = try TemporaryProjectPackage.make()
        let destination = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 4, in: original.url)
        let plyPath = "Outputs/Gaussians/frame-00000004.ply"
        let plyData = Data("synthetic-sharp-ply".utf8)
        try plyData.write(to: original.url.appending(path: plyPath))
        let manifest = ProjectManifest(
            reconstructionPlan: .sharp(SharpReconstructionConfiguration(inputFrameIndex: 4)),
            outputState: .gaussian(GaussianSceneOutput(
                sourceFrameIndex: 4,
                plyRelativePath: plyPath,
                gaussianCount: 1
            )),
            frames: [frame],
            sessionState: SessionState(
                phase: .completed,
                capturedCount: 1,
                queuedCount: 1,
                processedCount: 1
            )
        )
        try manifest.writeAtomically(to: original.url)

        let factory = HarnessEngineFactory(engine: HarnessEngine(packageURL: original.url))
        let controller = SessionController(
            manifest: manifest,
            packageURL: original.url,
            dependencies: .harness(
                engineFactory: factory,
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: HarnessEffects()
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.updatePackageURL(destination.url)
        await controller.flush()

        XCTAssertEqual(try Data(contentsOf: destination.url.appending(path: plyPath)), plyData)
        XCTAssertEqual(
            try ProjectManifest.load(from: destination.url).reconstructionPlan.modeID,
            .sharpGaussian
        )
        XCTAssertEqual(factory.calls, 0)
    }

    func testManifestWriteFailurePreventsEngineAdmission() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let engine = HarnessEngine(packageURL: package.url)
        let store = HarnessManifestStore(failingWrite: 4)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: store,
                importer: HarnessRecordingImporter(frames: [frame]),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        let failed = try await effects.next { $0.phase == .failed }
        let enqueued = await engine.enqueued

        XCTAssertEqual(failed.capturedCount, 0)
        XCTAssertEqual(failed.queuedCount, 0)
        XCTAssertEqual(enqueued, [])
        XCTAssertEqual(try ProjectManifest.load(from: package.url).frames, [])
    }

    func testDuplicateSourceFrameFailsWithoutReplacingDurableFrame() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: [frame, frame]),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        let failed = try await effects.next { $0.phase == .failed }
        let enqueued = await engine.enqueued

        XCTAssertEqual(failed.capturedCount, 1)
        XCTAssertEqual(failed.queuedCount, 1)
        XCTAssertEqual(enqueued.map(\.index), [0])
        XCTAssertEqual(try ProjectManifest.load(from: package.url).frames.map(\.index), [0])
    }

    func testResumeReplaysExactGappedSuffixAndAdmitsDurableTail() async throws {
        let package = try TemporaryProjectPackage.make()
        let frames = try [UInt32(2), 5, 9].map {
            try WorkspaceTestFiles.writeJPEG(frameIndex: $0, in: package.url)
        }
        let firstArtifact = try WorkspaceTestFiles.writeArtifacts(
            frameIndex: 2,
            windowIndex: 0,
            in: package.url
        )
        let secondArtifact = try WorkspaceTestFiles.writeArtifacts(
            frameIndex: 5,
            windowIndex: 0,
            in: package.url
        )
        try WorkspaceTestFiles.writePointChunk(
            windowIndex: 0,
            firstFrame: 2,
            lastFrame: 5,
            in: package.url
        )
        var configuration = EngineConfiguration()
        configuration.scaleFrames = 1
        configuration.windowSize = 2
        configuration.windowOverlap = 1
        var manifest = ProjectManifest.fixture()
        manifest.engineConfiguration = configuration
        manifest.frames = frames
        manifest.completedWindows = [CompletedWindow(
            index: 0,
            inferenceFrameStart: 2,
            frameStart: 2,
            frameEnd: 5,
            pointChunkRelativePath: WorkerArtifactPath.points(windowIndex: 0),
            alignmentRowMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ],
            lastProcessedFrameIndex: 5,
            inlierCount: 2,
            durationSeconds: 0,
            frameArtifacts: [firstArtifact, secondArtifact]
        )]
        manifest.sessionState = SessionState(
            phase: .processing,
            capturedCount: 3,
            queuedCount: 2,
            processedCount: 2
        )
        try manifest.writeAtomically(to: package.url)

        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        let enqueued = await engine.enqueued
        let begunProjects = await engine.begunProjects
        let appendedRanges = await effects.appendedRanges

        XCTAssertEqual(enqueued.map(\.index), [5, 9])
        XCTAssertEqual(begunProjects.first?.resumeCheckpoint, ResumeCheckpoint(
            lastCommittedFrameIndex: 5,
            replayFromFrameIndex: 5,
            nextWindowIndex: 1
        ))
        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.sessionState.capturedCount, 3)
        XCTAssertEqual(disk.sessionState.queuedCount, 3)
        XCTAssertEqual(disk.sessionState.processedCount, 2)
        XCTAssertEqual(appendedRanges, [2...5])
        let finishInputCount = await engine.finishInputCount
        XCTAssertEqual(finishInputCount, 1)
    }

    func testReplayOnlyResumeClosesFiniteInput() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 4, in: package.url)
        let artifact = try WorkspaceTestFiles.writeArtifacts(
            frameIndex: 4,
            windowIndex: 0,
            in: package.url
        )
        try WorkspaceTestFiles.writePointChunk(
            windowIndex: 0,
            firstFrame: 4,
            lastFrame: 4,
            in: package.url
        )
        var manifest = ProjectManifest.fixture()
        manifest.frames = [frame]
        manifest.completedWindows = [CompletedWindow(
            index: 0,
            inferenceFrameStart: 4,
            frameStart: 4,
            frameEnd: 4,
            pointChunkRelativePath: WorkerArtifactPath.points(windowIndex: 0),
            alignmentRowMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ],
            lastProcessedFrameIndex: 4,
            inlierCount: 1,
            durationSeconds: 0,
            frameArtifacts: [artifact]
        )]
        manifest.sessionState = SessionState(
            phase: .processing,
            capturedCount: 1,
            queuedCount: 1,
            processedCount: 1
        )
        try manifest.writeAtomically(to: package.url)

        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: manifest,
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()

        let enqueuedIndices = await engine.enqueued.map(\.index)
        XCTAssertEqual(enqueuedIndices, [4])
        let finishInputCount = await engine.finishInputCount
        XCTAssertEqual(finishInputCount, 1)
        XCTAssertEqual(try ProjectManifest.load(from: package.url).sessionState.phase, .processing)
    }

    func testSaveAsDuringCaptureRelocatesDurableWorkAndFinishesInput() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        try ProjectManifest.fixture().writeAtomically(to: relocatedPackage.url)
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: originalPackage.url)
        let cameraPayload = CameraPersistedFrame(
            lifecycleID: 73,
            sequence: 0,
            frame: frame
        )
        let camera = HarnessCameraInput(completion: CameraLifecycleCompletion(
            lifecycleID: 73,
            durablePersistedEventCount: 1,
            terminalFailure: nil,
            durablePersistedEvents: [cameraPayload]
        ))
        let originalEngine = HarnessEngine(packageURL: originalPackage.url)
        let relocatedEngine = HarnessEngine(packageURL: relocatedPackage.url)
        let sequence = HarnessEngineSequence([originalEngine, relocatedEngine])
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { try sequence.make() },
                cameraFactory: { _, _ in camera },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 2)
        camera.emit(.persisted(cameraPayload))
        _ = try await effects.next { $0.queuedCount == 1 }

        await controller.updatePackageURL(relocatedPackage.url)
        let relocated = try await effects.next {
            $0.phase == .processing && $0.queuedCount == 1
        }

        XCTAssertNil(relocated.errorText)
        XCTAssertEqual(sequence.calls, 2)
        XCTAssertEqual(try ProjectManifest.load(from: originalPackage.url).frames, [frame])
        XCTAssertEqual(try ProjectManifest.load(from: relocatedPackage.url).frames, [frame])
        XCTAssertNotNil(try Data(contentsOf: relocatedPackage.url.appending(path: frame.relativePath)))
        let originalBegunURL = await originalEngine.begunProjects.first?.packageURL
        let relocatedBegunURL = await relocatedEngine.begunProjects.first?.packageURL
        let relocatedEnqueuedIndices = await relocatedEngine.enqueued.map(\.index)
        let relocatedFinishInputCount = await relocatedEngine.finishInputCount
        let cameraState = await camera.state
        XCTAssertEqual(originalBegunURL?.standardizedFileURL, originalPackage.url.standardizedFileURL)
        XCTAssertEqual(relocatedBegunURL?.standardizedFileURL, relocatedPackage.url.standardizedFileURL)
        XCTAssertEqual(relocatedEnqueuedIndices, [0])
        XCTAssertEqual(relocatedFinishInputCount, 1)
        XCTAssertEqual(cameraState, .stopped)
    }

    func testSaveAsReconcilesPersistedCameraTailBeforeRelocation() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: originalPackage.url)
        let persistedEvent = CameraFrameSourceEvent.persisted(CameraPersistedFrame(
            lifecycleID: 91,
            sequence: 0,
            frame: frame
        ))
        let camera = HarnessCameraInput(
            completion: CameraLifecycleCompletion(
                lifecycleID: 91,
                durablePersistedEventCount: 1,
                terminalFailure: nil,
                durablePersistedEvents: [CameraPersistedFrame(
                    lifecycleID: 91,
                    sequence: 0,
                    frame: frame
                )]
            ),
            terminalEvents: [persistedEvent]
        )
        let originalEngine = HarnessEngine(packageURL: originalPackage.url)
        let relocatedEngine = HarnessEngine(packageURL: relocatedPackage.url)
        let sequence = HarnessEngineSequence([originalEngine, relocatedEngine])
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { try sequence.make() },
                cameraFactory: { _, _ in camera },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 2)
        await controller.updatePackageURL(relocatedPackage.url)
        _ = try await effects.next { $0.phase == .processing && $0.queuedCount == 1 }

        let disk = try ProjectManifest.load(from: relocatedPackage.url)
        let relocatedEnqueuedIndices = await relocatedEngine.enqueued.map(\.index)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
        XCTAssertEqual(relocatedEnqueuedIndices, [0])
    }

    func testCloseReconcilesPersistedCameraTailBeforeClosing() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let persistedEvent = CameraFrameSourceEvent.persisted(CameraPersistedFrame(
            lifecycleID: 92,
            sequence: 0,
            frame: frame
        ))
        let camera = HarnessCameraInput(
            completion: CameraLifecycleCompletion(
                lifecycleID: 92,
                durablePersistedEventCount: 1,
                terminalFailure: nil,
                durablePersistedEvents: [CameraPersistedFrame(
                    lifecycleID: 92,
                    sequence: 0,
                    frame: frame
                )]
            ),
            terminalEvents: [persistedEvent]
        )
        let engine = HarnessEngine(packageURL: package.url)
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: HarnessEffects(),
                cameraFactory: { _, _ in camera }
            )
        )

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 2)
        await controller.close()

        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.phase, .processing)
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
    }

    func testSaveAsWaitsForPersistedRecordingTailBeforeRelocation() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: originalPackage.url)
        let importer = BlockingRecordingImporter(frame: frame)
        let originalEngine = HarnessEngine(packageURL: originalPackage.url)
        let relocatedEngine = HarnessEngine(packageURL: relocatedPackage.url)
        let sequence = HarnessEngineSequence([originalEngine, relocatedEngine])
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { try sequence.make() },
                manifestStore: HarnessManifestStore(),
                recordingImporter: importer,
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        await importer.waitUntilStarted()
        await controller.updatePackageURL(relocatedPackage.url)
        await importer.releasePersistedFrame()
        _ = try await effects.next { $0.phase == .processing && $0.queuedCount == 1 }
        await controller.flush()

        let disk = try ProjectManifest.load(from: relocatedPackage.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
    }

    func testSaveAsAfterRecordingFailureDrainsStaleTaskBeforeRelocation() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: originalPackage.url)
        let importer = BlockingRecordingImporter(
            frame: frame,
            throwCancellationAfterFrame: false
        )
        let engine = HarnessEngine(packageURL: originalPackage.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: importer,
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        await importer.waitUntilStarted()
        await engine.fail(WorkspaceHarnessError.injectedEnqueueFailure)
        _ = try await effects.next { $0.phase == .failed }

        await controller.updatePackageURL(relocatedPackage.url)
        let pendingRelocation = try await effects.next {
            $0.phase == .failed
                && $0.setupText == "Finishing the current frame before moving this project"
        }
        await importer.releasePersistedFrame()
        _ = try await effects.next {
            $0.phase == .failed
                && $0.setupText == nil
                && $0.revision > pendingRelocation.revision
        }

        let disk = try ProjectManifest.load(from: relocatedPackage.url)
        XCTAssertEqual(disk.sessionState.phase, .failed)
        XCTAssertTrue(disk.frames.isEmpty)
    }

    func testRelocationClaimsCancelRegisteredDuringCameraDrain() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        let camera = BlockingStopCameraInput(lifecycleID: 94)
        let engine = ShutdownRequiresCancelEngine()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                cameraFactory: { _, _ in camera },
                manifestStore: HarnessManifestStore(),
                recordingImporter: HarnessRecordingImporter(frames: []),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 2)
        let relocationTask = Task { await controller.updatePackageURL(relocatedPackage.url) }
        await camera.waitUntilStopStarts()

        let cancelEntered = expectation(description: "cancel entered")
        let cancelFinished = expectation(description: "cancel completed")
        let cancelTask = Task {
            cancelEntered.fulfill()
            await controller.cancel()
            cancelFinished.fulfill()
        }
        await fulfillment(of: [cancelEntered], timeout: 1)
        try await Task.sleep(for: .milliseconds(20))
        await camera.releaseStop()

        let result = await XCTWaiter().fulfillment(of: [cancelFinished], timeout: 1)
        if result != .completed {
            await engine.forceReleaseShutdown()
        }
        await cancelTask.value
        await relocationTask.value

        let cancelCount = await engine.cancelCount
        XCTAssertEqual(result, .completed)
        XCTAssertGreaterThanOrEqual(cancelCount, 1)
        let disk = try ProjectManifest.load(from: relocatedPackage.url)
        XCTAssertEqual(disk.sessionState.phase, .cancelled)
    }

    func testCloseWaitsForPersistedRecordingTailBeforeClosing() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let importer = BlockingRecordingImporter(frame: frame)
        let engine = HarnessEngine(packageURL: package.url)
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: importer,
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener()
            )
        )

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        await importer.waitUntilStarted()
        let closeTask = Task { await controller.close() }
        try await Task.sleep(for: .milliseconds(20))
        await importer.releasePersistedFrame()
        await closeTask.value

        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.phase, .processing)
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
    }

    func testCancelAfterSaveAsPreservesPersistedRecordingTailInDestination() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: originalPackage.url)
        let importer = BlockingRecordingImporter(frame: frame)
        let engine = HarnessEngine(packageURL: originalPackage.url)
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { engine },
                manifestStore: HarnessManifestStore(),
                recordingImporter: importer,
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener()
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        await importer.waitUntilStarted()
        await controller.updatePackageURL(relocatedPackage.url)
        let cancelTask = Task { await controller.cancel() }
        try await Task.sleep(for: .milliseconds(20))
        await importer.releasePersistedFrame()
        await cancelTask.value

        let disk = try ProjectManifest.load(from: relocatedPackage.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.phase, .cancelled)
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
    }

    func testCancelReconcilesPersistedCameraTailBeforeCancelling() async throws {
        let package = try TemporaryProjectPackage.make()
        let frame = try WorkspaceTestFiles.writeJPEG(frameIndex: 0, in: package.url)
        let payload = CameraPersistedFrame(
            lifecycleID: 93,
            sequence: 0,
            frame: frame
        )
        let camera = HarnessCameraInput(
            completion: CameraLifecycleCompletion(
                lifecycleID: 93,
                durablePersistedEventCount: 1,
                terminalFailure: nil,
                durablePersistedEvents: [payload]
            ),
            terminalEvents: [.persisted(payload)]
        )
        let engine = HarnessEngine(packageURL: package.url)
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: HarnessEffects(),
                cameraFactory: { _, _ in camera }
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 2)
        await controller.cancel()

        let disk = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(disk.frames, [frame])
        XCTAssertEqual(disk.sessionState.phase, .cancelled)
        XCTAssertEqual(disk.sessionState.capturedCount, 1)
        XCTAssertEqual(disk.sessionState.queuedCount, 1)
    }

    func testSaveAsWhileReadyRebindsEngineBeforeNewInput() async throws {
        let originalPackage = try TemporaryProjectPackage.make()
        let relocatedPackage = try TemporaryProjectPackage.make()
        let relocatedFrame = try WorkspaceTestFiles.writeJPEG(
            frameIndex: 0,
            in: relocatedPackage.url
        )
        let originalEngine = HarnessEngine(packageURL: originalPackage.url)
        let relocatedEngine = HarnessEngine(packageURL: relocatedPackage.url)
        let sequence = HarnessEngineSequence([originalEngine, relocatedEngine])
        let store = HarnessManifestStore()
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: originalPackage.url,
            dependencies: SessionControllerDependencies(
                engineFactory: { try sequence.make() },
                manifestStore: store,
                recordingImporter: HarnessRecordingImporter(frames: [relocatedFrame]),
                jpegValidator: ProductionJPEGValidator(),
                pointChunkOpener: ProductionPointChunkOpener(),
                effects: SessionControllerEffects(
                    adoptManifest: { await effects.adopt($0) },
                    appendPointChunk: { await effects.append($0) },
                    publishSnapshot: { await effects.publish($0) }
                )
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try ProjectManifest.load(from: originalPackage.url).writeAtomically(
            to: relocatedPackage.url
        )

        await controller.updatePackageURL(relocatedPackage.url)
        await controller.flush()
        XCTAssertEqual(sequence.calls, 2)
        let originalBegunURL = await originalEngine.begunProjects.first?.packageURL
        let relocatedBegunURL = await relocatedEngine.begunProjects.first?.packageURL
        XCTAssertEqual(originalBegunURL?.standardizedFileURL, originalPackage.url.standardizedFileURL)
        XCTAssertEqual(relocatedBegunURL?.standardizedFileURL, relocatedPackage.url.standardizedFileURL)

        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        _ = try await effects.next { $0.queuedCount == 1 }

        XCTAssertTrue(try ProjectManifest.load(from: originalPackage.url).frames.isEmpty)
        XCTAssertEqual(try ProjectManifest.load(from: relocatedPackage.url).frames, [relocatedFrame])
    }

    func testSessionCompletionCounterMismatchFailsDurably() async throws {
        let package = try TemporaryProjectPackage.make()
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        await engine.emit(.sessionCompleted(processedFrames: 1, windowCount: 0, durationSeconds: 0))
        let failed = try await effects.next { $0.phase == .failed }

        XCTAssertEqual(failed.errorText, SessionControllerError.completionCounterMismatch.localizedDescription)
        XCTAssertEqual(try ProjectManifest.load(from: package.url).sessionState.phase, .failed)
    }

    func testSessionCompletionCannotStrandAdmittedFrame() async throws {
        let package = try TemporaryProjectPackage.make()
        let frames = try [UInt32(0), 1].map {
            try WorkspaceTestFiles.writeJPEG(frameIndex: $0, in: package.url)
        }
        let artifact = try WorkspaceTestFiles.writeArtifacts(
            frameIndex: 0,
            windowIndex: 0,
            in: package.url
        )
        try WorkspaceTestFiles.writePointChunk(
            windowIndex: 0,
            firstFrame: 0,
            lastFrame: 0,
            in: package.url
        )
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: frames),
                effects: effects
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.importRecording(URL(filePath: "/recording.mov"), framesPerSecond: 1)
        _ = try await effects.next { $0.queuedCount == 2 }
        await engine.emit(.frameStarted(frameIndex: 0, windowIndex: 0))
        await engine.emit(.frameCompleted(artifact))
        await engine.emit(.windowCompleted(WindowResult(
            windowIndex: 0,
            inferenceFrameStart: 0,
            frameStart: 0,
            frameEnd: 0,
            pointChunkRelativePath: WorkerArtifactPath.points(windowIndex: 0),
            alignmentRowMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ],
            lastProcessedFrameIndex: 0,
            inlierCount: 1,
            durationSeconds: 0
        )))
        _ = try await effects.next { $0.processedCount == 1 }

        await engine.emit(.sessionCompleted(
            processedFrames: 1,
            windowCount: 1,
            durationSeconds: 0
        ))
        let failed = try await effects.next { $0.phase == .failed }

        XCTAssertEqual(failed.queuedCount, 2)
        XCTAssertEqual(failed.processedCount, 1)
        XCTAssertEqual(failed.errorText, SessionControllerError.completionCounterMismatch.localizedDescription)
    }

    func testCameraStopWaitsForAllSeventeenDurableEventsBeforeFinishInput() async throws {
        let package = try TemporaryProjectPackage.make()
        let frames = try (0..<17).map {
            try WorkspaceTestFiles.writeJPEG(frameIndex: UInt32($0), in: package.url)
        }
        let cameraPayloads = frames.enumerated().map { sequence, frame in
            CameraPersistedFrame(
                lifecycleID: 41,
                sequence: UInt64(sequence),
                frame: frame
            )
        }
        let camera = HarnessCameraInput(completion: CameraLifecycleCompletion(
            lifecycleID: 41,
            durablePersistedEventCount: 17,
            terminalFailure: nil,
            durablePersistedEvents: cameraPayloads
        ))
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects,
                cameraFactory: { _, _ in camera }
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 5)
        try await controller.pause()
        for payload in cameraPayloads {
            camera.emit(.persisted(payload))
        }
        try await controller.stopCamera()

        _ = try await effects.next { $0.queuedCount == 17 }
        await controller.flush()
        let finishInputCount = await engine.finishInputCount
        let finishManifest = await engine.finishInputManifest
        XCTAssertEqual(finishInputCount, 1)
        XCTAssertEqual(finishManifest?.frames.map(\.index), (0..<17).map(UInt32.init))
        XCTAssertEqual(finishManifest?.sessionState.capturedCount, 17)
        XCTAssertEqual(finishManifest?.sessionState.queuedCount, 17)
        XCTAssertEqual(finishManifest?.sessionState.phase, .paused)

        try await controller.resume()
        XCTAssertEqual(
            try ProjectManifest.load(from: package.url).sessionState.phase,
            .processing
        )
    }

    func testRuntimeCameraFailureFailsProjectWithoutManualStop() async throws {
        let package = try TemporaryProjectPackage.make()
        let camera = HarnessCameraInput(completion: CameraLifecycleCompletion(
            lifecycleID: 91,
            durablePersistedEventCount: 0,
            terminalFailure: .cameraDisconnected
        ))
        let engine = HarnessEngine(packageURL: package.url)
        let effects = HarnessEffects()
        let controller = SessionController(
            manifest: .fixture(),
            packageURL: package.url,
            dependencies: .harness(
                engineFactory: HarnessEngineFactory(engine: engine),
                store: HarnessManifestStore(),
                importer: HarnessRecordingImporter(frames: []),
                effects: effects,
                cameraFactory: { _, _ in camera }
            )
        )
        defer { Task { await controller.close() } }

        try await controller.open()
        await controller.flush()
        try await controller.useCamera(deviceID: "camera-a", sampleRate: 2)
        camera.emit(.failed(CameraFrameSourceFailureEvent(
            lifecycleID: 91,
            sequence: 0,
            failure: .cameraDisconnected
        )))

        let failed = try await effects.next { $0.phase == .failed }
        XCTAssertEqual(
            failed.errorText,
            SessionControllerError.cameraFailure(.cameraDisconnected).localizedDescription
        )
        XCTAssertFalse(failed.isCapturing)
        XCTAssertEqual(try ProjectManifest.load(from: package.url).sessionState.phase, .failed)
    }
}

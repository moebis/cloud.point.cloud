import XCTest
@testable import CloudPoint

final class ProjectManifestTests: XCTestCase {
    func testManifestEncodesProjectIDAsLowercaseCanonicalUUIDForWorker() throws {
        var manifest = ProjectManifest.fixture()
        manifest.projectID = UUID(uuidString: "ABCDEF01-2345-4678-9ABC-DEF012345678")!

        let data = try ProjectManifest.encode(manifest)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(object["projectID"] as? String, "abcdef01-2345-4678-9abc-def012345678")
        XCTAssertEqual(try ProjectManifest.decode(data).projectID, manifest.projectID)
    }

    func testRecordingSourceCursorRoundTripsAndRejectsOutOfRangeProgress() throws {
        var manifest = ProjectManifest.fixture()
        manifest.recordingSource = RecordingSourceReference(
            bookmarkData: Data("bookmark".utf8),
            originalFilename: "Atrium.mov",
            fingerprint: RecordingSourceFingerprint(
                byteCount: 42,
                sha256: String(repeating: "a", count: 64)
            ),
            durationSeconds: 5,
            framesPerSecond: 2,
            expectedSampleCount: 10,
            nextSampleOrdinal: 4
        )
        manifest.frames = (0..<4).map {
            PersistedFrame(
                index: UInt32($0),
                sourceTimestamp: Double($0) / 2,
                relativePath: String(format: "Frames/%08u.jpg", $0)
            )
        }
        manifest.sessionState = SessionState(phase: .importing, capturedCount: 4)

        let decoded = try ProjectManifest.decode(ProjectManifest.encode(manifest))

        XCTAssertEqual(decoded.recordingSource, manifest.recordingSource)

        manifest.recordingSource?.nextSampleOrdinal = 11
        XCTAssertThrowsError(try ProjectManifest.validate(manifest)) {
            XCTAssertEqual($0 as? ProjectManifestError, .invalidRecordingSource)
        }
    }

    func testCameraDisplayMirroringRoundTripsAndDefaultsOnForLegacyProjects() throws {
        var manifest = ProjectManifest.fixture()
        manifest.cameraSource = CameraSourceReference(
            deviceID: "studio-display",
            deviceName: "Studio Display Camera",
            mirrorDisplay: false
        )

        let encoded = try ProjectManifest.encode(manifest)
        XCTAssertFalse(try ProjectManifest.decode(encoded).cameraSource?.mirrorDisplay ?? true)

        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        var legacyCamera = try XCTUnwrap(object["cameraSource"] as? [String: Any])
        legacyCamera.removeValue(forKey: "mirrorDisplay")
        object["cameraSource"] = legacyCamera
        let legacy = try JSONSerialization.data(withJSONObject: object)

        XCTAssertTrue(try XCTUnwrap(ProjectManifest.decode(legacy).cameraSource).mirrorDisplay)
    }

    func testManifestEnforcesDurableAndCommittedCounterOwnership() throws {
        var capturedMismatch = ProjectManifest.fixture()
        capturedMismatch.frames = [.fixture(index: 0)]
        XCTAssertThrowsError(try ProjectManifest.validate(capturedMismatch)) {
            XCTAssertEqual($0 as? ProjectManifestError, .invalidSessionState)
        }

        var processedMismatch = ProjectManifest.fixture()
        processedMismatch.frames = [.fixture(index: 0)]
        processedMismatch.completedWindows = [.fixture(index: 0)]
        processedMismatch.sessionState = SessionState(
            phase: .processing,
            capturedCount: 1,
            queuedCount: 1,
            processedCount: 0
        )
        XCTAssertThrowsError(try ProjectManifest.validate(processedMismatch)) {
            XCTAssertEqual($0 as? ProjectManifestError, .invalidSessionState)
        }

        processedMismatch.sessionState.processedCount = 1
        XCTAssertNoThrow(try ProjectManifest.validate(processedMismatch))
    }

    func testFractionalManifestDatesRoundTripExactlyAsRFC3339Strings() throws {
        var manifest = ProjectManifest.fixture()
        manifest.createdAt = Date(timeIntervalSince1970: 1_700_000_000.123_456_7)
        manifest.updatedAt = Date(timeIntervalSince1970: 1_700_000_001.987_654_2)

        let data = try ProjectManifest.encode(manifest)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let createdAt = try XCTUnwrap(object["createdAt"] as? String)
        let updatedAt = try XCTUnwrap(object["updatedAt"] as? String)
        let decoded = try ProjectManifest.decode(data)

        XCTAssertEqual(createdAt, "2023-11-14T22:13:20.1234567Z")
        XCTAssertEqual(updatedAt, "2023-11-14T22:13:21.9876542Z")
        XCTAssertEqual(decoded.createdAt, manifest.createdAt)
        XCTAssertEqual(decoded.updatedAt, manifest.updatedAt)
        XCTAssertEqual(decoded, manifest)
    }

    func testNegativeReferenceIntervalDateRoundTripsExactly() throws {
        var manifest = ProjectManifest.fixture()
        manifest.createdAt = Date(timeIntervalSinceReferenceDate: -0.1)

        let data = try ProjectManifest.encode(manifest)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let createdAt = try XCTUnwrap(object["createdAt"] as? String)
        let decoded = try ProjectManifest.decode(data)

        XCTAssertEqual(createdAt, "2000-12-31T23:59:59.9Z")
        XCTAssertEqual(
            decoded.createdAt.timeIntervalSinceReferenceDate.bitPattern,
            manifest.createdAt.timeIntervalSinceReferenceDate.bitPattern
        )
        XCTAssertEqual(decoded.createdAt, manifest.createdAt)
    }

    func testManifestDateCodecRoundTripsRepresentativeDoubleBitPatterns() throws {
        let referenceIntervals: [Double] = [
            -10_000_000_000.123_457,
            -1_234.567_89,
            -1.1,
            -1,
            -0.999_999_999_999_999_9,
            -0.5,
            -0.1,
            -Double.leastNonzeroMagnitude,
            0,
            Double.leastNonzeroMagnitude,
            0.1,
            0.5,
            0.999_999_999_999_999_9,
            1,
            1.1,
            1_234.567_89,
            10_000_000_000.123_457,
        ]

        for interval in referenceIntervals {
            var manifest = ProjectManifest.fixture()
            manifest.createdAt = Date(timeIntervalSinceReferenceDate: interval)

            let decoded = try ProjectManifest.decode(ProjectManifest.encode(manifest))

            XCTAssertEqual(
                decoded.createdAt.timeIntervalSinceReferenceDate.bitPattern,
                interval.bitPattern,
                "Did not preserve reference interval \(interval)"
            )
        }
    }

    func testManifestDateDecoderAcceptsWholeSecondRFC3339Strings() throws {
        let manifest = ProjectManifest.fixture()
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: ProjectManifest.encode(manifest)) as? [String: Any]
        )
        object["createdAt"] = "2001-01-01T00:16:40Z"
        object["updatedAt"] = "2001-01-01T00:33:20Z"

        let decoded = try ProjectManifest.decode(
            JSONSerialization.data(withJSONObject: object)
        )

        XCTAssertEqual(decoded.createdAt, manifest.createdAt)
        XCTAssertEqual(decoded.updatedAt, manifest.updatedAt)
    }

    func testLoadLeavesWorkerOwnedPartialArtifactsUntouched() throws {
        let package = try TemporaryProjectPackage.make()
        var manifest = ProjectManifest.fixture()
        manifest.frames = [.fixture(index: 0)]
        manifest.completedWindows = [.fixture(index: 0)]
        manifest.sessionState = SessionState(
            phase: .completed,
            capturedCount: 1,
            queuedCount: 1,
            processedCount: 1
        )

        try manifest.writeAtomically(to: package.url)
        FileManager.default.createFile(
            atPath: package.url.appending(path: "Points/window-1.cpc.partial").path,
            contents: Data("partial".utf8)
        )

        let loaded = try ProjectManifest.load(from: package.url)

        XCTAssertEqual(loaded.completedWindows.map(\.index), [0])
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: package.url.appending(path: "Points/window-1.cpc.partial").path
            )
        )
    }

    func testAtomicWriteReplacesAnExistingManifest() throws {
        let package = try TemporaryProjectPackage.make()
        var manifest = ProjectManifest.fixture()
        try manifest.writeAtomically(to: package.url)

        manifest.frames = [PersistedFrame(index: 3, sourceTimestamp: 0.6, relativePath: "Frames/00000003.jpg")]
        manifest.sessionState = SessionState(phase: .processing, capturedCount: 1)
        try manifest.writeAtomically(to: package.url)

        let loaded = try ProjectManifest.load(from: package.url)
        XCTAssertEqual(loaded.frames, manifest.frames)
        XCTAssertTrue(FileManager.default.fileExists(atPath: package.url.appending(path: "Manifest.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.url.appending(path: "Manifest.json.partial").path))
    }

    func testConcurrentAtomicWritesUseIndependentStagingFiles() async throws {
        let package = try TemporaryProjectPackage.make()
        let packageURL = package.url
        let manifests = (0..<32).map { ordinal in
            var manifest = ProjectManifest.fixture()
            manifest.projectID = UUID()
            manifest.updatedAt = Date(timeIntervalSinceReferenceDate: Double(ordinal))
            return manifest
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for manifest in manifests {
                group.addTask {
                    try manifest.writeAtomically(to: packageURL)
                }
            }
            try await group.waitForAll()
        }

        let loaded = try ProjectManifest.load(from: packageURL)
        XCTAssertTrue(manifests.contains(loaded))
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: packageURL.path)
                .filter { $0.hasSuffix(".partial") },
            []
        )
    }

    @MainActor
    func testDocumentDeclaresTheCloudPointProjectType() {
        XCTAssertEqual(CloudPointDocument.readableContentTypes, [.cloudPointProject])
    }

    @MainActor
    func testDocumentPackageWrapperCreatesTheRequiredDirectories() throws {
        let wrapper = try CloudPointDocument.packageWrapper(for: .fixture())

        XCTAssertEqual(
            Set(wrapper.fileWrappers?.keys.map { $0 } ?? []),
            ["Manifest.json", "Frames", "Predictions", "Points", "Logs"]
        )
    }

    @MainActor
    func testDocumentAdoptsControllerManifestForLaterSnapshotsAndSaves() throws {
        let document = CloudPointDocument()
        var committed = ProjectManifest.fixture()
        committed.frames = [.fixture(index: 0)]
        committed.sessionState = SessionState(
            phase: .processing,
            capturedCount: 1,
            queuedCount: 1
        )

        document.adoptCommittedManifest(committed)
        let snapshot = try document.snapshot(contentType: .cloudPointProject)
        let wrapper = try CloudPointDocument.packageWrapper(for: snapshot)
        let reloaded = try CloudPointDocument.loadManifest(from: wrapper)

        XCTAssertEqual(document.manifest, committed)
        XCTAssertEqual(snapshot, committed)
        XCTAssertEqual(reloaded, committed)
    }

    func testDocumentSnapshotIsSafeOnSerializationExecutor() async throws {
        let document = CloudPointDocument()
        var committed = ProjectManifest.fixture()
        committed.updatedAt = Date(timeIntervalSinceReferenceDate: 42)
        document.adoptCommittedManifest(committed)

        let snapshot = try await Task.detached {
            try document.snapshot(contentType: .cloudPointProject)
        }.value

        XCTAssertEqual(snapshot, committed)
    }

    @MainActor
    func testDocumentSaveRetainsExistingPayloadFiles() throws {
        let package = try TemporaryProjectPackage.make()
        let original = ProjectManifest.fixture()
        try original.writeAtomically(to: package.url)
        let payloads = [
            "Frames/existing.jpg": Data("frame".utf8),
            "Predictions/existing.depth-f16": Data("depth".utf8),
            "Points/window-0.cpc": Data("points".utf8),
            "Logs/worker.log": Data("log".utf8),
        ]

        for (relativePath, contents) in payloads {
            XCTAssertTrue(
                FileManager.default.createFile(
                    atPath: package.url.appending(path: relativePath).path,
                    contents: contents
                )
            )
        }

        var updated = original
        updated.frames = [PersistedFrame(index: 1, sourceTimestamp: 0.2, relativePath: "Frames/existing.jpg")]
        updated.sessionState = SessionState(phase: .processing, capturedCount: 1)
        let existingPackage = try FileWrapper(url: package.url, options: .immediate)
        let wrapper = try CloudPointDocument.packageWrapper(for: updated, preserving: existingPackage)
        try wrapper.write(
            to: package.url,
            options: FileWrapper.WritingOptions.atomic,
            originalContentsURL: package.url
        )

        XCTAssertEqual(try ProjectManifest.load(from: package.url).frames, updated.frames)
        for (relativePath, contents) in payloads {
            XCTAssertEqual(try Data(contentsOf: package.url.appending(path: relativePath)), contents)
        }
    }

    func testLoadRejectsAnUnsupportedFormatVersion() throws {
        let package = try TemporaryProjectPackage.make()
        let unsupported = try manifestData(formatVersion: 1)
        try unsupported.write(to: package.url.appending(path: "Manifest.json"))

        XCTAssertThrowsError(try ProjectManifest.load(from: package.url)) { error in
            XCTAssertEqual(error as? ProjectManifestError, .unsupportedFormatVersion(1))
        }
    }

    @MainActor
    func testDocumentReadRejectsAnUnsupportedFormatVersion() throws {
        let packageWrapper = FileWrapper(directoryWithFileWrappers: [
            "Manifest.json": FileWrapper(regularFileWithContents: try manifestData(formatVersion: 1)),
        ])

        XCTAssertThrowsError(try CloudPointDocument.loadManifest(from: packageWrapper)) { error in
            XCTAssertEqual(error as? ProjectManifestError, .unsupportedFormatVersion(1))
        }
    }

    func testAtomicWriteRejectsAnUnsupportedFormatVersion() throws {
        let package = try TemporaryProjectPackage.make()
        var unsupported = ProjectManifest.fixture()
        unsupported.formatVersion = 1

        XCTAssertThrowsError(try unsupported.writeAtomically(to: package.url)) { error in
            XCTAssertEqual(error as? ProjectManifestError, .unsupportedFormatVersion(1))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.url.appending(path: "Manifest.json").path))
    }

    func testApplicationExportsTheCloudPointPackageType() {
        let declarations = Bundle.main.object(forInfoDictionaryKey: "UTExportedTypeDeclarations") as? [[String: Any]]
        let identifiers = declarations?.compactMap { $0["UTTypeIdentifier"] as? String } ?? []

        XCTAssertTrue(identifiers.contains("cloud.point.cloud.project"))
    }

    func testApplicationRegistersMovieTypesOnlyAtAlternateRank() throws {
        let declarations = try XCTUnwrap(
            Bundle.main.object(forInfoDictionaryKey: "CFBundleDocumentTypes") as? [[String: Any]]
        )
        let movieDeclaration = try XCTUnwrap(declarations.first { declaration in
            let identifiers = declaration["LSItemContentTypes"] as? [String] ?? []
            return identifiers.contains("com.apple.quicktime-movie")
        })

        XCTAssertEqual(movieDeclaration["LSHandlerRank"] as? String, "Alternate")
        XCTAssertEqual(
            Set(movieDeclaration["LSItemContentTypes"] as? [String] ?? []),
            ["com.apple.quicktime-movie", "public.mpeg-4", "com.apple.m4v-video"]
        )
        XCTAssertEqual(CloudPointDocument.readableContentTypes, [.cloudPointProject])
    }

    private func manifestData(formatVersion: Int) throws -> Data {
        let valid = try ProjectManifest.encode(.fixture())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: valid) as? [String: Any])
        object["formatVersion"] = formatVersion
        return try JSONSerialization.data(withJSONObject: object)
    }
}

extension ProjectManifest {
    static func fixture() -> ProjectManifest {
        ProjectManifest(
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            createdAt: Date(timeIntervalSinceReferenceDate: 1_000),
            updatedAt: Date(timeIntervalSinceReferenceDate: 2_000),
            frames: [],
            completedWindows: [],
            sessionState: .empty
        )
    }
}

extension CompletedWindow {
    static func fixture(index: UInt32) -> CompletedWindow {
        CompletedWindow(
            index: index,
            inferenceFrameStart: index,
            frameStart: index,
            frameEnd: index,
            pointChunkRelativePath: String(format: "Points/window-%08u.cpc", index),
            alignmentRowMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ],
            lastProcessedFrameIndex: index,
            inlierCount: 1,
            durationSeconds: 0,
            frameArtifacts: [.fixture(frameIndex: index, windowIndex: index)]
        )
    }
}

private extension PersistedFrame {
    static func fixture(index: UInt32) -> PersistedFrame {
        PersistedFrame(
            index: index,
            sourceTimestamp: Double(index),
            relativePath: String(format: "Frames/%08u.jpg", index)
        )
    }
}

private extension FrameArtifacts {
    static func fixture(frameIndex: UInt32, windowIndex: UInt32) -> FrameArtifacts {
        FrameArtifacts(
            frameIndex: frameIndex,
            windowIndex: windowIndex,
            depthRelativePath: WorkerArtifactPath.depth(frameIndex: frameIndex),
            confidenceRelativePath: WorkerArtifactPath.confidence(frameIndex: frameIndex),
            geometryRelativePath: WorkerArtifactPath.geometry(frameIndex: frameIndex),
            durationSeconds: 0
        )
    }
}

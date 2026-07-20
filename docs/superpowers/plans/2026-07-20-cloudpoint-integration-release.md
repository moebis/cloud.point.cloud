# CloudPoint 0.1 Integration and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the native CloudPoint foundation with the real Lingbot MLX worker and finish a locally testable CloudPoint 0.1 app that can set up its model, reconstruct imported and live captures, recover, export, and pass the release gate.

**Architecture:** Keep Python and MLX behind the existing `ReconstructionEngine` boundary: the native app resolves one repository-managed Python 3.12 runtime, supervises `cloudpoint-worker`, and exchanges protocol-v1 file references over framed stdin/stdout with diagnostics on stderr. `SessionController` remains the single orchestration actor; it accumulates frame artifacts and commits each completed-window transaction atomically before publishing immutable UI updates, while model setup, exporting, and performance recording are focused services injected through `AppDependencies`.

**Tech Stack:** Swift 6, SwiftUI, AppKit, AVFoundation, Metal/MetalKit, URLSession, CryptoKit, XCTest/XCUITest, XcodeGen, Python 3.12.11, uv, MLX 0.32.0, safetensors 0.5.3, pytest.

## Global Constraints

- Target Apple Silicon Macs running macOS 15 or later; use the working bundle identifier `cloud.point.cloud.CloudPoint`.
- The app must never import Python or MLX; Python launch details remain in `CloudPoint/Engine` and behind `ReconstructionEngine`.
- Resolve the worker only from the `CLOUDPOINT_WORKER_RUNTIME` Xcode build setting; never fall back to a `python3`, `uv`, or `cloudpoint-worker` found on `PATH`.
- Use Python 3.12.11 and the fully pinned `worker/uv.lock`; runtime entry points are `<runtime>/bin/cloudpoint-worker` and `<runtime>/bin/cloudpoint-model`.
- Support only `robbyant/lingbot-map` revision `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`; `lingbot-map-long.pt` is exactly 4,632,303,465 bytes with SHA-256 `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`.
- Pin the reference implementation to Git commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`; converted weights remain in Application Support and are not redistributed.
- Model download is the worker's only prerequisite network activity and is performed by native URLSession; the worker binds no TCP port and makes no network requests.
- IPC is four-byte big-endian length plus canonical UTF-8 JSON over stdin/stdout,
  protocol version 1, maximum message size 1,048,576 bytes, exact malformed-input
  dispositions, one acknowledgement or structured error per decoded valid command,
  and an immediate heartbeat after hello ACK followed by at least one every five
  seconds during loading, inference, finalization, and idle.
- Sampling is timestamp-based, 1 through 10 FPS with a default of 5 FPS; persist oriented JPEG at quality 0.92 before enqueueing it.
- Use Float16 inference, eight anchor frames, 32-frame windows, eight-frame overlap,
  keyframe interval one, four camera-refinement passes, confidence threshold 1.5, and
  voxel size 0.01 by default.
- Display no more than five million points; release acceptance requires at least 30 renderer FPS at two million points and peak combined app-plus-worker resident memory below 48 GB on the development M1 Ultra with 64 GB unified memory.
- Store only sampled frames and derived data in `.cloudpoint`; never copy raw source recordings. Warn above a 10 GB estimated package or below 20 GB available volume capacity.
- Commit recovery only at completed-window boundaries. Native derives exact replay
  from committed artifact records; the worker alone performs exact-pattern output
  orphan cleanup before begin ACK and never mutates the manifest.
- A worker allocation failure gets exactly one clean restart of the current window with a 16-frame window and four-frame overlap; a second allocation failure stops without deleting captured frames.
- Coordinates are “reconstruction units,” never meters. Version 0.1 exports binary little-endian PLY and JSON camera trajectory only.
- App Store packaging, sandbox hardening, dense meshes, texture output, Gaussian splats, USDZ, and arbitrary user-selected model files remain out of scope.

### Prerequisite contracts used by every task

The native-foundation and Lingbot-worker plans must be complete before executing this plan. Do not rename these boundaries during integration:

```swift
public protocol ReconstructionEngine: Sendable {
    func prepare(configuration: EngineConfiguration) async throws
    func begin(project: ProjectDescriptor) async throws
    func enqueue(_ frame: PersistedFrame) async throws
    func finishInput() async throws
    func pause() async throws
    func resume() async throws
    func cancel() async
    func events() -> AsyncThrowingStream<EngineEvent, Error>
    func shutdown() async
}

public struct PersistedFrame: Codable, Sendable, Equatable {
    public let index: UInt32
    public let sourceTimestamp: Double
    public let relativePath: String
}

public struct ResumeCheckpoint: Codable, Sendable, Equatable {
    public let lastCommittedFrameIndex: UInt32
    public let replayFromFrameIndex: UInt32
    public let nextWindowIndex: UInt32
}

public extension ProjectManifest {
    static func load(from packageURL: URL) throws -> ProjectManifest
    func writeAtomically(to packageURL: URL, fileManager: FileManager = .default) throws
}
```

Manifest format 2 is the only newly written schema; v1 is rejected through the
actionable unsupported-version path. Persist and validate the complete configuration
(defaults 8/32/8/1/4/1.5/0.01), UInt32 frame/window IDs and inclusive bounds,
UInt64 counts/inliers, finite Double timestamps/durations/transforms, safe paths,
exactly 16 alignment values, ordered matching frame artifacts, and strictly
increasing source/window order while permitting source gaps. Each completed window
owns its inference start, unique output bounds, CPC, alignment, last processed index,
inliers, duration, and ordered committed frame artifacts.
Validate window size `1...1024`, scale frames `1...windowSize`, zero-legal overlap
less than window size, positive keyframe/refinement counts, and positive finite
confidence/voxel values.

Canonical outputs are `Predictions/%08u.depth-f16`,
`Predictions/%08u.confidence-f16`, `Predictions/%08u.geometry.json`, and
`Points/window-%08u.cpc` (minimum width eight, expanding through UInt32). Exclusive
siblings are `.<final-basename>.<lowercase-UUID>.partial`; the worker synchronizes
and promotes them without clobber. Before begin ACK, only the worker reads manifest
references and descriptor-relatively removes exact-pattern
partials plus unreferenced exact-pattern finals without following symlinks or
deleting unknown names; native never guesses output orphans.

Native derives replay from the final `max(windowOverlap, 1)` actual committed
artifacts, uses checked next-window addition, and never subtracts source indices.
Replay is read-only/eventless, absent from CPC, and excluded from queued/processed
counters. `queuedFrames` is cumulative unique-output admission this invocation, not
pending depth; `processedFrames <= queuedFrames`, backlog is their difference, and
all counters exclude replay. A native
`PendingWindowAccumulator` requires the exact ordered expected unique output IDs and
atomically commits artifacts/window before renderer publication.

The capture contracts are `AssetFrameSource.frames(at:) -> AsyncThrowingStream<CapturedFrame, Error>`, `CameraFrameSource.start(deviceID:sampleRate:)`, and injected `FramePersistence` that returns a `PersistedFrame` only after the JPEG rename. The rendering contracts are `PointChunk.open(url:limits:) throws -> PointChunk` and `PointCloudRenderer.append(_:)`.

The Swift worker boundary is the native plan's `WorkerEnvelope`, `WorkerCommand`,
`WorkerEvent`, `LengthPrefixedJSONCodec`, and
`WorkerProcess.start(executable:arguments:environment:)`. It mirrors the engine
plan's strict lower-camel protocol-v1 schema: the complete configure payload includes
voxel size; begin carries a nullable structured checkpoint; frame completion owns
only depth/confidence/geometry; window completion owns the CPC plus inference and
unique-output bounds; hello carries `supportedProtocolVersions: [UInt32]` containing
1. ACK carries top-level `commandId` and payload `{command}`;
command errors carry non-null `commandId`, asynchronous errors carry null
`commandId`, and both retain `{code,message,recoverable,details}`. ACK always
precedes lifecycle output: hello negotiates, configure validates/stores, begin
validates checkpoint/manifest and completes cleanup, enqueue admits to the bounded
ordered queue, finish closes admission, pause records the request, resume releases
it after cooperative quiescence with the queue retained, and cancel/shutdown are
distinct from later completion/cleanup/exit.
MLX/model work runs on one dedicated serial executor so framing, controls, and the
immediate-then-five-second heartbeat schedule remain live during loading, inference,
finalization, and idle.

Generated wire JSON uses sorted keys, lowercase UUIDs, no insignificant whitespace,
finite shortest-round-trip Double tokens, integral values without `.0`, normalized
negative zero, and lowercase `e` without `+` or redundant exponent zeroes. Relayed
raw numeric tokens in structured error details retain their exact lexemes. Encoders
emit and decoders require lowercase canonical UUID text. Invalid lengths/truncation
close without response; invalid JSON
or no recoverable complete command header gets at most one asynchronous fault before
close; a complete header with bad type/payload gets one command error and remains
open with state unchanged; an unsupported version gets one flushed command error and
then closes.

The worker development command is
`cd worker && uv run --frozen cloudpoint-worker serve --project ABSOLUTE_PATH --model ABSOLUTE_PATH`.
Production launches `<runtime>/bin/cloudpoint-worker` with the same `serve`
arguments. The process creates no listener or network connection. Its model
directory contains `lingbot-map-long-f16.safetensors`, `weights-manifest.json`, and
`model-manifest.json`. One CPC per output window uses the exact 32-byte little-endian
CPC1 header followed by stride-24 `<fff4BeHI` vertices; native code opens it only
through `PointChunk.open(url:limits:)`.

---

### Task 1: Bind the Native Engine to the Pinned Worker Runtime

**Files:**
- Modify: `Config/Local.example.xcconfig`
- Modify: `.gitignore`
- Modify: `Config/Base.xcconfig`
- Modify: `CloudPoint/Resources/Info.plist`
- Modify: `project.yml`
- Modify: `scripts/bootstrap`
- Modify: `CloudPoint/Engine/ReconstructionEngine.swift`
- Create: `CloudPoint/Engine/WorkerRuntime.swift`
- Create: `CloudPoint/Engine/PythonMLXEngine.swift`
- Modify: `CloudPoint/Engine/WorkerProcess.swift`
- Test: `CloudPointTests/WorkerRuntimeTests.swift`
- Test: `CloudPointTests/RealWorkerBridgeTests.swift`
- Create: `CloudPointTests/TestSupport/RealWorkerFixture.swift`

**Interfaces:**
- Consumes: `WorkerProcess.start(executable:arguments:environment:)`,
  `WorkerEnvelope`, `WorkerCommand`, `WorkerEvent`, `LengthPrefixedJSONCodec`, and
  `cloudpoint-worker serve --project ABSOLUTE_PATH --model ABSOLUTE_PATH`.
- Produces: `ReconstructionEngineFactory.makeEngine(modelDirectory:) throws -> any ReconstructionEngine`, `WorkerRuntime.resolve(bundleValue:environment:validateFiles:fileManager:) throws`, and a concrete engine that launches only the configured runtime.

- [ ] **Step 1: Write failing runtime-resolution and launch-contract tests**

```swift
import Foundation
import XCTest
@testable import CloudPoint

final class WorkerRuntimeTests: XCTestCase {
    func testMissingBuildSettingNeverFallsBackToPATH() {
        XCTAssertThrowsError(try WorkerRuntime.resolve(
            bundleValue: nil,
            environment: ["PATH": "/usr/local/bin:/usr/bin"]
        )) { error in
            XCTAssertEqual(error as? WorkerRuntimeError, .buildSettingMissing)
        }
    }

    func testRuntimeUsesExactManagedExecutables() throws {
        let root = URL(fileURLWithPath: "/tmp/cloudpoint-worker-runtime", isDirectory: true)
        let runtime = try WorkerRuntime.resolve(bundleValue: root.path, environment: [:], validateFiles: false)
        XCTAssertEqual(runtime.workerExecutable.path, root.appending(path: "bin/cloudpoint-worker").path)
        XCTAssertEqual(runtime.modelExecutable.path, root.appending(path: "bin/cloudpoint-model").path)
    }

    func testServeArgumentsContainOnlyResolvedAbsolutePaths() throws {
        let launch = try WorkerLaunch(
            runtime: .unchecked(root: URL(fileURLWithPath: "/opt/cloudpoint/.venv")),
            project: URL(fileURLWithPath: "/tmp/Test.cloudpoint"),
            model: URL(fileURLWithPath: "/tmp/model")
        )
        XCTAssertEqual(launch.arguments, [
            "serve", "--project", "/tmp/Test.cloudpoint",
            "--model", "/tmp/model"
        ])
    }
}
```

- [ ] **Step 2: Run the tests and verify the missing boundary fails**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/WorkerRuntimeTests CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL to compile with `cannot find 'WorkerRuntime' in scope` and `cannot find type 'ReconstructionEngineFactory' in scope`.

- [ ] **Step 3: Add the exact runtime and factory interfaces**

Add to `ReconstructionEngine.swift` and `WorkerRuntime.swift`:

```swift
public protocol ReconstructionEngineFactory: Sendable {
    func makeEngine(modelDirectory: URL) throws -> any ReconstructionEngine
}

public enum WorkerRuntimeError: Error, Equatable {
    case buildSettingMissing
    case runtimeMustBeAbsolute(String)
    case executableMissing(String)
}

public struct WorkerRuntime: Sendable, Equatable {
    public let root: URL
    public let workerExecutable: URL
    public let modelExecutable: URL

    public static func resolve(
        bundleValue: String?,
        environment: [String: String],
        validateFiles: Bool = true,
        fileManager: FileManager = .default
    ) throws -> Self {
        guard let raw = bundleValue?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            throw WorkerRuntimeError.buildSettingMissing
        }
        let root = URL(fileURLWithPath: raw, isDirectory: true).standardizedFileURL
        guard root.path.hasPrefix("/") else { throw WorkerRuntimeError.runtimeMustBeAbsolute(raw) }
        let worker = root.appending(path: "bin/cloudpoint-worker")
        let model = root.appending(path: "bin/cloudpoint-model")
        if validateFiles {
            for executable in [worker, model] where !fileManager.isExecutableFile(atPath: executable.path) {
                throw WorkerRuntimeError.executableMissing(executable.path)
            }
        }
        return Self(root: root, workerExecutable: worker, modelExecutable: model)
    }

    static func unchecked(root: URL) -> Self {
        Self(root: root,
             workerExecutable: root.appending(path: "bin/cloudpoint-worker"),
             modelExecutable: root.appending(path: "bin/cloudpoint-model"))
    }
}

public struct WorkerLaunch: Sendable, Equatable {
    public let executable: URL
    public let arguments: [String]

    public init(runtime: WorkerRuntime, project: URL, model: URL) throws {
        for url in [project, model] where !url.path.hasPrefix("/") {
            throw WorkerRuntimeError.runtimeMustBeAbsolute(url.path)
        }
        executable = runtime.workerExecutable
        arguments = ["serve", "--project", project.path, "--model", model.path]
    }
}
```

`PythonMLXEngineFactory` must create a fresh `PythonMLXEngine` per call.
`PythonMLXEngine` delegates subprocess lifecycle and framed pipes to the existing
`WorkerProcess`, maps every protocol-v1 `WorkerEvent` to the lossless native
`EngineEvent`, and passes an exact replacement environment containing only `HOME`,
`TMPDIR`, `PATH=/usr/bin:/bin`, `PYTHONNOUSERSITE=1`, `PYTHONHASHSEED=0`,
`LC_ALL=C`, and `LANG=C`. Never merge the inherited environment or pass Hugging Face
or shell credential variables. After hello ACK it calls idempotent
`markProtocolReady()`; launch alone never arms missed-heartbeat accounting and the
first valid heartbeat may also arm/reset it.

- [ ] **Step 4: Wire the local build setting and install the locked worker**

Use these exact settings and bootstrap commands:

```xcconfig
// Config/Local.example.xcconfig
CLOUDPOINT_WORKER_RUNTIME = $(SRCROOT)/worker/.venv
```

```xml
<!-- CloudPoint/Resources/Info.plist -->
<key>CloudPointWorkerRuntime</key>
<string>$(CLOUDPOINT_WORKER_RUNTIME)</string>
```

```bash
cd worker
uv sync --frozen --group dev --extra model-prep --extra reference
uv run --frozen cloudpoint-worker --help
uv run --frozen cloudpoint-model --help
```

`scripts/bootstrap` must copy `Config/Local.example.xcconfig` to ignored `Config/Local.xcconfig` only when the local file is absent, run the three commands above, then run `scripts/generate-project`.

- [ ] **Step 5: Add the opt-in real-worker bridge smoke test**

```swift
final class RealWorkerBridgeTests: XCTestCase {
    func testPreparedLingbotWorkerCompletesNineFrameFixture() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let runtimePath = environment["CLOUDPOINT_WORKER_RUNTIME"],
              let modelPath = environment["CLOUDPOINT_REAL_MODEL_DIR"] else {
            throw XCTSkip("Set CLOUDPOINT_WORKER_RUNTIME and CLOUDPOINT_REAL_MODEL_DIR")
        }
        let runtime = try WorkerRuntime.resolve(bundleValue: runtimePath, environment: environment)
        let fixture = try IntegrationFixture.courthouseNineFrames()
        let engine = try PythonMLXEngineFactory(runtime: runtime).makeEngine(
            modelDirectory: URL(fileURLWithPath: modelPath)
        )
        try await engine.prepare(configuration: fixture.engineConfiguration)
        try await engine.begin(project: fixture.project)
        for frame in fixture.frames { try await engine.enqueue(frame) }
        try await engine.finishInput()
        let events = try await engine.events().collect(until: { $0.isSessionCompleted })
        XCTAssertEqual(events.frameCompletionCount, 9)
        XCTAssertEqual(events.windowCompletionCount, 1)
        await engine.shutdown()
    }
}
```

- [ ] **Step 6: Run focused, protocol, and real bridge tests**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/WorkerRuntimeTests CODE_SIGNING_ALLOWED=NO
(cd worker && uv run --frozen pytest tests/protocol -q && uv run --frozen pytest tests/test_session_server.py -q)
CLOUDPOINT_WORKER_RUNTIME="$PWD/worker/.venv" CLOUDPOINT_REAL_MODEL_DIR="$HOME/Library/Application Support/cloud.point.cloud.CloudPoint/Models/robbyant-lingbot-map/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9" xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/RealWorkerBridgeTests CODE_SIGNING_ALLOWED=NO
```

Expected: the first two commands PASS; with prepared weights, the real smoke test PASSes with nine `frameCompleted` events and one `sessionCompleted` event. Without the two explicit environment variables, it reports one skipped test rather than searching `PATH`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore Config/Base.xcconfig Config/Local.example.xcconfig project.yml scripts/bootstrap CloudPoint/Engine CloudPointTests
git commit -m "feat: connect native engine to pinned worker runtime"
```

---

### Task 2: Download, Verify, Convert, and Report Model Health

**Files:**
- Modify: `project.yml`
- Create: `CloudPoint/ModelSetup/LingbotModelRelease.swift`
- Create: `CloudPoint/ModelSetup/URLSessionModelDownloader.swift`
- Create: `CloudPoint/ModelSetup/ModelInstaller.swift`
- Create: `CloudPoint/ModelSetup/ModelHealth.swift`
- Create: `CloudPoint/ModelSetup/ModelSetupViewModel.swift`
- Create: `CloudPoint/ModelSetup/ModelSetupView.swift`
- Modify: `CloudPoint/App/AppDependencies.swift`
- Modify: `CloudPoint/App/CloudPointApp.swift`
- Test: `CloudPointTests/ModelInstallerTests.swift`
- Test: `CloudPointTests/URLSessionModelDownloaderTests.swift`

**Interfaces:**
- Consumes: `WorkerRuntime.modelExecutable`, native Application Support, and the engine plan's exact `cloudpoint-model prepare --checkpoint ABSOLUTE_PATH --destination ABSOLUTE_EMPTY_DIRECTORY` command, which calls `verify_checkpoint(path: Path) -> VerifiedArtifact` and `prepare_model(checkpoint:destination:specs:) -> ModelManifest` without network access.
- Produces: `ModelInstalling.prepare() async -> AsyncThrowingStream<ModelSetupEvent, Error>`, `health() async -> ModelHealth`, resumable URLSession downloads, and `ModelSetupView` routing for unavailable models.

- [ ] **Step 1: Write failing provenance, checksum, and cancellation tests**

```swift
func testOfficialReleaseIsFullyPinned() {
    let release = LingbotModelRelease.v0_1
    XCTAssertEqual(release.revision, "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9")
    XCTAssertEqual(release.sourceBytes, 4_632_303_465)
    XCTAssertEqual(release.sourceSHA256, "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409")
    XCTAssertEqual(release.downloadURL.absoluteString,
        "https://huggingface.co/robbyant/lingbot-map/resolve/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/lingbot-map-long.pt?download=true")
}

func testBadDigestNeverRunsConverter() async throws {
    let download = FakeModelDownload(bytes: Data("wrong".utf8))
    let converter = RecordingModelConverter()
    let installer = ModelInstaller(release: .fixture(bytes: 5, sha256: String(repeating: "0", count: 64)),
                                   directories: try .temporary(), download: download,
                                   converter: converter)
    await XCTAssertThrowsErrorAsync { for try await _ in await installer.prepare() {} }
    XCTAssertEqual(await converter.invocationCount, 0)
}

func testCompletedInstallRequiresAllThreeConvertedArtifacts() async throws {
    let installer = try ModelInstaller.completedFixture(missing: "weights-manifest.json")
    XCTAssertEqual(await installer.health(), .invalid(.missingConvertedArtifact("weights-manifest.json")))
}
```

- [ ] **Step 2: Run the tests and verify they fail before the model module exists**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ModelInstallerTests -only-testing:CloudPointTests/URLSessionModelDownloaderTests CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL to compile with `cannot find 'LingbotModelRelease' in scope`.

- [ ] **Step 3: Implement the pinned release and installer contracts**

```swift
public struct LingbotModelRelease: Sendable, Equatable {
    public let repository: String
    public let revision: String
    public let filename: String
    public let sourceBytes: Int64
    public let sourceSHA256: String
    public let downloadURL: URL

    public static let v0_1 = Self(
        repository: "robbyant/lingbot-map",
        revision: "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9",
        filename: "lingbot-map-long.pt",
        sourceBytes: 4_632_303_465,
        sourceSHA256: "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
        downloadURL: URL(string: "https://huggingface.co/robbyant/lingbot-map/resolve/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/lingbot-map-long.pt?download=true")!
    )
}

public enum ModelSetupEvent: Sendable, Equatable {
    case downloading(received: Int64, expected: Int64)
    case verifying(bytesRead: Int64, expected: Int64)
    case converting
    case validating
    case ready(ModelInstallation)
}

public struct ModelInstallation: Codable, Sendable, Equatable {
    public let directory: URL
    public let sourceRevision: String
    public let sourceSHA256: String
    public let convertedSHA256: String
    public let engineVersion: String
}

public enum ModelHealthFailure: Error, Sendable, Equatable {
    case missingConvertedArtifact(String)
    case manifestMismatch(String)
    case operationFailed(String)
}

public enum ModelHealth: Sendable, Equatable {
    case absent
    case preparing(ModelSetupEvent)
    case ready(ModelInstallation)
    case invalid(ModelHealthFailure)
}

public protocol ModelInstalling: Sendable {
    func health() async -> ModelHealth
    func prepare() async -> AsyncThrowingStream<ModelSetupEvent, Error>
    func cancel() async
}
```

Use this Application Support directory exactly:

```text
~/Library/Application Support/cloud.point.cloud.CloudPoint/Models/
  robbyant-lingbot-map/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/
    source/lingbot-map-long.pt
    converted/lingbot-map-long-f16.safetensors
    converted/weights-manifest.json
    converted/model-manifest.json
    download.resume
```

Download to a sibling `.download.partial`; stream SHA-256 with CryptoKit in 8 MiB reads; compare byte count and digest before invoking the converter. Run the converter in a dedicated temporary directory with this exact invocation and sanitized environment:

```text
<runtime>/bin/cloudpoint-model prepare
  --checkpoint <verified-partial-path>
  --destination <converted.partial-directory>
```

The engine plan's converter re-verifies the native-downloaded checkpoint, never downloads, and produces the canonical SafeTensors plus `weights-manifest.json` and `model-manifest.json`. Only after it exits zero and all three outputs validate should the installer atomically rename source and converted artifacts into place. On cancellation preserve only URLSession resume data; on checksum or conversion failure delete source and converted partials.

- [ ] **Step 4: Implement URLSession progress and resume behavior**

```swift
public protocol ModelDownloading: Sendable {
    func download(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (_ received: Int64, _ expected: Int64) -> Void
    ) async throws -> URL
    func cancel() async -> Data?
}

public final class URLSessionModelDownloader: NSObject, ModelDownloading, URLSessionDownloadDelegate,
    @unchecked Sendable {
    // Own exactly one ephemeral URLSessionDownloadTask. Reject redirects whose final
    // host is not huggingface.co or cdn-lfs.huggingface.co, require HTTP 200/206,
    // publish delegate byte counts, and return URLSession resumeData on cancellation.
}
```

The test URL protocol must return three chunks, assert monotonic progress ending at content length, cancel after the first chunk, then prove the second task is constructed from the saved resume data. The production session uses `.ephemeral`, `waitsForConnectivity = true`, no cookie storage, and a 24-hour resource timeout.

- [ ] **Step 5: Add the setup sheet and disclose trusted-artifact conversion**

```swift
@MainActor
final class ModelSetupViewModel: ObservableObject {
    @Published private(set) var health: ModelHealth = .absent
    @Published private(set) var isPreparing = false
    let installer: any ModelInstalling

    func prepare() async {
        isPreparing = true
        defer { isPreparing = false }
        do {
            for try await event in await installer.prepare() { health = .preparing(event) }
            health = await installer.health()
        } catch { health = .invalid(.operationFailed(String(describing: error))) }
    }

    func cancel() { Task { await installer.cancel() } }
}
```

`ModelSetupView` must show source repository, pinned revision, 4.32 GiB download size, determinate download/verify progress, conversion activity, final health, Retry, Cancel, and “Continue” only for `.ready`. Display: “CloudPoint verifies the exact upstream checkpoint before a local trusted-artifact conversion. The checkpoint and converted weights are not redistributed.” Both **Open Recording** and **Use Camera** present this sheet when model health is not `.ready`.

- [ ] **Step 6: Run model tests and verify the setup UI builds**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ModelInstallerTests -only-testing:CloudPointTests/URLSessionModelDownloaderTests CODE_SIGNING_ALLOWED=NO
scripts/generate-project
xcodebuild build -project CloudPoint.xcodeproj -scheme CloudPoint -configuration Debug -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

Expected: all model setup tests PASS and `** BUILD SUCCEEDED **` appears. Network tests use the local URL protocol only; no test downloads the 4.6 GB checkpoint.

- [ ] **Step 7: Commit**

```bash
git add CloudPoint/ModelSetup CloudPoint/App project.yml CloudPointTests
git commit -m "feat: add verified model setup flow"
```

---

### Task 3: Orchestrate Imported and Live Sessions with Atomic Resume

**Files:**
- Modify: `CloudPoint/Domain/ProjectModels.swift`
- Modify: `CloudPoint/Persistence/ProjectManifest.swift`
- Modify: `CloudPoint/Workspace/SessionController.swift`
- Modify: `project.yml`
- Modify: `CloudPoint/Workspace/WorkspaceViewModel.swift`
- Modify: `CloudPoint/Persistence/CloudPointDocument.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Test: `CloudPointTests/SessionOrchestrationTests.swift`
- Test: `CloudPointTests/ProjectResumeTests.swift`
- Test: `CloudPointTests/ImportedRecordingIntegrationTests.swift`
- Test: `CloudPointTests/LiveCaptureIntegrationTests.swift`

**Interfaces:**
- Consumes: `AssetFrameSource.frames(at:)`, `CameraFrameSource.start(deviceID:sampleRate:)`, `FramePersistence`, `ProjectManifest.load(from:)`, `ProjectManifest.writeAtomically(to:fileManager:)`, `ReconstructionEngine.events()`, and `PointChunk.open(url:limits:)`.
- Produces: atomic `CompletedWindow` commits, `SessionController.resume(project:modelDirectory:)`, independent capture/processing fields on the existing `WorkspaceSnapshot`, and renderer-facing `WorkspaceEffect.appendPointChunk(PointChunk)`.

- [ ] **Step 1: Write failing orchestration and resume tests**

```swift
func testImportedFramesPersistBeforeEngineEnqueueAndFinish() async throws {
    let rig = try SessionRig.imported(frameTimestamps: [0, 0.2, 0.4])
    try await rig.controller.start(source: rig.source, project: rig.project, configuration: .defaults)
    await rig.waitForState(.completed)
    XCTAssertEqual(rig.trace.actions, [
        .persist(0), .saveManifest, .enqueue(0),
        .persist(1), .saveManifest, .enqueue(1),
        .persist(2), .saveManifest, .enqueue(2), .finishInput
    ])
}

func testStopCaptureFinishesInputWithoutCancellingBacklog() async throws {
    let rig = try SessionRig.live()
    try await rig.start()
    await rig.source.emit(indices: 0...9)
    await rig.controller.stopCapture()
    XCTAssertFalse(await rig.snapshot.isCapturing)
    XCTAssertTrue(await rig.snapshot.isProcessing)
    XCTAssertEqual(await rig.engine.cancelCount, 0)
    await rig.engine.completeQueuedFrames()
    await rig.waitForState(.completed)
}

func testResumeReplaysActualCommittedOverlapThenNewFrames() async throws {
    let rig = try SessionRig.interrupted(completedWindow: 1, lastFrame: 31, persistedFrames: 48)
    try await rig.controller.resume(project: rig.project)
    XCTAssertEqual(await rig.engine.begunDescriptor?.resumeCheckpoint,
                   ResumeCheckpoint(lastCommittedFrameIndex: 31,
                                    replayFromFrameIndex: 24,
                                    nextWindowIndex: 2))
    XCTAssertEqual(await rig.engine.enqueuedIndices, Array(24..<48))
    XCTAssertEqual(rig.store.nativeRemovedOutputOrphans, [])
}
```

- [ ] **Step 2: Run the focused tests and verify orchestration is incomplete**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/SessionOrchestrationTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ProjectResumeTests CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL because `SessionController.resume(project:)` is absent and the start flow does not commit manifest state before enqueueing.

- [ ] **Step 3: Add explicit checkpoint and snapshot state**

```swift
public struct CompletedWindow: Codable, Sendable, Equatable {
    public var index: UInt32
    public var inferenceFrameStart: UInt32
    public var frameStart: UInt32
    public var frameEnd: UInt32
    public var pointChunkRelativePath: String
    public var alignmentRowMajor: [Double] // exactly 16 finite values
    public var lastProcessedFrameIndex: UInt32
    public var inlierCount: UInt64
    public var durationSeconds: Double
    public var frameArtifacts: [FrameArtifacts]
}

// Add this field to the native plan's existing ProjectDescriptor.
public var resumeCheckpoint: ResumeCheckpoint?

public struct WorkspaceSnapshot: Sendable, Equatable {
    public var phase: SessionPhase
    public var isCapturing: Bool
    public var isProcessing: Bool
    public var capturedCount: UInt64
    public var queuedCount: UInt64
    public var processedCount: UInt64
    public var failedCount: UInt64
    public var currentWindow: UInt32?
    public var backlogCount: UInt64 {
        queuedCount >= processedCount ? queuedCount - processedCount : 0
    }
    public var recoverableAction: RecoveryAction?
}

public enum WorkspaceEffect: Sendable {
    case appendPointChunk(PointChunk)
    case selectSourceFrame(UInt32)
}
```

For every sampled frame, execute this order inside `SessionController`: receive the
`PersistedFrame` returned by `FramePersistence`, append it to a local manifest copy,
write that copy atomically, adopt the new revision in memory, then call
`engine.enqueue`. One view-model event-consumption task starts before any engine
method and passes each `EngineEvent` to `SessionController.apply(engineEvent:)`.
Generation tokens and monotonic manifest revisions prevent actor reentrancy from
letting stale engine/source work overwrite a newer session.
Publish/increment `queuedCount` only after successful enqueue admission, never
decrement it, and exclude replay; backlog is `queuedCount - processedCount`.

On `frameCompleted`, validate the three prediction paths and add the artifact to the
pure `PendingWindowAccumulator`; do not advance committed processed counts. On
`windowCompleted`, select the ordered expected unique output IDs from the session's
persisted/enqueued records, require the pending artifact IDs to match exactly, open
the window CPC, append the complete artifacts and window to one local manifest copy,
write it atomically, adopt it, update unique current-invocation counts, and only then
return `.appendPointChunk(chunk)`. Reject missing, extra, duplicate, cross-window,
and out-of-order artifacts; source-index gaps are legal, so bounds alone are not
proof of completeness. Native alone owns this pending/manifest transaction; worker
events are proposals until committed.

- [ ] **Step 4: Implement imported and live termination semantics**

```swift
public extension SessionController {
    func stopCapture() async {
        guard snapshot.isCapturing else { return }
        await source?.stop()
        snapshot.isCapturing = false
        publishSnapshot()
        do { try await engine.finishInput() }
        catch { await fail(error, preservingCapturedFrames: true) }
    }

    func pause() async throws {
        try await engine.pause()
        snapshot.phase = .paused
        // isCapturing deliberately remains unchanged for live capture.
        publishSnapshot()
    }
}
```

Imported source exhaustion and **Stop Capture** call `finishInput`; neither calls `cancel`. Pause affects worker progress only. Cancel stops both source and worker, leaves persisted frames intact, saves `.cancelled`, and never marks the package exportable. Before starting, estimate package bytes from sampled count and recent JPEG size; surface warnings for estimates over 10 GB or available capacity under 20 GB.

- [ ] **Step 5: Implement completed-window recovery and renderer restoration**

```swift
public func resume(project: ProjectDescriptor, modelDirectory: URL) async throws {
    var manifest = try ProjectManifest.load(from: project.packageURL)
    let completed = try manifest.completedWindows.map { window in
        let url = project.packageURL.appending(path: window.pointChunkRelativePath)
        return try PointChunk.open(url: url, limits: .renderer)
    }
    self.project = project
    self.manifest = manifest
    for chunk in completed { effectContinuation.yield(.appendPointChunk(chunk)) }
    let engine = try engineFactory.makeEngine(modelDirectory: modelDirectory)
    self.engine = engine
    startEventConsumption()
    try await engine.prepare(configuration: manifest.engineConfiguration)
    let checkpoint = try manifest.resumeCheckpoint()
    var resumedProject = project
    resumedProject.resumeCheckpoint = checkpoint
    try await engine.begin(project: resumedProject)
    let replayStart = checkpoint?.replayFromFrameIndex
    for frame in manifest.frames where replayStart == nil || frame.index >= replayStart! {
        try await engine.enqueue(frame)
    }
    try await engine.finishInput()
}
```

`ProjectManifest.load(from:)` plus `PointChunk.open(url:limits:)` reject invalid
completed chunks. Native never scans for or guesses removable output files. The
worker reads manifest references and performs exact-pattern, descriptor-relative,
no-follow orphan cleanup before begin ACK. Derive the checkpoint by checked next
window addition and by selecting the final `max(windowOverlap, 1)` actual committed
artifact records across window boundaries; the first selected index is replay start,
never arithmetic subtraction. Enqueue replay through the committed boundary and all
new frames in strict source order. Replay is read-only, eventless, absent from CPC,
and excluded from queued/processed counts. Reopened live projects resume as finite
recording-style jobs. `CloudPointDocument` loads recovery state on open and exposes
a resume action instead of starting automatically.

- [ ] **Step 6: Run unit and integration tests**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/SessionOrchestrationTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ProjectResumeTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ImportedRecordingIntegrationTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/LiveCaptureIntegrationTests CODE_SIGNING_ALLOWED=NO
```

Expected: all four suites PASS; the live test records `capturedCount > processedCount` before stop, continues processing afterward, and ends with equal counts.

- [ ] **Step 7: Commit**

```bash
git add CloudPoint/Domain CloudPoint/Persistence CloudPoint/Workspace CloudPointTests
git commit -m "feat: orchestrate resumable imported and live sessions"
```

---

### Task 4: Recover from Worker Crash, Missed Heartbeats, and One OOM

**Files:**
- Modify: `CloudPoint/Engine/WorkerProcess.swift`
- Modify: `CloudPoint/Engine/PythonMLXEngine.swift`
- Modify: `CloudPoint/Domain/ProjectModels.swift`
- Modify: `CloudPoint/Workspace/SessionController.swift`
- Test: `CloudPointTests/WorkerFailureIntegrationTests.swift`
- Test: `CloudPointTests/SessionRetryTests.swift`

**Interfaces:**
- Consumes: worker structured error code `ALLOCATION_FAILED`, process exit/stderr, five-second heartbeat cadence, and `SessionController.retry()`.
- Produces: `EngineFailure`, persisted `WindowRetryState`, clean process-group termination, automatic one-time 16/4 OOM retry, and user-triggered restart from checkpoint for crashes/timeouts.

- [ ] **Step 1: Write failing crash, heartbeat, and OOM tests**

```swift
func testThreeMissedHeartbeatsTerminateAndOfferRetry() async throws {
    let clock = TestClock()
    let rig = try FailureRig.busyWorker(clock: clock)
    await clock.advance(by: .seconds(15))
    await rig.waitForFailure()
    XCTAssertEqual(rig.snapshot.recoverableAction, .restartFromWindow(2))
    XCTAssertEqual(await rig.process.terminationSignals, [.terminateProcessGroup, .killProcessGroup])
    XCTAssertTrue(rig.workerLog.contains("heartbeat timeout"))
}

func testAllocationFailureRetriesCurrentWindowExactlyOnceWith16And4() async throws {
    let rig = try FailureRig.allocationFailures(count: 1)
    await rig.run()
    XCTAssertEqual(rig.factory.configurations.map { ($0.windowSize, $0.windowOverlap) }, [(32, 8), (16, 4)])
    XCTAssertEqual(rig.manifest.retryState, .allocationRetryConsumed(windowIndex: 2))
    XCTAssertEqual(rig.snapshot.phase, .completed)
}

func testSecondAllocationFailureStopsAndPreservesFrames() async throws {
    let rig = try FailureRig.allocationFailures(count: 2)
    await rig.run()
    XCTAssertEqual(rig.snapshot.phase, .failed)
    XCTAssertNil(rig.snapshot.recoverableAction)
    XCTAssertEqual(rig.persistedFrameCountAfterFailure, rig.persistedFrameCountBeforeFailure)
}
```

- [ ] **Step 2: Run tests and verify retry policy fails**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/SessionRetryTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/WorkerFailureIntegrationTests CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL because `EngineFailure` classification and persisted OOM retry state do not exist.

- [ ] **Step 3: Add bounded failure classification and supervision**

```swift
public enum EngineFailure: Error, Sendable, Equatable {
    case processExited(status: Int32, stderrLog: URL)
    case heartbeatTimeout(lastHeartbeat: ContinuousClock.Instant)
    case allocationFailed(windowIndex: UInt32)
    case worker(code: String, message: String, recoverable: Bool,
                details: [String: JSONValue])
}

public enum WindowRetryState: Codable, Sendable, Equatable {
    case none
    case allocationRetryConsumed(windowIndex: UInt32)
}
```

Process launch proves process-group ownership and writable framed stdio only; it
must not start a heartbeat timer. After hello ACK, call idempotent
`markProtocolReady()`; the first valid heartbeat may also arm/reset supervision.
Use local arrival time, never the worker's telemetry clock, and fail after three
missed five-second intervals regardless of busy/idle state. Capture bounded stderr
to `Logs/worker.log`, rotate at 10 MiB, and include exit status. On failure close all
pipes, send SIGTERM to the process group, wait two seconds using the injected clock,
then SIGKILL the same group if still alive. Never leave the process, reader,
escalation, or watchdog task detached.

- [ ] **Step 4: Implement checkpoint restart and exact OOM policy**

```swift
private func handle(_ failure: EngineFailure) async {
    await engine.shutdown()
    switch (failure, manifest.retryState) {
    case let (.allocationFailed(window), .none):
        manifest.retryState = .allocationRetryConsumed(windowIndex: window)
        manifest.engineConfiguration.windowSize = 16
        manifest.engineConfiguration.windowOverlap = 4
        try? manifest.writeAtomically(to: project.packageURL)
        await restartFromManifestCheckpoint(automatic: true)
    case (.allocationFailed, .allocationRetryConsumed):
        await fail(failure, recoverableAction: nil, preservingCapturedFrames: true)
    case (.processExited, _), (.heartbeatTimeout, _):
        let checkpoint = try? manifest.resumeCheckpoint()
        await fail(failure,
                   recoverableAction: checkpoint.map { .restartFromWindow($0.nextWindowIndex) },
                   preservingCapturedFrames: true)
    default:
        await fail(failure, recoverableAction: nil, preservingCapturedFrames: true)
    }
}
```

`retry()` must be legal only when `recoverableAction` exists, create a fresh engine,
revalidate model and the full manifest-derived checkpoint, then enqueue the exact
committed replay artifact range through the boundary followed by new frames. It must
not compute replay from `lastProcessedFrameIndex` arithmetic. Retain the previous
worker log. A crash never retries automatically; the UI action is **Restart from
window N**.

- [ ] **Step 5: Run failure suites and the complete core test set**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/SessionRetryTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/WorkerFailureIntegrationTests CODE_SIGNING_ALLOWED=NO
scripts/test-native
```

Expected: all tests PASS; the OOM fixture launches two engines at most, and the heartbeat fixture leaves no child process according to `waitpid(..., WNOHANG)`.

- [ ] **Step 6: Commit**

```bash
git add CloudPoint/Domain CloudPoint/Persistence CloudPoint/Workspace CloudPoint/Engine CloudPointTests
git commit -m "feat: recover safely from worker failures"
```

---

### Task 5: Stream Binary PLY and Camera Trajectory Exports

**Files:**
- Create: `CloudPoint/Export/ProjectExporter.swift`
- Create: `CloudPoint/Export/TrajectoryDocument.swift`
- Modify: `CloudPoint/Domain/ProjectModels.swift`
- Create: `CloudPoint/Workspace/ExportCommands.swift`
- Modify: `CloudPoint/App/CloudPointApp.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Test: `CloudPointTests/ProjectExporterTests.swift`

**Interfaces:**
- Consumes: `ProjectManifest.completedWindows`, `PointChunk.open(url:limits:)`, and engine-plan geometry JSON containing row-major camera-to-world/intrinsics arrays.
- Produces: `ProjectExporting.exportPLY(project:to:)`, `exportTrajectory(project:to:)`, atomic destination replacement, and completed-project export commands.

- [ ] **Step 1: Write byte-exact PLY and decoded JSON tests**

```swift
func testPLYIsBinaryLittleEndianAndStreamsAllChunks() async throws {
    let rig = try ExportRig.twoChunks(pointsPerChunk: 2)
    try await rig.exporter.exportPLY(project: rig.project, to: rig.plyURL)
    let file = try Data(contentsOf: rig.plyURL)
    let header = String(decoding: file.prefix(while: { $0 != 0x00 }), as: UTF8.self)
    XCTAssertTrue(header.contains("format binary_little_endian 1.0"))
    XCTAssertTrue(header.contains("comment coordinate_units reconstruction_units"))
    XCTAssertTrue(header.contains("element vertex 4"))
    XCTAssertEqual(try PLYFixtureReader(url: rig.plyURL).vertices.count, 4)
    XCTAssertLessThan(rig.exporter.maximumBufferedBytes, 8 * 1_024 * 1_024)
}

func testTrajectoryUsesCameraToWorldAndSourceTimestamps() async throws {
    let rig = try ExportRig.trajectoryFixture()
    try await rig.exporter.exportTrajectory(project: rig.project, to: rig.jsonURL)
    let document = try JSONDecoder().decode(TrajectoryDocument.self, from: Data(contentsOf: rig.jsonURL))
    XCTAssertEqual(document.coordinateUnits, "reconstruction_units")
    XCTAssertEqual(document.poses[0].sourceFrameIndex, 7)
    XCTAssertEqual(document.poses[0].timestampSeconds, 1.4, accuracy: 1e-9)
    XCTAssertEqual(document.poses[0].cameraToWorld.count, 16)
    XCTAssertEqual(document.poses[0].intrinsics.count, 9)
}
```

- [ ] **Step 2: Run tests and verify exporters are absent**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ProjectExporterTests CODE_SIGNING_ALLOWED=NO
```

Expected: FAIL to compile with `cannot find 'ProjectExporter' in scope`.

- [ ] **Step 3: Define exact export contracts and schemas**

```swift
public protocol ProjectExporting: Sendable {
    func exportPLY(project: ProjectDescriptor, to destination: URL) async throws
    func exportTrajectory(project: ProjectDescriptor, to destination: URL) async throws
}

public struct TrajectoryDocument: Codable, Sendable, Equatable {
    public let formatVersion: Int
    public let coordinateUnits: String
    public let modelIdentifier: String
    public let poses: [TrajectoryPose]
}

public struct TrajectoryPose: Codable, Sendable, Equatable {
    public let timestampSeconds: Double
    public let sourceFrameIndex: UInt32
    public let cameraToWorld: [Float] // row-major, exactly 16 values
    public let intrinsics: [Float]    // row-major 3x3, exactly 9 values
}
```

Write this exact ASCII header before the packed vertices:

```text
ply
format binary_little_endian 1.0
comment generated_by CloudPoint_0.1
comment coordinate_units reconstruction_units
element vertex <sum of validated chunk pointCount values>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
property float confidence
property uint source_frame
end_header
```

Open every CPC1 file through `PointChunk.open(url:limits:)`; its already-validated vertex records use the engine plan's `<fff4BeHI` layout. Convert Float16 confidence to little-endian Float32 while streaming and preserve source-frame UInt32. Hold one `PointChunk` mapping at a time and write in chunks no larger than 8 MiB. Preflight every header and total count before creating the destination partial.

- [ ] **Step 4: Make both exports atomic and gate them on completion**

Export to `.<destination-name>.partial-<UUID>` in the destination directory, call `FileHandle.synchronize()`, close, then use `FileManager.replaceItemAt`. Remove the partial on error. Reject non-completed manifests, invalid geometry paths, NaN points, destinations without available capacity, and trajectory records whose arrays are not exactly 16 and 9 values.

`ExportCommands` must present `NSSavePanel` with `allowedContentTypes` for `.ply` and `.json`, disable actions unless `manifest.sessionState.phase == .completed`, announce progress as `.finalizing`, and restore `.completed` after success or a user-correctable error.

- [ ] **Step 5: Run exporter tests and inspect fixture signatures**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ProjectExporterTests CODE_SIGNING_ALLOWED=NO
file CloudPointTests/TestSupport/Fixtures/export-four-points.ply
jq -e '.formatVersion == 1 and .coordinateUnits == "reconstruction_units" and (.poses | length) > 0' CloudPointTests/TestSupport/Fixtures/trajectory.json
```

Expected: tests PASS; `file` reports `PLY model`; `jq` exits 0.

- [ ] **Step 6: Commit**

```bash
git add CloudPoint/Domain CloudPoint/Export CloudPoint/Workspace CloudPoint/App CloudPointTests
git commit -m "feat: export point clouds and camera trajectories"
```

---

### Task 6: Finish the Four-Region UI and Performance Instrumentation

**Files:**
- Create: `CloudPoint/Workspace/SourcePanel.swift`
- Create: `CloudPoint/Workspace/PointCloudViewport.swift`
- Create: `CloudPoint/Workspace/InspectorPanel.swift`
- Create: `CloudPoint/Workspace/SessionTimelineView.swift`
- Create: `CloudPoint/Workspace/SessionStatusText.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Modify: `CloudPoint/Capture/CameraPreviewView.swift`
- Modify: `CloudPoint/Rendering/PointCloudRenderer.swift`
- Create: `CloudPoint/Performance/PerformanceRecorder.swift`
- Modify: `CloudPoint/Workspace/SessionController.swift`
- Modify: `project.yml`
- Create: `CloudPointTests/UITests/CloudPointDocumentUITests.swift`
- Test: `CloudPointTests/PerformanceRecorderTests.swift`
- Test: `CloudPointTests/RendererPerformanceTests.swift`

**Interfaces:**
- Consumes: the native plan's immutable `WorkspaceSnapshot`, `WorkspaceEffect`, `PointCloudRenderer`, direct `AVCaptureSession` preview, and project log directory.
- Produces: four stable regions, accessibility identifiers, truthful status copy, JSONL/signpost performance data, and renderer FPS/load measurements.

- [ ] **Step 1: Write failing status-copy, metric, and UI tests**

```swift
func testBackloggedCaptureNeverClaimsRealTime() {
    let snapshot = WorkspaceSnapshot.fixture(isCapturing: true, isProcessing: true,
                                           captured: 40, processed: 12, reconstructionFPS: 1.8,
                                           incomingSampleFPS: 5)
    XCTAssertEqual(SessionStatusText(snapshot).primary, "Capturing — 28 frames waiting")
    XCTAssertFalse(SessionStatusText(snapshot).primary.localizedCaseInsensitiveContains("real-time"))
}

func testRecorderWritesBoundedStructuredMeasurements() throws {
    let sink = InMemoryPerformanceSink()
    let recorder = PerformanceRecorder(clock: TestClock(), sink: sink)
    recorder.record(.windowCompleted(index: 2, frames: 32, durationSeconds: 8, residentBytes: 4_000))
    XCTAssertEqual(sink.records.single.name, "window_completed")
    XCTAssertEqual(sink.records.single.fields["window_index"], .integer(2))
}
```

```swift
func testEmptyDocumentExposesTwoPrimaryActions() throws {
    let app = XCUIApplication()
    app.launchArguments = ["--ui-testing", "--empty-document", "--mock-model-ready"]
    app.launch()
    XCTAssertTrue(app.buttons["open-recording"].waitForExistence(timeout: 3))
    XCTAssertTrue(app.buttons["use-camera"].exists)
    XCTAssertTrue(app.otherElements["point-cloud-viewport"].exists)
}
```

- [ ] **Step 2: Run focused tests and verify UI/metrics are missing**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/PerformanceRecorderTests CODE_SIGNING_ALLOWED=NO
scripts/generate-project
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointUITests/CloudPointDocumentUITests CODE_SIGNING_ALLOWED=NO
```

Expected: the Swift test fails to compile for missing `PerformanceRecorder`; the UI test fails to find `open-recording`.

- [ ] **Step 3: Build the stable four-region document layout**

```swift
struct WorkspaceView: View {
    @ObservedObject var model: WorkspaceViewModel

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                SourcePanel(model: model).frame(minWidth: 260, idealWidth: 300, maxWidth: 360)
                PointCloudViewport(model: model).frame(minWidth: 520)
                InspectorPanel(model: model).frame(minWidth: 260, idealWidth: 300, maxWidth: 360)
            }
            Divider()
            SessionTimelineView(model: model).frame(height: 88)
        }
        .frame(minWidth: 1_100, minHeight: 700)
    }
}
```

The source panel contains recording import, camera selection, direct live preview, source metadata, and capture controls. The viewport contains grid, axes, append-only cloud, trajectory, selected frustum, reset, empty/loading/error overlays, orbit/pan/zoom, and no inference work. The inspector exposes sampling 1–10 FPS, confidence default 1.5, point size, display budget up to five million, engine/model health, and coordinate-unit copy. The timeline shows captured, cumulative admitted, processed, backlog (`admitted - processed`), current window, elapsed time, reconstruction FPS, pause/resume/cancel, and **Stop Capture** separately.

Add accessibility identifiers `source-panel`, `point-cloud-viewport`, `inspector-panel`, `session-timeline`, `open-recording`, `use-camera`, `stop-capture`, `pause-processing`, `resume-processing`, `cancel-processing`, and `restart-window`. Every icon-only control has an accessibility label and help text. Disabled controls remain visible with a reason.

- [ ] **Step 4: Map every state to explicit, truthful copy and actions**

```swift
struct SessionStatusText: Equatable {
    let primary: String
    let detail: String

    init(_ value: WorkspaceSnapshot) {
        switch value.phase {
        case .empty: (primary, detail) = ("Choose a recording or camera", "Processing stays on this Mac after model setup.")
        case .preparing: (primary, detail) = ("Preparing reconstruction", value.modelHealthDescription)
        case .importing: (primary, detail) = ("Importing recording", "Sampled frames are saved before processing.")
        case .capturing where value.backlogCount > 0:
            (primary, detail) = ("Capturing — \(value.backlogCount) frames waiting", "Preview continues at capture speed.")
        case .capturing: (primary, detail) = ("Capturing", "Reconstruction is keeping up with the sampled input.")
        case .processing: (primary, detail) = ("Reconstructing window \(value.currentWindow ?? 1)", "\(value.processedCount) of \(value.queuedCount) sampled frames processed.")
        case .paused: (primary, detail) = ("Reconstruction paused", value.isCapturing ? "Capture continues; disk backlog is growing." : "Resume when ready.")
        case .finalizing: (primary, detail) = ("Finalizing", "Validating project outputs.")
        case .completed: (primary, detail) = ("Reconstruction complete", "Coordinates are reconstruction units, not meters.")
        case .cancelled: (primary, detail) = ("Processing cancelled", "Saved sampled frames remain in this project.")
        case .failed: (primary, detail) = ("Reconstruction failed", value.failureDescription)
        case .ready: (primary, detail) = ("Ready", "Start an import or camera capture.")
        }
    }
}
```

- [ ] **Step 5: Record actionable performance without blocking rendering**

```swift
public enum PerformanceEvent: Sendable {
    case modelLoaded(durationSeconds: Double, residentBytes: UInt64)
    case frameCompleted(index: UInt32, durationSeconds: Double)
    case windowCompleted(index: UInt32, frames: UInt64, durationSeconds: Double, residentBytes: UInt64)
    case rendererSample(displayedPoints: Int, framesPerSecond: Double)
    case backlog(queued: UInt64, processed: UInt64)
    case export(kind: String, bytes: UInt64, durationSeconds: Double, peakResidentBytes: UInt64)
}
```

Write newline-delimited JSON to `Logs/performance.jsonl` on a utility actor and mirror intervals with `OSSignposter`. Record app RSS, worker RSS from `proc_pid_rusage`, and combined peak; never sample faster than once per second. Renderer FPS is a rolling 120-draw average and is published at most once per second. Add deterministic synthetic CPC fixtures for one, two, and five million displayed points; gate two million at 30 FPS only when `CLOUDPOINT_PERFORMANCE_GATE=1` on an M1 Ultra, otherwise record and skip the hardware assertion.

- [ ] **Step 6: Run UI, accessibility, and performance tests**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/PerformanceRecorderTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/RendererPerformanceTests CODE_SIGNING_ALLOWED=NO
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointUITests CODE_SIGNING_ALLOWED=NO
```

Expected: all deterministic tests PASS, UI tests find the four regions and exercise pause/resume/stop/retry with the mock engine, and the renderer suite prints one skip unless the explicit hardware gate is enabled.

- [ ] **Step 7: Commit**

```bash
git add CloudPoint CloudPointTests project.yml
git commit -m "feat: polish session UI and record performance"
```

---

### Task 7: Establish the 0.1 Verification and Local Release Gate

**Files:**
- Create: `scripts/check-protocol-compatibility`
- Create: `scripts/verify`
- Create: `scripts/run-real-model-smoke`
- Create: `scripts/run-performance-gate`
- Create: `docs/acceptance/0.1-manual-acceptance.md`
- Create: `THIRD_PARTY_NOTICES.md`
- Modify: `README.md`
- Modify: `project.yml`
- Test: `CloudPointTests/ProtocolCompatibilityTests.swift`

**Interfaces:**
- Consumes: all Swift/Python tests, protocol-v1 schemas, Debug/Release builds, prepared local model, performance JSONL, representative recording, and a physical camera.
- Produces: one `scripts/verify` release command, explicit opt-in hardware commands, auditable third-party provenance, and a completed local acceptance procedure.

- [ ] **Step 1: Write the failing protocol compatibility test and checker fixture**

```swift
func testSwiftAndPythonProtocolFixturesAreByteCompatible() throws {
    let fixtureURL = repositoryRoot.appending(path: "worker/tests/fixtures/protocol-v1.json")
    let fixture = try Data(contentsOf: fixtureURL)
    let corpus = try JSONDecoder().decode(ProtocolCompatibilityFixture.self, from: fixture)
    for entry in corpus.messages {
        var decoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: corpus.maximumMessageBytes)
        let decoded = try decoder.append(Data(entry.framedBytes))
        XCTAssertEqual(decoded.count, 1, entry.name)
        XCTAssertEqual(decoded[0].protocolVersion, 1, entry.name)
    }
    XCTAssertEqual(corpus.protocolVersion, 1)
    XCTAssertEqual(corpus.maximumMessageBytes, 1_048_576)
}
```

- [ ] **Step 2: Run compatibility and repository gates to capture the initial failures**

Run:

```bash
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/ProtocolCompatibilityTests CODE_SIGNING_ALLOWED=NO
test -x scripts/verify
```

Expected: compatibility FAILs until the shared fixture/checker exists; `test` exits 1 because `scripts/verify` does not yet exist.

- [ ] **Step 3: Create the single deterministic verification command**

Create executable `scripts/verify` with this exact gate order:

```bash
#!/bin/bash
set -euo pipefail
repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"

xcrun swift-format lint --recursive --strict CloudPoint CloudPointTests CloudPointMockWorker
scripts/generate-project
scripts/test-native
(cd worker && uv sync --frozen --all-extras && uv run --frozen pytest -m 'not real_model' -q)
scripts/check-protocol-compatibility
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild build -project CloudPoint.xcodeproj -scheme CloudPoint -configuration Debug -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild build -project CloudPoint.xcodeproj -scheme CloudPoint -configuration Release -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
git diff --exit-code -- CloudPoint.xcodeproj
```

`scripts/check-protocol-compatibility` must run `(cd worker && uv run --frozen cloudpoint-worker protocol-fixture --output tests/fixtures/protocol-v1.json && git diff --exit-code -- tests/fixtures/protocol-v1.json)`, then run the focused Swift test against that exact corpus. `ProtocolCompatibilityFixture.messages` is the ordered union of command, ACK, command-error, asynchronous-error, and event rows; each row is `{name,json,framedBytes}`. Include null/full structured checkpoints, complete configuration including voxel size, frame-artifact and window-CPC completion events, nested unknown/missing fields, lowercase UUIDs, integral/nonintegral/exponent Double tokens, negative zero normalization, and exact raw numeric lexemes relayed in error details. The checker asserts all discriminators, typed widths, canonical bytes, malformed-input disposition cases, and the 1,048,576-byte limit on both sides. It prints `Protocol compatibility: PASS (version 1)` only after both implementations pass.

- [ ] **Step 4: Add exact opt-in real-model and hardware gates**

`scripts/run-real-model-smoke` must require `CLOUDPOINT_REAL_MODEL_DIR`, run the worker differential tolerances, run `RealWorkerBridgeTests`, import the nine courthouse fixtures, and assert one valid CPC plus nine depth/confidence/geometry outputs:

```bash
#!/bin/bash
set -euo pipefail
: "${CLOUDPOINT_REAL_MODEL_DIR:?Set CLOUDPOINT_REAL_MODEL_DIR to the prepared model revision directory}"
repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"
(cd worker && uv run --frozen pytest -m real_model -q)
CLOUDPOINT_WORKER_RUNTIME="$repository_root/worker/.venv" \
CLOUDPOINT_REAL_MODEL_DIR="$CLOUDPOINT_REAL_MODEL_DIR" \
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/RealWorkerBridgeTests CODE_SIGNING_ALLOWED=NO
```

`scripts/run-performance-gate` must require an M1 Ultra with at least 64 GB, set `CLOUDPOINT_PERFORMANCE_GATE=1`, run the 1M/2M/5M renderer test and representative PLY export test, then use `jq` on the generated report to require combined peak RSS `< 51539607552` and two-million-point renderer FPS `>= 30`.

- [ ] **Step 5: Record third-party provenance and local setup instructions**

`THIRD_PARTY_NOTICES.md` must include these concrete entries and license boundaries:

```markdown
# Third-Party Notices

## MLX
CloudPoint's local worker uses Apple MLX 0.32.0 from https://github.com/ml-explore/mlx under the MIT License. MLX is installed into the repository-managed Python environment and is not linked into the native app.

## Lingbot Map source
Model architecture and preprocessing behavior are reproduced from the Lingbot Map reference implementation pinned at commit 7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2. The referenced source is Apache-2.0; preserve its copyright and NOTICE material with any distributed derivative source.

## Lingbot Map checkpoint
The official robbyant/lingbot-map long checkpoint is downloaded by each tester from Hugging Face revision 204754b72bb24f561f8d7e7e1e4e4cd9e809adf9. CloudPoint verifies source SHA-256 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 and does not redistribute the checkpoint or converted weights while repository metadata and model-card licensing remain ambiguous.

## Test fixtures
The nine courthouse PNG fixtures 000000.png through 000008.png retain their upstream Lingbot Map Apache-2.0 provenance and checksums in worker/tests/fixtures/parity/provenance.json. Generated PyTorch reference tensors are local ignored artifacts and are not distributed.
```

The README must document macOS 15+, Apple Silicon, Xcode version from `.xcode-version`, `uv`, `scripts/bootstrap`, local `Config/Local.xcconfig`, `scripts/verify`, model preparation through the app, and the two explicit hardware gates. It must state that coordinates are reconstruction units and that raw source recordings are not copied.

- [ ] **Step 6: Execute the complete automated release gate**

Run:

```bash
scripts/verify
```

Expected final lines include:

```text
Protocol compatibility: PASS (version 1)
** TEST SUCCEEDED **
** BUILD SUCCEEDED **
** BUILD SUCCEEDED **
```

The command exits 0 with no generated-project diff.

- [ ] **Step 7: Execute and record manual M1 Ultra acceptance**

Follow `docs/acceptance/0.1-manual-acceptance.md` in this order:

1. Use setup UI to download, verify, and convert the pinned checkpoint; relaunch and confirm health remains **Ready** without network access.
2. Import a representative indoor MOV/MP4 at 5 FPS; confirm incremental points/trajectory, complete processing, and no invalid window seam.
3. Capture exactly 60 seconds from a Mac or Continuity Camera while the preview stays responsive and processed count trails captured count.
4. Pause processing while capture continues, verify backlog/estimated size increases, resume, press **Stop Capture**, and allow the disk backlog to reach completion.
5. Close a second project during an active window, reopen it, choose **Resume**, and confirm processing begins after the last completed window without duplicate chunks.
6. Export binary PLY, open it in an independent point-cloud viewer, and confirm color/orientation; decode `trajectory.json` and compare camera direction plus reprojection with source frames.
7. Run `scripts/run-real-model-smoke` and `scripts/run-performance-gate`; retain their JSON reports under `Artifacts/0.1-acceptance/` outside git.
8. Confirm the performance report shows combined peak RSS below 48 GB, renderer FPS at two million points at least 30, and records model-load, per-frame, per-window, backlog, one/two/five-million renderer, and PLY export measurements.

The acceptance document must define each row with `PASS`/`FAIL`, the exact artifact filename, and remediation. Release is accepted only when every row is `PASS`; reconstruction FPS is recorded but never used as a pass threshold.

- [ ] **Step 8: Commit the release gate and notices**

```bash
git add scripts docs/acceptance/0.1-manual-acceptance.md THIRD_PARTY_NOTICES.md README.md project.yml CloudPointTests
git commit -m "chore: establish CloudPoint 0.1 release gate"
```

- [ ] **Step 9: Prove the repository ends in a locally testable 0.1 state**

Run:

```bash
scripts/verify
git status --short
```

Expected: `scripts/verify` exits 0 and `git status --short` prints nothing. On a prepared M1 Ultra, both optional gates also exit 0; the generated Debug and Release products launch, open a `.cloudpoint` document, and route **Open Recording** and **Use Camera** through model health before starting work.

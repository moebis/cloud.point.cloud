# CloudPoint Native Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable native macOS CloudPoint app that imports recordings or previews a camera, persists sampled frames, supervises a mock reconstruction worker, and progressively renders deterministic point-cloud chunks with Metal.

**Architecture:** A SwiftUI document app owns capture and project state through actors. A replaceable `ReconstructionEngine` protocol initially targets both an in-process deterministic mock and a supervised local worker protocol. The renderer consumes validated, memory-mapped CPC chunks through Metal and remains independent of the eventual Lingbot MLX engine.

**Tech Stack:** Swift 6.3, SwiftUI, AppKit, AVFoundation, Metal, MetalKit, XCTest, XcodeGen 2.46.0, macOS 15+

## Global Constraints

- Target Apple Silicon Macs running macOS 15 or later.
- Use bundle identifier `cloud.point.cloud.CloudPoint` and working product name `CloudPoint`.
- Keep capture preview independent of reconstruction throughput.
- Persist sampled live frames before enqueueing them; never grow an unbounded in-memory frame queue.
- Treat point-cloud coordinates as reconstruction units, not meters.
- Do not add MLX, PyTorch, OpenCV, RealityKit, SceneKit, network servers, or arbitrary model loading in this plan.
- Use Swift strict concurrency and make mutable cross-task owners actors or `@MainActor` types.
- Do not commit generated DerivedData, model files, project documents, or local runtime paths.

---

## File Structure

```text
project.yml                                      XcodeGen source of truth
Config/Base.xcconfig                             shared build settings
CloudPoint/Resources/Info.plist                  camera permission and document type
CloudPoint/Resources/CloudPoint.entitlements     local developer entitlements
CloudPoint/App/CloudPointApp.swift               app/document entry point
CloudPoint/Domain/ProjectModels.swift             shared Codable/Sendable values
CloudPoint/Domain/SessionState.swift              legal session transitions
CloudPoint/Persistence/ProjectManifest.swift      manifest schema and atomic writes
CloudPoint/Persistence/CloudPointDocument.swift   SwiftUI document package adapter
CloudPoint/Engine/ReconstructionEngine.swift      replaceable engine contract
CloudPoint/Engine/MockReconstructionEngine.swift  deterministic in-process geometry
CloudPoint/Engine/WorkerProtocol.swift            length-prefixed JSON messages
CloudPoint/Engine/WorkerProcess.swift             subprocess/framed-stdio supervision
CloudPoint/Capture/FrameSampler.swift             timestamp sampling policy
CloudPoint/Capture/AssetFrameSource.swift         AVAssetReader import
CloudPoint/Capture/CameraFrameSource.swift        AVCaptureSession capture actor
CloudPoint/Capture/CameraPreviewView.swift        AppKit preview bridge
CloudPoint/Rendering/PointChunk.swift              validated CPC reader
CloudPoint/Rendering/PointCloudRenderer.swift      Metal buffers and draw loop
CloudPoint/Rendering/PointCloudView.swift          SwiftUI MTKView bridge
CloudPoint/Rendering/PointShaders.metal            point vertex/fragment shaders
CloudPoint/Workspace/WorkspaceViewModel.swift      UI orchestration
CloudPoint/Workspace/SessionController.swift       actor-owned workflow transitions
CloudPoint/Workspace/WorkspaceView.swift           four-region workspace
CloudPointMockWorker/main.swift                    protocol test executable
CloudPointTests/TestSupport/...                    deterministic package/video/harness fixtures
CloudPointTests/...                                native unit/integration tests
scripts/bootstrap                                 install/check deterministic tools
scripts/generate-project                          regenerate Xcode project
scripts/test-native                               native test gate
```

### Task 1: Reproducible Native Project and Build Gate

**Files:**
- Create: `.gitignore`
- Create: `project.yml`
- Create: `Config/Base.xcconfig`
- Create: `CloudPoint/Resources/Info.plist`
- Create: `CloudPoint/Resources/CloudPoint.entitlements`
- Create: `CloudPoint/App/CloudPointApp.swift`
- Create: `CloudPoint/Workspace/WorkspaceView.swift`
- Create: `scripts/bootstrap`
- Create: `scripts/generate-project`
- Create: `scripts/test-native`

**Interfaces:**
- Produces: Xcode schemes `CloudPoint`, `CloudPointTests`, and `CloudPointMockWorker`.
- Produces: `scripts/test-native`, the native verification entry point used by every later task.

- [ ] **Step 1: Write the bootstrap contract**

Create executable `scripts/bootstrap` that checks Xcode, installs XcodeGen 2.46.0 when absent, installs the Metal Toolchain when `xcodebuild -showComponent MetalToolchain -json` reports `uninstalled`, and then invokes `scripts/generate-project`:

```bash
#!/usr/bin/env bash
set -euo pipefail

required_xcodegen="2.46.0"
actual_xcodegen="$(xcodegen --version 2>/dev/null | awk '{print $2}' || true)"
if [[ "$actual_xcodegen" != "$required_xcodegen" ]]; then
  brew install xcodegen
fi

if xcodebuild -showComponent MetalToolchain -json | grep -q '"status" : "uninstalled"'; then
  xcodebuild -downloadComponent MetalToolchain
fi

"$(dirname "$0")/generate-project"
```

- [ ] **Step 2: Add the XcodeGen specification and minimal app**

Define a macOS application target, unit-test target, and command-line mock-worker target in `project.yml`. The application must set `MACOSX_DEPLOYMENT_TARGET: 15.0`, `SWIFT_VERSION: 6.0`, `SWIFT_STRICT_CONCURRENCY: complete`, use the committed Info.plist, and include `.metal` sources. Use this minimal entry point:

```swift
import SwiftUI

@main
struct CloudPointApp: App {
    var body: some Scene {
        WindowGroup("CloudPoint") {
            WorkspaceView()
        }
    }
}
```

and this temporary compile target:

```swift
import SwiftUI

struct WorkspaceView: View {
    var body: some View {
        ContentUnavailableView(
            "Create a 3D map",
            systemImage: "point.3.connected.trianglepath.dotted",
            description: Text("Open a recording or use a camera.")
        )
        .frame(minWidth: 960, minHeight: 640)
    }
}
```

- [ ] **Step 3: Generate and build the project**

Run:

```bash
chmod +x scripts/bootstrap scripts/generate-project scripts/test-native
scripts/bootstrap
scripts/test-native
```

Expected: XcodeGen creates `CloudPoint.xcodeproj`; all three schemes compile; `CloudPointTests` reports zero failures.

- [ ] **Step 4: Commit the project skeleton**

```bash
git add .gitignore project.yml Config CloudPoint CloudPointMockWorker scripts
git commit -m "build: scaffold CloudPoint macOS app"
```

### Task 2: Project Models, State Machine, and Atomic Manifest

**Files:**
- Create: `CloudPoint/Domain/ProjectModels.swift`
- Create: `CloudPoint/Domain/SessionState.swift`
- Create: `CloudPoint/Persistence/ProjectManifest.swift`
- Create: `CloudPoint/Persistence/CloudPointDocument.swift`
- Create: `CloudPointTests/SessionStateTests.swift`
- Create: `CloudPointTests/ProjectManifestTests.swift`
- Create: `CloudPointTests/TestSupport/TemporaryProjectPackage.swift`
- Modify: `project.yml`

**Interfaces:**
- Produces: `PersistedFrame(index:sourceTimestamp:relativePath:)` with a UInt32
  source index and finite nonnegative Double timestamp.
- Produces: `SessionPhase`, `SessionEvent`, and `SessionState.applying(_:) throws -> SessionState`.
- Produces: `ProjectManifest.load(from:)` and `writeAtomically(to:fileManager:)`.
- Produces: `.cloudpoint` package document support.

- [ ] **Step 1: Write failing state transition tests**

```swift
import XCTest
@testable import CloudPoint

final class SessionStateTests: XCTestCase {
    func testCaptureCanTrailIntoProcessing() throws {
        var state = SessionState.empty
        state = try state.applying(.prepare)
        state = try state.applying(.ready)
        state = try state.applying(.startCapture)
        state = try state.applying(.enqueueFrame)
        state = try state.applying(.stopCapture)
        XCTAssertEqual(state.phase, .processing)
        XCTAssertFalse(state.isCapturing)
        XCTAssertEqual(state.capturedCount, 1)
    }

    func testCompletedCannotRestartCapture() throws {
        let completed = SessionState(phase: .completed)
        XCTAssertThrowsError(try completed.applying(.startCapture))
    }
}
```

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS' -only-testing:CloudPointTests/SessionStateTests`

Expected: FAIL because `SessionState` is undefined.

- [ ] **Step 3: Implement immutable state and shared values**

Define `SessionPhase: String, Codable, Sendable` with `empty`, `preparing`, `ready`, `importing`, `capturing`, `processing`, `paused`, `finalizing`, `completed`, `cancelled`, and `failed`. Define explicit transition switches; reject every unspecified pair with `SessionTransitionError.illegal(from:event:)`. Define IDs as UUIDs, frame/window indices as UInt32, counts as UInt64, frame timestamps as finite nonnegative `Double` seconds, and paths as safe package-relative strings.

- [ ] **Step 4: Write failing manifest-version and validation tests**

Create `TemporaryProjectPackage` as an XCTest support value that uses
`FileManager.default.temporaryDirectory.appending(path: UUID().uuidString).appendingPathExtension("cloudpoint")`, creates the four package subdirectories, and removes its URL in `deinit`. Add deterministic `fixture()` factories in the test target for `ProjectManifest` and `CompletedWindow`. Assert format v2 round trips, v1 is rejected by the actionable unsupported-version path, malformed safe paths/nonfinite values/orderings fail, and the final completed window remains recoverable:

```swift
func testVersionTwoRoundTripsCompletedWindow() throws {
    let package = try TemporaryProjectPackage.make()
    var manifest = ProjectManifest.fixture()
    manifest.completedWindows = [.fixture(index: 0)]
    try manifest.writeAtomically(to: package.url)
    let loaded = try ProjectManifest.load(from: package.url)
    XCTAssertEqual(loaded.formatVersion, 2)
    XCTAssertEqual(loaded.completedWindows.map(\.index), [0])
}
```

- [ ] **Step 5: Implement versioned manifest and document package**

Use format version `2` only, sorted-key JSON, ISO-8601 dates, and a synchronized
same-directory manifest temporary followed by atomic replacement. Never silently
decode v1 as v2. Persist the complete engine configuration and validate all paths,
finite timestamps/durations/transforms, inclusive ranges, artifact/window ownership,
and strictly increasing source/window order while permitting source-index gaps.
`CloudPointDocument` must conform to `ReferenceFileDocument`, declare UTType
`cloud.point.cloud.project`, and create `Frames`, `Predictions`, `Points`, and `Logs`
directories on first save. The worker owns output-orphan cleanup; document loading
must not guess which artifact files are removable.

- [ ] **Step 6: Run tests and commit**

Run: `scripts/test-native`

Expected: PASS with state, manifest, and document tests green.

```bash
git add project.yml CloudPoint/Domain CloudPoint/Persistence CloudPointTests
git commit -m "feat: add recoverable CloudPoint projects"
```

### Task 3: Reconstruction Engine Contract and Deterministic Mock

**Files:**
- Create: `CloudPoint/Engine/ReconstructionEngine.swift`
- Create: `CloudPoint/Engine/MockReconstructionEngine.swift`
- Create: `CloudPointTests/MockReconstructionEngineTests.swift`

**Interfaces:**
- Consumes: `PersistedFrame` from Task 2.
- Produces: the exact `ReconstructionEngine` protocol from the approved design.
- Produces: `EngineConfiguration`, `ProjectDescriptor`, and `EngineEvent`.

- [ ] **Step 1: Write the failing async engine test**

```swift
func testMockEmitsOneChunkPerFrameThenCompletes() async throws {
    let engine = MockReconstructionEngine(clock: .immediate)
    try await engine.prepare(configuration: .fixture())
    try await engine.begin(project: .fixture())
    let events = await engine.events()
    try await engine.enqueue(.fixture(index: 7))
    try await engine.finishInput()

    var received: [EngineEvent] = []
    for try await event in events {
        received.append(event)
        if case .sessionCompleted = event { break }
    }
    XCTAssertTrue(received.contains { $0.frameIndex == 7 })
    XCTAssertTrue(received.contains(.sessionCompleted))
}
```

- [ ] **Step 2: Verify the test fails**

Run the focused test. Expected: FAIL because `ReconstructionEngine` and `MockReconstructionEngine` do not exist.

- [ ] **Step 3: Implement the engine types and actor**

Use `AsyncThrowingStream<EngineEvent, Error>` and an actor-owned continuation. Add a private `MockCPCWriter` in `MockReconstructionEngine.swift` so this task has no forward dependency on Task 7. It writes the exact approved 32-byte CPC1 header and 24-byte vertex records using explicit little-endian integers and bit patterns. For each new mock frame, create deterministic depth, confidence-above-1.5, and geometry artifacts, then one one-frame window CPC. Emit `frameStarted`, `frameCompleted(FrameArtifacts)` without a CPC, and `windowCompleted(WindowResult)` with the CPC, in that order and only after the files exist. Its color and Z offset derive from the source index so tests and screenshots are reproducible. Pause suspends consumption; cancel closes the stream with `.cancelled`. Check UInt32 arithmetic and honor replay checkpoints without rewriting or emitting events for replay frames.

- [ ] **Step 4: Run and commit**

Run: `scripts/test-native`

Expected: PASS, including async cancellation and pause/resume tests.

```bash
git add CloudPoint/Engine CloudPointTests/MockReconstructionEngineTests.swift
git commit -m "feat: define reconstruction engine boundary"
```

### Task 4: Versioned Worker Protocol and Supervision

**Files:**
- Create: `CloudPoint/Engine/WorkerProtocol.swift`
- Create: `CloudPoint/Engine/WorkerProcess.swift`
- Create: `CloudPointMockWorker/main.swift`
- Create: `CloudPointTests/WorkerProtocolTests.swift`
- Create: `CloudPointTests/WorkerProcessTests.swift`

**Interfaces:**
- Consumes: `EngineConfiguration`, `PersistedFrame`, and `EngineEvent`.
- Produces: `WorkerEnvelope`, `WorkerCommand`, `WorkerEvent`, and `LengthPrefixedJSONCodec`.
- Produces: `WorkerProcess.start(executable:arguments:environment:)` and heartbeat supervision.

- [ ] **Step 1: Write failing codec tests**

```swift
func testCodecRoundTripsFragmentedFrames() throws {
    let envelope = WorkerEnvelope.command(.hello(protocolVersion: 1))
    let bytes = try LengthPrefixedJSONCodec.encode(envelope)
    var decoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: 1_048_576)
    XCTAssertEqual(try decoder.append(bytes.prefix(3)), [])
    XCTAssertEqual(try decoder.append(bytes.dropFirst(3)), [envelope])
}

func testCodecRejectsOversizedPayload() {
    var decoder = LengthPrefixedJSONCodec.Decoder(maxPayloadBytes: 16)
    XCTAssertThrowsError(try decoder.append(Data([0, 0, 0, 17])))
}
```

- [ ] **Step 2: Verify codec tests fail, then implement framing**

Use a four-byte big-endian unsigned length and the canonical JSON rules frozen in
Task 7.5. Enforce the exact no-response/async-fault/command-error disposition for
invalid framing, JSON, headers, payloads, and versions; reject duplicate command IDs
without repeating mutation. Run focused tests and expect PASS.

- [ ] **Step 3: Write failing worker lifecycle and protocol-readiness tests**

Launch `CloudPointMockWorker --mode heartbeat`, perform `hello`, assert its ACK is
followed immediately by a heartbeat, call idempotent
`WorkerProcess.markProtocolReady()`, then terminate it and assert `WorkerProcess`
emits `.processExited(status:)` exactly once and closes file descriptors. With an
injected clock, assert launch requests no heartbeat timer, duplicate readiness arms
one timer, the first heartbeat may arm/reset it, and three missed intervals after
arming are terminal.

- [ ] **Step 4: Implement process supervision and mock worker**

Use `Process`, explicit framed stdin/stdout pipes, a diagnostics-only stderr pipe, a
process-group termination path, and an actor for decoder state. Supply the child
environment as an exact replacement containing only `HOME`, `TMPDIR`,
`PATH=/usr/bin:/bin`, `PYTHONNOUSERSITE=1`, `PYTHONHASHSEED=0`, `LC_ALL=C`, and
`LANG=C`, plus explicit mock controls in tests; prove an inherited sentinel secret
is absent. The mock worker supports `normal`,
`heartbeat`, `crash-after-ready`, and `silent` modes and writes protocol frames only
to stdout. Launcher readiness proves process ownership and writable stdio only.
Protocol readiness after hello ACK or the first valid heartbeat arms supervision;
three missed five-second intervals then produce `WorkerProcessError.unresponsive`.
Exit/shutdown races produce one terminal result with no timer, task, pipe, or child
leak.

- [ ] **Step 5: Run and commit**

Run: `scripts/test-native`

Expected: PASS for fragmented messages, malformed lengths, crash detection, heartbeat timeout with an injected test clock, and clean shutdown.

```bash
git add CloudPoint/Engine CloudPointMockWorker CloudPointTests/WorkerProtocolTests.swift CloudPointTests/WorkerProcessTests.swift
git commit -m "feat: supervise reconstruction workers"
```

### Task 5: Deterministic Recording Sampling and Persistence

**Files:**
- Create: `CloudPoint/Capture/FrameSampler.swift`
- Create: `CloudPoint/Capture/AssetFrameSource.swift`
- Create: `CloudPointTests/FrameSamplerTests.swift`
- Create: `CloudPointTests/AssetFrameSourceTests.swift`
- Create: `CloudPointTests/TestSupport/VideoFixtureFactory.swift`

**Interfaces:**
- Produces: `FrameSamplingPlan.timestamps(duration:rate:) -> [CMTime]`.
- Produces: `AssetFrameSource.frames(at:) -> AsyncThrowingStream<CapturedFrame, Error>`.
- Consumes: project `Frames` directory and produces `PersistedFrame` only after an atomic JPEG write.

- [ ] **Step 1: Write failing timestamp tests**

```swift
func testFiveFPSSamplingUsesTimeNotSourceFrameCount() throws {
    let plan = try FrameSamplingPlan(duration: .seconds(1.0), framesPerSecond: 5)
    XCTAssertEqual(plan.timestamps.map { $0.seconds }, [0.0, 0.2, 0.4, 0.6, 0.8], accuracy: 0.000_001)
}

func testRateOutsideOneThroughTenIsRejected() {
    XCTAssertThrowsError(try FrameSamplingPlan(duration: .seconds(1), framesPerSecond: 11))
}
```

- [ ] **Step 2: Implement and pass the sampling policy**

Use `CMTime(value:timescale:)` with timescale 600 and stop strictly before duration. Avoid accumulated floating-point addition by deriving each timestamp from its integer sample index.

- [ ] **Step 3: Write failing asset integration tests**

Have `VideoFixtureFactory` generate a one-second, 640-by-360 HEVC or H.264 movie in the test temporary directory with presentation timestamps `[0, 1/30, 2/30, 4/30, 7/30, 11/30, 16/30, 22/30, 29/30]` using `AVAssetWriter`; give each frame a deterministic solid color and set a 90-degree preferred transform. Decode that generated fixture at 5 FPS, assert timestamps are within half a source-frame duration of the plan, orientation is applied, output width/height are positive, and cancellation yields no additional frames. This keeps the fixture reproducible and avoids committing an opaque binary.

- [ ] **Step 4: Implement AVAssetReader source and atomic JPEG persistence**

Use `AVAssetReaderTrackOutput`, request native bi-planar YUV when supported,
transform orientation before JPEG encoding, convert the capture counter exactly to
UInt32, write and synchronize a sibling temporary for `Frames/%08u.jpg`, promote it,
then emit `PersistedFrame`. Do not load the complete recording into memory.

- [ ] **Step 5: Run and commit**

Run: `scripts/test-native`

Expected: PASS for timestamp, VFR fixture, cancellation, and persistence tests.

```bash
git add CloudPoint/Capture CloudPointTests
git commit -m "feat: import recordings with deterministic sampling"
```

### Task 6: Camera Capture and Preview Boundary

**Files:**
- Create: `CloudPoint/Capture/CameraFrameSource.swift`
- Create: `CloudPoint/Capture/CameraPreviewView.swift`
- Create: `CloudPointTests/CameraFrameSourceTests.swift`

**Interfaces:**
- Produces: `CameraCatalog.devices() async -> [CameraDescriptor]`.
- Produces: `CameraFrameSource.start(deviceID:sampleRate:)` plus independent preview session access.
- Consumes: `FrameSamplingPlan` and emits persisted sampled frames through an injected `FramePersistence` protocol.

- [ ] **Step 1: Write failing tests against injected capture samples**

Feed 30 synthetic timestamps over one second into `CameraSampleGate(rate: 5)` and assert exactly five persistence calls while all 30 samples reach the preview observer. Assert device-disconnect changes state to `.failed(.cameraDisconnected)` without deleting prior frames.

- [ ] **Step 2: Implement the timestamp gate and camera actor**

Use `AVCaptureDevice.DiscoverySession`, `AVCaptureSession`, `AVCaptureVideoDataOutput`, and a dedicated serial delegate queue. Request camera authorization explicitly. Preview frames update `AVCaptureVideoPreviewLayer`; sampled frames use the same atomic persistence interface as recordings. Probe intrinsic delivery at runtime but do not require it.

- [ ] **Step 3: Implement the SwiftUI preview bridge**

Wrap an `NSView` whose backing layer is `AVCaptureVideoPreviewLayer`. Set `.resizeAspect`; never convert every preview frame through SwiftUI images.

- [ ] **Step 4: Run and commit**

Run: `scripts/test-native`

Expected: PASS with synthetic capture samples; the physical-camera test remains an opt-in manual test named `CameraManualTests` and is excluded from CI.

```bash
git add CloudPoint/Capture CloudPointTests/CameraFrameSourceTests.swift
git commit -m "feat: capture sampled camera frames"
```

### Task 7: CPC Validation and Metal Point Renderer

**Files:**
- Create: `CloudPoint/Rendering/PointChunk.swift`
- Create: `CloudPoint/Rendering/PointCloudRenderer.swift`
- Create: `CloudPoint/Rendering/PointCloudView.swift`
- Create: `CloudPoint/Rendering/PointShaders.metal`
- Create: `CloudPointTests/PointChunkTests.swift`
- Create: `CloudPointTests/RendererBufferTests.swift`

**Interfaces:**
- Produces: `PointChunk.open(url:limits:) throws -> PointChunk` for window-owned CPC1 stride-24 files.
- Produces: `PointCloudRenderer.append(_:)`, `setConfidenceThreshold(_:)`, and camera controls.
- Consumes: CPC files emitted by either mock or real engine.

- [ ] **Step 1: Write binary-validation tests**

Create fixture bytes for one valid point and assert its position, RGBA, confidence, flags, and full UInt32 source frame. Add separate tests for bad magic, version, stride, declared count/file-size mismatch, NaN position, and a count beyond configured limits.

- [ ] **Step 2: Implement CPC parsing**

Memory-map read-only data, validate the 32-byte header before multiplication, use overflow-checked size arithmetic, and validate vertices in bounded chunks. Expose the vertex bytes without copying after validation.

- [ ] **Step 3: Write renderer-buffer tests**

Append two chunks and assert point counts and draw ranges. Lower the five-million display limit in the test to ten points and assert deterministic compaction returns the same source indices across runs.

- [ ] **Step 4: Implement Metal drawing**

Create a single render pipeline using `.point`; the vertex shader applies view/projection matrices and passes packed RGBA/confidence. The fragment shader discards confidence below the current threshold and draws circular point sprites. Grow shared vertex buffers geometrically, cap displayed points at five million, and retain only validated chunks.

- [ ] **Step 5: Run and commit**

Run: `scripts/test-native`

Expected: all CPC and buffer tests PASS; `xcodebuild build` compiles `PointShaders.metal` with the downloaded toolchain.

```bash
git add CloudPoint/Rendering CloudPointTests/PointChunkTests.swift CloudPointTests/RendererBufferTests.swift
git commit -m "feat: render validated point-cloud chunks"
```

### Task 7.5: Reconcile Protocol, Manifest, and Window Ownership

**Files:**
- Modify: `CloudPoint/Domain/ProjectModels.swift`
- Modify: `CloudPoint/Persistence/ProjectManifest.swift`
- Modify: `CloudPoint/Engine/ReconstructionEngine.swift`
- Modify: `CloudPoint/Engine/MockReconstructionEngine.swift`
- Modify: `CloudPoint/Engine/WorkerProtocol.swift`
- Modify: `CloudPoint/Engine/WorkerProcess.swift`
- Modify: `CloudPointMockWorker/main.swift`
- Test: focused manifest/protocol/mock/process/CPC suites

**Interfaces:**
- Produces: canonical UInt32 `ResumeCheckpoint`, `FrameArtifacts`, and
  window-owned `WindowResult` domain values reused by protocol and manifest code.
- Produces: manifest format 2, a pure `PendingWindowAccumulator`, and tested exact
  checkpoint derivation over ordered committed artifacts.
- Produces: framed-stdio disposition/canonical-JSON fixtures and protocol-readiness
  heartbeat supervision.

- [ ] **Step 1: Freeze shared values and complete configuration**

`ProjectDescriptor` carries `resumeCheckpoint: ResumeCheckpoint?`. The checkpoint
has UInt32 `lastCommittedFrameIndex`, actual `replayFromFrameIndex`, and
`nextWindowIndex`. `EngineConfiguration` persists UInt32 `scaleFrames`,
`windowSize`, `windowOverlap`, `keyframeInterval`, and
`cameraRefinementIterations`, plus finite positive Double `confidenceThreshold` and
`voxelSize`. Defaults are 8, 32, 8, 1, 4, 1.5, and 0.01 respectively. Validate
window size `1...1024`, scale frames
`1...windowSize`, overlap less than window size (zero is legal), positive
keyframe/refinement values,
and positive finite confidence/voxel values.

`FrameArtifacts` carries UInt32 frame/window IDs, depth/confidence/geometry paths,
and finite nonnegative Double duration. `WindowResult` carries UInt32 window index,
inference start, inclusive unique-output start/end, CPC path, 16 finite Double
row-major alignment values, UInt32 last processed frame, UInt64 inlier count, and
finite nonnegative duration. `EngineEvent` keeps frame and window completion
separate and uses UInt64 queued/processed/model-progress counts. Structured async
errors preserve code, message, recoverability, and canonical detail values.
`queuedFrames` is cumulative unique-output admission this invocation, not pending
depth; replay is excluded, `processedFrames <= queuedFrames`, and backlog is their
difference.

- [ ] **Step 2: Reset and validate manifest v2 plus checkpoint derivation**

Create only format v2 and reject v1 through the actionable unsupported-version
path. A completed window stores its full window result plus ordered committed frame
artifacts. Validate safe paths, finite nonnegative timestamps/durations, exactly 16
finite transform values, matching window IDs, inclusive ranges, artifact uniqueness,
and strictly increasing source/window order while permitting source gaps.

Derive the resume checkpoint from the final completed window and the final
`max(windowOverlap, 1)` actual committed artifacts across window boundaries.
`replayFromFrameIndex` is the first selected source index, never subtraction;
`nextWindowIndex` is checked addition. Test gapped frames, zero overlap,
cross-window selection, UInt32 maximums, and next-window overflow.

- [ ] **Step 3: Freeze protocol commands, events, and response ownership**

`beginSession` carries the nullable checkpoint object. `configure` includes
`voxelSize`. `frameCompleted` contains only frame/window IDs, depth/confidence/
geometry paths, and duration. `windowCompleted` contains window index, inference
start, unique-output bounds, CPC path, 16-number transform, last processed index,
inliers, and duration. Require
`inferenceFrameStart <= frameStart <= frameEnd <= lastProcessedFrameIndex`.
`hello.supportedProtocolVersions` is `[UInt32]` and must contain 1.

The real integration launches
`cloudpoint-worker serve --project ABSOLUTE_PATH --model ABSOLUTE_PATH`, sends
framed commands on stdin, receives framed responses/events on protocol-only stdout,
and captures diagnostics-only stderr. There is no listener or network transport.

Canonical JSON uses sorted keys, lowercase UUIDs, no whitespace, finite
shortest-round-trip typed Doubles, integral values without `.0`, normalized negative
zero, and lowercase `e` without `+` or redundant exponent zeroes. Relayed raw
numeric tokens in error details retain their exact lexeme. Encoders emit and
decoders require lowercase canonical UUID text.
Test null/full checkpoints, nested missing/unknown fields, integral/nonintegral/
exponent Doubles, UUID casing, configuration, both completion events, and raw detail
lexemes.

Disposition is explicit: invalid lengths/truncation close without response; invalid
JSON or no complete recoverable header gets at most one async fault then close; a
complete header with bad type/payload gets exactly one command error and leaves the
transport/state open; an unsupported version gets one flushed command error then
close. Every decoded valid command gets exactly one ACK/error.

- [ ] **Step 4: Freeze native pending-window and worker output transactions**

`PendingWindowAccumulator` accepts artifacts by window and finalizes only when a
matching window result is paired with the ordered expected unique output IDs selected
from persisted/enqueued frame records. Require exact ID-list equality; reject
duplicate, missing, extra, cross-window, and out-of-order artifacts. Native Task 8
will atomically persist the finalized transaction and only then publish the CPC.

Canonical paths are `Predictions/%08u.depth-f16`,
`Predictions/%08u.confidence-f16`, `Predictions/%08u.geometry.json`, and
`Points/window-%08u.cpc`, where UInt32 IDs may expand to ten digits. Exclusive
siblings are `.<final-basename>.<lowercase-UUID>.partial`; promotion is no-clobber.
Before begin ACK, worker/mock read manifest references and descriptor-relatively
remove only exact-pattern partials and unreferenced exact-pattern finals, without
following symlinks or deleting unknown names. Native never removes guessed output
orphans. The worker's manifest access is read-only; only native writes it.

- [ ] **Step 5: Correct mocks, replay, ACKs, and heartbeat arming**

Mocks write artifacts before events, keep confidence above 1.5, emit frame artifacts
then window CPC, consume replay through the committed boundary without events or
writes, accept strictly increasing source IDs beginning at the actual replay start
before new frames, and number new windows from `nextWindowIndex`. Source gaps are
legal. Replay is excluded from CPC
and queued/processed counters. ACK always precedes lifecycle output: hello completes
negotiation and precedes the immediate heartbeat; configure stores the complete
validated configuration; begin follows manifest/checkpoint validation and exact
cleanup; enqueue means bounded queue admission; finish closes admission; pause
records a request, resume releases it after cooperative quiescence with the queue
retained, and cancel/shutdown ACKs precede later completion, cleanup, or exit.

The mock worker waits for hello, emits its sole ACK, then an immediate heartbeat
before ready/model/lifecycle output. `WorkerProcess.markProtocolReady()` is
idempotent, launcher handoff does not arm the timer, and the first heartbeat may
arm/reset it. The exact child environment is the allowlist above, not a merge with
inherited values. The real worker contract reserves one dedicated serial executor
for MLX/model work so framing, controls, and an at-most-five-second heartbeat cadence
stay live during loading, inference, finalization, and idle.

- [ ] **Step 6: Verify and commit the reconciled boundary**

Run focused manifest/protocol/mock/process/CPC tests, `scripts/test-native`, and
`git diff --check`. Cross-language work must not begin until an independent review
approves the base-to-head diff.

```bash
git add CloudPoint CloudPointMockWorker CloudPointTests project.yml docs/superpowers
git commit -m "fix: reconcile reconstruction protocol ownership"
```

### Task 8: Runnable Vertical-Slice Workspace

**Files:**
- Create: `CloudPoint/Workspace/SessionController.swift`
- Create: `CloudPoint/Workspace/WorkspaceViewModel.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Modify: `CloudPoint/App/CloudPointApp.swift`
- Create: `CloudPointTests/WorkspaceViewModelTests.swift`
- Create: `CloudPointTests/TestSupport/WorkspaceTestHarness.swift`
- Create: `docs/DEVELOPMENT.md`

**Interfaces:**
- Consumes: document package, frame sources, `ReconstructionEngine`, and renderer.
- Produces: the four-region UI and end-to-end mock reconstruction flow.

- [ ] **Step 1: Write failing orchestration tests**

Implement `WorkspaceTestHarness` in the test target with injected `MockCameraFrameSource`, a manually-drained reconstruction engine, a temporary project package, and its `WorkspaceViewModel`. `emitPersistedFrames(count:)` must write real deterministic JPEG fixtures before returning; `completeQueuedFrames()` releases the engine's suspended queue so the test controls backlog timing.

```swift
@MainActor
func testStoppingCaptureLetsBacklogFinish() async throws {
    let harness = WorkspaceHarness.makeMock()
    try await harness.viewModel.startCamera(deviceID: "fixture")
    await harness.camera.emitPersistedFrames(count: 3)
    try await harness.viewModel.stopCapture()
    XCTAssertFalse(harness.viewModel.snapshot.isCapturing)
    XCTAssertEqual(harness.viewModel.snapshot.capturedCount, 3)
    await harness.engine.completeQueuedFrames()
    XCTAssertEqual(harness.viewModel.snapshot.phase, .completed)
    XCTAssertEqual(harness.viewModel.snapshot.processedCount, 3)
}
```

Add tests for recording import, pause/resume, cancel, engine failure, project close, and reopening a mock-completed package.

- [ ] **Step 2: Implement `SessionController` and `WorkspaceViewModel`**

Make the view model `@MainActor`, expose a single `WorkspaceSnapshot`, and delegate mutable workflow state to `SessionController` actor. The controller owns `SessionState`, ordered persisted/enqueued frame records, one monotonically revised manifest snapshot, and manifest mutations; each method applies a legal domain event and writes the manifest before returning its snapshot. Consume engine events in one view-model-owned task that starts before engine calls and cancel it on document close. Feed frame artifacts through `PendingWindowAccumulator`; on matching window completion select the exact expected unique output IDs, atomically commit artifacts plus window, and only then publish the CPC. Persist manifest changes before counts. Event generations and revisions prevent reentrant stale work from overwriting newer state.

Treat queued count as cumulative successful unique-output admissions for the current
invocation: increment only after enqueue returns its ACK-equivalent, never decrement,
exclude replay, and derive backlog as queued minus processed.

- [ ] **Step 3: Implement the four-region workspace**

Use `NavigationSplitView` for source and inspector panels, `PointCloudView` in the center, an optional `CameraPreviewView` overlay, and a bottom safe-area timeline. Provide **Open Recording**, **Use Camera**, **Stop Capture**, **Pause**, **Resume**, **Cancel**, reset-view, point-size, confidence, and sampling-rate controls. Disable controls from state, not ad-hoc booleans.

- [ ] **Step 4: Wire the document app and mock engine launch argument**

Use `DocumentGroup(newDocument:)`. In Debug builds, `--mock-engine` selects `MockReconstructionEngine`; without it, show a non-fatal “Lingbot engine not installed yet” setup state. The app must not pretend real reconstruction is available.

- [ ] **Step 5: Document and verify the native slice**

Document:

```bash
scripts/bootstrap
open CloudPoint.xcodeproj
xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS'
open "$(find ~/Library/Developer/Xcode/DerivedData -path '*/Build/Products/Debug/CloudPoint.app' -print -quit)" --args --mock-engine
```

Run `scripts/test-native`, launch with `--mock-engine`, import the fixture recording, and confirm the deterministic colored plane appears progressively.

- [ ] **Step 6: Commit the working native foundation**

```bash
git add CloudPoint CloudPointTests docs/DEVELOPMENT.md project.yml
git commit -m "feat: deliver native CloudPoint vertical slice"
```

## Plan Completion Check

Run:

```bash
scripts/test-native
git status --short
```

Expected: Debug and Release builds succeed, all native tests pass, the mock-worker lifecycle tests pass, the working tree is clean, and the app can import the committed fixture into a progressively rendered deterministic point cloud.

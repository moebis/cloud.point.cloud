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
CloudPoint/Engine/WorkerProcess.swift             subprocess/socket supervision
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
- Produces: `PersistedFrame(index:sourceTimestamp:relativePath:)`.
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

Define `SessionPhase: String, Codable, Sendable` with `empty`, `preparing`, `ready`, `importing`, `capturing`, `processing`, `paused`, `finalizing`, `completed`, `cancelled`, and `failed`. Define explicit transition switches; reject every unspecified pair with `SessionTransitionError.illegal(from:event:)`. Define IDs as UUIDs, frame timestamps as `Double` seconds, and paths as package-relative strings.

- [ ] **Step 4: Write failing manifest recovery tests**

Create `TemporaryProjectPackage` as an XCTest support value that uses
`FileManager.default.temporaryDirectory.appending(path: UUID().uuidString).appendingPathExtension("cloudpoint")`, creates the four package subdirectories, and removes its URL in `deinit`. Add deterministic `fixture()` factories in the test target for `ProjectManifest` and `CompletedWindow`. Then write a manifest, leave a `Points/window-1.cpc.partial`, reload, and assert the incomplete path is omitted while the last completed window remains recoverable:

```swift
func testLoadIgnoresPartialArtifacts() throws {
    let package = try TemporaryProjectPackage.make()
    var manifest = ProjectManifest.fixture()
    manifest.completedWindows = [.fixture(index: 0)]
    try manifest.writeAtomically(to: package.url)
    FileManager.default.createFile(
        atPath: package.url.appending(path: "Points/window-1.cpc.partial").path,
        contents: Data("partial".utf8)
    )
    let loaded = try ProjectManifest.load(from: package.url)
    XCTAssertEqual(loaded.completedWindows.map(\.index), [0])
}
```

- [ ] **Step 5: Implement versioned manifest and document package**

Use format version `1`, sorted-key JSON, ISO-8601 dates, and same-directory `Manifest.json.partial` followed by `FileManager.replaceItemAt`. `CloudPointDocument` must conform to `ReferenceFileDocument`, declare UTType `cloud.point.cloud.project`, and create `Frames`, `Predictions`, `Points`, and `Logs` directories on first save.

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

Use `AsyncThrowingStream<EngineEvent, Error>` and an actor-owned continuation. Add a private `MockCPCWriter` in `MockReconstructionEngine.swift` so this task has no forward dependency on Task 7. It writes the exact approved 32-byte CPC1 header and 24-byte vertex records using explicit little-endian integers and bit patterns. `MockReconstructionEngine.enqueue` creates a deterministic 64-by-64 RGB plane centered on the origin, writes the valid CPC chunk beneath the project `Points` directory, and emits `.frameCompleted(FrameResult(...))`. Its color and Z offset derive from the frame index so tests and screenshots are reproducible. Pause suspends consumption; cancel closes the stream with `.cancelled`.

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

Use a four-byte big-endian unsigned length and sorted-key JSON. Reject zero-length payloads, payloads over 1 MB, unknown protocol versions, duplicate command IDs, and trailing malformed JSON. Run focused tests and expect PASS.

- [ ] **Step 3: Write a failing worker lifecycle test**

Launch `CloudPointMockWorker --mode heartbeat`, wait for `.ready`, then terminate it and assert `WorkerProcess` emits `.processExited(status:)` exactly once and closes file descriptors.

- [ ] **Step 4: Implement process supervision and mock worker**

Use `Process`, explicit pipes for stdin/stdout/stderr, a process-group termination path, and an actor for decoder state. The mock worker supports `normal`, `heartbeat`, `crash-after-ready`, and `silent` modes. A five-second heartbeat interval and three missed heartbeats produce `WorkerProcessError.unresponsive`.

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

Use `AVAssetReaderTrackOutput`, request native bi-planar YUV when supported, transform orientation before JPEG encoding, write `Frames/%08d.jpg.partial`, synchronize, rename, then emit `PersistedFrame`. Do not load the complete recording into memory.

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
- Produces: `PointChunk.open(url:limits:) throws -> PointChunk` for CPC1 stride-24 files.
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

Make the view model `@MainActor`, expose a single `WorkspaceSnapshot`, and delegate mutable workflow state to `SessionController` actor. The controller owns `SessionState`, ordered queued frames, and manifest mutations; each method applies a legal domain event and writes the manifest before returning its snapshot. Consume engine events in one view-model-owned task and cancel that task on document close. Persist manifest changes before publishing counts to the UI.

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

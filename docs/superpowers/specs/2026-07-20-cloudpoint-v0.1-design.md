# CloudPoint 0.1 Design

**Status:** Approved architecture and product scope; technical design prepared for final review
**Date:** 2026-07-20
**Working product name:** CloudPoint
**Target:** Apple Silicon Macs running macOS 15 or later

## 1. Purpose

CloudPoint is a native Mac application that turns an imported video or a live camera feed into a colored 3D point cloud and estimated camera trajectory. Version 0.1 prioritizes a real, locally testable reconstruction loop over real-time throughput. Capture may run faster than inference, and reconstruction may continue after capture stops.

The first inference engine reproduces Lingbot Map's feed-forward reconstruction pipeline on Apple Silicon through MLX. The app itself remains independent of that implementation through a narrow `ReconstructionEngine` boundary so a native MLX Swift engine can replace the initial Python MLX worker later.

## 2. Success Criteria

Version 0.1 is successful when all of the following are true:

1. The app imports MOV and MP4 recordings and obtains frames at deterministic timestamps.
2. The app previews a Mac or Continuity Camera at capture speed while sampling frames into a disk-backed reconstruction queue.
3. The MLX worker produces depth, confidence, camera pose, intrinsics, colored point chunks, and progress events from real Lingbot Map weights.
4. The point cloud and camera path appear incrementally in a responsive native Metal viewport.
5. A stopped live capture continues processing its queued frames until complete.
6. A session can be saved as a `.cloudpoint` package, reopened, and resumed from its last completed reconstruction window.
7. The completed cloud exports as binary PLY and the camera trajectory exports as JSON.
8. The app remains responsive during capture and inference, bounds in-memory queues, and records actionable failures.
9. On the development M1 Ultra with 64 GB unified memory, peak process memory remains below 48 GB and the renderer sustains at least 30 FPS with two million displayed points.
10. Golden-reference tests establish that the MLX model's geometry agrees with the pinned PyTorch implementation within the tolerances in Section 14.

Reconstruction FPS is measured and displayed but is not a release gate for 0.1.

## 3. Product Scope

### Included

- Native SwiftUI application shell and macOS document workflow.
- MOV/MP4 import through AVFoundation.
- Mac and Continuity Camera capture through AVFoundation.
- Live preview independent of reconstruction throughput.
- Disk-backed sampled-frame queue with separate captured and processed counts.
- MLX-based Lingbot Map inference in a supervised local worker process.
- Progressive colored point-cloud and camera-trajectory rendering with Metal.
- Orbit, pan, zoom, reset-view, point-size, confidence, and timeline controls.
- Pause, resume, stop-capture, cancel-processing, and retry controls.
- `.cloudpoint` project persistence.
- Binary PLY point-cloud export and JSON trajectory export.
- Model download, verification, conversion, and health reporting.
- Local-only processing after the model has been downloaded.

### Excluded

- Guaranteed real-time reconstruction.
- Dense mesh generation, texturing, Gaussian splats, or USDZ output.
- App Store packaging and sandbox-hardening.
- iPhone/iPad ARKit or LiDAR streaming.
- Multi-camera synchronization.
- Semantic labels or object detection.
- Editing or merging separately captured maps.
- A user-facing arbitrary-model loader.
- Training or fine-tuning.

## 4. Architecture

CloudPoint uses a native front end and a replaceable local reconstruction engine.

```text
AVFoundation camera / AVAssetReader
                 |
                 v
        Frame acquisition actor
        - preview stream
        - deterministic sampler
        - project frame writer
                 |
                 v
        ReconstructionEngine protocol
                 |
                 v
       supervised Python MLX worker
       - Lingbot model + KV cache
       - direct/windowed orchestration
       - confidence filtering
       - depth unprojection
                 |
                 v
        project prediction chunks
                 |
                 v
       Metal point-cloud renderer
       + trajectory/timeline UI
```

The app never imports Python or MLX into its process. `PythonMLXEngine` owns a worker subprocess and implements the same Swift protocol that a future `NativeMLXEngine` will implement.

The worker is an internal implementation detail. Views consume session state and geometry events, never worker-specific messages.

For developer version 0.1, the worker runs from a repository-managed Python 3.12 virtual environment produced from a fully pinned lock file. The Xcode build receives the worker-runtime path through a local build setting and does not silently use whichever `python3` happens to be on `PATH`. Self-contained runtime bundling is deferred with App Store packaging.

## 5. Native App Components

### `CloudPointApp`

Creates document windows, application commands, the model-setup window, and dependency injection. Version 0.1 uses the working bundle identifier `cloud.point.cloud.CloudPoint`.

### `CloudPointDocument`

Owns a `.cloudpoint` package and its `ProjectManifest`. It coordinates package creation, atomic manifest writes, reopening, recovery, and export eligibility.

### `SessionController`

An actor that owns the session state machine and is the single coordinator for capture, sampling, inference, persistence, and renderer updates. UI code sends intent to this actor and observes immutable snapshots.

### `VideoFrameSource`

A protocol implemented by:

- `AssetFrameSource`, backed by `AVAssetReader` for deterministic recording timestamps.
- `CameraFrameSource`, backed by `AVCaptureSession` and `AVCaptureVideoDataOutput` for live frames.

Both emit a shared `CapturedFrame` containing an index, presentation timestamp, orientation, pixel buffer, source metadata, and optional camera intrinsics.

### `FrameSampler`

Selects frames at a requested rate from 1 through 10 FPS, defaulting to 5 FPS. It writes selected frames as oriented JPEG files at quality 0.92. The sampler uses timestamps rather than frame counts, so variable-frame-rate recordings behave deterministically.

Live preview does not pass through the sampler. Sampled live frames are written to disk immediately, allowing inference to lag without growing an in-memory frame queue.

### `ReconstructionEngine`

The Swift boundary is:

```swift
protocol ReconstructionEngine: Sendable {
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
```

`PythonMLXEngine` launches, monitors, and terminates the initial worker. No view or persistence type references Python, framed stdio, NumPy, or MLX.

### `PointCloudRenderer`

An `MTKView`-backed renderer draws `.point` primitives from append-only GPU buffers. It also draws a trajectory polyline, selected camera frustum, grid, and origin axes. Camera controls are implemented in native Swift and do not depend on RealityKit or SceneKit.

The renderer displays at most five million points. When the project contains more, deterministic voxel compaction selects the displayed subset while keeping the full reconstructed data on disk.

### `ProjectExporter`

Streams project point chunks into a binary little-endian PLY without loading the complete cloud into memory. It also emits `trajectory.json` with timestamps, 4-by-4 camera-to-world matrices, intrinsics, and source-frame indices.

## 6. User Interface

The main document window has four stable regions:

1. **Left source panel:** recording import, camera selection, live preview, source metadata, and capture controls.
2. **Center viewport:** Metal point cloud, trajectory, camera frustum, grid, axes, and empty/loading/error states.
3. **Right inspector:** sampling rate, confidence threshold, point size, rendering budget, reconstruction status, and model health.
4. **Bottom timeline:** captured-frame, cumulative admitted-frame, and processed-frame counts; current window; elapsed time; backlog (`admitted - processed`); and pause/resume/cancel controls.

Opening an empty document first presents two primary actions: **Open Recording** and **Use Camera**. If the worker or model is unavailable, either action routes through the setup sheet before creating a reconstruction session.

The UI always distinguishes these states:

- Capturing video.
- Waiting in the reconstruction backlog.
- Reconstructing a frame or window.
- Finalizing/exporting.
- Complete, paused, cancelled, or failed.

It never describes causal inference as real-time unless measured throughput actually meets the incoming sample rate.

## 7. Project Package

A `.cloudpoint` document is a package directory:

```text
Example.cloudpoint/
  Manifest.json
  Frames/
    00000001.jpg
    00000002.jpg
  Predictions/
    00000001.depth-f16
    00000001.confidence-f16
    00000001.geometry.json
  Points/
    window-00000001.cpc
  Logs/
    worker.log
```

`Manifest.json` uses format version 2. Version 0.1 creates only v2 packages and
rejects v1 through an actionable unsupported-version error rather than silently
decoding it as v2. It contains:

- Format version.
- Project identifier and creation/update dates.
- Source type and non-sensitive source metadata.
- Complete sampling and reconstruction configuration: UInt32 scale-frame, window,
  overlap, keyframe, and camera-refinement values plus finite positive Double
  confidence threshold and voxel size (default `0.01`).
- Model identifier, source revision, converted-weight checksum, and engine version.
- UInt32 per-frame indices, finite nonnegative timestamps, and relative paths.
- Strictly increasing UInt32 window indices; inference and inclusive unique-output
  bounds; window CPC path; exactly 16 finite Double Sim3 values; last processed
  frame, UInt64 inlier count, nonnegative duration, and ordered committed frame
  artifacts.
- Captured, queued, processed, failed, and exported counts.
- Session state; the recoverable checkpoint is derived from ordered completed-window
  artifacts rather than stored as an arithmetic marker.

Native code is the sole manifest writer. It validates every package-relative path,
timestamp, duration, transform, range, and ordering rule before writing a sibling
temporary file, synchronizing, and atomically renaming it. Source-frame gaps are
legal. The worker may read committed manifest state for replay and orphan cleanup
but never mutates it.

Raw source recordings are not copied into the package. Only sampled frames and derived data are stored. The app estimates required space before starting and warns when the package exceeds 10 GB or the volume has less than 20 GB free.

## 8. Worker Protocol

The app starts
`cloudpoint-worker serve --project ABSOLUTE_PATH --model ABSOLUTE_PATH` with an
exactly replaced child environment containing only `HOME`, `TMPDIR`,
`PATH=/usr/bin:/bin`, `PYTHONNOUSERSITE=1`, `PYTHONHASHSEED=0`, `LC_ALL=C`, and
`LANG=C` (plus explicit mock controls in tests). Commands use framed stdin, responses
and events use framed stdout, and diagnostics use stderr. The worker creates no
listener, binds no network port, and makes no network request. Messages use a
four-byte big-endian length followed by canonical UTF-8 JSON; large frames and
tensors never travel through JSON and messages reference files inside the project
package.

### App-to-worker commands

- `hello`
- `configure`
- `beginSession`
- `enqueueFrame`
- `finishInput`
- `pause`
- `resume`
- `cancel`
- `shutdown`

Each command has a lowercase canonical UUID, protocol version, project ID, and exact
typed payload. Encoders emit and decoders require lowercase UUID text; unknown or
missing fields are rejected at every level. `hello.supportedProtocolVersions` is a
UInt32 array containing `1`. Generated JSON has
sorted keys, no insignificant whitespace, finite shortest-round-trip Double tokens,
integral Doubles without `.0`, normalized negative zero, lowercase `e` without `+`
or redundant exponent zeroes. Relayed raw numeric
tokens inside structured error `details` preserve their original lexemes.

`configure` carries UInt32 `scaleFrames`, `windowSize`, `windowOverlap`,
`keyframeInterval`, and `cameraRefinementIterations`, plus finite positive Double
`confidenceThreshold` and `voxelSize`. `beginSession` carries
`resumeCheckpoint: null | ResumeCheckpoint`, whose fields are UInt32
`lastCommittedFrameIndex`, `replayFromFrameIndex`, and `nextWindowIndex`. A decoded
valid command receives exactly one acknowledgement or
structured error. Zero, oversized, or incomplete frames close without a response;
invalid JSON or an unrecoverable header produces at most one asynchronous protocol
fault before close; a recoverable complete header with bad type/payload gets one
command error without state mutation; an unsupported version gets one flushed
command error and then closes.

Configuration validation requires window size `1...1024`, scale frames
`1...windowSize`, overlap less than window size (zero is legal), positive
keyframe/refinement counts, and positive finite confidence/voxel values.

### Worker-to-app events

- `ready`
- `modelProgress`
- `frameStarted`
- `frameCompleted`
- `windowCompleted`
- `sessionCompleted`
- `paused`
- `cancelled`
- `warning`
- `error`
- `heartbeat`

`frameCompleted` identifies exactly the depth, confidence, and geometry artifacts
for one unique output frame. `windowCompleted` separately owns one CPC containing
only the window's unique output range; it carries the inference start, inclusive
unique-output bounds, global alignment, last processed index, inlier count, and
duration. Replay/overlap context emits no completion events and is never duplicated
in CPC output.

Wire frame/window IDs and bounds are UInt32; queued/processed/model-progress counts
and inliers are UInt64; durations, heartbeat time, and transforms are finite Double.
Protocol queued/processed/window counts cover unique current-invocation output only
and exclude replay context. `queuedFrames` is cumulative descriptors admitted, not
pending depth; `processedFrames <= queuedFrames`, and backlog is their difference.

ACKs precede lifecycle events. Hello ACK completes version negotiation and is
followed immediately by a heartbeat; configure ACK means the complete configuration
was stored; begin ACK means checkpoint/manifest validation and exact orphan cleanup
succeeded. Enqueue ACK means admission to the bounded ordered descriptor queue, not
completed inference. Finish ACK closes admission only; pause ACK records the request
before later quiescence with the queue retained; resume ACK releases paused work;
cancel ACK records cancellation before later safe-boundary cleanup/event; shutdown ACK is flushed
before active-session quiescence/cleanup or direct idle exit.

Native owns a pending-window accumulator. It groups frame artifacts by window and
commits them with a matching window completion only when their ordered IDs exactly
match the expected unique output IDs selected from persisted/enqueued frames.
Bounds alone are insufficient because source indices may have gaps. The atomic
manifest commit occurs before the CPC is published to the renderer.

Immediately after the `hello` ACK the worker emits a heartbeat, then emits at least
one every five seconds during loading, inference, finalization, and idle. MLX work
runs on one dedicated serial executor so framing, control, and heartbeat scheduling
remain live. Process launch does not arm the native watchdog: protocol readiness
after the hello ACK, or the first valid heartbeat, arms it idempotently. Three missed
intervals after arming mark the engine unresponsive and offer restart-from-window.

## 9. Point Chunk Format

`.cpc` is a private, versioned, little-endian format designed for memory mapping:

```text
Header:
  magic             4 bytes  "CPC1"
  version           UInt16   1
  vertexStride      UInt16   24
  pointCount        UInt64
  frameStart        UInt32
  frameEnd          UInt32
  reserved          8 bytes

Vertex, repeated pointCount times:
  position          3 x Float32
  rgba              4 x UInt8
  confidence        Float16
  flags             UInt16
  sourceFrame       UInt32
```

All numeric fields are validated before mapping. Invalid magic, version, stride, file size, NaN position, or out-of-range count rejects the chunk without exposing it to Metal.

## 10. Model and Inference Pipeline

### Model provenance and preparation

Version 0.1 supports only the official `robbyant/lingbot-map` long-sequence checkpoint. Model preparation is pinned to Hugging Face revision `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`; `lingbot-map-long.pt` must be exactly 4,632,303,465 bytes with SHA-256 `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`. The reference source implementation is pinned separately to Git commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`.

The setup process:

1. Downloads `lingbot-map-long.pt` from the pinned Hugging Face repository revision.
2. Verifies the expected file size and SHA-256 before deserialization.
3. Runs the converter only on that exact verified artifact in a dedicated temporary working directory, without credentials or inherited sensitive environment variables.
4. Converts tensors to Float16 SafeTensors with explicit handling for convolution and transposed-convolution layouts.
5. Produces a manifest listing every source key, destination key, shape, dtype, and checksum.
6. Rejects missing, extra, duplicated, or shape-mismatched tensors.
7. Stores converted weights in Application Support, not in the app bundle or project package.

The app and worker never deserialize an arbitrary `.pt` file. The converter first attempts restricted `weights_only` loading. If the pinned checkpoint requires ordinary pickle loading, only the exact verified digest above is accepted and the setup UI discloses that trusted-artifact conversion step. Converted weights are not redistributed in version 0.1; each developer/tester downloads and converts the upstream checkpoint locally while checkpoint redistribution terms are clarified.

### Preprocessing

- Apply source orientation and discard alpha.
- Convert to sRGB RGB.
- Resize the longest dimension to 518 pixels.
- Crop or pad dimensions to multiples of the 14-pixel patch size using the pinned upstream rules.
- Normalize pixels exactly as the pinned Lingbot implementation.
- Retain a transform from model pixels to source pixels for inspection and export metadata.

### Reconstruction defaults

- Float16 MLX computation.
- Eight initial anchor/scale frames.
- 32-frame windows.
- Eight overlapping frames between windows.
- Keyframe interval one.
- Four iterative camera refinement passes.
- Confidence threshold 1.5.
- Voxel size 0.01 reconstruction units.
- Direct processing when the session fits one window; overlapping-window processing otherwise.

These defaults are recorded in every project. Advanced model/cache controls remain hidden in 0.1; only sampling rate, confidence threshold, and point-display settings are user-facing.

### Outputs and geometry

The first engine enables the depth and camera heads and disables the optional world-point head. Each processed frame produces:

- A dense depth map.
- Dense depth confidence.
- Nine-value camera encoding.
- Decoded intrinsics.
- Camera-to-world pose.

Depth is unprojected using the decoded intrinsics and transformed by camera-to-world pose. Points below the stored confidence floor or with non-finite/non-positive depth are discarded. RGB comes from the corresponding preprocessed frame.

Adjacent windows are aligned with a robust Sim(3) estimate over all valid overlapping camera centers and sampled high-confidence 3D correspondences. The implementation must not use only one overlapping pose or one median-depth ratio. Degenerate overlap fails the window with a recoverable alignment error rather than silently accepting identity.

The worker performs deterministic voxel reduction before writing each `.cpc` chunk. Raw depth, confidence, pose, and intrinsics remain available so point chunks can be regenerated for changed filtering rules.

Lingbot reconstruction scale is treated as internally consistent but not guaranteed metric scale. The UI and exported PLY metadata describe coordinates as reconstruction units, never meters, unless a later calibration feature establishes a metric transform.

## 11. Recording and Live Data Flow

### Imported recording

1. `AssetFrameSource` validates that AVFoundation can decode the selected asset.
2. `FrameSampler` computes deterministic target timestamps at the selected FPS.
3. Frames are decoded, oriented, persisted, entered atomically in the manifest, and enqueued.
4. The worker loads its model once, establishes anchor frames, and processes windows.
5. Frame artifacts accumulate provisionally; a matching completed window commits
   its exact unique-frame set atomically, then its validated point chunk is appended
   to the renderer and committed progress advances.
6. `finishInput` closes the input stream, the final partial window is processed, and the package becomes exportable.

### Live camera

1. Camera permission and selected format are resolved before session creation.
2. The preview consumes capture frames directly and remains independent of inference.
3. The timestamp sampler persists frames at the selected FPS and enqueues only persisted paths.
4. The worker processes the FIFO disk-backed backlog. Captured and processed counts remain visible.
5. **Stop Capture** ends input but does not cancel reconstruction.
6. Reconstruction catches up, processes the final window, and finalizes the project.

Pause stops worker progress but does not implicitly stop a live capture. If capture continues while reconstruction is paused, the disk backlog and estimated size remain visible.

## 12. State and Recovery

The session state machine is:

```text
empty
  -> preparing
  -> ready
  -> importing | capturing
  -> processing
  -> paused
  -> finalizing
  -> completed | cancelled | failed
```

Capture and processing are orthogonal flags inside the active states, allowing camera capture to continue while processing trails behind.

Recovery occurs only at completed-window boundaries. Native validates manifest v2
and completed chunks, derives `lastCommittedFrameIndex` from the last window's
inclusive output end, derives the next window with checked addition, and selects the
last `max(windowOverlap, 1)` actual committed frame-artifact records across window
boundaries. The first selected source index becomes `replayFromFrameIndex`; it is
never computed by subtraction. The worker reads those committed artifacts from the
manifest, consumes replay through the committed boundary without writes/events or
counter increments, then processes strictly increasing new saved frames. Saved
sampled frames make both imported and formerly live sessions resumable as finite
recording-style jobs.

Canonical artifacts are `Predictions/%08u.depth-f16`,
`Predictions/%08u.confidence-f16`, `Predictions/%08u.geometry.json`, and
`Points/window-%08u.cpc` (minimum width eight; UInt32 may require ten digits).
Exclusive siblings use `.<final-basename>.<lowercase-UUID>.partial`. Before
promotion, the worker synchronizes and never clobbers an existing final. Before
`beginSession` ACK, the worker compares exact-pattern outputs with the
manifest-referenced set and descriptor-relatively removes only exact-pattern partials and
unreferenced canonical finals. It does not follow symlinks or delete unknown names;
native never guesses orphan files.

## 13. Error Handling and Safety

### User-correctable errors

- Camera permission denied or camera disconnected.
- Unsupported or corrupt recording.
- Insufficient project disk space.
- Model absent, checksum mismatch, or conversion incomplete.
- Worker runtime missing or incompatible.
- Export destination unavailable.

These retain the document and present an action that resolves the specific problem.

### Engine failures

- Worker crash or missed heartbeat.
- MLX allocation failure.
- Invalid model output.
- Degenerate cross-window alignment.
- Invalid or truncated point chunk.

On a worker crash, the app records stderr, preserves the project, terminates the process group, and offers restart from the last completed window. On an allocation failure, it retries the current window once with a 16-frame window and four-frame overlap after a clean worker restart. A second failure stops processing without deleting captured frames.

All paths passed to the worker are resolved beneath the project or managed model directory. Protocol messages have a 1 MB maximum. Frame dimensions, point counts, tensor shapes, and generated file sizes are bounded before allocation.

The worker binds no TCP port and makes no network requests. Model download is performed by the native setup layer before worker launch.

## 14. Testing Strategy

### Swift unit tests

- Timestamp-based sampling for constant and variable frame rates.
- Session state transitions and illegal-transition rejection.
- Atomic manifest save and interrupted-save recovery.
- Framed stdin/stdout size limits, canonical JSON, response-disposition rules,
  acknowledgements, and malformed messages.
- Manifest-v2 checkpoint derivation, gapped source indices, ordered pending-window
  commits, and exact worker orphan cleanup.
- Worker heartbeat supervision and process cleanup.
- `.cpc` header/length/NaN validation.
- Depth unprojection with known intrinsics and pose.
- Sim(3) recovery with known scale, rotation, translation, noise, and outliers.
- PLY and trajectory export structure.
- Deterministic displayed-point compaction.

### Python unit tests

- Exact image preprocessing against pinned upstream fixtures.
- Strict checkpoint conversion with complete key/shape coverage.
- Convolution and transposed-convolution layout conversion.
- KV-cache append, skip, eviction, reset, and window boundaries.
- Pose decoding and camera convention.
- Depth unprojection and confidence filtering.
- Robust multi-overlap Sim(3) alignment and degeneracy detection.
- Protocol schema validation, cancellation, and atomic output writes.

### Differential model tests

A small fixed image sequence is evaluated by the pinned PyTorch SDPA implementation and the MLX worker using the same checkpoint.

- Float32 leaf-layer outputs: `rtol <= 1e-3`, `atol <= 1e-4`.
- Aggregator feature cosine similarity: at least `0.995` at selected layers.
- Float16 depth correlation: at least `0.99` with median relative error no greater than `0.03` on valid pixels.
- Pose translation relative error: no greater than `0.03` after shared scale normalization.
- Pose rotation geodesic error: no greater than `1.0` degree per fixture frame.
- Decoded intrinsics relative error: no greater than `0.01`.

Golden tests explicitly settle the repository's inconsistent camera convention by comparing reprojected points, not only matrix values.

### Integration and UI tests

- Mock worker covering ready, progress, pause, crash, retry, completion, and cancellation.
- Short recording import through AVFoundation.
- Mock camera source without requiring physical permission in CI.
- Document close/reopen/recover during active processing.
- Progressive renderer append and selected-frame synchronization.
- Real-model smoke test on the development Mac, opt-in because of model size and runtime.

### Performance checks on M1 Ultra

- Model load time and peak memory.
- Per-frame and per-window reconstruction latency.
- Sustained capture while inference is backlogged.
- Renderer FPS at one, two, and five million displayed points.
- PLY export time and peak memory.

## 15. Verification Gate for 0.1

The release gate is a single `scripts/verify` command that runs:

1. Swift formatting/lint checks selected during implementation.
2. Swift unit and integration tests.
3. Python worker tests.
4. Protocol schema compatibility checks.
5. A Debug and Release app build.

Manual acceptance on the M1 Ultra additionally requires:

1. Importing and completing a representative indoor recording.
2. Capturing a 60-second live camera session while processing trails behind.
3. Stopping capture and allowing the backlog to complete.
4. Closing and reopening a partially processed project, then resuming it.
5. Exporting PLY and opening it in an independent point-cloud viewer.
6. Confirming camera trajectory direction and reprojection against the source frames.
7. Recording peak memory, reconstruction FPS, renderer FPS, and any visible window seams.

## 16. Dependencies and Provenance

- SwiftUI, AppKit, AVFoundation, Metal, MetalKit, Accelerate, and simd are Apple frameworks.
- MLX is pinned to a tested version from Apple's `ml-explore/mlx` project.
- Lingbot Map source concepts and checkpoint provenance are pinned to exact upstream revisions and attributed under Apache-2.0.
- The unofficial MLX port may inform implementation and test fixtures but is not imported as a package or runtime dependency.
- Every third-party source, model, and fixture is recorded in `THIRD_PARTY_NOTICES.md` before distribution.
- The checkpoint is downloaded by the tester and is not redistributed until its repository metadata and model-card license ambiguity are resolved.

## 17. Evolution Path

After 0.1 proves reconstruction quality and establishes performance profiles:

1. Port the verified MLX model modules to MLX Swift behind `ReconstructionEngine`.
2. Replace file-path IPC with in-process tensors while retaining project formats and UI contracts.
3. Optimize cache layout, quantization, and static subgraphs from measured bottlenecks.
4. Add optional Core ML auxiliary models only where they improve measured behavior.
5. Add mesh/splat exports and ARKit capture as separate, independently specified features.

The 0.1 document format, renderer, capture layer, and engine protocol are intentionally useful before and after the native-engine replacement.

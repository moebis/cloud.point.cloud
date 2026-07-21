# SHARP Gaussian Reconstruction Mode Design

**Date:** 2026-07-21
**Target release:** CloudPoint 1.1.0
**Status:** Approved

## Objective

CloudPoint 1.1 adds an experimental Apple SHARP reconstruction mode that turns
one selected video frame or one camera capture into a metric 3D Gaussian
splatting scene on Apple silicon. The release also corrects the existing
LingBot point-cloud display orientation and makes camera mirroring explicit and
consistent.

SHARP is a single-image nearby-view synthesis model. It does not estimate a
shared trajectory or merge several video frames into one map. CloudPoint must
describe that distinction plainly and must never imply that independent SHARP
outputs form a registered multi-frame reconstruction.

Matrix3D, Depth Pro, ARKit capture, NeRF generation, and native Apple
photogrammetry are outside the 1.1 scope.

## User experience

### Source-first project creation

The welcome screen retains three primary actions: **Open Video**, **Use
Camera**, and **Open Project**. Selecting a recording or camera probes the
source first and then presents a **New Reconstruction** sheet before a project
is created.

The sheet contains two working mode cards:

- **Point Cloud** — the existing LingBot multi-frame reconstruction.
- **Gaussian Scene — SHARP** — an experimental, single-frame metric Gaussian
  scene intended for nearby viewpoint movement.

Each card states its input behavior, output type, model setup state, and major
constraint. The Start action is enabled only after the selected mode and source
are ready. Existing projects bypass mode selection and reopen with their
persisted mode.

### Recording workflow

When SHARP is selected for MOV, MP4, or M4V input, CloudPoint displays a
timeline with a recommended key frame. The recommendation favors a sharp,
normally exposed frame near the temporal center; the user may scrub and choose
another frame. Selecting **Create Gaussian Scene** creates the managed project,
persists the full-resolution oriented JPEG, and starts SHARP once.

### Camera workflow

When SHARP is selected for a camera, CloudPoint displays the live preview and a
single **Capture & Reconstruct** action. Pressing it persists one
full-resolution frame, freezes the preview as the source thumbnail, and starts
SHARP. It does not enter the continuous 1-10 fps LingBot capture workflow.

### Viewer and export

Completed SHARP projects open in a native Metal Gaussian viewer with orbit,
pan, zoom, reset-view, background, exposure, and splat-size controls. The
project stores and exports Apple's complete standard 3DGS PLY. **Export
Output** exports the PLY; **Share Project** shares the `.cloudpoint` package.

The existing point-cloud viewer remains available for LingBot projects. Viewer
settings are visually separated from reconstruction settings.

## Coordinate and mirroring contract

Worker artifacts remain in their source convention: OpenCV right-handed camera
coordinates with positive X to image right, positive Y down, and positive Z
forward. Stored CPC and PLY data are not rewritten solely for display.

Both Metal viewers apply the same display-basis transform:

```text
OpenCV to Metal = diagonal(1, -1, -1, 1)
```

This makes image-up appear screen-up and places positive-Z source geometry in
front of the default Metal camera. Imported recordings are never mirrored by
default.

Live camera projects persist unmirrored sensor frames. A persisted
`mirrorDisplay` preference controls both the live preview and the result
viewer, defaults to on for the familiar webcam experience, and can be toggled
without mutating geometry. The preview connection and renderer must derive from
the same value so they cannot silently disagree.

## Reconstruction-mode architecture

Input source and reconstruction method are separate concepts. Add stable mode
identifiers:

```text
cloudpoint.lingbot.point-cloud.v1
cloudpoint.apple.sharp-gaussian.v1
```

A small mode registry owns mode descriptors, availability, input requirements,
engine construction, result restoration, viewer selection, and export
capabilities. `WorkspaceSourceMode` continues to describe recording, camera,
or existing-project input; it does not carry reconstruction behavior.

The common session lifecycle remains:

1. persist admitted input;
2. run the selected engine out of process;
3. validate a complete result;
4. atomically commit the manifest and artifact reference;
5. publish the committed result to the selected viewer.

LingBot keeps its strict worker protocol v1 unchanged. SHARP uses a separate
`SharpReconstructionEngine` and JSON-lines subprocess contract so Gaussian
events cannot weaken or ambiguously extend LingBot's protocol fixtures.

## Project manifest v3

Manifest v3 persists:

- the stable reconstruction mode ID;
- a tagged mode configuration;
- source and display-mirroring metadata;
- engine provenance;
- tagged output state;
- a general list of validated relative artifact references.

SHARP output state records the input frame, PLY path, Gaussian count, PLY
metadata, model provenance, and completion state. New outputs live under
`Outputs/Gaussians/`; legacy `Predictions/` and `Points/` locations remain
valid.

Manifest decoding probes `formatVersion` before decoding. Version 2 projects
migrate in memory to `cloudpoint.lingbot.point-cloud.v1` without moving files or
rewriting a view-only project. The first later mutation atomically writes v3.
Unknown v3 modes open read-only with a precise unavailable-mode explanation;
they never fall back to LingBot.

Before any v3 LingBot project runs, the Python worker's manifest reader gains a
v3 LingBot branch. Its standard-I/O wire protocol remains byte-for-byte v1.

Project relocation enumerates artifact paths from the manifest instead of a
hardcoded list so Gaussian outputs cannot be omitted.

## SHARP runtime and model setup

CloudPoint vendors the Apple `ml-sharp` inference source pinned to commit
`1eaa046834b81852261262b41b0919f5c1efdd2e`, together with Apple's license,
model license, and acknowledgements. The CUDA-only gsplat renderer and render
CLI are excluded from the packaged runtime.

The existing isolated CPython runtime supplies PyTorch 2.8, torchvision 0.23,
NumPy, Pillow, and SciPy. Add only the pinned inference dependencies required by
the vendored code. A packaging test rejects CUDA binaries and an accidental
`gsplat` dependency.

The official `sharp_2572gikvuh.pt` checkpoint is never committed to the public
repository or bundled in the app. Model setup:

1. presents Apple's research-only terms and required attribution;
2. requires explicit acceptance;
3. downloads from Apple's official URL into Application Support;
4. supports cancellation and HTTP resume;
5. verifies the pinned byte length and SHA-256 digest;
6. atomically publishes the verified checkpoint and model manifest.

The model manifest and every SHARP project record the checkpoint digest, Apple
source URL, upstream commit, license identifier, execution device, and runtime
versions. The release remains usable for LingBot projects without accepting or
downloading SHARP.

## SHARP engine

The engine receives exactly one durable oriented JPEG plus focal length or
intrinsics when available. It loads the model once per worker, selects MPS on
supported Apple-silicon systems, and falls back to CPU when MPS is unavailable.
Inference uses `torch.inference_mode()` and never imports Python into the app
process.

SHARP resizes the image internally to 1536 by 1536 and emits 1,179,648
Gaussians for the pinned model. Output uses Apple's binary PLY schema:

- mean: `x`, `y`, `z`;
- degree-zero spherical harmonics: `f_dc_0...2`;
- opacity logit;
- logarithmic scale: `scale_0...2`;
- quaternion: `rot_0...3`;
- intrinsic, image-size, extrinsic, color-space, and version metadata.

The engine writes to a staging file, validates all required fields, finite
values, positive scene depth, Gaussian count, and metadata, then atomically
renames to `Outputs/Gaussians/<frame-id>.ply`. Cancellation or failure removes
staging output and leaves the last committed project state untouched.

The subprocess emits setup, loading, inference, validation, commit, completed,
warning, and failure events. Long inference is not governed by the existing
15-second command acknowledgement timeout. The app remains responsive and can
cancel the subprocess at every stage.

## Native Gaussian rendering

CloudPoint pins `scier/MetalSplatter` 1.0.1 at commit
`71ff248e3016ac43c0a9271e322538421b28c360`. Its MIT-licensed Swift/Metal
reader accepts SHARP's standard PLY vertex properties and ignores SHARP's
supplementary metadata elements after CloudPoint records them.

The renderer loads the full PLY asynchronously, shows determinate loading
progress, converts the OpenCV display basis in the view matrix, and renders
elliptical anisotropic Gaussians with depth sorting and premultiplied alpha.
Full PLY data remains the export authority even if a lower-memory Mac uses a
deterministic display-only level of detail.

Renderer errors do not invalidate completed reconstruction output. An
unsupported GPU or failed PLY load produces a useful fallback panel with
**Export PLY**, **Reveal Project**, and retry actions.

## Failure behavior

- A missing or invalid checkpoint returns the user to SHARP setup without
  creating or corrupting a project.
- A model download never silently starts and never publishes unverified bytes.
- MPS allocation or unsupported-operation failure offers an explicit CPU retry
  and reports that it may be substantially slower.
- Disk-space preflight includes checkpoint, inference staging, PLY output, and
  safety margin.
- Selecting too many inputs is impossible because the SHARP plan accepts one
  frame by type.
- Closing during inference asks for confirmation; cancellation leaves the
  durable source frame and a resumable pre-result state.
- Reopening a completed project never requires the SHARP checkpoint merely to
  view or export its PLY.

## Test strategy

Implementation follows red-green-refactor. Required automated coverage:

- OpenCV-to-Metal projection golden cases for up, right, depth, reset-view,
  and optional camera mirroring.
- Shared camera preview/result mirroring policy and unmirrored persisted input.
- Video key-frame extraction and orientation for MOV, MP4, and M4V.
- No project creation before source, mode, and key frame are confirmed.
- Manifest-v2-to-v3 fixture migration, lazy rewrite, unknown-mode fail-closed,
  and generic artifact relocation.
- LingBot protocol-v1 fixtures remain unchanged; both v2 and v3 LingBot
  manifests remain worker-readable.
- SHARP process framing, progress, cancellation, timeout, CPU fallback, and
  atomic output commit using a deterministic fake worker.
- Synthetic PLY compatibility covering position, SH0 color, opacity, scale,
  quaternion, metadata, nonfinite rejection, and OpenCV display calibration.
- Mode-aware viewer and exporter dispatch on initial completion and project
  reopen.
- Runtime packaging contains pinned SHARP dependencies and notices but no
  checkpoint, CUDA binary, or gsplat renderer.
- Opt-in real-model smoke: one fixed image produces exactly 1,179,648 finite
  Gaussians and a PLY the native Metal viewer can load.

Manual release validation uses `/Users/moebis/Downloads/IMG_2285.MOV` and the
existing Studio Display capture frame. It compares source and rendered
orientation, exercises video scrubbing and camera mirroring, generates and
reopens a SHARP project, exports its PLY, and verifies both viewers at compact
and full window sizes.

## Documentation and release

CloudPoint 1.1 documentation must state that SHARP is experimental,
research-only, single-image, local after setup, and reliable only for nearby
novel viewpoints rather than unseen room geometry. README, development guide,
third-party notices, and changelog record all pinned commits, licenses, model
provenance, setup size, and test commands.

After all native, Python, packaging, real-input, and visual gates pass, publish
the source and compiled arm64 app directly to `main`, tag `v1.1.0`, and attach
the distributable archive to the public GitHub release. The checkpoint remains
external.

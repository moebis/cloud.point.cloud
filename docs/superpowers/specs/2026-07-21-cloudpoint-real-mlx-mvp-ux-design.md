# CloudPoint Real-MLX MVP and UX Design

**Status:** implementation baseline from user feedback  
**Date:** 2026-07-21  
**Supersedes:** the mock engine as a releasable product and the save-before-input workflow

## Product truth

CloudPoint 1.0 is not complete until a recording or camera feed produces scene
geometry from the pinned Lingbot Map checkpoint through MLX on Apple Silicon.
The deterministic mock remains a test fixture only. It must never be selected by
a normal or release launch, and it must never be described as reconstruction.

The public repository and `v1.0.0` release stay blocked until the supplied
recording completes real inference and produces inspectable depth, confidence,
camera pose, intrinsics, and colored point-cloud artifacts.

## Chosen architecture

Use a native SwiftUI `WindowGroup` with an `AppCoordinator` and a managed,
autosaved project store. Reconstruction is a durable job, not an untitled text
document. `CloudPointDocument` remains the package format and import/export
boundary, while the coordinator owns opening inputs, creating projects, recent
items, and workspace windows.

Projects live by default under Application Support in a `Projects` directory.
Each uses a UUID for identity and a source-derived display name. Frame and
completed-window transactions continue to be the durability boundaries.
Users may reveal, move, or copy the package, but no explicit save is required
before input.

## Launch and input flow

The first window is a branded welcome view with:

1. **Open Video…** as the primary action and Command-O shortcut.
2. **Use Camera** as the secondary action.
3. **Open CloudPoint Project…** as the tertiary action.
4. Recent projects with source name, state, and last-opened date.
5. Drag and drop for MOV, MP4, M4V, and `.cloudpoint` packages.
6. A compact engine state: ready, setup required, downloading, converting, or repair required.

Every video entry point routes through one coordinator method. It validates the
asset with AVFoundation, probes metadata and a deterministic sampling plan,
creates a project immediately, opens the workspace, prepares the engine, and
starts reconstruction. Finder registrations use Alternate rank for movie types
so CloudPoint never replaces QuickTime Player as the default.

## Model setup

The setup view explains that the pinned `lingbot-map-long.pt` download is
4,632,303,465 bytes and that conversion requires additional temporary space.
Native URLSession performs a resumable download from revision
`204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`, verifies SHA-256
`832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`,
and hands the file to the isolated converter. Download, verification, conversion,
and loading have distinct progress and recovery states. Once prepared, normal
reconstruction is local and network-free.

## Workspace

The central Metal viewport is primary. A native toolbar contains source title,
pause/resume, cancel, reset view, inspector, and export. A collapsible inspector
contains reconstruction quality and display controls. Source navigation is a
sidebar only when it adds value; recording and camera controls are never mixed in
one permanent panel.

User-facing stages replace implementation terms:

- Reading video — frame count and stable total
- Preparing reconstruction model
- Reconstructing scene — processed count and stable total
- Processing remaining camera frames
- Finalizing point cloud
- Complete, paused, cancelled, or failed

Sampling and inference use separate progress when they overlap. Errors always
include a primary recovery action such as Choose Another Video, Locate Original,
Repair Model, Move Project, or Resume from Last Checkpoint.

## Camera flow

Camera starts in a dedicated preflight workspace: permission explanation, large
preview, camera selection, sampling quality, then one prominent Start Capture
button. During capture it becomes Stop Capture. Stopping input keeps the window
open while queued frames finish reconstructing. Closing an active capture asks
to stop and close; closing ordinary reconstruction checkpoints and pauses safely.

## Verification gates

- Unit and integration tests cover managed-project creation, input routing,
  MOV/MP4/M4V decode, model lifecycle, and every recovery action.
- MLX layers and end-to-end outputs pass the pinned PyTorch differential tolerances.
- The supplied `IMG_2285.MOV` runs through the production engine, not the mock.
- A reopened project restores committed geometry and resumes only from a completed window.
- A release build cannot reference or select `MockReconstructionEngine`.
- UI validation covers the welcome, setup, reconstruction, completion, and failure
  states at normal and compact macOS window sizes.

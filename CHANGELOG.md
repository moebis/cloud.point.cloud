# Changelog

All notable changes to CloudPoint are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-21

### Added

- Experimental Apple SHARP reconstruction from a selected video frame or a
  one-click live camera snapshot, using MPS with a recoverable CPU fallback.
- Guided SHARP setup with explicit Apple research-license acceptance, resumable
  download, exact size/SHA-256 verification, provenance, and no bundled model
  weights.
- Native Metal Gaussian-splat viewer with orbit, pan, zoom, reset, validated PLY
  loading, standalone PLY export, and durable project reopen.
- Source-first mode chooser with seven ranked video key frames and a live SHARP
  camera preview before any project is created.
- Manifest schema v3 with stable reconstruction-mode identifiers, Gaussian
  output provenance, and read-only compatibility for unknown future modes.
- Separate strict JSON-lines SHARP worker protocol with progress, heartbeat,
  cancellation, atomic artifact publication, and finite-output validation.
- Pinned MetalSplatter/SplatIO dependency and Apple SHARP source snapshot with
  preserved upstream notices.

### Changed

- Reworked opening a recording or camera into an explicit Point Cloud versus
  Gaussian Scene decision instead of silently creating a project.
- Separated Point Cloud and Gaussian workspace controls, status text, export
  actions, and empty/loading states.
- Kept completed SHARP projects viewable without requiring the model to remain
  installed; only unfinished reconstruction requires model health.

### Fixed

- Corrected the OpenCV-to-Metal basis so reconstructed geometry is upright and
  forward-facing.
- Synchronized camera-preview and reconstructed-scene mirroring and fixed
  off-axis Gaussian centering when display mirroring is enabled.
- Hardened Gaussian PLY/provenance validation against symlinks, non-finite
  values, invalid depth, malformed schemas, truncation, and size overflow.

### Release notes

- Requires Apple Silicon and macOS 15.0 or newer.
- SHARP is single-image nearby-view synthesis, not multi-view SLAM,
  photogrammetry, or a watertight mesh generator.
- SHARP setup downloads a 2,809,738,232-byte research checkpoint after the user
  accepts Apple's terms; approximately 6 GiB free is required.
- The v1.1.0 app is ad-hoc signed and not Apple-notarized.

## [1.0.0] - 2026-07-21

### Added

- Real LingBot-Map scene reconstruction on Apple Silicon through MLX, including
  depth, confidence, camera estimates, and colored CPC point-cloud output.
- Import-first MOV, MP4, and M4V workflow with AVFoundation probing and a
  deterministic 2 fps default sampling plan.
- Live built-in, external, and Continuity Camera capture with preflight preview,
  1–10 fps sampling, stop-and-drain behavior, and camera permission recovery.
- Native Metal point-cloud viewer with orbit, pan, zoom, point-size controls,
  confidence filtering, incremental windows, and reset-view support.
- Managed `.cloudpoint` project packages with automatic creation, autosave,
  recent-project history, durable frames and predictions, atomic window commits,
  and checkpoint recovery.
- Guided one-time model download, SHA-256 verification, local conversion,
  health validation, cancellation, retry, and repair.
- Pinned LingBot-Map checkpoint revision
  `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9` and reproducible converted-model
  trust anchors.
- Self-contained arm64 release runtime with pinned CPython, MLX, conversion, and
  worker dependencies; model weights remain external.
- Branded welcome screen, drag-and-drop input, Finder movie/project opening,
  recent projects, clear progress metrics, recovery actions, and a new CloudPoint
  app icon.
- Native and Python test coverage for input routing, project transactions,
  protocol framing, model conversion, worker lifecycle, real MLX integration,
  recovery, and runtime packaging.

### Changed

- Replaced the original save-before-input document flow with automatic project
  creation after a video or camera is selected.
- Routed normal Debug and Release launches through the production MLX engine.
  The deterministic engine remains available only through an explicit Debug
  test flag.
- Made worker startup, shutdown, cancellation, and failed-session reopening
  interruptible and checkpoint-safe.

### Fixed

- Made Release-sandbox project access compatible with macOS security-scoped
  paths while retaining no-symlink validation for every path component.

### Release notes

- Requires Apple Silicon and macOS 15.0 or newer.
- Initial setup downloads a 4.32 GiB checkpoint and creates 2.16 GiB of converted
  weights; approximately 8 GiB of free space is recommended.
- The v1.0.0 app is ad-hoc signed and not Apple-notarized.

[1.1.0]: https://github.com/moebis/cloud.point.cloud/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/moebis/cloud.point.cloud/releases/tag/v1.0.0

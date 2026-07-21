# CloudPoint

![CloudPoint app icon](CloudPoint/Resources/Assets.xcassets/AppIcon.appiconset/icon_256x256.png)

> Local 3D reconstruction for Mac

CloudPoint turns a video recording or live camera feed into a colored 3D point
cloud on Apple Silicon. It uses the real LingBot-Map reconstruction model
through Apple MLX, renders completed geometry with Metal, and keeps every
project autosaved in a `.cloudpoint` package.

## What CloudPoint 1.0 does

- Opens MOV, MP4, and M4V recordings directly—there is no blank project to
  save first.
- Captures from built-in, external, or Continuity Camera devices with a live
  preview and an adjustable sampling rate.
- Runs real LingBot-Map depth and camera inference locally with MLX; CUDA is
  not required.
- Adds colored point-cloud windows to an interactive Metal viewport as they
  finish. Drag to orbit, Shift-drag to pan, and scroll to zoom.
- Autosaves sampled frames, predictions, point chunks, model provenance, and
  reconstruction checkpoints in a `.cloudpoint` project package.
- Reopens recent projects, restores committed geometry, and can resume failed
  or interrupted work from its last committed window.
- Verifies and converts the pinned model with a guided one-time setup flow.

CloudPoint produces a colored point cloud, not a watertight textured mesh. The
first release prioritizes a correct local reconstruction path over real-time
speed.

## Requirements

- An Apple Silicon Mac (M-series)
- macOS 15.0 or newer
- An internet connection for the initial model download
- About 8 GiB of free space during model setup

The release does not contain model weights. On first use, CloudPoint downloads
the pinned `lingbot-map-long.pt` checkpoint (4.32 GiB / 4,632,303,465 bytes),
verifies it, and converts it locally into 2.16 GiB of MLX-compatible weights.
Reconstruction is local after setup.

## Install the app

1. Download `CloudPoint-v1.0.0-macOS-arm64.zip` from the
   [latest release](https://github.com/moebis/cloud.point.cloud/releases/latest).
2. Unzip it and move `CloudPoint.app` to `/Applications`.
3. The v1.0.0 build is ad-hoc signed and is not Apple-notarized. On first
   launch, Control-click or right-click the app in Finder, choose **Open**, and
   confirm macOS's prompt.

Only use a release downloaded from this repository. A notarized Developer ID
build is not included in v1.0.0.

## Make your first point cloud

1. Launch CloudPoint and choose **Open Video…**, press Command-O, drag a video
   onto the welcome window, or open a supported movie with CloudPoint from
   Finder.
2. If setup is required, choose **Download and Prepare**. CloudPoint verifies
   the exact upstream checkpoint before conversion, then continues the pending
   import automatically.
3. The app creates and autosaves a project under Application Support. You do
   not need to choose a project name or save location before reconstruction.
4. Watch the separate Read, Queued, and Reconstructed counts. Geometry appears
   in the viewport after each reconstruction window commits.
5. Return to **All Projects** to reopen an autosaved project, or use **Export**
   after completion to share its `.cloudpoint` package.

For live input, choose **Use Camera**, grant camera access, review the preview
and sampling rate, and press **Start Capture**. **Stop Capture** ends input but
keeps processing already captured frames until the queue finishes.

### Supported recordings

CloudPoint accepts files with `.mov`, `.mp4`, and `.m4v` extensions. Decoding
uses AVFoundation, so the video track must use a codec available on the Mac; a
supported filename extension alone does not guarantee that every file can be
decoded.

## Projects and privacy

Video frames and scene inference stay on the Mac. The only production network
operation is the explicit model download during setup.

CloudPoint does not copy the original recording into the project. It stores a
secure reference to the source plus sampled JPEG frames and derived data. Keep
the source recording available until importing finishes. A project package
contains:

```text
Example-<uuid>.cloudpoint/
├── Manifest.json
├── Frames/
├── Predictions/
├── Points/
└── Logs/
```

The manifest and completed windows are committed atomically. Project packages
can become large because they retain durable reconstruction inputs and outputs.

## How it works

1. Swift and AVFoundation sample a recording or live camera into durable
   oriented frames.
2. A bundled, relocatable Python runtime launches a separate worker with the
   verified converted model.
3. LingBot-Map runs through MLX and produces depth, confidence, camera
   parameters, and colored geometry.
4. Completed CPC point chunks are transactionally added to the project and
   displayed by the native Metal renderer.

The worker binds no network port. Its framed standard-I/O protocol keeps the
native UI, project transactions, and ML runtime isolated from one another.

## Build from source

Development requires Xcode with the macOS 15 SDK and Swift 6, plus
[`uv`](https://docs.astral.sh/uv/) for the locked Python environment. From the
repository root:

```sh
scripts/bootstrap
open CloudPoint.xcodeproj
```

Select the **CloudPoint** scheme and **My Mac** in Xcode. To bootstrap, build,
and launch a Debug app in one command:

```sh
scripts/run-first-version
```

The bootstrap script installs the pinned XcodeGen 2.46.0 when needed, prepares
the MLX worker environment, installs Apple's Metal toolchain when needed, and
generates `CloudPoint.xcodeproj` from `project.yml`.

Run the native test gate with:

```sh
scripts/bootstrap
scripts/test-native
```

See [Development](docs/DEVELOPMENT.md) for worker tests, real-model integration
tests, release packaging, and the developer-only deterministic test engine.

## Model and upstream provenance

CloudPoint 1.0 supports the official `robbyant/lingbot-map` long-sequence
checkpoint at Hugging Face revision
`204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`. The source checkpoint and
converted weights are deliberately excluded from both this repository and the
compiled app.

- [LingBot-Map source](https://github.com/Robbyant/lingbot-map/tree/7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2)
- [Pinned model revision](https://huggingface.co/robbyant/lingbot-map/tree/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9)
- [Apple MLX](https://github.com/ml-explore/mlx)

The LingBot-Map project and pinned model card identify the work as Apache-2.0.
CloudPoint also contains attributed source from an unofficial MLX port. See
[Third-party notices](THIRD_PARTY_NOTICES.md) for exact commits, checksums, and
license boundaries.

## License

CloudPoint is licensed under the [Apache License 2.0](LICENSE). Third-party
components and the separately downloaded model remain subject to their own
license notices.

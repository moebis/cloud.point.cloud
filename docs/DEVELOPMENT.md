# CloudPoint development

CloudPoint 1.0 is a native macOS 15 SwiftUI app backed by the production
LingBot-Map MLX worker. A normal Debug or Release launch reconstructs real scene
geometry; it does not substitute deterministic output when the worker or model
is unavailable.

## Architecture

- SwiftUI and `AppCoordinator` own the welcome screen, input routing, model
  setup, managed projects, recent projects, and workspace lifecycle.
- AVFoundation validates and samples MOV, MP4, and M4V recordings and captures
  live camera frames.
- `SessionController` persists every admitted frame and commits predictions,
  CPC point chunks, and schema-v2 manifest state at reconstruction-window
  boundaries.
- `PythonMLXEngine` launches the worker as a child process over framed standard
  input/output. The worker does not bind a network port.
- The worker loads the verified converted LingBot-Map model with Apple MLX and
  emits depth, confidence, camera, and colored point-cloud artifacts.
- `PointCloudRenderer` displays committed CPC chunks in a native Metal view.

The native app never imports Python or MLX into its process. Model download is
performed by the native setup layer; worker inference is network-free.

## Prerequisites

- Apple Silicon Mac
- macOS 15.0 or newer
- Xcode with the macOS 15 SDK and Swift 6
- [`uv`](https://docs.astral.sh/uv/)

## Bootstrap and run

From the repository root:

```sh
scripts/bootstrap
open CloudPoint.xcodeproj
```

`scripts/bootstrap`:

1. verifies Xcode;
2. installs the pinned XcodeGen 2.46.0 if it is not already available;
3. installs Apple's Metal toolchain if Xcode reports it missing;
4. creates `Config/Local.xcconfig` from the checked-in example;
5. resolves the locked worker, model-preparation, reference, and test
   dependencies into `worker/.venv`; and
6. regenerates `CloudPoint.xcodeproj` from `project.yml`.

In Xcode, select the **CloudPoint** scheme and **My Mac**, then Run. The fastest
terminal path performs the same bootstrap, builds an arm64 Debug app in a
temporary DerivedData directory, prints its exact path, and opens it:

```sh
scripts/run-first-version
```

Use `scripts/run-first-version --build-only` to compile without launching.

Open a recording from the welcome screen. CloudPoint probes it first, routes
through model setup if needed, then automatically creates an autosaved project
and begins import. There is no untitled document and no save-before-input step.

## Development and release runtimes

Debug builds use the absolute worker path generated in the ignored
`Config/Local.xcconfig`:

```xcconfig
CLOUDPOINT_WORKER_RUNTIME = $(SRCROOT)/worker/.venv
```

Release builds ignore that machine-local value and run
`scripts/package-worker-runtime` during the build. It creates a relocatable,
arm64 CPython 3.12.11 environment under
`CloudPoint.app/Contents/Resources/WorkerRuntime`, installs the exact locked
packages, adds native launchers, removes build paths and bytecode, and verifies
the result before publication. Model weights are never placed in the app.

The app launches only its bundled runtime or the exact build-configured Debug
runtime. It deliberately ignores inherited `PATH` lookups.

## Model setup and trust anchors

CloudPoint supports the official `robbyant/lingbot-map` long-sequence checkpoint
at Hugging Face revision
`204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`.

### Source checkpoint trust anchor

- File: `lingbot-map-long.pt`
- Exact size: 4,632,303,465 bytes
- SHA-256:

  ```text
  832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
  ```

### Converted model trust anchor

- File: `lingbot-map-long-f16.safetensors`
- Exact size: 2,316,040,080 bytes
- SHA-256:

  ```text
  eb966484923b5a205677b3ce7316d079c46fc6503bc9b6ac256b6e11560ea2e5
  ```

Native `URLSession` downloads and resumes only that checkpoint. The installer
verifies its size and SHA-256 before invoking the isolated converter. The
converter accepts only the exact verified artifact, produces 1,342 tensors,
and publishes converted weights only after the output digest and manifests
validate. Allow approximately 8 GiB of free space for setup.

The normal inference process reads SafeTensors and does not accept `.pt` input.
The source checkpoint, converted weights, and model manifests live under the
sandboxed app's Application Support directory, not in a project package.

## Input and project lifecycle

`AppCoordinator.defaultSamplingRate` is 2 fps for recording import. Camera
sampling is adjustable from 1 through 10 fps before capture. Selected frames
are written to the package before they are admitted to reconstruction.

Managed projects are created under the sandboxed Application Support
`CloudPoint/Projects` directory with a source-derived name and UUID. A package
contains `Manifest.json`, `Frames/`, `Predictions/`, `Points/`, and `Logs/`.
The original recording is referenced with a security-scoped bookmark rather
than copied into the package.

Stopping camera capture closes input but drains all durable queued frames.
Closing during active capture requires confirmation. Completed geometry is
restored when a project reopens; interrupted or failed work resumes from the
last committed reconstruction window.

## Native tests

Regenerate the project whenever `project.yml`, target resources, or Swift source
membership changes, then run the complete native gate:

```sh
scripts/bootstrap
scripts/test-native
```

The script builds the app, test bundle, mock worker, and worker launcher before
running the full `CloudPoint` scheme test suite on arm64 macOS.

### Recording-ingest regression

Run the portable recording smoke against a local movie:

```sh
scripts/test-recording-smoke '/absolute/path/to/video.mov'
```

This test fingerprints the source, stages a read-only copy inside the test
host's sandbox, drives the production recording importer and project
transaction path, validates all generated schema-v2 artifacts, and proves the
source remained unchanged. It intentionally uses the deterministic test engine
to isolate ingest and durability behavior; it is not evidence of MLX scene
reconstruction.

### Real native-to-worker MLX bridge

After preparing the pinned model through CloudPoint, package the locked worker
runtime inside CloudPoint's test-support container. The sandbox intentionally
cannot execute the repository runtime from `Documents`, so the CLI build
setting below points the Debug test host at that sandbox-readable copy.

```sh
model_root="$HOME/Library/Containers/cloud.point.cloud.CloudPoint/Data/Library/Application Support/cloud.point.cloud.CloudPoint/Models"
test_support="$HOME/Library/Containers/cloud.point.cloud.CloudPoint/Data/Library/Application Support/CloudPoint/TestSupport"
runtime="$test_support/WorkerRuntime"

scripts/package-worker-runtime "$runtime"

CLOUDPOINT_REAL_MODEL_DIR="$model_root/robbyant-lingbot-map/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/converted" \
xcodebuild test -quiet \
  -project CloudPoint.xcodeproj \
  -scheme CloudPoint \
  -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/cloudpoint-real-bridge \
  "CLOUDPOINT_WORKER_RUNTIME=$runtime" \
  -only-testing:CloudPointTests/RealWorkerBridgeTests
```

This opt-in test sends the nine pinned courthouse frames through the real Swift
bridge and MLX worker, waits for terminal completion, opens the resulting CPC,
and requires more than 1,000 points. The packaged runtime is the same locked,
relocatable runtime produced for Release builds; Debug is used only so the
`@testable` test target and its test doubles remain available.

## Worker tests

Run the locked Python checks from the repository root:

```sh
(
  cd worker
  uv run --frozen --extra model-prep ruff check src tests
  uv run --frozen --extra model-prep pytest -q -m 'not real_model'
)
```

The real-model Python health test is opt-in:

```sh
(
  cd worker
  CLOUDPOINT_MODEL_DIR='/absolute/path/to/converted' \
    uv run --frozen --extra model-prep \
    pytest -q -m real_model
)
```

Protocol fixtures have a dedicated regeneration and verification command:

```sh
worker/scripts/verify-protocol
```

The release-runtime static contract is checked with:

```sh
scripts/tests/test-worker-runtime-packaging
```

## Deterministic test engine

`MockReconstructionEngine` is a developer fixture for fast native input,
persistence, and failure-path tests. It is compiled for Debug only and can be
selected only with the exact `--mock-engine` process argument. Ordinary Debug
launches and all Release launches use the production MLX route. A missing or
invalid runtime/model produces a visible setup or repair state instead of mock
geometry.

## Build a release

Release builds package and verify the relocatable runtime, so they take longer
and produce a much larger app than Debug builds:

```sh
scripts/bootstrap

xcodebuild build -quiet \
  -project CloudPoint.xcodeproj \
  -scheme CloudPoint \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/cloudpoint-release

codesign --verify --deep --strict --verbose=2 \
  /tmp/cloudpoint-release/Build/Products/Release/CloudPoint.app
```

To create the distributable archive:

```sh
ditto -c -k --sequesterRsrc --keepParent \
  /tmp/cloudpoint-release/Build/Products/Release/CloudPoint.app \
  CloudPoint-v1.0.0-macOS-arm64.zip
```

The public v1.0.0 artifact is ad-hoc signed and not notarized. Do not describe it
as Developer ID signed or notarized unless a later release pipeline actually
performs and verifies those steps.

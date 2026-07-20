# CloudPoint development

CloudPoint is a macOS 15 SwiftUI document app. The first runnable version can
sample recordings or a live camera into durable JPEG frames and exercise the
complete schema-v2 reconstruction workflow with the deterministic mock engine.
The production Apple/MLX reconstruction engine is not wired yet, so a normal
launch reports **Lingbot engine not installed yet** and never substitutes mock
output.

## Bootstrap and open in Xcode

From the repository root:

```sh
scripts/bootstrap
open CloudPoint.xcodeproj
```

`scripts/bootstrap` verifies Xcode, installs the pinned XcodeGen 2.46.0 when
needed, installs Apple's Metal toolchain when needed, and generates the ignored
Xcode project. Run `scripts/generate-project` again whenever Swift source
membership or `project.yml` changes.

In Xcode, select the **CloudPoint** scheme and **My Mac**, then Run. A new
untitled document must be saved as a `.cloudpoint` package before recording or
camera input is enabled.

## Run the deterministic app

Mock reconstruction is intentionally available only in Debug and only with the
exact `--mock-engine` argument. In Xcode, add it under Scheme > Run > Arguments,
or use the one-command first-version launcher from Terminal:

```sh
scripts/run-first-version
```

The launcher verifies the pinned tools, builds the arm64 Debug app in a temporary
DerivedData directory, prints the exact `.app` path, and launches it with the
mock flag. To compile without opening the app, use
`scripts/run-first-version --build-only`.

The equivalent manual build and launch is:

```sh
xcodebuild build \
  -project CloudPoint.xcodeproj \
  -scheme CloudPoint \
  -configuration Debug \
  -destination 'platform=macOS,arch=arm64'

open "$HOME/Library/Developer/Xcode/DerivedData/CloudPoint-"*/Build/Products/Debug/CloudPoint.app \
  --args --mock-engine
```

Save the project, choose **Open Recording**, and select a movie. CloudPoint
persists each selected frame before admitting it to reconstruction. A window is
shown only after its predictions, CPC point chunk, manifest transaction, and
document adoption all succeed. **Use Camera** follows the same transaction path;
**Stop Capture** drains every durable camera event before closing engine input.

## Tests

Run the complete native gate:

```sh
scripts/generate-project
scripts/test-native
```

Run the portable real-recording smoke against any local movie without embedding
its path in the test suite:

```sh
scripts/test-recording-smoke '/absolute/path/to/video.mov'
```

The helper fingerprints the caller-provided movie, stages a read-only copy
inside the test host's sandbox container, and removes that copy afterward. The
original is never passed to the test process and its fingerprint must remain
unchanged. The smoke drives the same production `AssetRecordingImporter` used
by the app through `SessionController` and the deterministic
`MockReconstructionEngine` until terminal completion. It then reloads and
validates the schema-v2 manifest; checks captured, queued, and processed parity;
checks the exact JPEG, three-per-frame prediction, and CPC file sets; opens
every JPEG and CPC; and rejects any remaining `.partial` transaction file. All
generated artifacts live in a temporary `.cloudpoint` package. If the variable
is absent, the test is skipped.

Build the shipping configuration explicitly:

```sh
xcodebuild build \
  -project CloudPoint.xcodeproj \
  -scheme CloudPoint \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64'
```

Release and ordinary Debug builds do not enable mock reconstruction. The next
engine milestone is an Apple Silicon implementation using Metal/MLX rather than
CUDA, connected through the existing `ReconstructionEngine` boundary.

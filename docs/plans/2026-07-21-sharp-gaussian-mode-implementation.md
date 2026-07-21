# SHARP Gaussian Reconstruction Mode Implementation Plan

> **Execution rule:** Complete each task test-first, keep LingBot protocol v1
> fixtures byte-for-byte stable, and never add the pre-existing untracked
> `CloudPoint/App/CloudPointApp 2.swift` file to a commit.

**Goal:** Release CloudPoint 1.1.0 with correct point-cloud orientation,
consistent camera mirroring, an experimental single-frame Apple SHARP mode,
native Metal Gaussian viewing, safe v1.0 project migration, and a compiled
GitHub release.

**Architecture:** Preserve canonical OpenCV geometry and correct it only at the
viewer boundary. Add stable reconstruction mode IDs and manifest v3 while
adapting existing LingBot state rather than rewriting its strict worker
protocol. Run SHARP in a separate packaged Python subprocess, commit one
validated PLY atomically, and render it through a pinned native Metal library.

**Technology:** Swift 6, SwiftUI, AVFoundation, MetalKit, MetalSplatter 1.0.1,
CPython 3.12, PyTorch 2.8 MPS/CPU, Apple ml-sharp, XcodeGen, XCTest, pytest.

---

## Task 1: Correct the point-cloud display basis

**Files:**

- Modify: `CloudPoint/Rendering/PointCloudRenderer.swift`
- Modify: `CloudPointTests/RendererBufferTests.swift`

1. Add projection tests proving positive OpenCV X appears right, positive
   OpenCV Y appears down, positive OpenCV Z is visible in front of the default
   camera, and reset view preserves the convention.
2. Run the focused renderer tests and confirm the new assertions fail.
3. Add one reusable `openCVToMetal` matrix equal to
   `diagonal(1, -1, -1, 1)` and compose it after the view matrix.
4. Run focused tests, then `scripts/test-native`.
5. Commit as `fix(renderer): correct OpenCV display orientation`.

## Task 2: Make camera mirroring explicit and consistent

**Files:**

- Modify: `CloudPoint/Capture/CameraPreviewView.swift`
- Modify: `CloudPoint/Capture/CameraFrameSource.swift`
- Modify: `CloudPoint/Domain/ProjectModels.swift`
- Modify: `CloudPoint/Rendering/PointCloudRenderer.swift`
- Modify: `CloudPoint/Workspace/WorkspaceViewModel.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Modify: `CloudPointTests/CameraFrameSourceTests.swift`
- Modify: `CloudPointTests/RendererBufferTests.swift`
- Modify: `CloudPointTests/WorkspaceViewModelTests.swift`

1. Add tests for a shared `CameraDisplayPolicy`: sensor frames remain
   unmirrored; camera preview and result viewer use the same persisted display
   mirror flag; recording imports default unmirrored.
2. Confirm the focused tests fail with the independent AVFoundation connection
   defaults.
3. Configure both preview and video-data connections explicitly, persist
   `mirrorDisplay`, and apply optional X reflection only in the viewer matrix.
4. Add an accessible Mirror toggle for camera projects and verify it never
   rewrites CPC data.
5. Run the focused tests and native gate.
6. Commit as `fix(camera): synchronize preview and reconstruction mirroring`.

## Task 3: Introduce reconstruction modes and manifest v3

**Files:**

- Create: `CloudPoint/Domain/ReconstructionMode.swift`
- Modify: `CloudPoint/Persistence/ProjectManifest.swift`
- Modify: `CloudPoint/Persistence/ManagedProjectStore.swift`
- Modify: `CloudPoint/Persistence/CloudPointDocument.swift`
- Modify: `CloudPoint/Engine/ReconstructionEngine.swift`
- Modify: `CloudPoint/Workspace/SessionController.swift`
- Modify: `CloudPointTests/ProjectManifestTests.swift`
- Modify: `CloudPointTests/ManagedProjectStoreTests.swift`
- Modify: `CloudPointTests/SessionControllerTests.swift`

1. Add byte fixtures and tests for lazy v2-to-v3 LingBot migration, stable mode
   IDs, tagged configurations and outputs, unknown-mode fail-closed behavior,
   mirror persistence, and generic artifact enumeration.
2. Run the focused persistence tests and confirm failures.
3. Add `ReconstructionModeID`, `ReconstructionPlan`, tagged output state, and
   v3 decoding with an explicit legacy-v2 type. Keep existing completed-window
   data in place and rewrite only on the next mutation.
4. Make project creation add `Outputs/Gaussians` and relocation use manifest
   artifact references rather than hardcoded directories.
5. Adapt the existing LingBot controller path behind the v3 LingBot mode
   without changing runtime behavior.
6. Run persistence/controller tests and the native gate.
7. Commit as `feat(projects): add versioned reconstruction modes`.

## Task 4: Preserve LingBot worker compatibility with v3 projects

**Files:**

- Modify: `worker/src/cloudpoint_worker/session.py`
- Modify: `worker/tests/test_session_server.py`
- Modify: `worker/tests/test_cli_smoke.py`
- Modify: `CloudPointTests/ProtocolOwnershipContractTests.swift`

1. Add Python fixtures for equivalent v2 and v3 LingBot manifests and tests
   proving both produce the same validated session description.
2. Confirm the v3 fixture fails while all protocol-v1 fixtures still pass.
3. Dispatch on manifest version in the worker and extract only the v3 LingBot
   configuration; reject SHARP and unknown modes.
4. Run Python session/protocol tests and Swift protocol ownership tests.
5. Verify `worker/tests/fixtures/protocol-v1.json` is unchanged.
6. Commit as `feat(worker): read LingBot manifest v3`.

## Task 5: Add source-first mode selection and SHARP key-frame capture

**Files:**

- Create: `CloudPoint/App/NewReconstructionView.swift`
- Create: `CloudPoint/Capture/VideoKeyFrameSelector.swift`
- Modify: `CloudPoint/App/AppCoordinator.swift`
- Modify: `CloudPoint/App/WelcomeView.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Modify: `CloudPoint/Workspace/WorkspaceViewModel.swift`
- Modify: `CloudPointTests/AppCoordinatorTests.swift`
- Modify: `CloudPointTests/AssetFrameSourceTests.swift`
- Create: `CloudPointTests/VideoKeyFrameSelectorTests.swift`

1. Add coordinator tests for probe -> pending source -> mode sheet -> project
   creation, proving no empty project exists before confirmation.
2. Add deterministic key-frame ranking tests for sharpness, exposure, temporal
   preference, orientation, and user override across MOV, MP4, and M4V.
3. Confirm the new tests fail against immediate project creation.
4. Add accessible Point Cloud and Gaussian Scene mode cards. For SHARP video,
   add timeline thumbnails and a recommended frame; for camera, add one
   Capture & Reconstruct action.
5. Persist the selected full-resolution oriented JPEG before engine admission.
6. Run focused coordinator/capture tests and native gate.
7. Commit as `feat(app): add reconstruction mode selection`.

## Task 6: Vendor and package the pinned SHARP inference source

**Files:**

- Create: `worker/src/cloudpoint_worker/sharp/`
- Create: `worker/src/cloudpoint_worker/model/_vendor/ml_sharp/`
- Create: `worker/src/cloudpoint_worker/model/_vendor/ml_sharp/PROVENANCE.md`
- Modify: `worker/pyproject.toml`
- Modify: `worker/uv.lock`
- Modify: `worker/THIRD_PARTY_NOTICES.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `scripts/package-worker-runtime`
- Modify: `scripts/tests/test-worker-runtime-packaging`

1. Add packaging tests requiring the pinned SHARP source, licenses,
   acknowledgements, `timm`, and `plyfile`, while rejecting bundled weights,
   CUDA libraries, and `gsplat`.
2. Confirm the packaging test fails.
3. Vendor only inference/model/PLY-writing code from Apple commit
   `1eaa046834b81852261262b41b0919f5c1efdd2e`, preserving notices and recording
   modifications. Exclude CUDA render code.
4. Lock the minimal Python dependencies compatible with CPython 3.12 and the
   existing torch runtime.
5. Run ruff, non-real-model pytest, runtime packaging, and runtime verification.
6. Commit as `feat(worker): package Apple SHARP inference`.

## Task 7: Add SHARP research-model setup

**Files:**

- Create: `CloudPoint/ModelSetup/SharpModelRelease.swift`
- Create: `CloudPoint/ModelSetup/SharpModelInstaller.swift`
- Modify: `CloudPoint/ModelSetup/ModelSetupView.swift`
- Modify: `CloudPoint/ModelSetup/ModelSetupViewModel.swift`
- Modify: `CloudPoint/ModelSetup/URLSessionModelDownloader.swift`
- Modify: `CloudPoint/App/AppServices.swift`
- Create: `CloudPointTests/SharpModelInstallerTests.swift`
- Modify: `CloudPointTests/ModelInstallerTests.swift`

1. Add tests for explicit license acceptance, official HTTPS origin, resumable
   download, disk preflight, exact size/digest verification, cancellation,
   atomic publication, provenance manifest, and no network on ordinary reopen.
2. Confirm focused tests fail.
3. Implement SHARP-specific setup using the shared downloader primitives and a
   separately namespaced Application Support model directory.
4. Present the full research notice before download and retain the accepted
   license text beside the verified checkpoint.
5. Run model setup tests and native gate.
6. Commit as `feat(models): add verified SHARP setup`.

## Task 8: Implement the SHARP subprocess engine and atomic PLY result

**Files:**

- Create: `CloudPoint/Engine/SharpReconstructionEngine.swift`
- Create: `CloudPoint/Engine/SharpWorkerProtocol.swift`
- Create: `worker/src/cloudpoint_worker/sharp/cli.py`
- Create: `worker/src/cloudpoint_worker/sharp/session.py`
- Create: `worker/tests/test_sharp_session.py`
- Create: `CloudPointTests/SharpWorkerProtocolTests.swift`
- Create: `CloudPointTests/SharpReconstructionEngineTests.swift`
- Modify: `CloudPoint/Workspace/SessionController.swift`
- Modify: `CloudPoint/App/AppServices.swift`

1. Add strict JSON-lines fixtures and fake-worker tests for setup, loading,
   inference, validation, commit, completion, warning, failure, cancellation,
   CPU fallback, and long-operation liveness.
2. Add Python tests using a deterministic fake predictor for the exact PLY
   schema, finite values, positive depth, Gaussian count, metadata, staging
   cleanup, and atomic rename.
3. Confirm Swift and Python tests fail.
4. Implement the separate SHARP process and engine adapter. Use MPS by default,
   `torch.inference_mode()`, supplied intrinsics where available, and explicit
   CPU retry after recoverable MPS failure.
5. Commit only a validated `Outputs/Gaussians/<frame-id>.ply` and provenance
   record through the common session durability boundary.
6. Run focused Swift/Python tests, worker gate, and native gate.
7. Commit as `feat(sharp): generate Gaussian scenes on Apple silicon`.

## Task 9: Add the native Metal Gaussian viewer and PLY export

**Files:**

- Modify: `project.yml`
- Create: `CloudPoint/Rendering/GaussianSplatView.swift`
- Create: `CloudPoint/Rendering/GaussianSplatRenderer.swift`
- Create: `CloudPoint/Rendering/ReconstructionViewer.swift`
- Modify: `CloudPoint/Workspace/WorkspaceView.swift`
- Modify: `CloudPoint/Workspace/WorkspaceViewModel.swift`
- Create: `CloudPointTests/GaussianSplatRendererTests.swift`
- Modify: `CloudPointTests/WorkspaceViewModelTests.swift`

1. Pin MetalSplatter 1.0.1 at commit
   `71ff248e3016ac43c0a9271e322538421b28c360` in XcodeGen.
2. Add a tiny synthetic SHARP-compatible binary PLY fixture and tests for
   position, SH0 color, log scale, opacity logit, quaternion, ignored metadata,
   OpenCV-to-Metal calibration, optional mirroring, cancellation, viewer
   dispatch, reopen, and full PLY export.
3. Confirm tests fail before viewer implementation.
4. Implement asynchronous PLY loading, mode-aware viewer dispatch, orbit/pan/
   zoom/reset, accessible display controls, renderer fallback, Export Output,
   and a separate Share Project action.
5. Regenerate the Xcode project, run focused tests, and run the native gate in
   Debug and Release configurations.
6. Commit as `feat(renderer): display and export SHARP Gaussian scenes`.

## Task 10: Real-model, input, visual, packaging, and release validation

**Files:**

- Create: `worker/tests/test_sharp_real_model.py`
- Create: `scripts/test-sharp-smoke`
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `CHANGELOG.md`
- Modify: `CloudPoint/Resources/Info.plist`
- Modify: `Config/Base.xcconfig`

1. Download the SHARP checkpoint through the app setup flow and record its
   verified trust anchors in code and documentation.
2. Run a pinned still-image real-model smoke and require exactly 1,179,648
   finite Gaussians plus a natively loadable PLY.
3. Run `/Users/moebis/Downloads/IMG_2285.MOV` through key-frame selection,
   SHARP inference, viewer reopen, and PLY export.
4. Validate the stored Studio Display frame and live preview against the fixed
   point-cloud and Gaussian coordinate/mirror contracts.
5. Run all gates:

   ```sh
   scripts/bootstrap
   scripts/test-native
   (
     cd worker
     uv run --frozen --extra model-prep ruff check src tests
     uv run --frozen --extra model-prep pytest -q -m 'not real_model'
   )
   scripts/tests/test-worker-runtime-packaging
   scripts/verify-worker-runtime <packaged-runtime>
   scripts/test-recording-smoke /Users/moebis/Downloads/IMG_2285.MOV
   scripts/test-sharp-smoke /Users/moebis/Downloads/IMG_2285.MOV
   ```

6. Build Release, verify code signing and sandbox launch, inspect compact/full
   layouts, and archive `CloudPoint-v1.1.0-macOS-arm64.zip`.
7. Update README, development guide, notices, changelog, and version metadata.
8. Commit as `release: CloudPoint 1.1.0`, push `main`, tag `v1.1.0`, and publish
   the GitHub release with the compiled archive and research-use notes.

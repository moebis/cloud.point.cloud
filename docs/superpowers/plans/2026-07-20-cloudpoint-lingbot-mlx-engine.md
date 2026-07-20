# CloudPoint Lingbot MLX Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python 3.12/MLX Lingbot Map worker that prepares the one pinned checkpoint, matches the pinned PyTorch SDPA reference, streams versioned reconstruction events, writes recoverable geometry artifacts, and runs independently from the command line.

**Architecture:** A locked `uv` project exposes a network-free worker process and a separate model-preparation command. Focused MLX modules mirror upstream Lingbot Map commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`; a session runner owns preprocessing, KV-cached direct/windowed inference, robust Sim(3) alignment, atomic prediction/CPC output, and the JSON-over-Unix-socket adapter.

**Tech Stack:** Python 3.12.11, uv, MLX 0.32.0, NumPy 2.3.1, Pillow 11.3.0, SafeTensors 0.5.3, msgspec 0.19.0, pytest 8.4.1, optional PyTorch 2.8.0/torchvision 0.23.0 reference tooling

## Global Constraints

- Target Apple Silicon Macs running macOS 15 or later; reject non-Apple-Silicon MLX execution with a structured `RUNTIME_INCOMPATIBLE` error.
- The repository-managed runtime is Python 3.12.11 from `.python-version`; every install and run uses the committed `uv.lock` with `--frozen`.
- MLX is exactly `0.32.0`; production inference uses Float16, while camera decoding, Sim(3), and parity probes explicitly promote to Float32.
- Support only `robbyant/lingbot-map` revision `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`, file `lingbot-map-long.pt`, exact size `4,632,303,465`, SHA-256 `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`.
- Port and attribute source topology from commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`; do not import an unofficial MLX port as a package or runtime dependency.
- The production worker never accepts or deserializes `.pt`; only the separate converter may deserialize the exact verified checkpoint in a scrubbed subprocess.
- IPC is protocol version `1`: four-byte big-endian length plus UTF-8 JSON, maximum message length `1,048,576` bytes, exactly one acknowledgement or structured error per command, and no tensors or images in JSON.
- The native setup layer downloads the checkpoint with URLSession before conversion or worker launch. Both `cloudpoint-model` and `cloudpoint-worker` bind no TCP port and make no network requests.
- Reconstruction defaults are eight scale frames, 32-frame windows, eight-frame overlap, keyframe interval one, one camera-refinement pass, confidence threshold `1.5`, direct mode at 32 frames or fewer, and overlapping-window mode above 32 frames.
- Enable depth and camera heads and disable the optional world-point head; emit dense depth/confidence, nine-value camera encoding, decoded intrinsics, camera-to-world pose, and filtered colored points.
- Coordinates are reconstruction units, not meters. Reject non-finite/non-positive depth, confidence below the stored floor, degenerate alignment, malformed messages, unsafe paths, excessive tensor dimensions, and oversized outputs.
- Recovery boundaries are completed windows only. Atomic artifacts use a sibling `.partial`, `fsync`, and rename; startup removes stale `.partial` files and resumes after the last completed window.
- CPC `frameStart` and `frameEnd` are inclusive UInt32 source-frame indices; internal Python windows use `start` inclusive and `stop` exclusive, so writers/events use `frameEnd = stop - 1`.
- Bound inputs before allocation to an 8,192-pixel maximum dimension, 33,554,432 source pixels, 134,217,728 tensor elements, 50,000,000 CPC points, and `1,200,000,032` CPC bytes. The native reader uses the same point/file bounds.
- Real-model differential thresholds are: Float32 leaf layers `rtol <= 1e-3`, `atol <= 1e-4`; selected aggregator cosine similarity `>= 0.995`; Float16 depth correlation `>= 0.99` and median relative error `<= 0.03`; translation relative error `<= 0.03` after shared-scale normalization; rotation geodesic error `<= 1.0` degree; intrinsics relative error `<= 0.01`.

---

## File Structure

Create the worker as an isolated package under `worker/`:

```text
worker/
  .python-version                         # exact Python toolchain
  pyproject.toml                          # runtime, setup/reference extras, entry points
  uv.lock                                 # fully resolved cross-command lock
  src/cloudpoint_worker/
    __init__.py                           # engine/protocol versions
    errors.py                             # stable structured error codes
    cli.py                                # `cloudpoint-worker serve|run|health|protocol-fixture`
    protocol/framing.py                   # bounded canonical four-byte JSON framing
    protocol/schema.py                    # version-1 commands, acks, errors, events
    protocol/fixtures.py                  # canonical Swift/Python compatibility fixture
    model_prep/provenance.py              # pinned source constants and fd verification
    model_prep/convert.py                 # scrubbed strict PyTorch-to-SafeTensors converter
    model_prep/cli.py                     # `cloudpoint-model prepare|verify`
    model/config.py                       # immutable Lingbot topology/defaults
    model/weight_specs.py                 # exhaustive PyTorch-to-MLX tensor mapping
    model/layers.py                       # MLX linear/conv/attention/MLP/LayerScale blocks
    model/rope.py                         # pinned 2D and temporal rotary embeddings
    model/backbone.py                     # DINOv2 ViT-L/14 register-token backbone
    model/cache.py                        # append/skip/evict/reset streaming KV cache
    model/aggregator.py                   # 24 alternating frame/global blocks
    model/heads.py                        # DPT depth/confidence and causal camera heads
    model/lingbot.py                      # weight specs and direct streaming forward API
    preprocess.py                         # orientation, sRGB, 518/14 transform and normalization
    sim3.py                               # deterministic robust overlap alignment
    geometry.py                           # pose decode, unprojection, filtering, voxel reduction
    cpc.py                                # CPC1 little-endian atomic writer/reader
    outputs.py                            # depth/confidence/geometry atomic artifacts
    windows.py                            # 32/8 orchestration and checkpoint records
    session.py                            # queue, lifecycle, pause/cancel/recovery
    server.py                             # Unix-socket command/event adapter and heartbeat
  tools/export_reference.py              # pinned PyTorch SDPA golden generator
  tests/
    fixtures/courthouse/000000.png through 000008.png
    fixtures/courthouse/provenance.json
    fixtures/parity/generated/.gitkeep    # local, ignored real-model tensors
    fixtures/protocol-v1.json             # framed canonical command/event corpus
    protocol/test_framing.py
    protocol/test_schema.py
    protocol/test_fixtures.py
    test_model_prep.py
    test_preprocess.py
    test_layers_parity.py
    test_model_parity.py
    test_cache_windows_sim3.py
    test_geometry_cpc.py
    test_session_server.py
    test_cli_smoke.py
```

The generated golden directory must be ignored except for `.gitkeep`; checkpoint-derived tensors are regenerated locally and are not redistributed.

### Task 1: Lock the Runtime and Freeze Protocol Version 1

**Files:**
- Create: `worker/.python-version`
- Create: `worker/pyproject.toml`
- Create: `worker/uv.lock`
- Create: `worker/src/cloudpoint_worker/__init__.py`
- Create: `worker/src/cloudpoint_worker/errors.py`
- Create: `worker/src/cloudpoint_worker/protocol/__init__.py`
- Create: `worker/src/cloudpoint_worker/protocol/framing.py`
- Create: `worker/src/cloudpoint_worker/protocol/schema.py`
- Create: `worker/src/cloudpoint_worker/protocol/fixtures.py`
- Create: `worker/tests/protocol/test_framing.py`
- Create: `worker/tests/protocol/test_schema.py`
- Create: `worker/tests/protocol/test_fixtures.py`
- Create: `worker/tests/fixtures/protocol-v1.json`

**Interfaces:**
- Consumes: byte streams implementing `read(n) -> bytes` and `write(bytes) -> int`.
- Produces: `read_json_frame(stream: BinaryIO) -> dict[str, object]`, `write_json_frame(stream: BinaryIO, value: object) -> None`, `decode_command(value: object) -> Command`, `ack(command: Command) -> Ack`, `command_error(command: Command, fault: WorkerFault) -> ErrorMessage`, and `write_protocol_fixture(path: Path) -> None`.

- [ ] **Step 1: Write framing and schema tests first**

```python
def test_frame_is_big_endian_and_round_trips() -> None:
    stream = io.BytesIO()
    write_json_frame(stream, {"type": "hello", "protocolVersion": 1})
    raw = stream.getvalue()
    assert int.from_bytes(raw[:4], "big") == len(raw) - 4
    assert raw[4:] == b'{"protocolVersion":1,"type":"hello"}'
    stream.seek(0)
    assert read_json_frame(stream)["type"] == "hello"

def test_frame_rejects_more_than_one_megabyte_before_body_read() -> None:
    class RecordingReader:
        def __init__(self, data: bytes) -> None:
            self._stream, self.requested = io.BytesIO(data), []
        def read(self, count: int) -> bytes:
            self.requested.append(count)
            return self._stream.read(count)
    stream = RecordingReader((1_048_577).to_bytes(4, "big"))
    with pytest.raises(WorkerFault, match="MESSAGE_TOO_LARGE"):
        read_json_frame(stream)
    assert stream.requested == [4]

def test_command_requires_version_uuid_project_and_typed_payload() -> None:
    value = {"protocolVersion": 2, "id": str(uuid.uuid4()),
             "projectId": str(uuid.uuid4()), "type": "pause", "payload": {}}
    with pytest.raises(msgspec.ValidationError):
        decode_command(value)

def test_zero_length_and_duplicate_command_ids_are_rejected() -> None:
    with pytest.raises(WorkerFault, match="INVALID_MESSAGE_LENGTH"):
        read_json_frame(io.BytesIO(b"\0\0\0\0"))
    tracker = CommandIDTracker(capacity=4096)
    tracker.insert(COMMAND_ID)
    with pytest.raises(WorkerFault, match="DUPLICATE_COMMAND_ID"):
        tracker.insert(COMMAND_ID)
```

- [ ] **Step 2: Run the tests and verify the missing package failure**

Run: `cd worker && uv run --with pytest==8.4.1 --python 3.12.11 pytest tests/protocol -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'cloudpoint_worker'`.

- [ ] **Step 3: Add the exact environment and public protocol types**

Use this dependency surface in `pyproject.toml`:

```toml
[project]
name = "cloudpoint-worker"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
  "mlx==0.32.0",
  "msgspec==0.19.0",
  "numpy==2.3.1",
  "Pillow==11.3.0",
  "safetensors==0.5.3",
]

[project.optional-dependencies]
model-prep = ["torch==2.8.0"]
reference = [
  "einops==0.8.1", "huggingface-hub==0.33.4", "scipy==1.16.0",
  "torch==2.8.0", "torchvision==0.23.0", "tqdm==4.67.1",
]

[dependency-groups]
dev = ["pytest==8.4.1", "pytest-timeout==2.4.0", "ruff==0.12.4"]

[project.scripts]
cloudpoint-worker = "cloudpoint_worker.cli:main"
cloudpoint-model = "cloudpoint_worker.model_prep.cli:main"

[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["real_model: requires the verified 4.6 GB checkpoint and Apple Silicon"]
```

Set `.python-version` to `3.12.11`, `PROTOCOL_VERSION = 1`, `ENGINE_VERSION = "0.1.0"`, and implement envelopes with lower-camel JSON field names. `WorkerFault` is the stable internal error transported by `ErrorMessage`:

```python
@dataclass(frozen=True)
class WorkerFault(Exception):
    code: str
    message: str
    recoverable: bool
    details: dict[str, object] = field(default_factory=dict)

class CommandHeader(msgspec.Struct, rename="camel"):
    protocol_version: Literal[1] = msgspec.field(name="protocolVersion")
    id: uuid.UUID
    project_id: uuid.UUID = msgspec.field(name="projectId")
    type: CommandType

class PauseCommand(msgspec.Struct, rename="camel"):
    protocol_version: Literal[1]
    id: uuid.UUID
    project_id: uuid.UUID
    type: Literal["pause"]
    payload: EmptyPayload

class ErrorPayload(msgspec.Struct, rename="camel"):
    code: str
    message: str
    recoverable: bool
    details: dict[str, object]
```

`Command` is the union of nine concrete structs. Their payloads are exact: `hello(clientVersion, supportedProtocolVersions=[1])`; `configure(scaleFrames, windowSize, windowOverlap, keyframeInterval, cameraRefinementIterations, confidenceThreshold)`; `beginSession(resumeAfterFrameIndex: int | None)`; `enqueueFrame(frameIndex: int, sourceTimestamp: float, relativePath: str)`; and `{}` for `finishInput`, `pause`, `resume`, `cancel`, and `shutdown`. Only `configure` may carry the 16/4 retry values; validate `0 <= resumeAfterFrameIndex <= UInt32.max`, `0 <= frameIndex <= UInt32.max`, finite non-negative timestamps, and package-relative frame paths.

All response/event envelopes carry `protocolVersion`, their own UUID `id`, and `projectId`. `Ack` has `type: "ack"`, `commandId`, and payload `{command}`. A command `ErrorMessage` has `type: "error"`, non-null `commandId`, and `ErrorPayload`; an asynchronous error event uses the same discriminator with `commandId: null`. Event payloads are:

- `ready`: `engineVersion`, `modelIdentifier`, `modelRevision`, `convertedWeightsSHA256`.
- `modelProgress`: `phase` (`validating` or `loading`), `completed`, `total`.
- `frameStarted`: `frameIndex`, `windowIndex`.
- `frameCompleted`: `frameIndex`, `windowIndex`, package-relative `depthPath`, `confidencePath`, `geometryPath`, `pointChunkPath`, and `durationSeconds`.
- `windowCompleted`: `windowIndex`, inclusive `frameStart`, inclusive `frameEnd`, `pointChunkPath`, row-major 16-number `alignmentTransform`, `lastProcessedFrameIndex`, `inlierCount`, and `durationSeconds`.
- `sessionCompleted`: `processedFrames`, `windowCount`, `durationSeconds`.
- `paused`: `queuedFrames`, `processedFrames`; `cancelled`: `lastCompletedWindowIndex` (nullable).
- `warning` and asynchronous `error`: `ErrorPayload`; `heartbeat`: `busy`, `monotonicSeconds`, `queuedFrames`, `processedFrames`, `currentWindow` (nullable).

`write_protocol_fixture` writes sorted-key JSON containing `protocolVersion`, `maximumMessageBytes`, and every command/ack/error/event as `{name, json, framedBytes}` where `framedBytes` is the complete frame represented as an array of UInt8. Generate it deterministically with UUIDs `00000000-0000-0000-0000-000000000001` upward so Swift's compatibility decoder can consume the same corpus.

- [ ] **Step 4: Implement bounded framing without partial-read ambiguity**

```python
MAX_MESSAGE_BYTES = 1_048_576

def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise WorkerFault("TRUNCATED_MESSAGE", "peer closed mid-frame", False)
        chunks.extend(chunk)
    return bytes(chunks)

def read_json_frame(stream: BinaryIO) -> dict[str, object]:
    length = int.from_bytes(_read_exact(stream, 4), "big")
    if length == 0:
        raise WorkerFault("INVALID_MESSAGE_LENGTH", "zero-length JSON frame", False)
    if length > MAX_MESSAGE_BYTES:
        raise WorkerFault("MESSAGE_TOO_LARGE", f"{length} exceeds 1048576", False)
    return msgspec.json.decode(_read_exact(stream, length), type=dict[str, object])

def write_json_frame(stream: BinaryIO, value: object) -> None:
    canonical = msgspec.to_builtins(value)
    body = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise WorkerFault("MESSAGE_TOO_LARGE", f"{len(body)} exceeds 1048576", False)
    stream.write(len(body).to_bytes(4, "big") + body)
    stream.flush()
```

- [ ] **Step 5: Lock, test, and commit**

Run: `cd worker && uv lock --python 3.12.11 && uv sync --frozen --group dev && uv run --frozen python -m cloudpoint_worker.protocol.fixtures --output tests/fixtures/protocol-v1.json && uv run --frozen pytest tests/protocol -q`

Expected: lock creation succeeds, the canonical fixture is written, and all protocol tests PASS.

```bash
git add worker/.python-version worker/pyproject.toml worker/uv.lock worker/src/cloudpoint_worker worker/tests/protocol worker/tests/fixtures/protocol-v1.json
git commit -m "feat(worker): lock runtime and protocol v1"
```

### Task 2: Re-verify and Strictly Convert the Native-Downloaded Checkpoint

**Files:**
- Create: `worker/src/cloudpoint_worker/model_prep/__init__.py`
- Create: `worker/src/cloudpoint_worker/model_prep/provenance.py`
- Create: `worker/src/cloudpoint_worker/model_prep/convert.py`
- Create: `worker/src/cloudpoint_worker/model_prep/cli.py`
- Create: `worker/src/cloudpoint_worker/model/__init__.py`
- Create: `worker/src/cloudpoint_worker/model/config.py`
- Create: `worker/src/cloudpoint_worker/model/weight_specs.py`
- Create: `worker/tests/test_model_prep.py`

**Interfaces:**
- Consumes: the absolute `--checkpoint` path downloaded and size/SHA-verified by native URLSession plus an empty absolute `--destination` converted directory.
- Produces: `verify_checkpoint(path: Path) -> VerifiedArtifact`, `build_weight_specs(config: ModelConfig) -> tuple[WeightSpec, ...]`, `prepare_model(checkpoint: Path, destination: Path, specs: tuple[WeightSpec, ...]) -> ModelManifest`, `lingbot-map-long-f16.safetensors`, `weights-manifest.json`, and `model-manifest.json`.

- [ ] **Step 1: Write rejection and layout-conversion tests**

```python
def test_checkpoint_must_match_size_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "lingbot-map-long.pt"
    path.write_bytes(b"wrong")
    with pytest.raises(WorkerFault, match="MODEL_CHECKSUM_MISMATCH"):
        verify_artifact(path, expected_size=5, expected_sha256="00" * 32)

def test_layouts_are_explicit_and_coverage_is_bijective() -> None:
    state = {
        "conv.weight": torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2),
        "up.weight": torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2),
    }
    specs = [
        WeightSpec("conv.weight", "conv.weight", (2, 3, 2, 2), (2, 2, 2, 3), "conv2d"),
        WeightSpec("up.weight", "up.weight", (3, 2, 2, 2), (2, 2, 2, 3), "conv_transpose2d"),
    ]
    converted, rows = convert_state_dict(state, specs)
    assert converted["conv.weight"].shape == (2, 2, 2, 3)
    assert converted["up.weight"].shape == (2, 2, 2, 3)
    with pytest.raises(WorkerFault, match="MODEL_EXTRA_TENSOR"):
        convert_state_dict({**state, "surprise": torch.zeros(1)}, specs)

def test_prepare_has_no_network_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("network used"))
    assert set(model_prepare_parser().parse_args(["prepare", "--checkpoint", "/tmp/source.pt",
                                                   "--destination", "/tmp/converted"]).__dict__) == {
        "command", "checkpoint", "destination"
    }
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `cd worker && uv run --frozen --extra model-prep pytest tests/test_model_prep.py -q`

Expected: FAIL with `ModuleNotFoundError: cloudpoint_worker.model_prep`.

- [ ] **Step 3: Define immutable topology/weight specifications and fd-based provenance verification**

```python
MODEL_REPO = "robbyant/lingbot-map"
MODEL_REVISION = "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
MODEL_FILENAME = "lingbot-map-long.pt"
MODEL_SIZE = 4_632_303_465
MODEL_SHA256 = "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
SOURCE_COMMIT = "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"

@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    heads: int = 16
    register_tokens: int = 4
    selected_layers: tuple[int, ...] = (4, 11, 17, 23)

@dataclass(frozen=True)
class WeightSpec:
    source_key: str
    destination_key: str
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]
    transform: Literal["identity", "conv2d", "conv_transpose2d"]

@dataclass(frozen=True)
class VerifiedArtifact:
    path: Path
    size: int
    sha256: str

@dataclass(frozen=True)
class ModelManifest:
    schema_version: Literal[1]
    model_identifier: str
    model_revision: str
    source_sha256: str
    converted_sha256: str
    tensor_count: int
    mlx_version: str
    engine_version: str

def verify_artifact(path: Path, expected_size: int, expected_sha256: str) -> VerifiedArtifact:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb", closefd=True) as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise WorkerFault("MODEL_CHECKSUM_MISMATCH", "size mismatch", True)
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        if not hmac.compare_digest(digest, expected_sha256):
            raise WorkerFault("MODEL_CHECKSUM_MISMATCH", "SHA-256 mismatch", True)
    return VerifiedArtifact(path.resolve(), expected_size, digest)
```

`build_weight_specs(ModelConfig())` deterministically enumerates the DINOv2 backbone, 24 frame blocks, 24 global blocks, camera/four-register/scale tokens, DPT depth head, and one-pass causal camera head. It assigns one explicit transform to every source tensor. Task 4 must prove that this destination set and the flattened MLX parameter set are identical before any real weight load.

- [ ] **Step 4: Implement the strict converter and atomic manifests**

Conversion rules are exhaustive: Linear/LayerNorm/parameters use identity; PyTorch Conv2d `OIHW -> OHWI` uses `(0, 2, 3, 1)`; PyTorch ConvTranspose2d `IOHW -> OHWI` uses `(1, 2, 3, 0)`. Cast every floating tensor to little-endian Float16, reject non-floating tensors unless its `WeightSpec` declares an identity scalar, and reject duplicate destinations, missing sources, extra sources, or either-side shape mismatch.

Open the native-provided checkpoint with `O_NOFOLLOW`, hash that descriptor, seek it back to zero, and attempt `torch.load(file_object, map_location="cpu", weights_only=True)` first. Accept either a direct tensor mapping or the tensor mapping under the sole model payload key `model`; top-level non-model metadata never participates in tensor coverage. Only after re-verifying the same open file descriptor may the converter retry `weights_only=False`; execute that retry in a dedicated temporary child whose environment is exactly `PATH`, `TMPDIR`, `PYTHONNOUSERSITE=1`, `PYTHONHASHSEED=0`, `LC_ALL=C`, and no home, token, key, cookie, credential, or proxy variables. Write SafeTensors plus these canonical JSON rows:

```python
row = {
    "sourceKey": "depth_head.resize_layers.0.weight",
    "destinationKey": "depth_head.resize_layers.0.weight",
    "sourceShape": [256, 256, 4, 4],
    "destinationShape": [256, 4, 4, 256],
    "sourceDtype": "float32",
    "destinationDtype": "float16",
    "transform": "conv_transpose2d",
    "sha256": hashlib.sha256(converted_little_endian_bytes).hexdigest(),
}
```

Sort rows by `destinationKey`, write `.partial`, `flush`, `os.fsync`, and `os.replace`. `model-manifest.json` records schema version `1`, all pinned provenance, converted-file SHA-256, tensor count, MLX version, engine version, and conversion UTC timestamp. The converter accepts exactly `cloudpoint-model prepare --checkpoint ABSOLUTE_PATH --destination ABSOLUTE_EMPTY_DIRECTORY` and `cloudpoint-model verify --checkpoint ABSOLUTE_PATH`; it never downloads. `prepare` prints sorted-key JSON Lines for `verifying`, `restrictedLoading`, optional `trustedArtifactLoading`, `converting`, `validating`, and `ready`, then exits `0`. It exits `2` for a digest/path error and `4` for conversion/coverage failure.

- [ ] **Step 5: Test and commit**

Run: `cd worker && uv run --frozen --extra model-prep pytest tests/test_model_prep.py -q`

Expected: PASS, including the no-network assertion, required `--checkpoint`, restricted-load-first, environment scrubbing, bad digest, extra/missing key, both convolution layouts, atomic rename, and deterministic manifest ordering.

```bash
git add worker/src/cloudpoint_worker/model_prep worker/src/cloudpoint_worker/model/config.py worker/src/cloudpoint_worker/model/weight_specs.py worker/tests/test_model_prep.py
git commit -m "feat(worker): add verified strict model conversion"
```

### Task 3: Reproduce Preprocessing and Generate Pinned PyTorch Fixtures

**Files:**
- Create: `worker/.gitignore`
- Create: `worker/src/cloudpoint_worker/preprocess.py`
- Create: `worker/tools/export_reference.py`
- Create: `worker/tests/fixtures/courthouse/000000.png` through `000008.png`
- Create: `worker/tests/fixtures/courthouse/provenance.json`
- Create: `worker/tests/fixtures/parity/generated/.gitkeep`
- Create: `worker/tests/fixtures/parity/generated/.gitignore`
- Create: `worker/tests/fixtures/preprocess/orientation-6-rgba.png`
- Create: `worker/tests/test_preprocess.py`

**Interfaces:**
- Consumes: an oriented JPEG/PNG path and pinned upstream checkout/reference checkpoint paths.
- Produces: `preprocess_image(path: Path) -> PreprocessedFrame`, plus local `preprocess.safetensors`, `leaf-fp32.safetensors`, `e2e-fp32.safetensors`, and `reference-manifest.json`.

- [ ] **Step 1: Write transform, color, and upstream-fixture tests**

```python
def test_preprocess_applies_orientation_rgb_518_patch_grid_and_inverse() -> None:
    fixture_root = Path(__file__).parent / "fixtures"
    result = preprocess_image(fixture_root / "preprocess/orientation-6-rgba.png")
    assert result.source_size == (20, 40)  # EXIF orientation 6 applied to a 40x20 source
    assert max(result.model_size) == 518
    assert result.model_size[0] % 14 == result.model_size[1] % 14 == 0
    assert result.rgb.shape[-1] == 3
    model_corner = np.array([result.model_size[0] - 1.0, result.model_size[1] - 1.0, 1.0])
    source_corner = result.model_to_source @ model_corner
    round_trip = np.linalg.inv(result.model_to_source) @ source_corner
    np.testing.assert_allclose(round_trip, model_corner, atol=1e-9)

def test_courthouse_fixture_matches_pinned_reference_pixels() -> None:
    fixture_root = Path(__file__).parent / "fixtures"
    actual = preprocess_image(fixture_root / "courthouse/000000.png")
    expected = safetensors.numpy.load_file(
        fixture_root / "parity/generated/preprocess.safetensors"
    )["frame.0.normalized"]
    np.testing.assert_allclose(actual.normalized, expected, rtol=0, atol=1 / 255)
```

- [ ] **Step 2: Run and verify fixture absence is explicit**

Run: `cd worker && uv run --frozen pytest tests/test_preprocess.py -q`

Expected: the pure transform test FAILS because `preprocess_image` is absent; the reference test SKIPS only with `reference fixture not generated`.

- [ ] **Step 3: Implement deterministic preprocessing and its affine metadata**

Apply EXIF orientation, convert embedded profiles through Pillow `ImageCms` to sRGB, composite alpha over white and discard it, resize with bicubic sampling so the longest dimension is 518, round each output dimension to the nearest positive multiple of 14, and center crop/pad with white according to the pinned rule. Return HWC Float32 `[0,1]` RGB and `(rgb - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`; record a Float64 3-by-3 `model_to_source` transform that includes orientation, scale, crop, and padding.

```python
@dataclass(frozen=True)
class PreprocessedFrame:
    rgb: np.ndarray
    normalized: np.ndarray
    model_to_source: np.ndarray
    source_size: tuple[int, int]
    model_size: tuple[int, int]

def _patch_extent(value: float) -> int:
    return max(14, min(518, round(value / 14) * 14))
```

Treat `source_size` and `model_size` as `(width, height)`. The committed orientation fixture is a 40-by-20 RGBA PNG with EXIF orientation 6; construct it deterministically in the test-fixture creation step with Pillow, transparent pixels containing non-white RGB, and assert conversion composites those pixels over white before alpha removal.

- [ ] **Step 4: Add traceable Apache-2.0 inputs and the reference exporter**

Copy upstream `example/courthouse/000000.png` through `000008.png` from commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`. `provenance.json` records source URL, commit, Apache-2.0 license, and these filename/SHA-256 pairs; fixture loading fails on any mismatch:

```json
{
  "000000.png":"8b07befd90f26e87411e64f7fd433c6b96b2dec6d640ce1e3c78f35402043b8c",
  "000001.png":"382233d33a4e6c4c596d09cf8d44c361962df502a180bfa9c1a64e8cf10e3425",
  "000002.png":"234711bff562f68da4e8515d0788404d406591a93f0efe4391c34ff6c7f3f1e2",
  "000003.png":"54549d7a0a6386d1e2279060836edddd20b68c4c5f3acfe7198ed20cc0c869b4",
  "000004.png":"52a808711be5309c8be9041acd947d2adba0d754da0c35da6537c1fb1b210c9f",
  "000005.png":"7d44bf8ecaa77e06567a431cb0d731bf31c486fa7c397ee76d8571d43a4f6c87",
  "000006.png":"b4c6aa89f28d696c3acdb6b06917ed7d51a21dbb1a770b6e7b4ec2262fc8030a",
  "000007.png":"e94422026dee320f1c1455ec461df9dbff80c10557a582ec5f62d0a57dd66eea",
  "000008.png":"b18bfeb7f53b4a24e873eb466ec10f6dae41a430defbe82566c9b2d587c6a3ad"
}
```

The exporter verifies upstream HEAD and checkpoint digest, instantiates upstream `GCTStream(use_sdpa=True, enable_point=False, enable_depth=True, camera_num_iterations=1)`, registers hooks for patch embedding, first/middle/final frame and global blocks, DPT resize layers, and camera trunk, then writes Float32 preprocessing, leaf tensors, selected aggregator layers `[4,11,17,23]`, pose/depth/confidence/intrinsics, and reprojected points. Generated tensor files remain ignored, as does `worker/.reference/`.

Run:

```bash
cd worker
test -d .reference/lingbot-map/.git || git clone --filter=blob:none https://github.com/Robbyant/lingbot-map.git .reference/lingbot-map
git -C .reference/lingbot-map fetch --depth 1 origin 7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
git -C .reference/lingbot-map checkout --detach FETCH_HEAD
uv run --frozen --extra reference python tools/export_reference.py --upstream .reference/lingbot-map --checkpoint "$CLOUDPOINT_LINGBOT_PT" --output tests/fixtures/parity/generated
```

Expected: exits `0`, reports source commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`, `9 frames`, `restricted load attempted`, and writes four manifest-verified files without modifying the checkpoint.

- [ ] **Step 5: Run tests and commit**

Run: `cd worker && uv run --frozen pytest tests/test_preprocess.py -q`

Expected: pure tests PASS; the local reference comparison PASS when generated and otherwise has exactly one documented skip.

```bash
git add worker/.gitignore worker/src/cloudpoint_worker/preprocess.py worker/tools/export_reference.py worker/tests/fixtures/courthouse worker/tests/fixtures/parity worker/tests/fixtures/preprocess worker/tests/test_preprocess.py
git commit -m "test(worker): pin preprocessing reference fixtures"
```

### Task 4: Port the Lingbot Topology and Pass Layer/End-to-End Differential Tests

**Files:**
- Modify: `worker/src/cloudpoint_worker/model/config.py`
- Modify: `worker/src/cloudpoint_worker/model/weight_specs.py`
- Create: `worker/src/cloudpoint_worker/model/layers.py`
- Create: `worker/src/cloudpoint_worker/model/rope.py`
- Create: `worker/src/cloudpoint_worker/model/backbone.py`
- Create: `worker/src/cloudpoint_worker/model/aggregator.py`
- Create: `worker/src/cloudpoint_worker/model/heads.py`
- Create: `worker/src/cloudpoint_worker/model/lingbot.py`
- Create: `worker/tests/test_layers_parity.py`
- Create: `worker/tests/test_model_parity.py`

**Interfaces:**
- Consumes: HWC `PreprocessedFrame.normalized`, HWC `PreprocessedFrame.rgb`, Task 2's `ModelConfig`/`WeightSpec` set, and an absolute converted model directory containing exactly `lingbot-map-long-f16.safetensors`, `weights-manifest.json`, and `model-manifest.json`.
- Produces: `LingbotMap.load(model_dir: Path) -> LingbotMap`, `LingbotMap.forward_scale(images: mx.array) -> FrameBatchPrediction`, `LingbotMap.forward_frame(image: mx.array, append_cache: bool) -> FramePrediction`, `LingbotMap.infer_direct(images: mx.array, scale_frames: int = 8) -> FrameBatchPrediction`, and `LingbotMap.weight_specs() -> tuple[WeightSpec, ...]`.

Define the prediction types in `model/lingbot.py` and use them unchanged in Tasks 5–7:

```python
@dataclass(frozen=True)
class FramePrediction:
    depth: mx.array               # H,W Float16
    confidence: mx.array          # H,W Float16
    pose_encoding: mx.array       # 9 Float32
    intrinsics: mx.array          # 3,3 Float32
    camera_to_world: mx.array      # 4,4 Float32

@dataclass(frozen=True)
class FrameBatchPrediction:
    frames: tuple[FramePrediction, ...]
    selected_features: dict[int, mx.array]
```

- [ ] **Step 1: Write exact topology and parity assertions**

```python
def test_topology_matches_pinned_upstream() -> None:
    model = LingbotMap(ModelConfig())
    assert (model.config.embed_dim, model.config.depth, model.config.heads) == (1024, 24, 16)
    assert model.config.patch_size == 14
    assert model.aggregator.selected_layers == (4, 11, 17, 23)
    assert len(model.aggregator.frame_blocks) == len(model.aggregator.global_blocks) == 24
    assert model.aggregator.patch_start == 6  # camera + four registers + scale
    assert model.point_head is None and model.depth_head is not None and model.camera_head is not None

@pytest.mark.real_model
def test_float32_leaf_and_aggregator_parity(real_model, reference) -> None:
    actual = real_model.probe(reference.inputs)
    for name, expected in reference.leaf.items():
        np.testing.assert_allclose(actual.leaf[name], expected, rtol=1e-3, atol=1e-4)
    for name, expected in reference.aggregator.items():
        assert cosine(actual.aggregator[name], expected) >= 0.995
```

- [ ] **Step 2: Run tests and verify missing implementation**

Run: `cd worker && uv run --frozen pytest tests/test_layers_parity.py tests/test_model_parity.py -q`

Expected: FAIL importing `cloudpoint_worker.model.lingbot`; real-model cases are separately marked, never silently selected by ordinary unit runs.

- [ ] **Step 3: Port leaf operations in Float32-first parity order**

Implement NHWC MLX Conv2d/ConvTranspose2d, Linear, GELU, SiLU, LayerNorm (`eps=1e-6` where upstream uses it), LayerScale (`0.01`), MLP ratio four, bicubic/bilinear resize with upstream `align_corners=True` coordinate behavior, 2D RoPE frequency `100`, temporal RoPE theta `10000`, q/k per-head normalization, and scaled dot-product attention. Each class accepts `dtype` and probe callbacks; parity tests cover one operation before composing blocks.

```python
def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None) -> mx.array:
    scale = q.shape[-1] ** -0.5
    scores = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
    if mask is not None:
        scores = mx.where(mask, scores, mx.array(-1e9, dtype=mx.float32))
    return (mx.softmax(scores, axis=-1) @ v.astype(mx.float32)).astype(q.dtype)
```

- [ ] **Step 4: Assemble the exact current upstream graph**

Port DINOv2 ViT-L/14 with four register tokens and 24 blocks. The streaming aggregator adds camera/four-register/scale tokens, runs 24 alternating same-frame and causal-global blocks, and concatenates frame/global outputs at groups 4, 11, 17, and 23 to 2048 channels. Port DPT projections `[256,512,1024,1024]`, transpose-convolution scales `[4,2,1]`, stride-two fourth projection, four 256-channel fusion blocks, exponential depth and `expm1` confidence activation. Port the 2048-channel camera token norm, four causal trunk blocks, one refinement iteration, nine-value `absT_quaR_FoV` output, and OpenCV world-to-camera convention.

`weight_specs()` must flatten every MLX parameter, map exactly one checkpoint key, annotate layout kind, and make Task 2's strict converter public CLI usable:

Run: `cd worker && uv run --frozen --extra model-prep cloudpoint-model prepare --checkpoint "$CLOUDPOINT_LINGBOT_PT" --destination "$CLOUDPOINT_MODEL_DIR"`

Expected: re-verifies the native-downloaded source digest, prints zero missing/extra/duplicate tensors, performs no network request, then writes only SafeTensors and the two manifests into the destination. `CLOUDPOINT_MODEL_DIR` is the native installation's `converted/` directory and is the exact directory later passed to `cloudpoint-worker --model`.

- [ ] **Step 5: Gate leaf and complete direct-sequence parity**

```python
@pytest.mark.real_model
def test_end_to_end_geometry_parity(real_model, reference) -> None:
    got = real_model.infer_direct(reference.inputs, scale_frames=8)
    valid = np.isfinite(reference.depth) & (reference.depth > 0)
    assert pearson(got.depth[valid], reference.depth[valid]) >= 0.99
    assert median_relative_error(got.depth[valid], reference.depth[valid]) <= 0.03
    assert shared_scale_translation_error(got.c2w, reference.c2w) <= 0.03
    assert max_rotation_geodesic_degrees(got.c2w, reference.c2w) <= 1.0
    assert max_relative_error(got.intrinsics, reference.intrinsics) <= 0.01
    np.testing.assert_allclose(reproject(got), reference.reprojected, rtol=0.03, atol=0.5)
```

Run: `cd worker && CLOUDPOINT_MODEL_DIR="$CLOUDPOINT_MODEL_DIR" uv run --frozen pytest -m real_model tests/test_layers_parity.py tests/test_model_parity.py -q`

Expected: every numerical threshold PASS; a camera-matrix-only agreement with failed reprojection is a FAIL.

- [ ] **Step 6: Commit the verified topology**

```bash
git add worker/src/cloudpoint_worker/model worker/tests/test_layers_parity.py worker/tests/test_model_parity.py
git commit -m "feat(worker): port Lingbot Map topology to MLX"
```

### Task 5: Add Streaming KV Cache, Windowing, and Robust Sim3 Alignment

**Files:**
- Create: `worker/src/cloudpoint_worker/model/cache.py`
- Create: `worker/src/cloudpoint_worker/sim3.py`
- Create: `worker/src/cloudpoint_worker/windows.py`
- Create: `worker/tests/test_cache_windows_sim3.py`

**Interfaces:**
- Consumes: `LingbotMap.forward_scale/forward_frame`, ordered `PersistedFrame` records, overlap camera centers and sampled high-confidence correspondences.
- Produces: `KVCache.append(frame_index, key, value, persistent)`, `KVCache.attend(frame_index, key, value, append)`, `KVCache.evict()`, `KVCache.reset()`, `plan_windows(frame_count: int, config: WindowConfig) -> list[Window]`, `estimate_sim3(source: np.ndarray, target: np.ndarray, frame_ids: np.ndarray, seed: int = 0) -> Sim3`, and `WindowRunner.run(frames: Sequence[PersistedFrame]) -> Iterator[WindowResult]`.

```python
@dataclass(frozen=True)
class PersistedFrame:
    index: int
    source_timestamp: float
    relative_path: str

@dataclass(frozen=True)
class WindowConfig:
    size: int = 32
    overlap: int = 8
    scale_frames: int = 8
    keyframe_interval: int = 1

@dataclass(frozen=True)
class Window:
    index: int
    start: int                 # inclusive sequence offset
    stop: int                  # exclusive sequence offset
    overlap_frames: tuple[int, ...]

@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray       # 3,3 Float64
    translation: np.ndarray    # 3 Float64
    inlier_mask: np.ndarray    # N bool

@dataclass(frozen=True)
class WindowResult:
    window: Window
    predictions: tuple[FramePrediction, ...]
    alignment: Sim3
```

- [ ] **Step 1: Write cache and deterministic window tests**

```python
def test_cache_keeps_anchors_skips_non_keyframes_evicts_live_and_resets() -> None:
    cache = KVCache(layers=2, heads=2, head_dim=4, scale_frames=2, live_frames=3)
    cache.append(frame=0, key=K0, value=V0, persistent=True)
    cache.append(frame=1, key=K1, value=V1, persistent=True)
    cache.attend(frame=2, key=K2, value=V2, append=False)
    for frame in range(2, 7):
        cache.append(frame, K(frame), V(frame), persistent=True)
    assert cache.frame_ids == (0, 1, 4, 5, 6)
    cache.reset()
    assert cache.frame_ids == ()

def test_window_plan_is_direct_then_32_with_eight_overlap() -> None:
    DEFAULTS = WindowConfig(size=32, overlap=8, scale_frames=8, keyframe_interval=1)
    assert plan_windows(32, DEFAULTS) == [Window(0, 0, 32, ())]
    assert plan_windows(57, DEFAULTS) == [
        Window(0, 0, 32, ()), Window(1, 24, 56, tuple(range(24, 32))),
        Window(2, 48, 57, tuple(range(48, 56))),
    ]
```

- [ ] **Step 2: Write noisy/outlier and degenerate Sim3 tests**

Generate 80 known correspondences from scale `1.7`, a 21-degree normalized-axis rotation, translation `[2,-3,0.5]`, Gaussian sigma `0.002`, and 20 deterministic outliers. Assert scale error `<0.01`, rotation `<0.2` degree, translation norm `<0.03`, and inliers `>=58`. Also assert collinear points, fewer than three non-collinear camera centers, or fewer than 12 total correspondences raise recoverable `ALIGNMENT_DEGENERATE`; identity is never returned as fallback.

- [ ] **Step 3: Run and verify failures**

Run: `cd worker && uv run --frozen pytest tests/test_cache_windows_sim3.py -q`

Expected: FAIL importing `KVCache`, `plan_windows`, and `estimate_sim3`.

- [ ] **Step 4: Implement bounded cache semantics**

Store per-layer Float16 K/V as `[heads, frames, tokens, head_dim]`, anchors separately from live frames, and camera-head caches per refinement pass. `attend(frame_index, key, value, append=False)` includes current K/V in attention but does not mutate cache; eviction preserves the first eight anchors and newest 24 live frames for the default 32-slot window. `reset()` clears arrays, frame IDs, temporal RoPE indices, and camera refinement state. Force `mx.eval` after each append/output so lazy graphs do not retain prior frames.

- [ ] **Step 5: Implement robust overlap alignment and window commits**

Use all finite overlap camera centers plus a deterministic grid sample of high-confidence unprojected correspondences. Run seeded RANSAC with 256 three-point hypotheses, reject rank `<2`, score normalized residuals with MAD threshold `max(3*MAD, 1e-4)`, require at least 12 inliers spanning three frames, then refit Umeyama Sim(3) over inliers. Apply scale/rotation/translation to current-window c2w translations and point data; rotate c2w bases without scale. Persist the transform and inlier diagnostics in `WindowResult`.

`WindowRunner` chooses direct mode for `<=32`; otherwise it advances by 24, resets cache per window, uses its first eight frames as window anchors, emits only non-duplicate frame outputs, and calls its checkpoint callback only after alignment and all atomic artifacts succeed. `Window.stop` is always exclusive; a completed 0–31 window is `Window(start=0, stop=32)` and produces CPC/event `frameStart=0`, `frameEnd=31`, and `lastProcessedFrameIndex=31`.

- [ ] **Step 6: Test and commit**

Run: `cd worker && uv run --frozen pytest tests/test_cache_windows_sim3.py -q`

Expected: PASS for append/skip/evict/reset, boundary/final-partial windows, robust recovery, determinism, and every degeneracy case.

```bash
git add worker/src/cloudpoint_worker/model/cache.py worker/src/cloudpoint_worker/sim3.py worker/src/cloudpoint_worker/windows.py worker/tests/test_cache_windows_sim3.py
git commit -m "feat(worker): add cached windowed reconstruction"
```

### Task 6: Decode Geometry and Atomically Write Predictions and CPC1 Chunks

**Files:**
- Create: `worker/src/cloudpoint_worker/geometry.py`
- Create: `worker/src/cloudpoint_worker/cpc.py`
- Create: `worker/src/cloudpoint_worker/outputs.py`
- Create: `worker/tests/test_geometry_cpc.py`

**Interfaces:**
- Consumes: pose encoding `[Txyz, quaternion wxyz, fovH, fovW]`, depth/confidence/RGB arrays, frame metadata, and window Sim3.
- Produces: `decode_camera(pose_encoding: np.ndarray, image_size: tuple[int, int]) -> DecodedCamera`, `unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray`, `filter_and_reduce_points(depth, confidence, rgb, intrinsics, camera_to_world, source_frame, confidence_floor, voxel_size, flags) -> tuple[CPCVertex, ...]`, `write_frame_outputs(project_root: Path, frame: PersistedFrame, prediction: FramePrediction, preprocessed: PreprocessedFrame) -> FrameArtifactPaths`, and `write_cpc(path: Path, frame_start: int, frame_end_inclusive: int, vertices: Sequence[CPCVertex]) -> CPCDescriptor`.

```python
@dataclass(frozen=True)
class CPCVertex:
    position: tuple[float, float, float]
    rgba: tuple[int, int, int, int]
    confidence: float
    flags: int
    source_frame: int

@dataclass(frozen=True)
class DecodedCamera:
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray

@dataclass(frozen=True)
class FrameArtifactPaths:
    depth_path: str
    confidence_path: str
    geometry_path: str

@dataclass(frozen=True)
class CPCDescriptor:
    relative_path: str
    point_count: int
    frame_start: int
    frame_end: int              # inclusive
```

- [ ] **Step 1: Write camera convention, filtering, and binary-format tests**

```python
def test_unprojection_uses_opencv_w2c_then_c2w() -> None:
    depth = np.array([[2.0]], np.float32)
    intrinsics = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], np.float32)
    c2w = translation_matrix(1, 2, 3)
    points = unproject_depth(depth, intrinsics, c2w)
    np.testing.assert_allclose(points[0, 0], [1, 2, 5])

def test_cpc1_exact_layout_and_rejects_nan(tmp_path: Path) -> None:
    THREE_VERTICES = make_vertices(position=[[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    path = tmp_path / "window-00000001.cpc"
    write_cpc(path, frame_start=1, frame_end_inclusive=2, vertices=THREE_VERTICES)
    raw = path.read_bytes()
    assert raw[:4] == b"CPC1"
    assert len(raw) == 32 + 3 * 24
    assert read_cpc(path).point_count == 3
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(path, 1, 2, vertices_with_nan())
```

- [ ] **Step 2: Run and verify missing geometry failure**

Run: `cd worker && uv run --frozen pytest tests/test_geometry_cpc.py -q`

Expected: FAIL importing `unproject_depth` and `write_cpc`.

- [ ] **Step 3: Implement pose decoding and geometry filtering**

Normalize quaternion `wxyz`; decode upstream OpenCV world-to-camera `[R|t]`; construct intrinsics with `fy=(H/2)/tan(fovH/2)`, `fx=(W/2)/tan(fovW/2)`, `cx=W/2`, `cy=H/2`; invert to camera-to-world. Unproject pixel centers with `K^-1 [u,v,1] * depth`, then transform by c2w. Reject depth `<=0`, non-finite values, confidence `< threshold`, invalid pose/intrinsics, and outputs outside configured counts. Carry preprocessed RGB and source-frame index.

Voxel reduction uses `floor(position / voxel_size)` Int64 keys, keeps the highest-confidence point per voxel, breaks ties by the lowest source-frame then original pixel index, and sorts final points by `(voxelX,voxelY,voxelZ)` for reproducible bytes. Set RGBA alpha to 255; flags bit 0 means keyframe and bit 1 means anchor, with bits 2–15 zero in version 1.

- [ ] **Step 4: Write exact atomic artifacts**

Depth and confidence are raw little-endian Float16 files; geometry JSON contains protocol/engine/model versions, source/model sizes, model-to-source transform, nine-value pose, row-major c2w and intrinsics, confidence floor, and reconstruction-unit label. CPC header is exactly 32 bytes: `4s magic`, `<H version=1`, `<H stride=24`, `<Q pointCount`, inclusive `<I frameStart`, inclusive `<I frameEnd`, eight zero reserved bytes. Each vertex is `<fff4BeHI`; validate finite position/confidence, UInt16 flags, UInt32 frame indices, `frameStart <= frameEnd`, count `<= 50_000_000`, and total `32 + count*24 <= 1_200_000_032` before rename.

Write each target as `name.partial`, flush/fsync, rename, then fsync its parent directory. Any failure removes only that operation's `.partial` and leaves earlier completed-window files intact.

- [ ] **Step 5: Test and commit**

Run: `cd worker && uv run --frozen pytest tests/test_geometry_cpc.py -q`

Expected: PASS for known intrinsics/pose, quaternion convention, filtering, deterministic voxel winners, little-endian bytes, NaN/count/stride/truncation rejection, and interrupted atomic writes.

```bash
git add worker/src/cloudpoint_worker/geometry.py worker/src/cloudpoint_worker/cpc.py worker/src/cloudpoint_worker/outputs.py worker/tests/test_geometry_cpc.py
git commit -m "feat(worker): emit recoverable prediction and CPC artifacts"
```

### Task 7: Supervise Sessions with Pause, Cancel, Heartbeats, and Recovery

**Files:**
- Create: `worker/src/cloudpoint_worker/session.py`
- Create: `worker/src/cloudpoint_worker/server.py`
- Create: `worker/tests/test_session_server.py`

**Interfaces:**
- Consumes: decoded protocol commands, project/model roots, frame-relative paths, `WindowRunner`, and an event sink.
- Produces: `SessionRunner(project_root: Path, model: LingbotMap, event_sink: EventSink, resume_after_frame_index: int | None)`, exactly one ack/error per command, all version-1 events, a heartbeat at most five seconds apart while busy, and restart from the native-provided completed-window boundary.

```python
class SessionState(enum.StrEnum):
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
```

- [ ] **Step 1: Write lifecycle and acknowledgement tests with a fake model**

```python
async def test_pause_cancel_and_recovery_are_boundary_safe(project, fake_model) -> None:
    runner = SessionRunner(project.root, fake_model, event_sink=events.append,
                           resume_after_frame_index=31)
    await runner.begin(CONFIG)
    assert runner.next_frame_index == 32
    for frame in persisted_frames(start=32, count=8):
        await runner.enqueue(frame)
    await runner.pause()
    assert runner.state == SessionState.PAUSED
    assert events[-1].type == "paused"
    await runner.resume()
    await runner.cancel()
    assert events[-1].type == "cancelled"
    assert not list(project.root.rglob("*.partial"))

async def test_every_command_has_exactly_one_terminal_response(server) -> None:
    await server.feed(valid_command("pause"))
    assert len(server.responses_for(COMMAND_ID)) == 1
    await server.feed(malformed_enqueue_outside_project())
    assert len(server.responses_for(BAD_ID)) == 1
    assert server.responses_for(BAD_ID)[0].payload.code == "PATH_OUTSIDE_PROJECT"
```

- [ ] **Step 2: Run and verify missing runner/server**

Run: `cd worker && uv run --frozen pytest tests/test_session_server.py -q`

Expected: FAIL importing `SessionRunner` and `WorkerServer`.

- [ ] **Step 3: Implement the queue and cooperative controls**

Maintain FIFO persisted-frame records, never image buffers. `pause` clears an `asyncio.Event`; check it before preprocessing, model forward, Sim3, and every atomic output. `cancel` sets a cancellation token, unblocks pause, discards the active incomplete window and its `.partial` files, resets MLX caches, emits `cancelled`, and preserves sampled frames/completed windows. `finishInput` closes enqueueing and processes a final partial window.

The worker never reads or writes `Manifest.json`; `SessionController` remains its sole owner. `beginSession.resumeAfterFrameIndex` is the native app's validated inclusive completed-window boundary. Reject any subsequent enqueue at or below that value, remove stale `.partial` only beneath `Predictions/` and `Points/`, validate referenced completed artifacts when they are reused, and begin with `next_frame_index = resumeAfterFrameIndex + 1` (or zero for null). `windowCompleted` carries a checkpoint candidate; only the native app commits it to the manifest. Map `MemoryError`/MLX allocation errors to `ALLOCATION_FAILED`; report it recoverably so the native supervisor can restart exactly once with 16-frame/four-overlap configuration. Map degenerate alignment to recoverable `ALIGNMENT_DEGENERATE` and invalid model/output to fatal stable codes.

- [ ] **Step 4: Implement the socket adapter and heartbeat**

`cloudpoint-worker serve --socket PATH --project PATH --model PATH` connects within ten seconds to the app-created `AF_UNIX` listener, verifies the socket/project root and exact converted model directory, then reads framed commands on a dedicated asyncio task. The app sends `hello` first; after its ack the worker validates/loads the model, emits `modelProgress`, and emits `ready`. Serialize all writes through one lock. Start heartbeat when work begins; emit `{type:"heartbeat", payload:{busy:true, monotonicSeconds, queuedFrames, processedFrames, currentWindow}}` immediately and every five seconds, stopping only when idle/completed/cancelled/failed.

Command handling validates the project UUID, legal state, and a bounded 4,096-entry recent command-ID set before mutation, sends its sole ack/error, then emits lifecycle events. A duplicate ID receives one `DUPLICATE_COMMAND_ID` error and never repeats the mutation. `shutdown` cancels tasks, resets model/cache, closes the socket, and exits `0`; EOF follows the same cleanup. Never log JSON bodies containing paths; log command IDs, relative artifact paths, error codes, and tracebacks to stderr.

- [ ] **Step 5: Test fault injection and commit**

Run: `cd worker && uv run --frozen pytest tests/test_session_server.py -q`

Expected: PASS for hello-before-ready, ready/progress/completion, pause/resume, pause-while-enqueueing, cancellation at each cooperative boundary, socket EOF, malformed/zero/oversize messages, duplicate IDs, exact ack/error count, five-second fake-clock heartbeats, stale partial cleanup, path traversal, native-provided resume boundaries, and proof that `Manifest.json` is unchanged.

```bash
git add worker/src/cloudpoint_worker/session.py worker/src/cloudpoint_worker/server.py worker/tests/test_session_server.py
git commit -m "feat(worker): add supervised resumable sessions"
```

### Task 8: Expose a Real CLI Smoke Run and Verify the Worker End to End

**Files:**
- Create: `worker/src/cloudpoint_worker/cli.py`
- Create: `worker/tests/test_cli_smoke.py`
- Modify: `worker/src/cloudpoint_worker/model_prep/cli.py`

**Interfaces:**
- Consumes: a verified converted model directory, project directory, and ordered persisted image paths.
- Produces: `cloudpoint-worker health`, socket `serve`, independently testable `run`, and `protocol-fixture`; `run` emits protocol-v1 JSON Lines to stdout and real project prediction/CPC files, while `protocol-fixture --output PATH` regenerates Task 1's Swift/Python compatibility corpus byte-for-byte.

- [ ] **Step 1: Write subprocess tests before the CLI**

```python
def test_health_is_machine_readable(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cloudpoint_worker.cli", "health", "--model", str(tmp_path)],
        text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "MODEL_UNAVAILABLE"

@pytest.mark.real_model
def test_real_run_writes_frame_and_chunk_outputs(real_model_dir, smoke_project) -> None:
    fixture_root = Path(__file__).parent / "fixtures/courthouse"
    courthouse_inputs = tuple(fixture_root / f"{index:06d}.png" for index in range(9))
    result = run_cli("run", "--project", smoke_project, "--model", real_model_dir,
                     "--frames", *courthouse_inputs)
    assert result.returncode == 0
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["type"] == "sessionCompleted"
    assert len(list((smoke_project / "Predictions").glob("*.geometry.json"))) == 9
    assert validate_cpc(next((smoke_project / "Points").glob("*.cpc"))).point_count > 0
```

- [ ] **Step 2: Run and verify CLI failure**

Run: `cd worker && uv run --frozen pytest tests/test_cli_smoke.py -q`

Expected: FAIL because `cloudpoint_worker.cli` is absent; the real-model case is explicitly deselected without `-m real_model`.

- [ ] **Step 3: Implement stable exit codes and direct-run wiring**

Use `argparse` subcommands with required absolute `--model` and `--project`. `health` validates architecture/macOS/Python/MLX, both manifests, converted SafeTensors digest, strict MLX weight loading, and prints one JSON object. `run` requires a path ending `.cloudpoint`, creates that root plus only `Frames`, `Predictions`, `Points`, and `Logs` when absent, and never creates or mutates `Manifest.json`; imports the ordered `--frames`, applies default direct/windowed configuration, passes events through the same `SessionRunner` as `serve`, and prints one compact sorted-key JSON event per line. Add `--resume-after-frame-index` with a null default for direct recovery testing.

Exit codes are `0` success, `2` correctable setup/input failure, `3` protocol/unsafe-path failure, `4` engine/model/output failure, and `130` cancellation/SIGINT. SIGINT calls cooperative cancel, waits for cleanup, prints `cancelled`, and exits `130`; a second SIGINT exits immediately after closing the output stream.

- [ ] **Step 4: Run all unit and protocol gates**

Run: `cd worker && uv sync --frozen --all-extras && uv run --frozen ruff check src tests tools && uv run --frozen pytest tests/protocol -q && uv run --frozen pytest -m "not real_model" -q`

Expected: Ruff exits `0`; all non-real tests PASS; only the documented locally generated parity tests are skipped.

- [ ] **Step 5: Prepare the pinned model and run the independent real worker**

```bash
cd worker
uv sync --frozen --group dev --extra model-prep --extra reference
uv run --frozen --extra model-prep cloudpoint-model prepare --checkpoint "$CLOUDPOINT_LINGBOT_PT" --destination "$CLOUDPOINT_MODEL_DIR"
uv run --frozen --extra reference python tools/export_reference.py --upstream .reference/lingbot-map --checkpoint "$CLOUDPOINT_LINGBOT_PT" --output tests/fixtures/parity/generated
CLOUDPOINT_MODEL_DIR="$CLOUDPOINT_MODEL_DIR" uv run --frozen pytest -m real_model -q
smoke_root="$(mktemp -d /tmp/cloudpoint-worker-smoke.XXXXXX)"
smoke_project="$smoke_root/Smoke.cloudpoint"
uv run --frozen cloudpoint-worker run --project "$smoke_project" --model "$CLOUDPOINT_MODEL_DIR" --frames tests/fixtures/courthouse/000000.png tests/fixtures/courthouse/000001.png tests/fixtures/courthouse/000002.png tests/fixtures/courthouse/000003.png tests/fixtures/courthouse/000004.png tests/fixtures/courthouse/000005.png tests/fixtures/courthouse/000006.png tests/fixtures/courthouse/000007.png tests/fixtures/courthouse/000008.png
uv run --frozen cloudpoint-worker protocol-fixture --output tests/fixtures/protocol-v1.json
git diff --exit-code -- tests/fixtures/protocol-v1.json
```

Expected: model preparation reports the exact pinned size/SHA and strict zero-key mismatch without network access; every real parity threshold passes; the CLI emits `ready`, nine `frameStarted`/`frameCompleted` pairs, one `windowCompleted` with `frameStart=0`/`frameEnd=8`, and `sessionCompleted`; the project contains nine depth/confidence/geometry sets and one valid non-empty CPC1 chunk with inclusive header bounds 0–8. The protocol fixture has no diff, the executable resolves under `worker/.venv`, and the socket/network-denial tests prove the worker opens only its passed Unix socket and no TCP socket.

- [ ] **Step 6: Commit the independently runnable worker**

```bash
git add worker/src/cloudpoint_worker/cli.py worker/src/cloudpoint_worker/model_prep/cli.py worker/tests/test_cli_smoke.py
git commit -m "feat(worker): ship CLI-testable MLX reconstruction worker"
```

The resulting subproject is independently usable without the Swift app: native URLSession or the tester supplies the pinned checkpoint, `cloudpoint-model prepare --checkpoint` re-verifies and converts it, `cloudpoint-worker health` validates the converted directory, `cloudpoint-worker run` reconstructs ordered files into project artifacts, and `cloudpoint-worker serve` exposes the identical engine through protocol version 1.

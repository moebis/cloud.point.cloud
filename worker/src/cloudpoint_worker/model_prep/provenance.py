"""Pinned checkpoint provenance and descriptor-based verification."""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from cloudpoint_worker.errors import WorkerFault

MODEL_REPO = "robbyant/lingbot-map"
MODEL_REVISION = "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
MODEL_FILENAME = "lingbot-map-long.pt"
MODEL_SIZE = 4_632_303_465
MODEL_SHA256 = "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
SOURCE_COMMIT = "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"


@dataclass(frozen=True)
class VerifiedArtifact:
    path: Path
    size: int
    sha256: str


def _checksum_fault(message: str) -> WorkerFault:
    return WorkerFault("MODEL_CHECKSUM_MISMATCH", message, True)


def verify_open_file(
    source: object, expected_size: int, expected_sha256: str
) -> VerifiedArtifact:
    """Verify one already-open file descriptor and rewind it for loading."""

    fileno = source.fileno()  # type: ignore[attr-defined]
    info = os.fstat(fileno)
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
        raise _checksum_fault("size mismatch")
    source.seek(0)  # type: ignore[attr-defined]
    digest = hashlib.file_digest(source, "sha256").hexdigest()  # type: ignore[arg-type]
    if not hmac.compare_digest(digest, expected_sha256):
        raise _checksum_fault("SHA-256 mismatch")
    source.seek(0)  # type: ignore[attr-defined]
    return VerifiedArtifact(Path(f"/dev/fd/{fileno}"), expected_size, digest)


def verify_artifact(
    path: Path, expected_size: int, expected_sha256: str
) -> VerifiedArtifact:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise _checksum_fault(
            "checkpoint is unavailable or not a regular file"
        ) from error
    with os.fdopen(fd, "rb", closefd=True) as source:
        verified = verify_open_file(source, expected_size, expected_sha256)
    return VerifiedArtifact(path.resolve(), verified.size, verified.sha256)


def verify_checkpoint(path: Path) -> VerifiedArtifact:
    return verify_artifact(path, MODEL_SIZE, MODEL_SHA256)


__all__ = [
    "MODEL_FILENAME",
    "MODEL_REPO",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "MODEL_SIZE",
    "SOURCE_COMMIT",
    "VerifiedArtifact",
    "verify_artifact",
    "verify_checkpoint",
    "verify_open_file",
]

"""Verified, network-free conversion of the pinned Lingbot Map checkpoint."""

from typing import TYPE_CHECKING

from cloudpoint_worker.model_prep.provenance import VerifiedArtifact, verify_checkpoint

if TYPE_CHECKING:
    from cloudpoint_worker.model_prep.convert import ModelManifest, prepare_model


def __getattr__(name: str) -> object:
    """Load torch-backed conversion APIs only when explicitly requested."""

    if name in {"ModelManifest", "prepare_model"}:
        from cloudpoint_worker.model_prep.convert import ModelManifest, prepare_model

        value = {
            "ModelManifest": ModelManifest,
            "prepare_model": prepare_model,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = ["ModelManifest", "VerifiedArtifact", "prepare_model", "verify_checkpoint"]

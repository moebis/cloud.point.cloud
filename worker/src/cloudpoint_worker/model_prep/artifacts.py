"""Torch-free filenames shared by model preparation and runtime loading."""

CONVERTED_FILENAME = "lingbot-map-long-f16.safetensors"
WEIGHTS_MANIFEST_FILENAME = "weights-manifest.json"
MODEL_MANIFEST_FILENAME = "model-manifest.json"

__all__ = [
    "CONVERTED_FILENAME",
    "MODEL_MANIFEST_FILENAME",
    "WEIGHTS_MANIFEST_FILENAME",
]

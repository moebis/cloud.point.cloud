"""Apple SHARP inference adapter without CUDA rendering dependencies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cloudpoint_worker.model._vendor.ml_sharp.models import (
    PredictorParams,
    create_predictor,
)
from cloudpoint_worker.model._vendor.ml_sharp.utils.gaussians import (
    Gaussians3D,
    unproject_gaussians,
)

ProgressEmitter = Callable[[dict[str, object]], None]


def predict(
    image: np.ndarray,
    focal_length_px: float,
    device_name: str,
    emit: ProgressEmitter,
    *,
    checkpoint: Path,
) -> Gaussians3D:
    """Load the pinned network and predict metric Gaussians for one RGB image."""
    del emit
    device = torch.device(device_name)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()
    predictor.to(device)

    internal_shape = (1536, 1536)
    with torch.inference_mode():
        image_tensor = (
            torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
        )
        _, height, width = image_tensor.shape
        disparity_factor = torch.tensor(
            [focal_length_px / width], dtype=torch.float32, device=device
        )
        resized = F.interpolate(
            image_tensor[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )
        gaussians_ndc = predictor(resized, disparity_factor)
        intrinsics = torch.tensor(
            [
                [focal_length_px, 0, width / 2, 0],
                [0, focal_length_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        intrinsics[0] *= internal_shape[0] / width
        intrinsics[1] *= internal_shape[1] / height
        return unproject_gaussians(
            gaussians_ndc,
            torch.eye(4, device=device),
            intrinsics,
            internal_shape,
        )

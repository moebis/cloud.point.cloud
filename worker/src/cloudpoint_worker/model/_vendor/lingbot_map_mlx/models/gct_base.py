"""GCTBase — MLX port. Inference only."""

from abc import ABC, abstractmethod
from typing import Optional, Dict

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.heads.dpt_head import DPTHead
from lingbot_map_mlx.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map_mlx.utils.geometry import closed_form_inverse_se3


class GCTBase(nn.Module, ABC):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        patch_embed: str = 'dinov2_vitl14_reg',
        disable_global_rope: bool = False,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_local_point: bool = False,
        enable_depth: bool = True,
        enable_track: bool = False,
        enable_camera_sliding_window: bool = False,
        enable_3d_rope: bool = False,
        enable_ulysses_cp: bool = False,
        enable_normalize: bool = False,
        pred_normalization: bool = False,
        pred_normalization_detach_scale: bool = False,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed_type = patch_embed
        self.disable_global_rope = disable_global_rope
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_local_point = enable_local_point
        self.enable_depth = enable_depth
        self.enable_camera_sliding_window = enable_camera_sliding_window
        self.enable_3d_rope = enable_3d_rope

        self.aggregator = self._build_aggregator()
        self.camera_head = self._build_camera_head() if enable_camera else None
        self.point_head = self._build_point_head() if enable_point else None
        self.local_point_head = self._build_local_point_head() if enable_local_point else None
        self.depth_head = self._build_depth_head() if enable_depth else None

    @abstractmethod
    def _build_aggregator(self):
        pass

    @abstractmethod
    def _build_camera_head(self):
        pass

    def _build_depth_head(self):
        return DPTHead(
            dim_in=2 * self.embed_dim, patch_size=self.patch_size,
            output_dim=2, activation="exp", conf_activation="expp1",
        )

    def _build_point_head(self):
        return DPTHead(
            dim_in=2 * self.embed_dim, patch_size=self.patch_size,
            output_dim=4, activation="inv_log", conf_activation="expp1",
        )

    def _build_local_point_head(self):
        return DPTHead(
            dim_in=2 * self.embed_dim, patch_size=self.patch_size,
            output_dim=4, activation="inv_log", conf_activation="expp1",
        )

    def _normalize_input(self, images, query_points=None):
        if images.ndim == 4:
            images = images[None]
        return images, query_points

    @abstractmethod
    def _aggregate_features(self, images, num_frame_for_scale=None,
                            sliding_window_size=None, num_frame_per_block=1, **kwargs):
        pass

    def _predict_camera(self, aggregated_tokens_list, mask=None,
                        causal_inference=False, num_frame_for_scale=None,
                        sliding_window_size=None, num_frame_per_block=1, **kwargs):
        if self.camera_head is None:
            return {}

        tokens_fp32 = [t.astype(mx.float32) for t in aggregated_tokens_list]
        camera_sw = sliding_window_size if self.enable_camera_sliding_window else -1

        pose_enc_list = self.camera_head(
            tokens_fp32, mask=mask, causal_inference=causal_inference,
            num_frame_for_scale=num_frame_for_scale if num_frame_for_scale is not None else -1,
            sliding_window_size=camera_sw,
            num_frame_per_block=num_frame_per_block,
        )
        return {"pose_enc": pose_enc_list[-1], "pose_enc_list": pose_enc_list}

    def _predict_depth(self, aggregated_tokens_list, images, patch_start_idx, **kwargs):
        if self.depth_head is None:
            return {}
        tokens_fp32 = [t.astype(mx.float32) for t in aggregated_tokens_list]
        depth, depth_conf = self.depth_head(tokens_fp32, images=images.astype(mx.float32),
                                            patch_start_idx=patch_start_idx)
        return {"depth": depth, "depth_conf": depth_conf}

    def _predict_points(self, aggregated_tokens_list, images, patch_start_idx, **kwargs):
        if self.point_head is None:
            return {}
        tokens_fp32 = [t.astype(mx.float32) for t in aggregated_tokens_list]
        pts3d, pts3d_conf = self.point_head(tokens_fp32, images=images.astype(mx.float32),
                                            patch_start_idx=patch_start_idx)
        return {"world_points": pts3d, "world_points_conf": pts3d_conf}

    def _predict_local_points(self, aggregated_tokens_list, images, patch_start_idx, **kwargs):
        if self.local_point_head is None:
            return {}
        tokens_fp32 = [t.astype(mx.float32) for t in aggregated_tokens_list]
        pts3d, pts3d_conf = self.local_point_head(tokens_fp32, images=images.astype(mx.float32),
                                                   patch_start_idx=patch_start_idx)
        return {"cam_points": pts3d, "cam_points_conf": pts3d_conf}

    def __call__(self, images, num_frame_for_scale=None, sliding_window_size=None,
                 num_frame_per_block=1, mask=None, causal_inference=False, **kwargs):
        images, _ = self._normalize_input(images)

        aggregated_tokens_list, patch_start_idx = self._aggregate_features(
            images, num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )

        predictions = {}
        predictions.update(self._predict_camera(
            aggregated_tokens_list, mask=mask, causal_inference=causal_inference,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        ))
        predictions.update(self._predict_depth(aggregated_tokens_list, images, patch_start_idx))
        predictions.update(self._predict_points(aggregated_tokens_list, images, patch_start_idx))
        predictions.update(self._predict_local_points(aggregated_tokens_list, images, patch_start_idx))
        predictions["images"] = images
        return predictions

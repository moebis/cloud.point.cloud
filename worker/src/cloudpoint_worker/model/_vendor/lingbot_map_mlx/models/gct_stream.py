"""GCTStream — MLX port. Streaming inference with KV cache."""

from typing import Optional, Dict
from tqdm.auto import tqdm

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.heads.camera_head import CameraCausalHead
from lingbot_map_mlx.models.gct_base import GCTBase
from lingbot_map_mlx.aggregator.stream import AggregatorStream


class GCTStream(GCTBase):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        patch_embed: str = 'dinov2_vitl14_reg',
        pretrained_path: str = '',
        disable_global_rope: bool = False,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_local_point: bool = False,
        enable_depth: bool = True,
        enable_track: bool = False,
        enable_normalize: bool = False,
        pred_normalization: bool = False,
        sliding_window_size: int = -1,
        num_frame_for_scale: int = 1,
        num_random_frames: int = 0,
        attend_to_special_tokens: bool = False,
        attend_to_scale_frames: bool = False,
        enable_stream_inference: bool = True,
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        enable_camera_3d_rope: bool = False,
        camera_rope_theta: float = 10000.0,
        use_scale_token: bool = True,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        use_sdpa: bool = False,
        use_gradient_checkpoint: bool = True,
        camera_num_iterations: int = 4,
    ):
        self.pretrained_path = pretrained_path
        self.sliding_window_size = sliding_window_size
        self.num_frame_for_scale = num_frame_for_scale
        self.num_random_frames = num_random_frames
        self.attend_to_special_tokens = attend_to_special_tokens
        self.attend_to_scale_frames = attend_to_scale_frames
        self.enable_stream_inference = enable_stream_inference
        self.enable_3d_rope = enable_3d_rope
        self.max_frame_num = max_frame_num
        self.enable_camera_3d_rope = enable_camera_3d_rope
        self.camera_rope_theta = camera_rope_theta
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only
        self.use_sdpa = use_sdpa
        self.camera_num_iterations = camera_num_iterations

        super().__init__(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            patch_embed=patch_embed, disable_global_rope=disable_global_rope,
            enable_camera=enable_camera, enable_point=enable_point,
            enable_local_point=enable_local_point, enable_depth=enable_depth,
            enable_track=enable_track, enable_normalize=enable_normalize,
            pred_normalization=pred_normalization, enable_3d_rope=enable_3d_rope,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )

    def _build_aggregator(self):
        return AggregatorStream(
            img_size=self.img_size, patch_size=self.patch_size,
            embed_dim=self.embed_dim, patch_embed=self.patch_embed_type,
            disable_global_rope=self.disable_global_rope,
            sliding_window_size=self.sliding_window_size,
            num_frame_for_scale=self.num_frame_for_scale,
            num_random_frames=self.num_random_frames,
            attend_to_special_tokens=self.attend_to_special_tokens,
            attend_to_scale_frames=self.attend_to_scale_frames,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
        )

    def _build_camera_head(self):
        return CameraCausalHead(
            dim_in=2 * self.embed_dim,
            sliding_window_size=self.sliding_window_size,
            attend_to_scale_frames=self.attend_to_scale_frames,
            num_iterations=self.camera_num_iterations,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
            enable_3d_rope=self.enable_camera_3d_rope,
            max_frame_num=self.max_frame_num,
            rope_theta=self.camera_rope_theta,
        )

    def _aggregate_features(self, images, num_frame_for_scale=None,
                            sliding_window_size=None, num_frame_per_block=1, **kwargs):
        return self.aggregator(
            images, selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )

    def clean_kv_cache(self):
        if hasattr(self.aggregator, 'clean_kv_cache'):
            self.aggregator.clean_kv_cache()
        if self.camera_head is not None and hasattr(self.camera_head, 'clean_kv_cache'):
            self.camera_head.clean_kv_cache()

    def _set_skip_append(self, skip: bool):
        if hasattr(self.aggregator, 'kv_cache') and self.aggregator.kv_cache is not None:
            self.aggregator.kv_cache["_skip_append"] = skip
        if self.camera_head is not None and self.camera_head.kv_cache is not None:
            for cache_dict in self.camera_head.kv_cache:
                cache_dict["_skip_append"] = skip

    def inference_streaming(
        self,
        images: mx.array,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
    ) -> Dict[str, mx.array]:
        """Streaming inference: scale frames first, then frame-by-frame.

        Args:
            images: (B, S, H, W, C) or (S, H, W, C) in NHWC
            num_scale_frames: Number of initial bidirectional frames
            keyframe_interval: Cache every N-th frame after scale frames

        Returns:
            Dict with pose_enc, depth, depth_conf, world_points, world_points_conf, images
        """
        if images.ndim == 4:
            images = images[None]
        B, S, H, W, C = images.shape

        scale_frames = num_scale_frames if num_scale_frames is not None else self.num_frame_for_scale
        scale_frames = min(scale_frames, S)

        self.clean_kv_cache()

        # Phase 1: Scale frames (bidirectional)
        print(f"Processing {scale_frames} scale frames...")
        scale_images = images[:, :scale_frames]
        scale_output = self(
            scale_images,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )

        all_pose_enc = [scale_output["pose_enc"]]
        all_depth = [scale_output["depth"]] if "depth" in scale_output else []
        all_depth_conf = [scale_output["depth_conf"]] if "depth_conf" in scale_output else []
        all_world_points = [scale_output["world_points"]] if "world_points" in scale_output else []
        all_world_points_conf = [scale_output["world_points_conf"]] if "world_points_conf" in scale_output else []

        # Phase 2: Stream frame-by-frame
        pbar = tqdm(range(scale_frames, S), desc='Streaming inference', initial=scale_frames, total=S)
        for i in pbar:
            frame_image = images[:, i:i+1]

            is_keyframe = (keyframe_interval <= 1) or ((i - scale_frames) % keyframe_interval == 0)

            if not is_keyframe:
                self._set_skip_append(True)

            frame_output = self(
                frame_image,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )

            if not is_keyframe:
                self._set_skip_append(False)

            all_pose_enc.append(frame_output["pose_enc"])
            if "depth" in frame_output:
                all_depth.append(frame_output["depth"])
            if "depth_conf" in frame_output:
                all_depth_conf.append(frame_output["depth_conf"])
            if "world_points" in frame_output:
                all_world_points.append(frame_output["world_points"])
            if "world_points_conf" in frame_output:
                all_world_points_conf.append(frame_output["world_points_conf"])

            mx.eval(frame_output["pose_enc"])

        predictions = {"pose_enc": mx.concatenate(all_pose_enc, axis=1)}
        if all_depth:
            predictions["depth"] = mx.concatenate(all_depth, axis=1)
        if all_depth_conf:
            predictions["depth_conf"] = mx.concatenate(all_depth_conf, axis=1)
        if all_world_points:
            predictions["world_points"] = mx.concatenate(all_world_points, axis=1)
        if all_world_points_conf:
            predictions["world_points_conf"] = mx.concatenate(all_world_points_conf, axis=1)
        predictions["images"] = images

        return predictions

    def inference_windowed(
        self,
        images: mx.array,
        window_size: int = 20,
        overlap_size: Optional[int] = None,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
    ) -> Dict[str, mx.array]:
        """Windowed inference for long sequences on limited memory.

        Processes the video in overlapping windows, each with a fresh KV cache.
        Overlapping frames are used to compute a similarity transform (scale,
        rotation, translation) that aligns each window into a common coordinate
        frame before stitching.

        Args:
            images: (B, S, H, W, C) or (S, H, W, C) in NHWC
            window_size: Frames per window (must fit in memory)
            overlap_size: Overlap between windows (default: num_scale_frames)
            num_scale_frames: Bidirectional frames per window
            keyframe_interval: Cache every N-th frame within each window
        """
        import numpy as np
        from lingbot_map_mlx.utils.rotation import quat_to_mat, mat_to_quat

        if images.ndim == 4:
            images = images[None]
        B, S, H, W, C = images.shape

        ws = num_scale_frames if num_scale_frames is not None else self.num_frame_for_scale
        ws = min(ws, S)
        overlap = min(overlap_size if overlap_size is not None else ws, S - 1)

        if window_size >= S:
            return self.inference_streaming(images, num_scale_frames=ws, keyframe_interval=keyframe_interval)

        step = max(window_size - overlap, 1)
        win_list = []
        for start in range(0, S, step):
            end = min(start + window_size, S)
            if end - start >= overlap or end == S:
                win_list.append((start, end))
            if end == S:
                break

        print(f"Windowed inference: {len(win_list)} windows of ~{window_size} frames, overlap={overlap}")

        all_window_preds = []
        for wi, (start, end) in enumerate(tqdm(win_list, desc='Windows')):
            window_preds = self.inference_streaming(
                images[:, start:end],
                num_scale_frames=min(ws, end - start),
                keyframe_interval=keyframe_interval,
            )
            window_preds.pop("images", None)
            for v in window_preds.values():
                if isinstance(v, mx.array):
                    mx.eval(v)
            all_window_preds.append(window_preds)

        # Align windows into first window's coordinate frame, then stitch
        warped = []
        for idx, raw in enumerate(all_window_preds):
            if idx == 0:
                warped.append(raw)
                continue

            prev = warped[-1]
            s_ab, R_ab, t_ab = self._pairwise_alignment(prev, raw, overlap)
            warped.append(self._warp_predictions(raw, R_ab, t_ab, s_ab))

        merged = self._stitch_windows(warped, overlap)
        merged["images"] = images
        return merged

    @staticmethod
    def _pairwise_alignment(prev_pred, curr_pred, overlap):
        """Compute similarity transform (scale, R, t) aligning curr to prev using overlap frames."""
        from lingbot_map_mlx.utils.rotation import quat_to_mat
        import numpy as np

        pe_prev = prev_pred.get("pose_enc")
        pe_curr = curr_pred.get("pose_enc")
        if pe_prev is None or pe_curr is None or overlap <= 0:
            return 1.0, mx.eye(3), mx.zeros(3)

        idx_a = max(pe_prev.shape[1] - overlap, 0)

        # Decompose: translation + quaternion rotation
        Ra = quat_to_mat(pe_prev[:, idx_a, 3:7])  # (B, 3, 3)
        ca = pe_prev[:, idx_a, :3]                  # (B, 3)
        Rb = quat_to_mat(pe_curr[:, 0, 3:7])
        cb = pe_curr[:, 0, :3]

        # R_ab = Ra @ Rb^T
        R_ab = Ra @ mx.transpose(Rb, axes=(0, 2, 1))

        # Scale from depth ratio
        s_ab = mx.array([1.0])
        da = prev_pred.get("depth")
        db = curr_pred.get("depth")
        if da is not None and db is not None and da.shape[1] > idx_a and db.shape[1] > 0:
            a = da[:, idx_a, ..., 0].reshape(-1).astype(mx.float32)
            b = db[:, 0, ..., 0].reshape(-1).astype(mx.float32)
            a_np = np.array(a)
            b_np = np.array(b)
            valid = np.isfinite(a_np) & np.isfinite(b_np) & (np.abs(b_np) > 1e-7)
            if valid.any():
                ratio = np.median(a_np[valid] / b_np[valid])
                s_ab = mx.array([float(np.clip(ratio, 1e-3, 1e3))])

        # t_ab = ca - s * R_ab @ cb
        cb_rot = (R_ab @ cb[..., None])[..., 0]  # (B, 3)
        t_ab = ca - s_ab * cb_rot

        return s_ab, R_ab, t_ab

    @staticmethod
    def _warp_predictions(pred, R, t, s):
        """Apply similarity transform to one window's predictions."""
        from lingbot_map_mlx.utils.rotation import quat_to_mat, mat_to_quat

        warped = {}

        pe = pred.get("pose_enc")
        if pe is not None:
            nf = pe.shape[1]
            local_rot = quat_to_mat(pe[:, :, 3:7])   # (B, nf, 3, 3)
            local_ctr = pe[:, :, :3]                    # (B, nf, 3)

            R_exp = mx.broadcast_to(R[:, None], (R.shape[0], nf, 3, 3))
            new_rot = R_exp @ local_rot
            new_ctr = s * (R_exp @ local_ctr[..., None])[..., 0] + t[:, None, :]

            new_quat = mat_to_quat(new_rot.reshape(-1, 3, 3)).reshape(pe.shape[0], nf, 4)
            out_pe = mx.concatenate([new_ctr, new_quat, pe[:, :, 7:]], axis=-1)
            warped["pose_enc"] = out_pe

        d = pred.get("depth")
        if d is not None:
            warped["depth"] = d * s.reshape(1, 1, 1, 1, 1)

        for k in ("depth_conf", "world_points", "world_points_conf"):
            if k in pred:
                warped[k] = pred[k]

        wp = pred.get("world_points")
        if wp is not None:
            b, nf, h, w, _ = wp.shape
            flat = wp.reshape(b, nf * h * w, 3)
            transformed = (flat @ mx.transpose(R, axes=(0, 2, 1))) * s.reshape(1, 1, 1)
            transformed = transformed + t[:, None, :]
            warped["world_points"] = transformed.reshape(b, nf, h, w, 3)

        return warped

    @staticmethod
    def _stitch_windows(windows, overlap):
        """Concatenate window predictions, removing overlapping frames."""
        if len(windows) == 1:
            return windows[0]

        keys = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]
        merged = {}
        n_win = len(windows)

        for key in keys:
            parts = []
            for wi, w in enumerate(windows):
                if key not in w:
                    continue
                tensor = w[key]
                total = tensor.shape[1]
                is_last = (wi == n_win - 1)
                end = total if is_last else max(total - overlap, 0)
                if end > 0:
                    parts.append(tensor[:, :end])

            if parts:
                merged[key] = mx.concatenate(parts, axis=1)

        return merged

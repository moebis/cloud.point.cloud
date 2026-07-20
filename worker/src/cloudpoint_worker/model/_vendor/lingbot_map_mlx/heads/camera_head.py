"""Camera prediction heads for MLX."""

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.layers.mlp import Mlp
from lingbot_map_mlx.layers.block import Block, CameraBlock
from lingbot_map_mlx.layers.rope import WanRotaryPosEmbed
from lingbot_map_mlx.heads.head_act import activate_pose
from lingbot_map_mlx.layers.swiglu_ffn import silu


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


class SiLU(nn.Module):
    def __call__(self, x):
        return silu(x)


class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        enable_ulysses_cp=False,
    ):
        super().__init__()
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        self.trunk = [
            Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
            for _ in range(trunk_depth)
        ]

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_pose_tokens = mx.zeros((1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        self.poseLN_modulation_0 = SiLU()
        self.poseLN_modulation_1 = nn.Linear(dim_in, 3 * dim_in, bias=True)

        self.adaln_norm = nn.LayerNorm(dim_in, affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def __call__(self, aggregated_tokens_list: list, num_iterations: int = 4, **kwargs) -> list:
        tokens = aggregated_tokens_list[-1]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)
        return self.trunk_fn(pose_tokens, num_iterations)

    def trunk_fn(self, pose_tokens: mx.array, num_iterations: int) -> list:
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(mx.broadcast_to(self.empty_pose_tokens, (B, S, self.target_dim)))
            else:
                module_input = self.embed_pose(mx.stop_gradient(pred_pose_enc))

            mod_out = self.poseLN_modulation_1(self.poseLN_modulation_0(module_input))
            shift_msa, scale_msa, gate_msa = mx.split(mod_out, 3, axis=-1)

            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for block in self.trunk:
                pose_tokens_modulated = block(pose_tokens_modulated)

            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


class CameraCausalHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        num_iterations=4,
        elementwise_attn_output_gate: bool = False,
        sliding_window_size: int = -1,
        attend_to_scale_frames: bool = False,
        num_random_frames: int = 0,
        enable_ulysses_cp: bool = False,
        attn_class: str = "flexflashattn_varlen",
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.sliding_window_size = sliding_window_size
        self.num_heads = num_heads
        self.enable_3d_rope = enable_3d_rope

        if enable_3d_rope:
            head_dim = dim_in // num_heads
            self.rope3d = WanRotaryPosEmbed(
                attention_head_dim=head_dim,
                patch_size=(max_frame_num, 1, 1),
                theta=rope_theta,
                fhw_dim=[40, 44, 44],
            )
        else:
            self.rope3d = None

        self.trunk = [
            CameraBlock(
                dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio,
                init_values=init_values,
                elementwise_attn_output_gate=elementwise_attn_output_gate,
                sliding_window_size=sliding_window_size,
                attend_to_scale_frames=attend_to_scale_frames,
                num_random_frames=num_random_frames,
                kv_cache_sliding_window=kv_cache_sliding_window,
                kv_cache_scale_frames=kv_cache_scale_frames,
                kv_cache_cross_frame_special=kv_cache_cross_frame_special,
                kv_cache_include_scale_frames=kv_cache_include_scale_frames,
                kv_cache_camera_only=kv_cache_camera_only,
            )
            for _ in range(trunk_depth)
        ]

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_pose_tokens = mx.zeros((1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        self.poseLN_modulation_0 = SiLU()
        self.poseLN_modulation_1 = nn.Linear(dim_in, 3 * dim_in, bias=True)

        self.adaln_norm = nn.LayerNorm(dim_in, affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

        self.num_iterations = num_iterations
        self.kv_cache = None
        self.frame_idx = 0

    def clean_kv_cache(self):
        self.kv_cache = None
        self.frame_idx = 0

    def __call__(
        self, aggregated_tokens_list: list, mask=None, num_iterations=None,
        causal_inference=False, num_frame_per_block=1, num_frame_for_scale=-1,
        sliding_window_size=None, **kwargs,
    ) -> list:
        if num_iterations is None:
            num_iterations = self.num_iterations

        effective_sw = sliding_window_size if sliding_window_size is not None else self.sliding_window_size

        tokens = aggregated_tokens_list[-1]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        if causal_inference and self.kv_cache is None:
            self.kv_cache = []
            for i in range(num_iterations):
                cache = {"_skip_append": False}
                for j in range(self.trunk_depth):
                    cache[f"k_{j}"] = None
                    cache[f"v_{j}"] = None
                self.kv_cache.append(cache)

        return self.trunk_fn(
            pose_tokens, mask, num_iterations,
            num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=effective_sw,
        )

    def trunk_fn(
        self, pose_tokens: mx.array, mask=None, num_iterations: int = 4,
        num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=None,
    ) -> list:
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        is_scale_frames = (self.kv_cache is not None and self.frame_idx == 0)

        pos3d = None
        if self.rope3d is not None:
            if self.kv_cache is not None:
                f_start = self.frame_idx
                f_end = self.frame_idx + S
            else:
                f_start = 0
                f_end = None

            pos3d = self.rope3d(
                ppf=S, pph=1, ppw=1, patch_start_idx=0,
                f_start=f_start, f_end=f_end,
            )  # Returns (cos, sin) tuple

        for i in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(mx.broadcast_to(self.empty_pose_tokens, (B, S, self.target_dim)))
            else:
                module_input = self.embed_pose(mx.stop_gradient(pred_pose_enc))

            mod_out = self.poseLN_modulation_1(self.poseLN_modulation_0(module_input))
            shift_msa, scale_msa, gate_msa = mx.split(mod_out, 3, axis=-1)

            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for idx in range(self.trunk_depth):
                pose_tokens_modulated = self.trunk[idx](
                    pose_tokens_modulated, pos=pos3d, video_mask=mask,
                    num_frames=S, frame_seqlen=1,
                    kv_cache=self.kv_cache[i] if self.kv_cache is not None else None,
                    global_idx=idx, num_frame_per_block=num_frame_per_block,
                    num_frame_for_scale=num_frame_for_scale,
                    sliding_window_size=sliding_window_size,
                    enable_3d_rope=self.enable_3d_rope,
                    is_scale_frames=is_scale_frames,
                )

            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        if self.kv_cache is not None:
            self.frame_idx += S

        return pred_pose_enc_list

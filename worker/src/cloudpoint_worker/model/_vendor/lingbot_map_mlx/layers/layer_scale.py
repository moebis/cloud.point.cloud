import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_values

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma

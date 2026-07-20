import mlx.core as mx
import mlx.nn as nn


def drop_path(x: mx.array, drop_prob: float = 0.0, training: bool = False) -> mx.array:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = mx.random.bernoulli(keep_prob, shape=shape).astype(x.dtype)
    if keep_prob > 0.0:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def __call__(self, x: mx.array) -> mx.array:
        return drop_path(x, self.drop_prob, self.training)

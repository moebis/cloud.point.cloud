# Lingbot Map MLX vendored source

This directory contains an unchanged inference-source snapshot from
[`anmolduainter/lingbot-map-mlx`](https://github.com/anmolduainter/lingbot-map-mlx)
commit `c2e0cf04072afa1c817755f1d665e71349d0a32d`, except for this package's
`__init__.py`, which installs a private import alias so the original absolute
imports continue to resolve under CloudPoint's namespace.

The snapshot is an unofficial Apple MLX port of
[`Robbyant/lingbot-map`](https://github.com/Robbyant/lingbot-map). CloudPoint's
strict converter and topology specification are pinned to original upstream
commit `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`.

The vendored work is distributed under Apache License 2.0. A copy is included
as `LICENSE.txt`. CloudPoint does not redistribute model weights.

# Third-party notices

## Lingbot Map MLX

CloudPoint includes inference source derived from the unofficial
[`anmolduainter/lingbot-map-mlx`](https://github.com/anmolduainter/lingbot-map-mlx)
Apple MLX port at commit `c2e0cf04072afa1c817755f1d665e71349d0a32d`.
The vendored snapshot and its Apache License 2.0 text are in
`src/cloudpoint_worker/model/_vendor/lingbot_map_mlx/`.

That project ports the Apache-2.0
[`Robbyant/lingbot-map`](https://github.com/Robbyant/lingbot-map) inference
topology. CloudPoint pins its model topology to upstream commit
`7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2` and downloads model weights
separately at runtime; weights are not included in this repository or package.

## Apple SHARP

CloudPoint includes the model-construction and Gaussian PLY inference source
from [`apple/ml-sharp`](https://github.com/apple/ml-sharp), pinned to commit
`1eaa046834b81852261262b41b0919f5c1efdd2e`. The Apple software license,
research-model license, acknowledgements, provenance, and a description of the
CloudPoint namespace-only modifications are retained in
`src/cloudpoint_worker/model/_vendor/ml_sharp/`.

CUDA rendering and `gsplat` are not included. The SHARP checkpoint is not part
of the repository or bundled runtime; it is downloaded separately after the
user accepts Apple's research-only model terms.

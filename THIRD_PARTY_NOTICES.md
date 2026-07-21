# Third-party notices

CloudPoint is licensed under Apache License 2.0. The following components keep
their original licenses and attribution. Model weights are downloaded separately
and are not part of this source repository or the compiled application.

## LingBot-Map source and model

CloudPoint reproduces model architecture, preprocessing, and inference behavior
from [Robbyant/lingbot-map][lingbot-source], pinned to source commit
`7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`. The upstream repository is
distributed under Apache License 2.0.

CloudPoint's one-time setup downloads `lingbot-map-long.pt` from
[robbyant/lingbot-map on Hugging Face][lingbot-model] at revision
`204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`. The pinned model card identifies
LingBot-Map as Apache-2.0.

The application accepts only these compiled trust anchors:

### Source checkpoint

- Size: 4,632,303,465 bytes
- SHA-256:

  ```text
  832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
  ```

### Converted SafeTensors

- Size: 2,316,040,080 bytes
- SHA-256:

  ```text
  eb966484923b5a205677b3ce7316d079c46fc6503bc9b6ac256b6e11560ea2e5
  ```

CloudPoint does not redistribute either artifact. Download and use remain
subject to the upstream model repository's terms.

## LingBot Map MLX source

CloudPoint includes inference source derived from the unofficial
[`anmolduainter/lingbot-map-mlx`][lingbot-mlx-port] port at commit
`c2e0cf04072afa1c817755f1d665e71349d0a32d`. The vendored snapshot is
Apache-2.0 and is stored at
`worker/src/cloudpoint_worker/model/_vendor/lingbot_map_mlx/`. Its full license
text and provenance are included beside the source.

CloudPoint contains adaptations and integration code around that snapshot; the
vendored files themselves remain separately identified.

## Apple SHARP source and research model

CloudPoint includes a minimal inference-only snapshot of
[`apple/ml-sharp`][apple-sharp] at commit
`1eaa046834b81852261262b41b0919f5c1efdd2e`. Apple retains copyright in that
source and distributes it under the license preserved at
`worker/src/cloudpoint_worker/model/_vendor/ml_sharp/LICENSE`.

The snapshot contains model construction and Gaussian PLY utilities. CloudPoint
excludes the upstream CLI, image/video utilities, CUDA renderer, and `gsplat`,
and modifies only absolute imports to use a private vendored namespace. Full
provenance and acknowledgements are stored beside the source.

The separately downloaded SHARP checkpoint is governed by Apple's Machine
Learning Research Model License Agreement, preserved as `LICENSE_MODEL`. It is
restricted to research purposes and is never committed to this repository or
redistributed inside the application.

## MetalSplatter and SplatIO

CloudPoint uses [`scier/MetalSplatter`][metal-splatter], including SplatIO and
PLYIO, pinned to commit `71ff248e3016ac43c0a9271e322538421b28c360`
(release 1.0.1). It is distributed under the MIT License. Copyright © 2026
Sean Cier.

MetalSplatter pulls [`scier/spz-swift`][spz-swift] 2.1.0 for its scene I/O
layer. It is distributed under the MIT License. Copyright © 2024 Niantic Labs;
Swift port by Sean Cier, 2026. SHARP projects currently publish and load PLY,
but this transitive library remains linked in the application.

## Courthouse test images

The nine courthouse images in `worker/tests/fixtures/courthouse/` come from the
LingBot-Map repository at source commit
`7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2` under Apache License 2.0. Exact
source URLs and SHA-256 checksums are recorded in the fixture's
`provenance.json`.

## Apple MLX

CloudPoint bundles [MLX](https://github.com/ml-explore/mlx) and MLX Metal
version 0.32.0.

> MIT License. Copyright © 2023 Apple Inc.

The complete MLX license text is preserved in each bundled Python
distribution's `.dist-info/licenses/` directory.

## Bundled Python runtime and packages

The arm64 release includes CPython 3.12.11 and the following locked runtime
packages. Package metadata and license files supplied by each distribution are
preserved inside the bundled runtime.

| Component | Version | License |
| --- | --- | --- |
| CPython | 3.12.11 | Python Software Foundation License |
| MLX / MLX Metal | 0.32.0 | MIT |
| msgspec | 0.19.0 | BSD-3-Clause |
| NumPy | 2.3.1 | BSD-3-Clause and bundled component licenses |
| Pillow | 11.3.0 | MIT-CMU |
| plyfile | 1.1.2 | GPL-3.0-only |
| safetensors | 0.5.3 | Apache-2.0 |
| SciPy | 1.16.2 | BSD-3-Clause and bundled component licenses |
| timm | 1.0.20 | Apache-2.0 |
| PyTorch | 2.8.0 | BSD-3-Clause and bundled component licenses |
| torchvision | 0.23.0 | BSD-3-Clause |
| tqdm | 4.67.1 | MPL-2.0 and MIT |
| filelock | 3.31.1 | MIT |
| fsspec | 2026.6.0 | BSD-3-Clause |
| Jinja2 | 3.1.6 | BSD-3-Clause |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD-3-Clause |
| NetworkX | 3.6.1 | BSD-3-Clause |
| setuptools | 83.0.0 | MIT |
| SymPy | 1.14.0 | BSD-3-Clause |
| typing_extensions | 4.16.0 | PSF-2.0 |

NumPy and PyTorch ship additional compatible third-party components. Their
complete notices are retained in the corresponding installed distribution
license files. This summary does not replace those license texts.

## System frameworks

CloudPoint links Apple platform frameworks including SwiftUI, AppKit,
AVFoundation, Metal, MetalKit, Core Video, and CryptoKit. Those frameworks are
provided by macOS and are not redistributed by this repository.

[lingbot-mlx-port]: https://github.com/anmolduainter/lingbot-map-mlx/tree/c2e0cf04072afa1c817755f1d665e71349d0a32d
[lingbot-model]: https://huggingface.co/robbyant/lingbot-map/tree/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9
[lingbot-source]: https://github.com/Robbyant/lingbot-map/tree/7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
[apple-sharp]: https://github.com/apple/ml-sharp/tree/1eaa046834b81852261262b41b0919f5c1efdd2e
[metal-splatter]: https://github.com/scier/MetalSplatter/tree/71ff248e3016ac43c0a9271e322538421b28c360
[spz-swift]: https://github.com/scier/spz-swift/tree/2.1.0

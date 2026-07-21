# Apple SHARP vendored inference source

- Upstream: <https://github.com/apple/ml-sharp>
- Source commit: `1eaa046834b81852261262b41b0919f5c1efdd2e`
- Retrieved: 2026-07-21
- Upstream license: `LICENSE`
- Model license: `LICENSE_MODEL`
- Upstream acknowledgements: `ACKNOWLEDGEMENTS`

CloudPoint vendors the model construction and Gaussian PLY utilities required
for local inference. The upstream CLI, image/video utilities, visualization,
CUDA renderer, and `gsplat` adapter are intentionally excluded. Model weights
are not included and are downloaded separately only after the user accepts the
research-model terms.

The vendored Python files differ from upstream only by mechanically rewriting
absolute `sharp.*` imports into CloudPoint's private package namespace. This
prevents collisions with unrelated Python installations and makes the bundled
runtime relocatable.

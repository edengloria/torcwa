# TORCWA v2 Refactor Notes

TORCWA v2 is an additive, accuracy-first refactor.  The legacy `torcwa.rcwa`
API remains available while the new `torcwa.v2` package grows a typed
configuration surface, reusable numerical kernels, and validation fixtures.
The v2 target runtime is Python 3.10+ with PyTorch 2.11+.

## Implemented Foundation

- `torcwa.v2.RCWAConfig`, `SolverOptions`, `FourierBasis`, `LayerSpec`,
  `MaterialGrid`, and `PortSpec` define explicit simulation state.
- `torcwa.v2.linalg` provides solve-based helpers for replacing
  `inv(A) @ B` and broadcast-based helpers for replacing explicit diagonal
  matrix products.
- `torcwa.v2.physics` centralizes kz branch selection, propagating-order
  classification, diffraction-order indexing, and Fresnel reference values.
- `torcwa.v2.eig` adds a GPU-resident stabilized eigendecomposition path.
- `torcwa.v2.RCWASolver` exposes the v2 public facade while delegating full
  RCWA solves to the legacy implementation during the validation transition.
- The legacy numerical path now uses solve-based helpers in the high-risk
  coupling, layer S-matrix, Redheffer product, eig backward, and field
  reconstruction paths instead of forming inverse matrices explicitly.
- Internal field reconstruction avoids explicit diagonal phase matrices in the
  layer propagation path, and `source_planewave(..., notation="ps")` now maps
  p/s amplitudes to x/y amplitudes by broadcasting rather than constructing a
  dense block-diagonal matrix.
- Empty-stack S-matrix initialization now uses matrix-shaped zero reflection
  blocks, which fixes zero-layer reflection queries.
- xz/yz field reconstruction now batches z samples by layer region and supports
  chunking through v2 solver options.
- Repeated solves reuse LU factorizations within a local computation, avoiding
  persistent autograd graph caches.
- Non-gradient material convolution matrices are cached across fixed-geometry
  sweeps; gradient-carrying materials are excluded from the cache.
- `RCWASolver.solve_sweep(...)` provides an experimental fixed-geometry
  frequency/angle sweep API for requested S-parameters.

## Accuracy Policy

Changes to the numerical core should be accepted only after they pass:

- analytical Fresnel/interface/slab checks,
- lossless power conservation and lossy non-negative absorption checks,
- legacy TORCWA regression checks on reduced notebook examples,
- external-solver fixtures for at least one grating and one multilayer case,
- gradient checks against finite differences in `complex128`.

## Current Compatibility Notes

- `torcwa.rcwa` still supports the legacy object-oriented workflow.
- `S_parameters(..., evanescent=...)` is the canonical spelling; the legacy
  `evanscent` spelling is retained as a deprecated alias.
- `torcwa.rcwa_geo` is still present for examples but new code should use
  `torcwa.geometry(...)` instances or the v2 dataclasses.

## Release Gate

This branch is a v2 developer-preview foundation, not yet a final physics
release.  The local QA pass covers analytical normal-incidence checks, lossless
power conservation for the interface case, CPU/CUDA smoke workloads, finite
gradients, and reproducible microbenchmarks.  Before promoting this to a stable
major release, add external-solver fixtures for representative gratings and
metasurfaces and broaden the analytical suite to oblique slab/Brewster/near
critical cases.

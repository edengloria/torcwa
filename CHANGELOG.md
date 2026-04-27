# Changelog

## 0.3.0.dev0

- Added the modern public `torcwa.RCWA + torcwa.Stack + torcwa.PlaneWave`
  workflow for researcher-facing simulations.
- Added `Result` helpers for transmission, reflection, diffraction tables,
  power balance, and field-plane reconstruction.
- Added `MaterialGrid`, `UnitCell`, and `torcwa.material.mix(...)` as the
  simple differentiable material/geometry path.
- Added `Output` and loop-backed `RCWA.sweep(...)` for wavelength sweeps.
- Added the `torcwa.core` helper namespace as the stable boundary for ongoing
  numerical-core modularization.
- Added modern API examples and regression tests against the legacy solver.
- Added `SolverOptions.store_fields` and `RCWA(..., store_fields=False)` /
  `solve(..., store_fields=False)` for S-parameter-only solves.
- Propagated `field_chunk_size` from modern `RCWA` options into
  `Result.fields.plane(...)`.
- Cached repeated `Result` order/S-parameter/diffraction table queries and
  routed `RCWA.sweep(...)` through the fixed-geometry v2 sweep backend.
- Added original-vs-current-vs-optimized performance comparison tooling and
  `docs/TORCWA_v3_PERFORMANCE_REPORT.md`.
- Added per-simulation legacy `S_parameters(...)` normalization caches for
  repeated diffraction-order and s/p polarization queries, with value caching
  disabled for gradient-bearing port materials and cache keys guarded by port
  tensor signatures.
- Added a `memory_mode="memory"` homogeneous-layer structured path that avoids
  dense homogeneous `P/Q` storage while preserving the balanced default path.
- Added a `memory_mode="memory"` eigenmode-streaming path for internal field
  reconstruction.
- Normalized fixed-geometry sweep output requests outside the wavelength loop.
- Added release-grade repeat controls and repeated diffraction-query coverage to
  the original/current/optimized benchmark runner.

## 0.2.0.dev1

- Added committed Stanford S4 external-reference fixtures for homogeneous,
  patterned, lossy, oblique-incidence, and multilayer RCWA validation.
- Added `tools/generate_s4_fixtures.py`, `tools/build_s4_no_sudo.sh`, and the
  `s4_live` pytest marker so S4 remains optional for normal QA but can refresh
  fixtures in configured environments.
- Added broader physical invariant tests covering Fabry-Perot slab phase,
  Brewster suppression, near-critical evanescent classification, reciprocity,
  field continuity, lossless conservation, lossy absorption, and gradient smoke.
- Fixed a diagonal-free homogeneous `P/Q` block sign regression introduced in
  the v2 optimization path; updated-vs-original TORCWA comparison now matches to
  roundoff for the S4 core cases.
- Tightened patterned S4 fixture tolerances enough to catch multilayer coupling
  regressions while still allowing expected S4 closed-form geometry versus
  TORCWA sampled-grid differences.
- Documented the S4 no-sudo build path, current validation status, and the
  original-vs-updated TORCWA regression diagnosis.

## 0.2.0.dev0

- Added the `torcwa.v2` developer-preview API with dataclass configuration,
  reusable linalg/physics/eig helpers, and an `RCWASolver` facade.
- Raised the target runtime to Python 3.10+ and PyTorch 2.11+.
- Made `S_parameters(..., evanescent=...)` the canonical spelling while keeping
  the legacy `evanscent` alias with a deprecation warning.
- Replaced key explicit inverse products in the legacy numerical path with
  `torch.linalg.solve`-based helpers.
- Replaced selected explicit diagonal products with broadcasting in source and
  field reconstruction paths.
- Kept eig backward on the active device and added a v2 stabilized eig helper
  with finite-gradient and `torch.func.vmap` smoke coverage.
- Fixed empty-stack reflection S-parameter queries by initializing zero
  reflection blocks as matrices.
- Added pytest coverage for analytical Fresnel checks, lossless power
  conservation, typo alias compatibility, v2 facade compatibility, finite field
  reconstruction, and finite geometry gradients.
- Added `benchmarks/v2_microbench.py` and v2 QA/migration documentation.
- Optimized developer-preview field reconstruction by batching xz/yz z-samples
  by layer and honoring `SolverOptions.field_chunk_size` through v2
  `field_plane(..., chunk_size=...)`.
- Reused LU factorizations for repeated solve RHS groups in layer, Redheffer,
  and field reconstruction paths.
- Reduced dense k-vector diagonal hot paths by using normalized vector storage
  internally while preserving `Kx_norm`/`Ky_norm` compatibility properties.
- Added a bounded material convolution cache for non-gradient material tensors
  and `rcwa.clear_material_cache()`.
- Added experimental `RCWASolver.solve_sweep(...)` for fixed-geometry
  frequency/angle S-parameter sweeps.
- Added `SolverOptions.memory_mode` with balanced, memory, and speed policies.
- Reduced balanced-mode peak memory by avoiding dense homogeneous-transform
  storage and replacing the layer-coupling `4M x 4M` solve with exact block
  symmetric `2M x 2M` solves.
- Streamed `field_xy` and added spatial-axis chunking for `field_xz`/`field_yz`.
- Strengthened material convolution cache keys and honored
  `MaterialGrid(cache_key=..., cache=False)`.
- Added a validation-only Fourier convolution operator prototype and review
  benchmark; it is not used by the solver path.
- Added the S4 external-validation harness, pending fixture manifest, optional
  fixture generator, `s4_live` pytest marker, and broader analytical/physical
  invariant tests.
- Added committed S4 core-gate fixtures and a no-sudo S4 build helper for
  environments without system Python development headers.
- Fixed a diagonal-free homogeneous `P/Q` block sign regression caught by
  original-TORCWA and S4 multilayer-stack comparison.

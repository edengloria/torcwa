# TORCWA v2 QA And Benchmark Report

Date: 2026-04-24

## Environment

- OS: Linux 6.6.87.2 WSL2 x86_64
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- GPU: NVIDIA GeForce RTX 5070
- Package install used for this workspace:
  `python3 -m pip install --user --break-system-packages -e . pytest numpy scipy`
- Reproducible dev install file added: `requirements-dev.txt`

## Automated QA

Command:

```bash
python3 -m pytest -q
```

Result:

```text
14 passed in 1.54s
```

Coverage added in this pass:

- v2 linalg helpers match inverse/explicit diagonal reference operations.
- v2 physics helpers cover Fourier basis bookkeeping, diffraction-order
  indexing, kz branch selection, propagating-order classification, and Fresnel
  reference values.
- `StabilizedEig` has finite gradients on-device and supports
  `torch.func.vmap` forward transforms.
- Legacy `rcwa` passes empty-stack identity/reflection, single-interface
  Fresnel amplitude, and lossless power conservation at normal incidence.
- `S_parameters(..., evanescent=...)` is canonical and the deprecated
  `evanscent` alias still matches with a `DeprecationWarning`.
- v2 `RCWASolver` facade matches the legacy solver on a patterned single layer,
  reconstructs finite xz fields, and supports p/s source notation.
- A small differentiable geometry case produces a finite radius gradient.

## Manual Smoke Checks

CPU and CUDA were both checked on a patterned rectangular layer with
`order=[2, 2]`, `grid=48x48`, `complex64`.

```text
cpu  legacy txx finite True  tensor([0.4765+0.8774j])
cpu  field shapes [(16, 11), (16, 11), (16, 11)] finite True
cpu  v2 matches legacy True   tensor([0.4765+0.8774j])
cuda legacy txx finite True  tensor([0.4765+0.8774j])
cuda field shapes [(16, 11), (16, 11), (16, 11)] finite True
cuda v2 matches legacy True   tensor([0.4765+0.8774j])
gradient finite True          tensor(-0.0002, dtype=torch.float64)
```

## Benchmark Commands

Standard benchmark:

```bash
python3 benchmarks/v2_microbench.py
```

CUDA stress benchmark:

```bash
python3 benchmarks/v2_microbench.py --quick --devices cuda --stress
```

## Standard Benchmark

```text
torch: 2.11.0+cu130
cuda: 13.0 device: NVIDIA GeForce RTX 5070

| workload | case | device | dtype | size | median ms | min ms | peak CUDA MB | check |
|---|---|---:|---|---|---:|---:|---:|---|
| linear_system | solve_left | cpu | complex64 | 128x128 rhs=32 | 0.305 | 0.301 |  | rel_error_vs_inverse=4.004e-07 |
| linear_system | inverse_matmul | cpu | complex64 | 128x128 rhs=32 | 0.426 | 0.410 |  | rel_error_vs_inverse=4.004e-07 |
| diagonal_product | broadcast_diag | cpu | complex64 | 768x768 | 0.056 | 0.053 |  | rel_error_vs_explicit=0.000e+00 |
| diagonal_product | explicit_diag_matmul | cpu | complex64 | 768x768 | 9.625 | 8.081 |  | rel_error_vs_explicit=0.000e+00 |
| rcwa_solve | patterned_single_layer | cpu | complex64 | order=[3, 3] grid=72x72 | 6.333 | 5.817 |  | finite=True |
| field_reconstruction | xz_plane | cpu | complex64 | order=[3, 3] samples=(40, 24) | 13.090 | 11.964 |  | finite=True |
| linear_system | solve_left | cuda | complex64 | 128x128 rhs=32 | 0.380 | 0.357 | 8.66 | rel_error_vs_inverse=5.801e-07 |
| linear_system | inverse_matmul | cuda | complex64 | 128x128 rhs=32 | 0.465 | 0.424 | 8.85 | rel_error_vs_inverse=5.801e-07 |
| diagonal_product | broadcast_diag | cuda | complex64 | 768x768 | 0.028 | 0.026 | 26.13 | rel_error_vs_explicit=4.251e-08 |
| diagonal_product | explicit_diag_matmul | cuda | complex64 | 768x768 | 0.367 | 0.357 | 30.63 | rel_error_vs_explicit=4.251e-08 |
| rcwa_solve | patterned_single_layer | cuda | complex64 | order=[3, 3] grid=72x72 | 41.085 | 29.818 | 10.52 | finite=True |
| field_reconstruction | xz_plane | cuda | complex64 | order=[3, 3] samples=(40, 24) | 120.925 | 111.044 | 9.87 | finite=True |
```

Interpretation:

- `torch.linalg.solve`-based left solves are faster than `inv(A) @ B` for the
  measured 128x128 repeated RHS case on both CPU and CUDA.
- Broadcast diagonal products remove the dense diagonal allocation and are much
  faster on CPU and CUDA.
- Small RCWA and field workloads remain CPU-favorable because Python overhead
  and many small CUDA kernels dominate.  This is expected and should guide the
  next batching/refactor phase.

## CUDA Stress Benchmark

```text
torch: 2.11.0+cu130
cuda: 13.0 device: NVIDIA GeForce RTX 5070

| workload | case | device | dtype | size | median ms | min ms | peak CUDA MB | check |
|---|---|---:|---|---|---:|---:|---:|---|
| linear_system | solve_left | cuda | complex64 | 64x64 rhs=16 | 0.962 | 0.834 | 8.25 | rel_error_vs_inverse=4.898e-07 |
| linear_system | inverse_matmul | cuda | complex64 | 64x64 rhs=16 | 0.459 | 0.414 | 8.31 | rel_error_vs_inverse=4.898e-07 |
| diagonal_product | broadcast_diag | cuda | complex64 | 256x256 | 0.108 | 0.054 | 10.13 | rel_error_vs_explicit=4.247e-08 |
| diagonal_product | explicit_diag_matmul | cuda | complex64 | 256x256 | 0.300 | 0.226 | 10.63 | rel_error_vs_explicit=4.247e-08 |
| rcwa_solve | patterned_single_layer | cuda | complex64 | order=[2, 2] grid=48x48 | 16.917 | 13.004 | 8.77 | finite=True |
| field_reconstruction | xz_plane | cuda | complex64 | order=[2, 2] samples=(20, 12) | 28.762 | 27.613 | 8.61 | finite=True |
| rcwa_stress | patterned_single_layer | cuda | complex64 | order=[10, 10] grid=160x160 | 285.391 | 272.000 | 199.39 | finite=True |
| rcwa_stress | patterned_single_layer | cuda | complex64 | order=[15, 15] grid=224x224 | 1156.557 | 1153.154 | 910.36 | finite=True |
| rcwa_stress | patterned_single_layer | cuda | complex64 | order=[20, 20] grid=300x300 | 3052.403 | 2984.157 | 2768.28 | finite=True |
```

Additional CPU/GPU order-10 comparison after CUDA warmup:

```text
cpu  order=[10,10] grid=160x160: runs 827.9 ms, 706.8 ms, 754.5 ms
cuda order=[10,10] grid=160x160: runs 1118.0 ms, 393.7 ms, 272.0 ms, peak 200.2 MB
```

Interpretation:

- CUDA has a substantial first-use cost from context/library initialization.
  Warmed order-10 and higher dense workloads show useful GPU acceleration.
- The order-20 patterned-layer stress point completed with finite output and
  about 2.77 GB peak CUDA allocation.

## Release Status

This QA pass is sufficient for a developer-preview v2 foundation:

- installable on Python 3.12 with PyTorch 2.11,
- automated tests passing,
- CPU/CUDA smoke workloads finite,
- large CUDA order-20 stress workload finite,
- reproducible benchmark script committed.
- follow-up optimization patches added batched xz/yz field reconstruction,
  local LU factor reuse, reduced k-vector diagonal hot paths, non-gradient
  material convolution caching, and experimental fixed-geometry `solve_sweep`.

## Follow-Up Optimization QA

Command:

```bash
python3 -m py_compile setup.py torcwa/*.py torcwa/v2/*.py tests/*.py benchmarks/*.py
python3 -m pytest -q
python3 benchmarks/v2_microbench.py --quick --devices cuda --stress
```

Result:

```text
17 passed in 1.61s
```

CUDA stress benchmark after the follow-up optimization series:

```text
| workload | case | device | dtype | size | median ms | min ms | peak CUDA MB | check |
|---|---|---:|---|---|---:|---:|---:|---|
| field_reconstruction | xz_plane | cuda | complex64 | order=[2, 2] samples=(20, 12) | 4.367 | 4.299 | 9.72 | finite=True |
| v2_sweep | fixed_geometry_three_freqs | cuda | complex64 | order=[1, 1] grid=40x40 | 23.150 | 22.897 | 8.47 | finite=True |
| rcwa_stress | patterned_single_layer | cuda | complex64 | order=[10, 10] grid=160x160 | 254.939 | 250.277 | 256.31 | finite=True |
| rcwa_stress | patterned_single_layer | cuda | complex64 | order=[15, 15] grid=224x224 | 928.573 | 890.135 | 1179.61 | finite=True |
| rcwa_stress | patterned_single_layer | cuda | complex64 | order=[20, 20] grid=300x300 | 2906.402 | 2731.858 | 3596.07 | finite=True |
```

Spot comparison against `18efa78`:

```text
interface_cpu   maxabs 0.000e+00 rel 0.000e+00 base 1.011 ms current 0.927 ms
field_cuda      maxabs 2.529e-07 rel 1.059e-07 base 52.028 ms current 40.722 ms peak 8.77 -> 9.73 MB
stress10_cuda   maxabs 1.333e-07 rel 1.335e-07 base 368.250 ms current 302.187 ms peak 199.40 -> 259.99 MB
stress20_cuda   maxabs 0.000e+00 rel 0.000e+00 base 3084.278 ms current 2696.103 ms peak 2769.16 -> 3631.98 MB
```

The optimization series improves field reconstruction and high-order CUDA
runtime in these spot checks, but peak CUDA memory increases for stress
workloads.  A later memory-focused pass should reduce LU workspace/block
assembly pressure before claiming whole-solver memory improvement.

## Memory-Balanced Optimization QA

Command:

```bash
python3 -m py_compile setup.py torcwa/*.py torcwa/v2/*.py tests/*.py benchmarks/*.py
python3 -m pytest -q
python3 benchmarks/v2_microbench.py --quick --devices cuda --stress --memory-modes balanced memory speed
python3 benchmarks/fourier_operator_review.py --quick --devices cpu
python3 benchmarks/fourier_operator_review.py --quick --devices cuda
```

Result:

```text
20 passed in 1.51s
```

Spot comparison against the pre-patch `21040fe` HEAD:

```text
interface_tss        maxabs 0.000e+00 rel 0.000e+00 pass
interface_rss        maxabs 0.000e+00 rel 0.000e+00 pass
pattern_txx_cpu      maxabs 0.000e+00 rel 0.000e+00 pass
pattern_tyy_cpu      maxabs 0.000e+00 rel 0.000e+00 pass
pattern_txx_cuda     maxabs 0.000e+00 rel 0.000e+00 pass
field_xz_Ex_cuda     maxabs 0.000e+00 rel 0.000e+00 pass
field_xz_Hz_cuda     maxabs 0.000e+00 rel 0.000e+00 pass
field_xy_Ex_cuda     maxabs 0.000e+00 rel 0.000e+00 pass
field_xy_Hz_cuda     maxabs 0.000e+00 rel 0.000e+00 pass
stress10_txx_cuda    maxabs 0.000e+00 rel 0.000e+00 pass
stress20_txx_cuda    maxabs 0.000e+00 rel 0.000e+00 pass_rel_3e-3
stress20_tyy_cuda    maxabs 0.000e+00 rel 0.000e+00 pass_rel_3e-3
```

CUDA stress memory policy benchmark:

```text
| mode | order/grid | median ms | peak CUDA MB | previous peak MB |
|---|---|---:|---:|---:|
| balanced | order=[10,10] grid=160x160 | 324.875 | 153.82 | 256.31 |
| balanced | order=[15,15] grid=224x224 | 898.422 | 700.40 | 1179.61 |
| balanced | order=[20,20] grid=300x300 | 2778.271 | 2129.93 | 3596.07 |
| memory | order=[20,20] grid=300x300 | 2797.326 | 2160.02 | 3596.07 |
| speed | order=[20,20] grid=300x300 | 2965.970 | 3570.05 | 3596.07 |
```

Interpretation:

- The default `balanced` mode now reduces order-20 stress peak CUDA allocation by
  about 41% versus the prior LU-heavy follow-up state, while preserving identical
  S-parameter outputs in the sampled cases.
- The reduction comes from structured homogeneous transforms, exact
  block-symmetric layer coupling solves, and conservative repeated-RHS solve
  policy.  The non-homogeneous dense eigensolver path is intentionally unchanged.
- `field_xy` now streams spatial tiles and `field_xz/field_yz` stream both z and
  transverse axes.  Small field planes may run slower when automatic chunks are
  selected, but peak field workspace stays bounded.

Fourier convolution operator review summary:

```text
CPU  c64 rectangle order=[3,3]: dense 0.243 ms, operator 0.806 ms, rel 1.42e-07
CPU  c128 rectangle order=[3,3]: dense 0.103 ms, operator 0.832 ms, rel 2.32e-16
CUDA c64 rectangle order=[3,3]: dense 0.320 ms, operator 5.414 ms, rel 1.59e-07
CUDA c128 rectangle order=[3,3]: dense 0.408 ms, operator 4.813 ms, rel 3.56e-16
```

The validation-only operator matches the dense convention and has finite
material gradients in the smoke cases, but the direct operator prototype is much
slower than dense assembly/matmul at the measured sizes.  Because the current
eigensolver still requires dense matrices, the operator is not connected to the
solver path.

## S4 External Validation Harness

Command:

```bash
python3 -m py_compile setup.py torcwa/*.py torcwa/v2/*.py tests/*.py benchmarks/*.py tools/*.py
python3 -m pytest -q
python3 -m pytest -q -m s4_live
```

Result in the current environment:

```text
28 passed, 7 skipped, 1 xfailed in 1.52s
1 skipped, 35 deselected in 1.19s
```

What changed:

- Added `references/s4/manifest.json` with the six-case S4 core gate.
- Added `tools/generate_s4_fixtures.py` for official Stanford S4 fixture
  generation.
- Added S4 fixture comparison tests that run automatically when committed
  `.npz` fixtures are present and skip pending fixture files otherwise.
- Added the `s4_live` pytest marker for environments with the official S4
  Python extension.
- Added broader analytical/physical invariant tests: Fabry-Perot slab phase,
  Brewster-angle TM reflection suppression, near-critical evanescent branch,
  lossless conservation, reciprocity, tangential field continuity, and
  finite-difference gradient comparison.

S4 installation attempt:

```text
git clone --depth 1 https://github.com/victorliu/S4.git /tmp/S4-stanford
make S4_pyext
```

The S4 core library compiled, but the Python extension failed because
`Python.h` is unavailable.  Installing `python3-dev` requires sudo credentials in
this environment, so genuine S4 `.npz` fixtures are still pending.  The PyPI
package named `S4` was checked and rejected because it is not the Stanford RCWA
solver.

Known physics gate:

- `test_lossy_slab_absorption_is_nonnegative` is marked `xfail`.  The current
  homogeneous lossy slab check produces gain-like `R+T>1`, so this remains a
  release-blocking convention/implementation issue to resolve against S4.

It is not yet sufficient for a final physics major release.  Required release
gates still outstanding:

- official S4 `.npz` fixtures for the six-case core gate,
- resolution of the lossy homogeneous slab absorption `xfail`,
- broader `complex128` gradcheck sweeps across eig broadening values,
- fully batched wavelength/angle/geometry sweep kernels beyond the current
  loop-backed v2 facade,
- external validation before enabling any Fourier-operator or iterative
  eigensolver path in production.

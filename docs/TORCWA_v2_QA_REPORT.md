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

It is not yet sufficient for a final physics major release.  Required release
gates still outstanding:

- external RCWA solver fixtures for at least one rectangular grating and one
  multilayer metasurface,
- oblique slab, Brewster, near-critical, lossy absorption, and reciprocity
  analytical tests,
- `complex128` gradcheck and finite-difference gradient sweeps across
  eig broadening values,
- batched wavelength/angle/geometry sweep API and memory-bounded field
  chunking beyond the current facade.

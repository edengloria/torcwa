# TORCWA v3 Performance And Memory Report

## Environment

- original: `0.1.4.2` from `51c0d24`
- current-modern-api: `0.3.0.dev0` snapshot before optimization
- optimized: `0.3.0.dev0` working tree after optimization
- torch: `2.11.0+cu130`
- cuda: `13.0` / `NVIDIA GeForce RTX 5070`

## Patch Summary

- Added solved-simulation caches for legacy `S_parameters(...)` order matching,
  xy power normalization, and s/p angular conversion terms.  Value caches are
  skipped for gradient-bearing port materials and cache keys include port tensor
  signatures.
- Added exact homogeneous structured transforms for `memory_mode="memory"`.
- Added eigenmode-chunked field Fourier reconstruction for
  `memory_mode="memory"`.
- Normalized fixed-geometry sweep output request dictionaries once outside the
  sweep loop.
- Added `--release-grade` benchmark repeats and a repeated diffraction-query
  benchmark case.

## Results

| workload | case | device | size | original ms | current ms | optimized ms | opt speedup vs original | opt speedup vs current | original MB | current MB | optimized MB | max abs vs current | rel L2 vs current |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| field | xy_plane | cpu | order=[2, 2] samples=20x20 | 0.485 | 0.524 | 0.508 | 0.96x | 1.03x |  |  |  | 0.000 | 0.000 |
| field | xy_plane | cuda | order=[2, 2] samples=20x20 | 1.954 | 2.578 | 2.382 | 0.82x | 1.08x | 8.82 | 8.61 | 8.61 | 0.000 | 0.000 |
| field | xz_plane | cpu | order=[2, 2] samples=20x12 | 3.501 | 1.934 | 1.347 | 2.60x | 1.44x |  |  |  | 0.000 | 0.000 |
| field | xz_plane | cuda | order=[2, 2] samples=20x12 | 26.168 | 18.421 | 13.011 | 2.01x | 1.42x | 8.71 | 9.71 | 9.71 | 0.000 | 0.000 |
| field | yz_plane | cpu | order=[2, 2] samples=20x12 | 3.296 | 1.724 | 1.344 | 2.45x | 1.28x |  |  |  | 0.000 | 0.000 |
| field | yz_plane | cuda | order=[2, 2] samples=20x12 | 18.350 | 19.803 | 5.938 | 3.09x | 3.33x | 8.71 | 9.71 | 9.71 | 0.000 | 0.000 |
| modern_api | solve_s_only | cpu | order=[2, 2] grid=48 |  | 2.069 | 3.163 |  | 0.65x |  |  |  | 0.000 | 0.000 |
| modern_api | solve_s_only | cuda | order=[2, 2] grid=48 |  | 27.374 | 11.977 |  | 2.29x |  | 8.64 | 8.64 | 0.000 | 0.000 |
| modern_api | sweep_16 | cpu | order=[2, 2] grid=48 |  | 37.955 | 37.697 |  | 1.01x |  |  |  | 0.000 | 0.000 |
| modern_api | sweep_16 | cuda | order=[2, 2] grid=48 |  | 317.874 | 257.064 |  | 1.24x |  | 8.89 | 8.90 | 0.000 | 0.000 |
| s_parameter | empty_stack | cpu | order=[0, 0] | 0.378 | 0.407 | 0.404 | 0.94x | 1.01x |  |  |  | 0.000 | 0.000 |
| s_parameter | empty_stack | cuda | order=[0, 0] | 1.930 | 8.472 | 5.937 | 0.33x | 1.43x | 0.02 | 0.02 | 0.02 | 0.000 | 0.000 |
| s_parameter | interface | cpu | order=[0, 0] | 1.082 | 1.500 | 1.060 | 1.02x | 1.42x |  |  |  | 0.000 | 0.000 |
| s_parameter | interface | cuda | order=[0, 0] | 6.115 | 18.190 | 18.744 | 0.33x | 0.97x | 8.16 | 8.16 | 8.16 | 0.000 | 0.000 |
| s_parameter | multilayer_patterned | cpu | order=[2, 2] grid=48 | 6.770 | 4.656 | 4.157 | 1.63x | 1.12x |  |  |  | 0.000 | 0.000 |
| s_parameter | multilayer_patterned | cuda | order=[2, 2] grid=48 | 63.813 | 52.110 | 46.273 | 1.38x | 1.13x | 9.52 | 9.59 | 9.59 | 0.000 | 0.000 |
| s_parameter | patterned_single_layer | cpu | order=[2, 2] grid=48 | 2.924 | 2.849 | 2.767 | 1.06x | 1.03x |  |  |  | 0.000 | 0.000 |
| s_parameter | patterned_single_layer | cuda | order=[2, 2] grid=48 | 13.386 | 26.988 | 10.503 | 1.27x | 2.57x | 8.75 | 8.62 | 8.62 | 0.000 | 0.000 |
| s_parameter | repeated_diffraction_query | cpu | order=[2, 2] grid=48 orders=25 | 1.226 | 1.223 | 0.487 | 2.52x | 2.51x |  |  |  | 0.000 | 0.000 |
| s_parameter | repeated_diffraction_query | cuda | order=[2, 2] grid=48 orders=25 | 18.363 | 16.460 | 3.507 | 5.24x | 4.69x | 8.44 | 8.41 | 8.40 | 0.000 | 0.000 |
| s_parameter | slab | cpu | order=[0, 0] | 1.126 | 1.217 | 1.079 | 1.04x | 1.13x |  |  |  | 0.000 | 0.000 |
| s_parameter | slab | cuda | order=[0, 0] | 9.123 | 21.343 | 7.796 | 1.17x | 2.74x | 8.16 | 8.16 | 8.16 | 0.000 | 0.000 |
| stress | order_10 | cuda | order=[10, 10] grid=160 | 416.178 | 379.486 | 235.241 | 1.77x | 1.61x | 193.32 | 153.83 | 153.83 | 0.000 | 0.000 |
| stress | order_15 | cuda | order=[15, 15] grid=224 | 1125.237 | 1070.858 | 930.158 | 1.21x | 1.15x | 882.16 | 700.42 | 700.42 | 0.000 | 0.000 |
| stress | order_20 | cuda | order=[20, 20] grid=300 | 4006.560 | 2681.180 | 2277.697 | 1.76x | 1.18x | 2682.01 | 2129.94 | 2129.94 | 0.000 | 0.000 |
| sweep | wavelength_16 | cpu | order=[1, 1] grid=40 | 34.476 | 21.385 | 19.832 | 1.74x | 1.08x |  |  |  | 0.000 | 0.000 |
| sweep | wavelength_16 | cuda | order=[1, 1] grid=40 | 179.511 | 297.659 | 177.049 | 1.01x | 1.68x | 8.43 | 8.44 | 8.44 | 0.000 | 0.000 |
| sweep | wavelength_3 | cpu | order=[1, 1] grid=40 | 6.489 | 5.610 | 4.157 | 1.56x | 1.35x |  |  |  | 0.000 | 0.000 |
| sweep | wavelength_3 | cuda | order=[1, 1] grid=40 | 31.643 | 43.806 | 22.207 | 1.42x | 1.97x | 8.42 | 8.43 | 8.43 | 0.000 | 0.000 |

## Notes

- Missing original rows are modern API workloads that do not exist in TORCWA 0.1.4.2.
- `max abs` and `rel L2` compare optimized against the current-modern-api snapshot.
- Correctness gates remain the pytest analytical/S4 fixtures; this report is a performance and memory summary.
- CUDA rows are quick sequential measurements.  Use
  `benchmarks/original_comparison.py --quick --release-grade` for release
  claims, especially for small CUDA workloads where launch and allocator noise
  can dominate.

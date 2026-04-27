# TORCWA v3 Performance And Memory Report

## Environment

- original: `0.1.4.2` from `51c0d24`
- current-modern-api: `0.3.0.dev0` snapshot before optimization
- optimized: `0.3.0.dev0` working tree after optimization
- torch: `2.11.0+cu130`
- cuda: `13.0` / `NVIDIA GeForce RTX 5070`

## Results

Implemented safe optimizations in this measurement:

- modern API field chunk propagation through `Result.fields.plane(...)`
- cached `Result` order/S-parameter/diffraction table queries
- `RCWA.sweep(...)` routed through the fixed-geometry v2 sweep backend
- optional S-parameter-only solve mode with `store_fields=False`

Deferred to the next isolated numerical-core patch:

- homogeneous layer structured eig/coupling path
- deeper eigenmode-level field streaming for large field planes

| workload | case | device | size | original ms | current ms | optimized ms | opt speedup vs original | opt speedup vs current | original MB | current MB | optimized MB | max abs vs current | rel L2 vs current |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| field | xy_plane | cpu | order=[2, 2] samples=20x20 | 0.974 | 1.019 | 1.256 | 0.78x | 0.81x |  |  |  | 0.000 | 0.000 |
| field | xy_plane | cuda | order=[2, 2] samples=20x20 | 2.677 | 1.889 | 2.187 | 1.22x | 0.86x | 8.82 | 8.61 | 8.61 | 0.000 | 0.000 |
| field | xz_plane | cpu | order=[2, 2] samples=20x12 | 6.191 | 3.017 | 2.697 | 2.30x | 1.12x |  |  |  | 0.000 | 0.000 |
| field | xz_plane | cuda | order=[2, 2] samples=20x12 | 18.102 | 19.007 | 5.427 | 3.34x | 3.50x | 8.71 | 9.71 | 9.71 | 0.000 | 0.000 |
| field | yz_plane | cpu | order=[2, 2] samples=20x12 | 6.649 | 2.627 | 2.567 | 2.59x | 1.02x |  |  |  | 0.000 | 0.000 |
| field | yz_plane | cuda | order=[2, 2] samples=20x12 | 18.544 | 4.682 | 5.686 | 3.26x | 0.82x | 8.71 | 9.71 | 9.71 | 0.000 | 0.000 |
| modern_api | solve_s_only | cpu | order=[2, 2] grid=48 |  | 5.478 | 2.147 |  | 2.55x |  |  |  | 0.000 | 0.000 |
| modern_api | solve_s_only | cuda | order=[2, 2] grid=48 |  | 10.077 | 10.381 |  | 0.97x |  | 8.64 | 8.64 | 0.000 | 0.000 |
| modern_api | sweep_16 | cpu | order=[2, 2] grid=48 |  | 52.255 | 27.126 |  | 1.93x |  |  |  | 0.000 | 0.000 |
| modern_api | sweep_16 | cuda | order=[2, 2] grid=48 |  | 183.878 | 159.851 |  | 1.15x |  | 8.89 | 8.89 | 0.000 | 0.000 |
| s_parameter | empty_stack | cpu | order=[0, 0] | 0.362 | 0.354 | 0.375 | 0.96x | 0.94x |  |  |  | 0.000 | 0.000 |
| s_parameter | empty_stack | cuda | order=[0, 0] | 2.618 | 8.741 | 2.235 | 1.17x | 3.91x | 0.02 | 0.02 | 0.02 | 0.000 | 0.000 |
| s_parameter | interface | cpu | order=[0, 0] | 1.077 | 1.011 | 1.115 | 0.97x | 0.91x |  |  |  | 0.000 | 0.000 |
| s_parameter | interface | cuda | order=[0, 0] | 7.945 | 14.861 | 9.416 | 0.84x | 1.58x | 8.16 | 8.16 | 8.16 | 0.000 | 0.000 |
| s_parameter | multilayer_patterned | cpu | order=[2, 2] grid=48 | 13.000 | 7.023 | 6.827 | 1.90x | 1.03x |  |  |  | 0.000 | 0.000 |
| s_parameter | multilayer_patterned | cuda | order=[2, 2] grid=48 | 51.059 | 36.763 | 34.103 | 1.50x | 1.08x | 9.52 | 9.59 | 9.59 | 0.000 | 0.000 |
| s_parameter | patterned_single_layer | cpu | order=[2, 2] grid=48 | 4.020 | 2.719 | 2.757 | 1.46x | 0.99x |  |  |  | 0.000 | 0.000 |
| s_parameter | patterned_single_layer | cuda | order=[2, 2] grid=48 | 13.140 | 10.240 | 10.321 | 1.27x | 0.99x | 8.75 | 8.62 | 8.62 | 0.000 | 0.000 |
| s_parameter | slab | cpu | order=[0, 0] | 1.076 | 2.508 | 1.112 | 0.97x | 2.25x |  |  |  | 0.000 | 0.000 |
| s_parameter | slab | cuda | order=[0, 0] | 10.018 | 6.828 | 8.011 | 1.25x | 0.85x | 8.16 | 8.16 | 8.16 | 0.000 | 0.000 |
| stress | order_10 | cuda | order=[10, 10] grid=160 | 311.290 | 279.015 | 287.060 | 1.08x | 0.97x | 193.32 | 153.83 | 153.83 | 0.000 | 0.000 |
| stress | order_15 | cuda | order=[15, 15] grid=224 | 1140.610 | 877.469 | 859.339 | 1.33x | 1.02x | 882.16 | 700.42 | 700.42 | 0.000 | 0.000 |
| stress | order_20 | cuda | order=[20, 20] grid=300 | 3482.167 | 2488.101 | 2315.638 | 1.50x | 1.07x | 2682.01 | 2129.94 | 2129.94 | 0.000 | 0.000 |
| sweep | wavelength_16 | cpu | order=[1, 1] grid=40 | 47.529 | 36.277 | 27.448 | 1.73x | 1.32x |  |  |  | 0.000 | 0.000 |
| sweep | wavelength_16 | cuda | order=[1, 1] grid=40 | 401.872 | 192.193 | 208.187 | 1.93x | 0.92x | 8.43 | 8.44 | 8.44 | 0.000 | 0.000 |
| sweep | wavelength_3 | cpu | order=[1, 1] grid=40 | 16.831 | 5.569 | 3.322 | 5.07x | 1.68x |  |  |  | 0.000 | 0.000 |
| sweep | wavelength_3 | cuda | order=[1, 1] grid=40 | 78.483 | 30.651 | 28.367 | 2.77x | 1.08x | 8.42 | 8.43 | 8.43 | 0.000 | 0.000 |

## Notes

- Missing original rows are modern API workloads that do not exist in TORCWA 0.1.4.2.
- `max abs` and `rel L2` compare optimized against the current-modern-api snapshot.
- Correctness gates remain the pytest analytical/S4 fixtures; this report is a performance and memory summary.

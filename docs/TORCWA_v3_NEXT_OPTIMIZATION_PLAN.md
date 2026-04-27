# TORCWA v3 Next Optimization Plan

## Current Patch Status

The current optimization patch keeps public APIs unchanged and preserves the
`618cb3d` v3 baseline numerically.  It intentionally activates the higher-risk
numerical-core changes only through existing policies such as
`memory_mode="memory"` so the default balanced path remains conservative.

Implemented in this patch:

- Legacy `S_parameters(...)` now caches diffraction-order matching, xy power
  normalization terms, and s/p angular conversion terms per solved simulation.
  Normalization value caches are skipped when port material tensors require
  gradients, and cache keys include port tensor signatures.
- Homogeneous layers can skip dense `P/Q` storage in `memory_mode="memory"` by
  using exact order-wise 2x2 block transforms.
- Internal field Fourier reconstruction has a `memory_mode="memory"` streaming
  path that accumulates eigenmode chunks instead of materializing the full
  `(2M, 2M, z_chunk)` intermediate.
- Fixed-geometry sweep request dictionaries are normalized once outside the
  wavelength loop.
- `benchmarks/original_comparison.py` has a `--release-grade` repeat mode and a
  repeated diffraction-query workload.

## Latest Measurement Summary

Quick benchmark results on RTX 5070 / PyTorch `2.11.0+cu130` are recorded in
`docs/TORCWA_v3_PERFORMANCE_REPORT.md`.

- Repeated S-parameter diffraction query:
  - CPU: current `1.223 ms`, optimized `0.487 ms` (`2.51x`).
  - CUDA: current `16.460 ms`, optimized `3.507 ms` (`4.69x`).
- Sweep workload:
  - CPU wavelength-3 legacy sweep improved from current `5.610 ms` to
    `4.157 ms`.
  - CUDA wavelength-3 legacy sweep improved from current `43.806 ms` to
    `22.207 ms`.
  - CUDA wavelength-16 legacy sweep improved from current `297.659 ms` to
    `177.049 ms`.
  - CUDA sweep quick results remain noisy and should be judged with
    `--release-grade` before release claims.
- CUDA stress memory is unchanged versus `618cb3d`:
  - order 20 peak memory remains `2129.94 MB`, versus original TORCWA
    `2682.01 MB`.
  - quick runtime improved from current `2681.180 ms` to `2277.697 ms`.

## Next Patch Order

1. **Release-grade benchmark stabilization**
   - Run original/current/optimized with `--quick --release-grade` before
     publishing performance claims.
   - Add a compact summary script option that reports median and min together
     for CUDA stress cases.

2. **Balanced-mode homogeneous structured promotion**
   - Keep the new structured path in `memory_mode="memory"` until release-grade
     benchmarks show no runtime regression.
   - If stable, promote homogeneous structured `P` solve to `balanced` for
     homogeneous-only stacks, while leaving dense debug comparison available in
     tests.

3. **Field streaming large-plane benchmark**
   - Add explicit large `xz/yz/xy` memory-mode benchmarks where streaming should
     reduce peak memory.
   - Keep small-plane balanced timings separate, because streaming overhead can
     dominate small fields.

4. **Sweep reuse beyond request normalization**
   - Reuse request order indices and output polarization mapping when a single
     legacy solver instance handles multiple requests.
   - Do not reuse frequency-dependent eigen/coupling data until an analytical
     proof and fixture comparison show equivalence.

5. **Autograd cache regression coverage**
   - Add a repeated-query-after-backward regression test for differentiable
     port-material workflows.
   - Extend the same policy to any future caches that may retain tensors with
     autograd history.

## Acceptance Criteria

- `python3 -m pytest -q` passes, including S4 committed fixtures.
- Optimized results match current baseline:
  - `complex128`: `rtol=1e-8`, `atol=1e-10`
  - `complex64`: `rtol=5e-4`, `atol=5e-5`
  - CUDA `order=[20,20]`: S-parameter relative diff `<=3e-3`
- Balanced-mode peak CUDA memory must not increase versus `618cb3d`.
- Runtime improvements are reported only from sequential benchmark runs; CUDA
  quick runs are treated as diagnostic, not release-grade claims.
- Do not merge a numerical-core optimization if it improves runtime but fails
  analytical, S4, or legacy-equivalence gates.

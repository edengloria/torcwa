# TORCWA v3 Next Optimization Plan

## Summary

The current v3 patch keeps numerical results identical to the current modern API
baseline while improving API-level sweep/result reuse and preserving the v2
memory reductions versus original TORCWA. The next optimization work should move
carefully into numerical-core internals where the potential gains are larger but
the accuracy risk is higher.

## Patch Order

1. **Homogeneous structured eig/coupling path**
   - Replace homogeneous-layer dense `P/Q` block assembly with exact order-wise
     structured helpers.
   - Keep a debug flag that materializes the dense legacy form and compares
     `P`, `Q`, `P@Q`, S-parameters, and fields.
   - Gate on slab, multilayer homogeneous stack, and S4 homogeneous fixtures.

2. **Large field streaming**
   - Add eigenmode chunking inside field reconstruction so
     `(2M, 2M, z_chunk)` intermediates are accumulated instead of materialized.
   - Enable it by default only for `memory_mode="memory"` first.
   - Compare `xz`, `yz`, and `xy` planes against the current dense field path.

3. **S-parameter normalization cache in legacy core**
   - Move repeated kz/polarization normalization terms out of each
     `S_parameters(...)` call.
   - Cache per solved simulation and invalidate only when k-vectors or ports
     change.
   - This should speed diffraction-table and power-balance workloads without
     changing the public API.

4. **Fixed-geometry sweep reuse beyond material convolution**
   - Reuse homogeneous interface transforms and request normalization across
     wavelength/angle sweep points where mathematically valid.
   - Keep material/eigen reuse disabled across frequency unless explicitly
     proven equivalent for that case.

5. **Benchmark stability improvements**
   - Add longer-repeat benchmark mode for release reporting.
   - Separate small-kernel benchmarks from full RCWA benchmarks so CUDA launch
     noise does not obscure solver changes.

## Acceptance Criteria

- `python3 -m pytest -q` passes, including S4 committed fixtures.
- Optimized results match current baseline:
  - `complex128`: `rtol=1e-8`, `atol=1e-10`
  - `complex64`: `rtol=5e-4`, `atol=5e-5`
  - CUDA `order=[20,20]`: S-parameter relative diff `<=3e-3`
- For each accepted patch, update
  `docs/TORCWA_v3_PERFORMANCE_REPORT.md` with original/current/optimized
  measurements.
- Do not merge a numerical-core optimization if it improves runtime but fails
  analytical, S4, or legacy-equivalence gates.

## Current Baseline To Preserve

From the v3 report on RTX 5070 / PyTorch `2.11.0+cu130`:

- CUDA stress `order=[20,20]`: original `3482 ms / 2682 MB`, optimized
  `2316 ms / 2130 MB`.
- CUDA stress `order=[15,15]`: original `1141 ms / 882 MB`, optimized
  `859 ms / 700 MB`.
- CPU modern sweep 16 wavelengths: current `52.3 ms`, optimized `27.1 ms`.

Future patches should improve these numbers without increasing peak memory in
balanced mode.

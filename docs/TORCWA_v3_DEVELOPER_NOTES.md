# TORCWA v3 Developer Notes

TORCWA v3 introduces a modern public API while keeping the validated legacy
solver path available. The design goal is a simple researcher-facing workflow:

```python
import torcwa as tw

stack = tw.Stack(period=(300.0, 300.0))
stack.add_layer(thickness=80.0, eps=2.25)

solver = tw.RCWA(wavelength=500.0, orders=(3, 3), device="cuda")
result = solver.solve(stack, tw.PlaneWave(polarization="x"))

txx = result.transmission(order=(0, 0), polarization="x")
balance = result.power_balance(input_polarization="x")
```

## Public API Boundary

- `torcwa.RCWA` owns wavelength/orders/dtype/device/options.
- `torcwa.Stack` owns periodic layers and input/output media.
- `torcwa.PlaneWave` owns incidence angle, direction, polarization, and source
  notation.
- `torcwa.Result` owns S-parameter queries, diffraction tables, power balance,
  and field reconstruction.
- `torcwa.MaterialGrid`, `torcwa.UnitCell`, and `torcwa.material.mix(...)`
  provide the simple differentiable geometry/material path.
- `torcwa.core` is the stable boundary for low-level basis, Fourier, linalg,
  eig, and physics helpers during future numerical-core modularization.

The legacy `torcwa.rcwa` API remains supported. New examples and README
quickstarts should prefer the v3 public API.

## Performance Features

- `SolverOptions.store_fields` controls whether global layer coupling history is
  retained for field reconstruction. Use `store_fields=False` for
  S-parameter-only workloads.
- `Result.fields.plane(...)` respects `SolverOptions.field_chunk_size` unless an
  explicit `chunk_size` override is passed.
- `Result` caches repeated order, S-parameter, and diffraction table queries for
  one solved object.
- `RCWA.sweep(...)` routes through the fixed-geometry v2 sweep backend and
  benefits from material convolution caching.

## Validation Commands

```bash
python3 -m py_compile setup.py torcwa/*.py torcwa/v2/*.py torcwa/core/*.py tests/*.py benchmarks/*.py tools/*.py example/*.py
python3 -m pytest -q
python3 benchmarks/v2_microbench.py --quick --devices cpu
python3 benchmarks/v2_microbench.py --quick --devices cuda --stress
```

External S4 validation remains fixture-backed for normal QA. Live S4 fixture
generation is optional:

```bash
python3 -m pytest -q -m s4_live
python3 tools/generate_s4_fixtures.py --overwrite
```

## Original TORCWA Comparison

The reference original is `51c0d24` / TORCWA `0.1.4.2`.

```bash
git worktree add --detach /tmp/torcwa-original-51c0d24 51c0d24
python3 benchmarks/original_comparison.py --label optimized --quick --devices auto --stress
```

The current report is recorded in
[`TORCWA_v3_PERFORMANCE_REPORT.md`](./TORCWA_v3_PERFORMANCE_REPORT.md).

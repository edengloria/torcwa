# TORCWA v2 Migration Guide

This branch introduces a developer-preview v2 surface while preserving the
legacy `torcwa.rcwa` workflow.

## Runtime Baseline

- Python: 3.10+
- PyTorch: 2.11+
- CUDA: optional but recommended for high-order dense workloads
- Package version in this branch: `0.2.0.dev0`

## Legacy Code Keeps Working

Existing scripts can continue using:

```python
import torcwa

sim = torcwa.rcwa(freq=1 / 500, order=[5, 5], L=[300.0, 300.0])
sim.set_incident_angle(0.0, 0.0)
sim.add_layer(thickness=80.0, eps=eps)
sim.solve_global_smatrix()
txx = sim.S_parameters([0, 0], polarization="xx")
```

The `S_parameters` typo is now handled as follows:

```python
sim.S_parameters([0, 0], evanescent=1e-3)  # canonical
sim.S_parameters([0, 0], evanscent=1e-3)   # deprecated alias with warning
```

## New v2 Facade

The v2 entrypoint makes solver state explicit with dataclasses.  During this
transition it delegates the full solve to the validated legacy implementation
while v2 numerical kernels are moved under the hood.

```python
import torch
import torcwa
from torcwa.v2 import RCWAConfig, RCWASolver, SolverOptions

config = RCWAConfig(
    freq=1 / 500,
    order=(5, 5),
    lattice=(300.0, 300.0),
    options=SolverOptions(dtype=torch.complex64, device=torch.device("cuda")),
)

solver = RCWASolver(config)
solver.add_layer(thickness=80.0, eps=eps)
solver.solve()
txx = solver.s_parameter([0, 0], polarization="xx")
```

## Field Chunking

`SolverOptions.field_chunk_size` limits the number of z samples handled by one
field reconstruction batch.  `None` uses one batch per layer region.

```python
config = RCWAConfig(
    freq=1 / 500,
    order=(5, 5),
    lattice=(300.0, 300.0),
    options=SolverOptions(dtype=torch.complex64, device=torch.device("cuda"), field_chunk_size=16),
)

solver = RCWASolver(config).add_layer(80.0, eps=eps).solve()
electric, magnetic = solver.field_plane(plane="xz", axis0=x_axis, axis1=z_axis, offset=150.0)
electric_small_chunks, magnetic_small_chunks = solver.field_plane(
    plane="xz",
    axis0=x_axis,
    axis1=z_axis,
    offset=150.0,
    chunk_size=4,
)
```

## Fixed-Geometry Sweeps

`solve_sweep` is an experimental loop-backed v2 API for fixed layer/material
stacks.  It reuses the validated legacy solve path and benefits from the
material convolution cache when the same non-gradient material tensors are used
across sweep points.

```python
result = solver.solve_sweep(
    freqs=torch.tensor([1 / 450, 1 / 500, 1 / 550], device=device),
    incident_angles=0.0,
    azimuth_angles=0.0,
    requests=[
        {"name": "txx", "orders": [0, 0], "polarization": "xx"},
        {"name": "tyy", "orders": [0, 0], "polarization": "yy"},
    ],
)

txx = result["txx"]
```

## Material Convolution Cache

Non-gradient material tensors are cached by tensor identity, storage, version,
shape, dtype, device, and Fourier order.  Tensors with `requires_grad=True` are
not cached.

```python
torcwa.rcwa.clear_material_cache()
```

## Numerical Behavior Changes

- Coupling, Redheffer product, layer S-matrix, eig backward, and field
  reconstruction paths avoid explicit inverse products where the solve form is
  available.
- Diagonal phase products in layer field reconstruction use broadcasting.
- p/s source conversion avoids allocating a dense block-diagonal conversion
  matrix.
- Empty stacks now produce matrix-shaped zero reflection blocks, so reflection
  S-parameter queries are valid even with no internal layers.

## Verification Commands

Run the automated QA:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

Run the standard benchmark:

```bash
python3 benchmarks/v2_microbench.py
```

Run the CUDA stress benchmark:

```bash
python3 benchmarks/v2_microbench.py --quick --devices cuda --stress
```

## Migration Recommendations

- Keep production notebooks on `torcwa.rcwa` until your target structures pass
  analytical, legacy, and external-solver fixtures.
- Use `torcwa.v2.RCWASolver` for new experiments that should be easier to batch
  and validate later.
- Replace `evanscent=` with `evanescent=` now; the alias remains only for
  compatibility.
- Prefer explicit `torch.device` and complex dtype selection in scripts so
  CPU/CUDA comparisons remain reproducible.

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch

import torcwa


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _flatten_result(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.reshape([-1])
    if isinstance(value, (list, tuple)):
        parts = [_flatten_result(item) for item in value]
        return torch.cat(parts) if parts else torch.empty([0])
    return torch.as_tensor(value).reshape([-1])


def _serialize_result(value) -> dict:
    flat = _flatten_result(value).detach().cpu()
    if not torch.is_complex(flat):
        flat = flat.to(torch.complex128)
    else:
        flat = flat.to(torch.complex128)
    return {
        "shape": list(flat.shape),
        "real": torch.real(flat).tolist(),
        "imag": torch.imag(flat).tolist(),
    }


def _measure(fn: Callable[[], object], *, device: torch.device, repeats: int, warmup: int):
    sample = fn()
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return sample, statistics.median(times), min(times), _peak_mb(device)


def _record(label, workload, case, device, dtype, size, fn, *, repeats, warmup):
    sample, median_ms, min_ms, peak_mb = _measure(fn, device=device, repeats=repeats, warmup=warmup)
    flat = _flatten_result(sample)
    finite = bool(torch.isfinite(torch.real(flat)).all() and torch.isfinite(torch.imag(flat)).all())
    return {
        "label": label,
        "workload": workload,
        "case": case,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "size": size,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "peak_cuda_mb": peak_mb,
        "finite": finite,
        "result": _serialize_result(sample),
    }


def _patterned_eps(device, *, grid, dtype=torch.float32):
    geo = torcwa.geometry(Lx=300.0, Ly=300.0, nx=grid, ny=grid, edge_sharpness=40.0, dtype=dtype, device=device)
    mask = geo.rectangle(Wx=120.0, Wy=90.0, Cx=150.0, Cy=150.0)
    return mask * 2.25 + (1.0 - mask)


def _legacy_solve(device, dtype, *, order, layers, output_eps=1.0, angle=0.0):
    sim = torcwa.rcwa(freq=1 / 500, order=list(order), L=[300.0, 300.0], dtype=dtype, device=device)
    if output_eps != 1.0:
        sim.add_output_layer(eps=output_eps, mu=1.0)
    sim.set_incident_angle(angle, 0.0)
    for thickness, eps in layers:
        sim.add_layer(thickness=thickness, eps=eps)
    sim.solve_global_smatrix()
    return sim


def _legacy_s_cases(label, device, *, quick):
    dtype = torch.complex64
    repeats = 3 if quick else 6
    warmup = 1 if quick else 2
    records = []

    cases = [
        ("empty_stack", (0, 0), [], 1.0, lambda sim: sim.S_parameters([0, 0], polarization="xx")),
        ("interface", (0, 0), [], 2.25, lambda sim: (sim.S_parameters([0, 0], polarization="ss"), sim.S_parameters([0, 0], polarization="ss", port="reflection"))),
        ("slab", (0, 0), [(120.0, 2.25)], 1.0, lambda sim: (sim.S_parameters([0, 0], polarization="ss"), sim.S_parameters([0, 0], polarization="ss", port="reflection"))),
    ]
    for name, order, layers, output_eps, extract in cases:
        def fn(order=order, layers=layers, output_eps=output_eps, extract=extract):
            sim = _legacy_solve(device, dtype, order=order, layers=layers, output_eps=output_eps)
            return extract(sim)

        records.append(_record(label, "s_parameter", name, device, dtype, f"order={list(order)}", fn, repeats=repeats, warmup=warmup))

    grid = 48 if quick else 96
    order = (2, 2) if quick else (3, 3)
    eps = _patterned_eps(device, grid=grid)

    def patterned():
        sim = _legacy_solve(device, dtype, order=order, layers=[(80.0, eps)])
        return sim.S_parameters([0, 0], polarization="xx")

    records.append(_record(label, "s_parameter", "patterned_single_layer", device, dtype, f"order={list(order)} grid={grid}", patterned, repeats=repeats, warmup=warmup))

    def multilayer():
        sim = _legacy_solve(device, dtype, order=order, layers=[(50.0, eps), (40.0, 1.44), (50.0, eps)])
        return sim.S_parameters([0, 0], polarization="xx")

    records.append(_record(label, "s_parameter", "multilayer_patterned", device, dtype, f"order={list(order)} grid={grid}", multilayer, repeats=repeats, warmup=warmup))
    return records


def _legacy_field_cases(label, device, *, quick):
    dtype = torch.complex64
    repeats = 3 if quick else 6
    warmup = 1
    grid = 48 if quick else 96
    order = (2, 2) if quick else (3, 3)
    eps = _patterned_eps(device, grid=grid)
    sim = _legacy_solve(device, dtype, order=order, layers=[(80.0, eps)])
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")
    n0 = 20 if quick else 48
    n1 = 12 if quick else 32
    x_axis = torch.linspace(0.0, 300.0, n0, device=device)
    y_axis = torch.linspace(0.0, 300.0, n0, device=device)
    z_axis = torch.linspace(-20.0, 110.0, n1, device=device)

    return [
        _record(label, "field", "xz_plane", device, dtype, f"order={list(order)} samples={n0}x{n1}", lambda: sim.field_xz(x_axis, z_axis, 150.0), repeats=repeats, warmup=warmup),
        _record(label, "field", "yz_plane", device, dtype, f"order={list(order)} samples={n0}x{n1}", lambda: sim.field_yz(y_axis, z_axis, 150.0), repeats=repeats, warmup=warmup),
        _record(label, "field", "xy_plane", device, dtype, f"order={list(order)} samples={n0}x{n0}", lambda: sim.field_xy(0, x_axis, y_axis, z_prop=20.0), repeats=repeats, warmup=warmup),
    ]


def _legacy_sweep_cases(label, device, *, quick):
    dtype = torch.complex64
    repeats = 2 if quick else 4
    warmup = 1
    grid = 40 if quick else 96
    order = (1, 1) if quick else (2, 2)
    counts = (3, 16) if quick else (16, 64)
    eps = _patterned_eps(device, grid=grid)
    records = []
    for count in counts:
        wavelengths = torch.linspace(450.0, 700.0, count, dtype=torch.float32, device=device)

        def sweep():
            values = []
            for wavelength in wavelengths:
                sim = torcwa.rcwa(freq=1 / wavelength, order=list(order), L=[300.0, 300.0], dtype=dtype, device=device)
                sim.set_incident_angle(0.0, 0.0)
                sim.add_layer(thickness=80.0, eps=eps)
                sim.solve_global_smatrix()
                values.append(sim.S_parameters([0, 0], polarization="xx"))
            return torch.stack(values, dim=0)

        records.append(_record(label, "sweep", f"wavelength_{count}", device, dtype, f"order={list(order)} grid={grid}", sweep, repeats=repeats, warmup=warmup))
    return records


def _legacy_stress_cases(label, device):
    if device.type != "cuda":
        return []
    dtype = torch.complex64
    records = []
    for order, grid, repeats in [(10, 160, 3), (15, 224, 2), (20, 300, 2)]:
        eps = _patterned_eps(device, grid=grid)

        def solve(order=order, eps=eps):
            sim = torcwa.rcwa(freq=1 / 500, order=[order, order], L=[300.0, 300.0], dtype=dtype, device=device)
            if hasattr(sim, "memory_mode"):
                sim.memory_mode = "balanced"
            sim.set_incident_angle(0.0, 0.0)
            sim.add_layer(thickness=80.0, eps=eps)
            sim.solve_global_smatrix()
            return sim.S_parameters([0, 0], polarization="xx")

        records.append(_record(label, "stress", f"order_{order}", device, dtype, f"order=[{order}, {order}] grid={grid}", solve, repeats=repeats, warmup=1))
    return records


def _modern_cases(label, device, *, quick):
    if not all(hasattr(torcwa, name) for name in ("RCWA", "Stack", "PlaneWave")):
        return []
    dtype = torch.complex64
    repeats = 3 if quick else 6
    warmup = 1
    period = (300.0, 300.0)
    grid = 48 if quick else 96
    order = (2, 2) if quick else (3, 3)
    eps = _patterned_eps(device, grid=grid)
    stack = torcwa.Stack(period=period)
    material = torcwa.MaterialGrid(eps, period) if hasattr(torcwa, "MaterialGrid") else eps
    stack.add_layer(thickness=80.0, eps=material)
    solver = torcwa.RCWA(wavelength=500.0, orders=order, dtype=dtype, device=device)

    def solve():
        try:
            result = solver.solve(stack, torcwa.PlaneWave(polarization="x"), store_fields=False)
        except TypeError:
            result = solver.solve(stack, torcwa.PlaneWave(polarization="x"))
        return result.transmission(polarization="x")

    records = [_record(label, "modern_api", "solve_s_only", device, dtype, f"order={list(order)} grid={grid}", solve, repeats=repeats, warmup=warmup)]

    wavelengths = torch.linspace(450.0, 700.0, 16 if quick else 64, dtype=torch.float32, device=device)

    def sweep():
        return solver.sweep(
            stack,
            source=torcwa.PlaneWave(polarization="x"),
            wavelength=wavelengths,
            outputs=[torcwa.Output.transmission(order=(0, 0), polarization="x", name="txx")],
        )["txx"]

    records.append(_record(label, "modern_api", f"sweep_{len(wavelengths)}", device, dtype, f"order={list(order)} grid={grid}", sweep, repeats=2 if quick else 4, warmup=1))
    return records


def _devices(selection):
    if selection == "cpu":
        return [torch.device("cpu")]
    if selection == "cuda":
        return [torch.device("cuda")] if torch.cuda.is_available() else []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original TORCWA-compatible workloads")
    parser.add_argument("--label", required=True)
    parser.add_argument("--devices", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    records = []
    for device in _devices(args.devices):
        records.extend(_legacy_s_cases(args.label, device, quick=args.quick))
        records.extend(_legacy_field_cases(args.label, device, quick=args.quick))
        records.extend(_legacy_sweep_cases(args.label, device, quick=args.quick))
        records.extend(_modern_cases(args.label, device, quick=args.quick))
        if args.stress:
            records.extend(_legacy_stress_cases(args.label, device))

    payload = {
        "label": args.label,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torcwa_version": getattr(torcwa, "__version__", "unknown"),
        "records": records,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

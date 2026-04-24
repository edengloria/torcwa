from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch

import torcwa
from torcwa.v2.linalg import diag_post_multiply, solve_left


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _measure(
    fn: Callable[[], torch.Tensor | tuple | list],
    *,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[float, float, float | None]:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    return statistics.median(times_ms), min(times_ms), _peak_mb(device)


def _complex_randn(shape, *, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator(device=device).manual_seed(seed)
    real = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    imag = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    return torch.complex(real, imag).to(dtype)


def bench_solve_vs_inverse(device: torch.device, *, quick: bool) -> list[dict]:
    dtype = torch.complex64
    n = 64 if quick else 128
    rhs = 16 if quick else 32
    repeats = 8 if quick else 20
    warmup = 2 if quick else 5
    raw = _complex_randn((n, n), dtype=dtype, device=device, seed=11)
    A = raw @ torch.conj(raw).T + 0.1 * torch.eye(n, dtype=dtype, device=device)
    B = _complex_randn((n, rhs), dtype=dtype, device=device, seed=12)

    solve_result = solve_left(A, B)
    inverse_result = torch.linalg.inv(A) @ B
    rel_error = torch.linalg.norm(solve_result - inverse_result) / torch.linalg.norm(inverse_result)

    records = []
    for name, fn in (
        ("solve_left", lambda: solve_left(A, B)),
        ("inverse_matmul", lambda: torch.linalg.inv(A) @ B),
    ):
        median_ms, min_ms, peak_mb = _measure(fn, device=device, repeats=repeats, warmup=warmup)
        records.append(
            {
                "workload": "linear_system",
                "case": name,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "size": f"{n}x{n} rhs={rhs}",
                "median_ms": median_ms,
                "min_ms": min_ms,
                "peak_cuda_mb": peak_mb,
                "check": f"rel_error_vs_inverse={float(rel_error):.3e}",
            }
        )
    return records


def bench_diag_broadcast(device: torch.device, *, quick: bool) -> list[dict]:
    dtype = torch.complex64
    n = 256 if quick else 768
    repeats = 10 if quick else 25
    warmup = 2 if quick else 5
    diagonal = _complex_randn((n,), dtype=dtype, device=device, seed=21)
    matrix = _complex_randn((n, n), dtype=dtype, device=device, seed=22)

    broadcast_result = diag_post_multiply(matrix, diagonal)
    explicit_result = matrix @ torch.diag(diagonal)
    rel_error = torch.linalg.norm(broadcast_result - explicit_result) / torch.linalg.norm(explicit_result)

    records = []
    for name, fn in (
        ("broadcast_diag", lambda: diag_post_multiply(matrix, diagonal)),
        ("explicit_diag_matmul", lambda: matrix @ torch.diag(diagonal)),
    ):
        median_ms, min_ms, peak_mb = _measure(fn, device=device, repeats=repeats, warmup=warmup)
        records.append(
            {
                "workload": "diagonal_product",
                "case": name,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "size": f"{n}x{n}",
                "median_ms": median_ms,
                "min_ms": min_ms,
                "peak_cuda_mb": peak_mb,
                "check": f"rel_error_vs_explicit={float(rel_error):.3e}",
            }
        )
    return records


def _patterned_eps(device: torch.device, *, grid: int) -> torch.Tensor:
    geo = torcwa.geometry(Lx=300.0, Ly=300.0, nx=grid, ny=grid, edge_sharpness=40.0, dtype=torch.float32, device=device)
    mask = geo.rectangle(Wx=120.0, Wy=90.0, Cx=150.0, Cy=150.0)
    return mask * 2.25 + (1.0 - mask)


def bench_rcwa(device: torch.device, *, quick: bool) -> list[dict]:
    dtype = torch.complex64
    order = [2, 2] if quick else [3, 3]
    grid = 48 if quick else 72
    repeats = 3 if quick else 6
    warmup = 1 if quick else 2
    eps = _patterned_eps(device, grid=grid)

    def solve_case():
        sim = torcwa.rcwa(freq=1 / 500, order=order, L=[300.0, 300.0], dtype=dtype, device=device)
        sim.set_incident_angle(0.0, 0.0)
        sim.add_layer(thickness=80.0, eps=eps)
        sim.solve_global_smatrix()
        return sim.S_parameters([0, 0], polarization="xx")

    sample = solve_case()
    finite = bool(torch.isfinite(torch.real(sample)).all() and torch.isfinite(torch.imag(sample)).all())
    median_ms, min_ms, peak_mb = _measure(solve_case, device=device, repeats=repeats, warmup=warmup)
    records = [
        {
            "workload": "rcwa_solve",
            "case": "patterned_single_layer",
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "size": f"order={order} grid={grid}x{grid}",
            "median_ms": median_ms,
            "min_ms": min_ms,
            "peak_cuda_mb": peak_mb,
            "check": f"finite={finite}",
        }
    ]

    sim = torcwa.rcwa(freq=1 / 500, order=order, L=[300.0, 300.0], dtype=dtype, device=device)
    sim.set_incident_angle(0.0, 0.0)
    sim.add_layer(thickness=80.0, eps=eps)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="ps")
    x_axis = torch.linspace(0.0, 300.0, 20 if quick else 40, device=device)
    z_axis = torch.linspace(-20.0, 110.0, 12 if quick else 24, device=device)

    def field_case():
        electric, magnetic = sim.field_xz(x_axis, z_axis, 150.0)
        return electric[0], magnetic[0]

    electric_x, magnetic_x = field_case()
    finite = bool(torch.isfinite(torch.real(electric_x)).all() and torch.isfinite(torch.real(magnetic_x)).all())
    median_ms, min_ms, peak_mb = _measure(field_case, device=device, repeats=repeats, warmup=warmup)
    records.append(
        {
            "workload": "field_reconstruction",
            "case": "xz_plane",
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "size": f"order={order} samples={tuple(electric_x.shape)}",
            "median_ms": median_ms,
            "min_ms": min_ms,
            "peak_cuda_mb": peak_mb,
            "check": f"finite={finite}",
        }
    )
    return records


def bench_sweep_api(device: torch.device, *, quick: bool) -> list[dict]:
    dtype = torch.complex64
    order = (1, 1) if quick else (2, 2)
    grid = 40 if quick else 64
    repeats = 2 if quick else 4
    warmup = 1
    eps = _patterned_eps(device, grid=grid)
    material = torcwa.v2.MaterialGrid(eps, (300.0, 300.0))
    freqs = torch.tensor([1 / 450, 1 / 500, 1 / 550], dtype=torch.float32, device=device)

    def sweep_case():
        torcwa.rcwa.clear_material_cache()
        config = torcwa.v2.RCWAConfig(
            freq=freqs[0],
            order=order,
            lattice=(300.0, 300.0),
            options=torcwa.v2.SolverOptions(dtype=dtype, device=device),
        )
        solver = torcwa.v2.RCWASolver(config).add_layer(80.0, eps=material)
        return solver.solve_sweep(freqs, requests=[{"name": "txx", "orders": [0, 0], "polarization": "xx"}])["txx"]

    sample = sweep_case()
    finite = bool(torch.isfinite(torch.real(sample)).all() and torch.isfinite(torch.imag(sample)).all())
    median_ms, min_ms, peak_mb = _measure(sweep_case, device=device, repeats=repeats, warmup=warmup)
    return [
        {
            "workload": "v2_sweep",
            "case": "fixed_geometry_three_freqs",
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "size": f"order={list(order)} grid={grid}x{grid}",
            "median_ms": median_ms,
            "min_ms": min_ms,
            "peak_cuda_mb": peak_mb,
            "check": f"finite={finite}",
        }
    ]


def bench_rcwa_stress(device: torch.device) -> list[dict]:
    dtype = torch.complex64
    cases = [(10, 160, 3)]
    if device.type == "cuda":
        cases.extend([(15, 224, 2), (20, 300, 2)])

    records = []
    for order, grid, repeats in cases:
        eps = _patterned_eps(device, grid=grid)

        def solve_case():
            sim = torcwa.rcwa(freq=1 / 500, order=[order, order], L=[300.0, 300.0], dtype=dtype, device=device)
            sim.set_incident_angle(0.0, 0.0)
            sim.add_layer(thickness=80.0, eps=eps)
            sim.solve_global_smatrix()
            return sim.S_parameters([0, 0], polarization="xx")

        sample = solve_case()
        finite = bool(torch.isfinite(torch.real(sample)).all() and torch.isfinite(torch.imag(sample)).all())
        median_ms, min_ms, peak_mb = _measure(solve_case, device=device, repeats=repeats, warmup=1)
        records.append(
            {
                "workload": "rcwa_stress",
                "case": "patterned_single_layer",
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "size": f"order=[{order}, {order}] grid={grid}x{grid}",
                "median_ms": median_ms,
                "min_ms": min_ms,
                "peak_cuda_mb": peak_mb,
                "check": f"finite={finite}",
            }
        )
    return records


def _devices(selection: str) -> list[torch.device]:
    if selection == "cpu":
        return [torch.device("cpu")]
    if selection == "cuda":
        return [torch.device("cuda")]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _print_markdown(records: list[dict]) -> None:
    print(f"torch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"cuda: {torch.version.cuda} device: {torch.cuda.get_device_name(0)}")
    print()
    print("| workload | case | device | dtype | size | median ms | min ms | peak CUDA MB | check |")
    print("|---|---|---:|---|---|---:|---:|---:|---|")
    for record in records:
        peak = "" if record["peak_cuda_mb"] is None else f"{record['peak_cuda_mb']:.2f}"
        print(
            f"| {record['workload']} | {record['case']} | {record['device']} | {record['dtype']} | "
            f"{record['size']} | {record['median_ms']:.3f} | {record['min_ms']:.3f} | {peak} | {record['check']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TORCWA v2 numerical-kernel and smoke workload benchmarks")
    parser.add_argument("--quick", action="store_true", help="Use smaller matrices and fewer repeats")
    parser.add_argument("--devices", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stress", action="store_true", help="Also run larger order/grid RCWA smoke benchmarks")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a markdown table")
    args = parser.parse_args()

    records: list[dict] = []
    for device in _devices(args.devices):
        if device.type == "cuda" and not torch.cuda.is_available():
            continue
        records.extend(bench_solve_vs_inverse(device, quick=args.quick))
        records.extend(bench_diag_broadcast(device, quick=args.quick))
        records.extend(bench_rcwa(device, quick=args.quick))
        records.extend(bench_sweep_api(device, quick=args.quick))
        if args.stress:
            records.extend(bench_rcwa_stress(device))

    if args.json:
        print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "records": records}, indent=2))
    else:
        _print_markdown(records)


if __name__ == "__main__":
    main()

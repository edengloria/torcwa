from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch

import torcwa
from torcwa.v2.fourier import material_convolution_apply, material_convolution_dense


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _measure(fn: Callable[[], torch.Tensor], *, device: torch.device, repeats: int, warmup: int) -> tuple[float, float, float | None]:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000)
    return statistics.median(times_ms), min(times_ms), _peak_mb(device)


def _complex_randn(shape, *, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.complex(
        torch.randn(shape, dtype=real_dtype, device=device, generator=generator),
        torch.randn(shape, dtype=real_dtype, device=device, generator=generator),
    ).to(dtype)


def _material(kind: str, *, grid: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    if kind == "random":
        return 1.0 + 0.3 * _complex_randn((grid, grid), dtype=dtype, device=device, seed=101)

    geo = torcwa.geometry(Lx=300.0, Ly=300.0, nx=grid, ny=grid, edge_sharpness=35.0, dtype=real_dtype, device=device)
    if kind == "rectangle":
        mask = geo.rectangle(Wx=120.0, Wy=80.0, Cx=150.0, Cy=150.0)
    elif kind == "circle":
        mask = geo.circle(R=70.0, Cx=150.0, Cy=150.0)
    else:
        raise ValueError(f"unknown material kind: {kind}")
    return (mask * 2.25 + (1.0 - mask)).to(dtype)


def _gradient_finite(material: torch.Tensor, order: tuple[int, int], vector: torch.Tensor) -> bool:
    if material.device.type == "cuda":
        return True
    probe = material.detach().clone().requires_grad_(True)
    output = material_convolution_apply(probe, order, vector)
    loss = torch.sum(torch.abs(output) ** 2)
    loss.backward()
    return probe.grad is not None and bool(torch.isfinite(torch.real(probe.grad)).all())


def benchmark(device: torch.device, *, quick: bool) -> list[dict]:
    order = (3, 3) if quick else (5, 5)
    grid = 32 if quick else 64
    repeats = 3 if quick else 8
    warmup = 1 if quick else 2
    mode_count = (2 * order[0] + 1) * (2 * order[1] + 1)
    records = []

    for dtype in (torch.complex64, torch.complex128):
        for kind in ("random", "rectangle", "circle"):
            material = _material(kind, grid=grid, dtype=dtype, device=device)
            vector = _complex_randn((mode_count, 3), dtype=dtype, device=device, seed=202)

            dense = material_convolution_dense(material, order)
            dense_result = dense @ vector
            operator_result = material_convolution_apply(material, order, vector)
            diff = dense_result - operator_result
            max_abs = torch.max(torch.abs(diff))
            relative_l2 = torch.linalg.norm(diff) / torch.clamp(torch.linalg.norm(dense_result), min=torch.finfo(torch.float32).tiny)

            dense_ms, dense_min_ms, dense_peak = _measure(lambda: material_convolution_dense(material, order) @ vector, device=device, repeats=repeats, warmup=warmup)
            op_ms, op_min_ms, op_peak = _measure(lambda: material_convolution_apply(material, order, vector), device=device, repeats=repeats, warmup=warmup)

            records.append(
                {
                    "case": kind,
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                    "size": f"order={list(order)} grid={grid}x{grid} rhs=3",
                    "dense_median_ms": dense_ms,
                    "operator_median_ms": op_ms,
                    "dense_min_ms": dense_min_ms,
                    "operator_min_ms": op_min_ms,
                    "dense_peak_cuda_mb": dense_peak,
                    "operator_peak_cuda_mb": op_peak,
                    "max_abs_diff": float(max_abs),
                    "relative_l2_diff": float(relative_l2),
                    "grad_finite": _gradient_finite(material, order, vector),
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
    print("| case | device | dtype | size | dense ms | operator ms | dense peak MB | operator peak MB | max abs | rel l2 | grad |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for record in records:
        dense_peak = "" if record["dense_peak_cuda_mb"] is None else f"{record['dense_peak_cuda_mb']:.2f}"
        op_peak = "" if record["operator_peak_cuda_mb"] is None else f"{record['operator_peak_cuda_mb']:.2f}"
        print(
            f"| {record['case']} | {record['device']} | {record['dtype']} | {record['size']} | "
            f"{record['dense_median_ms']:.3f} | {record['operator_median_ms']:.3f} | "
            f"{dense_peak} | {op_peak} | {record['max_abs_diff']:.3e} | {record['relative_l2_diff']:.3e} | "
            f"{record['grad_finite']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dense Fourier convolution against an operator prototype")
    parser.add_argument("--quick", action="store_true", help="Use smaller order/grid and fewer repeats")
    parser.add_argument("--devices", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    records = []
    for device in _devices(args.devices):
        if device.type == "cuda" and not torch.cuda.is_available():
            continue
        records.extend(benchmark(device, quick=args.quick))

    if args.json:
        print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "records": records}, indent=2))
    else:
        _print_markdown(records)


if __name__ == "__main__":
    main()

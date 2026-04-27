from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load(path):
    return json.loads(Path(path).read_text())


def _key(record):
    return (record["workload"], record["case"], record["device"], record["dtype"], record["size"])


def _array(record):
    result = record["result"]
    return np.asarray(result["real"], dtype=np.float64) + 1j * np.asarray(result["imag"], dtype=np.float64)


def _diff(current, reference):
    if current is None or reference is None:
        return math.nan, math.nan
    a = _array(current)
    b = _array(reference)
    if a.shape != b.shape:
        return math.nan, math.nan
    delta = a - b
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    denom = np.linalg.norm(b)
    rel_l2 = float(np.linalg.norm(delta) / denom) if denom > 0 else max_abs
    return max_abs, rel_l2


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return f"{numerator / denominator:.2f}x"


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _index(payload):
    return {_key(record): record for record in payload["records"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize original/current/optimized TORCWA benchmark JSON")
    parser.add_argument("--original", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    original = _load(args.original)
    current = _load(args.current)
    optimized = _load(args.optimized)
    original_idx = _index(original)
    current_idx = _index(current)
    optimized_idx = _index(optimized)
    keys = sorted(set(original_idx) | set(current_idx) | set(optimized_idx))

    lines = [
        "# TORCWA v3 Performance And Memory Report",
        "",
        "## Environment",
        "",
        f"- original: `{original.get('torcwa_version')}` from `51c0d24`",
        f"- current-modern-api: `{current.get('torcwa_version')}` snapshot before optimization",
        f"- optimized: `{optimized.get('torcwa_version')}` working tree after optimization",
        f"- torch: `{optimized.get('torch')}`",
        f"- cuda: `{optimized.get('cuda')}` / `{optimized.get('cuda_device')}`",
        "",
        "## Results",
        "",
        "| workload | case | device | size | original ms | current ms | optimized ms | opt speedup vs original | opt speedup vs current | original MB | current MB | optimized MB | max abs vs current | rel L2 vs current |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key in keys:
        opt = optimized_idx.get(key)
        cur = current_idx.get(key)
        orig = original_idx.get(key)
        ref = opt or cur or orig
        max_abs, rel_l2 = _diff(opt, cur)
        opt_ms = opt.get("median_ms") if opt else None
        cur_ms = cur.get("median_ms") if cur else None
        orig_ms = orig.get("median_ms") if orig else None
        opt_mb = opt.get("peak_cuda_mb") if opt else None
        cur_mb = cur.get("peak_cuda_mb") if cur else None
        orig_mb = orig.get("peak_cuda_mb") if orig else None
        lines.append(
            "| "
            + " | ".join(
                [
                    ref["workload"],
                    ref["case"],
                    ref["device"],
                    ref["size"],
                    _fmt(orig_ms),
                    _fmt(cur_ms),
                    _fmt(opt_ms),
                    _ratio(orig_ms, opt_ms),
                    _ratio(cur_ms, opt_ms),
                    _fmt(orig_mb, 2),
                    _fmt(cur_mb, 2),
                    _fmt(opt_mb, 2),
                    _fmt(max_abs, 3),
                    _fmt(rel_l2, 3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Missing original rows are modern API workloads that do not exist in TORCWA 0.1.4.2.",
            "- `max abs` and `rel L2` compare optimized against the current-modern-api snapshot.",
            "- Correctness gates remain the pytest analytical/S4 fixtures; this report is a performance and memory summary.",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

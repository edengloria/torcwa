#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "references" / "s4"


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "s4_uniform_slab",
        "kind": "homogeneous",
        "fixture": "s4_uniform_slab.npz",
        "freq": 1 / 500,
        "lattice": (300.0, 300.0),
        "num_basis": 25,
        "torcwa_order": (2, 2),
        "incidence_deg": (0.0, 0.0),
        "layers": [
            {"name": "front", "thickness": 0.0, "material": "air"},
            {"name": "slab", "thickness": 120.0, "material": "slab"},
            {"name": "back", "thickness": 0.0, "material": "air"},
        ],
        "materials": {"air": 1.0, "slab": 2.25},
        "polarizations": ("s", "p"),
    },
    {
        "name": "s4_oblique_interface",
        "kind": "homogeneous",
        "fixture": "s4_oblique_interface.npz",
        "freq": 1 / 500,
        "lattice": (300.0, 300.0),
        "num_basis": 25,
        "torcwa_order": (2, 2),
        "incidence_deg": (25.0, 0.0),
        "layers": [
            {"name": "front", "thickness": 0.0, "material": "air"},
            {"name": "back", "thickness": 0.0, "material": "glass"},
        ],
        "materials": {"air": 1.0, "glass": 2.25},
        "polarizations": ("s", "p"),
    },
    {
        "name": "s4_1d_binary_grating",
        "kind": "patterned",
        "fixture": "s4_1d_binary_grating.npz",
        "freq": 1 / 500,
        "lattice": (300.0, 300.0),
        "num_basis": 49,
        "torcwa_order": (3, 3),
        "incidence_deg": (0.0, 0.0),
        "layers": [
            {"name": "front", "thickness": 0.0, "material": "air"},
            {"name": "grating", "thickness": 80.0, "material": "air"},
            {"name": "back", "thickness": 0.0, "material": "air"},
        ],
        "materials": {"air": 1.0, "bar": 2.25},
        "rectangles": [{"layer": "grating", "material": "bar", "center": (0.0, 0.0), "angle": 0.0, "halfwidths": (60.0, 150.0)}],
        "polarizations": ("s", "p"),
    },
    {
        "name": "s4_2d_rect_metasurface",
        "kind": "patterned",
        "fixture": "s4_2d_rect_metasurface.npz",
        "freq": 1 / 500,
        "lattice": (300.0, 300.0),
        "num_basis": 49,
        "torcwa_order": (3, 3),
        "incidence_deg": (0.0, 0.0),
        "layers": [
            {"name": "front", "thickness": 0.0, "material": "air"},
            {"name": "pillar", "thickness": 80.0, "material": "air"},
            {"name": "back", "thickness": 0.0, "material": "air"},
        ],
        "materials": {"air": 1.0, "pillar": 2.25},
        "rectangles": [{"layer": "pillar", "material": "pillar", "center": (0.0, 0.0), "angle": 0.0, "halfwidths": (60.0, 45.0)}],
        "polarizations": ("s", "p"),
    },
    {
        "name": "s4_lossy_grating",
        "kind": "patterned",
        "fixture": "s4_lossy_grating.npz",
        "freq": 1 / 500,
        "lattice": (300.0, 300.0),
        "num_basis": 49,
        "torcwa_order": (3, 3),
        "incidence_deg": (0.0, 0.0),
        "layers": [
            {"name": "front", "thickness": 0.0, "material": "air"},
            {"name": "lossy", "thickness": 70.0, "material": "air"},
            {"name": "back", "thickness": 0.0, "material": "air"},
        ],
        "materials": {"air": 1.0, "lossy": 2.25 + 0.08j},
        "rectangles": [{"layer": "lossy", "material": "lossy", "center": (0.0, 0.0), "angle": 0.0, "halfwidths": (55.0, 45.0)}],
        "polarizations": ("s", "p"),
    },
    {
        "name": "s4_multilayer_stack",
        "kind": "patterned",
        "fixture": "s4_multilayer_stack.npz",
        "freq": 1 / 500,
        "lattice": (300.0, 300.0),
        "num_basis": 49,
        "torcwa_order": (3, 3),
        "incidence_deg": (0.0, 0.0),
        "layers": [
            {"name": "front", "thickness": 0.0, "material": "air"},
            {"name": "pattern_a", "thickness": 50.0, "material": "air"},
            {"name": "spacer", "thickness": 40.0, "material": "spacer"},
            {"name": "pattern_b", "thickness": 50.0, "copy": "pattern_a"},
            {"name": "back", "thickness": 0.0, "material": "air"},
        ],
        "materials": {"air": 1.0, "spacer": 1.44, "bar": 2.25},
        "rectangles": [{"layer": "pattern_a", "material": "bar", "center": (0.0, 0.0), "angle": 0.0, "halfwidths": (55.0, 45.0)}],
        "polarizations": ("s", "p"),
    },
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _import_s4():
    try:
        module = importlib.import_module("S4")
    except Exception as exc:
        raise SystemExit(
            "Official Stanford S4 Python extension is not importable.\n"
            "Build it from https://github.com/victorliu/S4 with `make S4_pyext` "
            "and add the generated build/lib.* directory to PYTHONPATH.\n"
            "The PyPI package named `S4` is not the Stanford RCWA solver."
        ) from exc
    if not (hasattr(module, "NewSimulation") or hasattr(module, "New")):
        raise SystemExit("Imported `S4`, but it does not expose NewSimulation/New. This is not the official Stanford S4 extension.")
    return module


def _new_sim(s4_module, spec: dict[str, Any]):
    lattice = ((spec["lattice"][0], 0.0), (0.0, spec["lattice"][1]))
    if hasattr(s4_module, "NewSimulation"):
        return s4_module.NewSimulation(Lattice=lattice, NumBasis=spec["num_basis"])
    return s4_module.New(Lattice=lattice, NumBasis=spec["num_basis"])


def _complex_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)


def _setup_case(s4_module, spec: dict[str, Any]):
    sim = _new_sim(s4_module, spec)
    sim.SetOptions(Verbosity=0, LatticeTruncation="Parallelogramic", DiscretizedEpsilon=False)
    for name, epsilon in spec["materials"].items():
        sim.SetMaterial(Name=name, Epsilon=epsilon)
    for layer in spec["layers"]:
        if "copy" in layer:
            sim.AddLayerCopy(Name=layer["name"], Thickness=layer["thickness"], Layer=layer["copy"])
        else:
            sim.AddLayer(Name=layer["name"], Thickness=layer["thickness"], Material=layer["material"])
    for rect in spec.get("rectangles", []):
        sim.SetRegionRectangle(
            Layer=rect["layer"],
            Material=rect["material"],
            Center=rect["center"],
            Angle=rect["angle"],
            Halfwidths=rect["halfwidths"],
        )
    sim.SetFrequency(spec["freq"])
    return sim


def _run_polarization(sim, spec: dict[str, Any], polarization: str) -> dict[str, np.ndarray | float]:
    s_amp = 1.0 if polarization == "s" else 0.0
    p_amp = 1.0 if polarization == "p" else 0.0
    sim.SetExcitationPlanewave(IncidenceAngles=spec["incidence_deg"], sAmplitude=s_amp, pAmplitude=p_amp, Order=0)
    front_layer = spec["layers"][0]["name"]
    back_layer = spec["layers"][-1]["name"]
    front_forw, front_back = sim.GetPowerFlux(Layer=front_layer, zOffset=0)
    back_forw, back_back = sim.GetPowerFlux(Layer=back_layer, zOffset=0)
    by_order_front = _complex_array(sim.GetPowerFluxByOrder(Layer=front_layer, zOffset=0))
    by_order_back = _complex_array(sim.GetPowerFluxByOrder(Layer=back_layer, zOffset=0))
    amp_front_forw, amp_front_back = sim.GetAmplitudes(Layer=front_layer, zOffset=0)
    amp_back_forw, amp_back_back = sim.GetAmplitudes(Layer=back_layer, zOffset=0)

    incident = np.real(front_forw)
    reflected = -np.real(front_back) / incident
    transmitted = np.real(back_forw) / incident
    return {
        f"{polarization}_front_power": np.asarray([front_forw, front_back], dtype=np.complex128),
        f"{polarization}_back_power": np.asarray([back_forw, back_back], dtype=np.complex128),
        f"{polarization}_front_power_by_order": by_order_front,
        f"{polarization}_back_power_by_order": by_order_back,
        f"{polarization}_front_amplitudes_forward": _complex_array(amp_front_forw),
        f"{polarization}_front_amplitudes_backward": _complex_array(amp_front_back),
        f"{polarization}_back_amplitudes_forward": _complex_array(amp_back_forw),
        f"{polarization}_back_amplitudes_backward": _complex_array(amp_back_back),
        f"{polarization}_R_total": float(reflected),
        f"{polarization}_T_total": float(transmitted),
        f"{polarization}_A_total": float(1.0 - reflected - transmitted),
    }


def _generate_case(s4_module, spec: dict[str, Any], out_dir: Path, overwrite: bool) -> dict[str, Any]:
    output = out_dir / spec["fixture"]
    if output.exists() and not overwrite:
        raise SystemExit(f"{output} exists. Pass --overwrite to replace it.")
    sim = _setup_case(s4_module, spec)
    arrays: dict[str, Any] = {
        "basis": np.asarray(sim.GetBasisSet(), dtype=np.int64),
        "freq": np.asarray(spec["freq"], dtype=np.float64),
        "lattice": np.asarray(spec["lattice"], dtype=np.float64),
        "torcwa_order": np.asarray(spec["torcwa_order"], dtype=np.int64),
        "incidence_deg": np.asarray(spec["incidence_deg"], dtype=np.float64),
    }
    for polarization in spec["polarizations"]:
        arrays.update(_run_polarization(sim, spec, polarization))
    np.savez_compressed(output, **arrays)
    return {"name": spec["name"], "fixture": spec["fixture"], "kind": spec["kind"], "generated": True}


def _write_manifest(out_dir: Path, generated_cases: list[dict[str, Any]]) -> None:
    manifest = {
        "schema_version": 1,
        "source": "Stanford S4",
        "source_url": "https://web.stanford.edu/group/fan/S4/",
        "s4_build": {
            "source_commit": os.environ.get("S4_SOURCE_COMMIT", "unknown"),
            "notes": os.environ.get("S4_BUILD_NOTES", "official Stanford S4 Python extension"),
        },
        "generated": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torcwa_commit": _git_commit(),
        "s4_options": {"LatticeTruncation": "Parallelogramic", "DiscretizedEpsilon": False, "Verbosity": 0},
        "tolerances": {
            "homogeneous": {"rtol": 1e-8, "atol": 1e-10},
            "patterned": {"power_rtol": 2e-2, "power_atol": 2e-3, "amplitude_rtol": 5e-2, "atol": 5e-5},
        },
        "cases": generated_cases,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TORCWA external-reference fixtures with the official Stanford S4 Python extension.")
    parser.add_argument("--output-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--case", choices=[case["name"] for case in CASES], action="append", help="Generate only selected case(s)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing fixture files and manifest")
    args = parser.parse_args()

    s4_module = _import_s4()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [case for case in CASES if args.case is None or case["name"] in args.case]
    generated = [_generate_case(s4_module, case, args.output_dir, args.overwrite) for case in selected]
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_path} exists. Pass --overwrite to replace it.")
    _write_manifest(args.output_dir, generated)
    print(f"Generated {len(generated)} S4 fixture(s) in {args.output_dir}")


if __name__ == "__main__":
    main()

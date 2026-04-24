import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torcwa


ROOT = Path(__file__).resolve().parents[1]
S4_DIR = ROOT / "references" / "s4"
MANIFEST = S4_DIR / "manifest.json"


def _load_manifest():
    return json.loads(MANIFEST.read_text())


def _orders(order):
    ox, oy = int(order[0]), int(order[1])
    return [[i, j] for i in range(-ox, ox + 1) for j in range(-oy, oy + 1)]


def _pattern_eps(case_name, lattice, *, device, dtype):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    geo = torcwa.geometry(Lx=lattice[0], Ly=lattice[1], nx=96, ny=96, edge_sharpness=40.0, dtype=real_dtype, device=device)
    if case_name == "s4_1d_binary_grating":
        mask = geo.rectangle(Wx=120.0, Wy=lattice[1], Cx=lattice[0] / 2, Cy=lattice[1] / 2)
        return mask * 2.25 + (1.0 - mask)
    if case_name == "s4_2d_rect_metasurface":
        mask = geo.rectangle(Wx=120.0, Wy=90.0, Cx=lattice[0] / 2, Cy=lattice[1] / 2)
        return mask * 2.25 + (1.0 - mask)
    if case_name == "s4_lossy_grating":
        mask = geo.rectangle(Wx=110.0, Wy=90.0, Cx=lattice[0] / 2, Cy=lattice[1] / 2)
        return mask * (2.25 + 0.08j) + (1.0 - mask)
    if case_name == "s4_multilayer_stack":
        mask = geo.rectangle(Wx=110.0, Wy=90.0, Cx=lattice[0] / 2, Cy=lattice[1] / 2)
        return mask * 2.25 + (1.0 - mask)
    raise ValueError(f"no TORCWA material builder for {case_name}")


def _build_torcwa(case_name, fixture):
    dtype = torch.complex128
    device = torch.device("cpu")
    freq = float(fixture["freq"])
    lattice = tuple(float(v) for v in fixture["lattice"])
    order = [int(v) for v in fixture["torcwa_order"]]
    incidence = fixture["incidence_deg"] * np.pi / 180.0
    sim = torcwa.rcwa(freq=freq, order=order, L=list(lattice), dtype=dtype, device=device)

    if case_name == "s4_uniform_slab":
        sim.set_incident_angle(float(incidence[0]), float(incidence[1]))
        sim.add_layer(thickness=120.0, eps=2.25, mu=1.0)
    elif case_name == "s4_oblique_interface":
        sim.add_output_layer(eps=2.25, mu=1.0)
        sim.set_incident_angle(float(incidence[0]), float(incidence[1]))
    elif case_name == "s4_multilayer_stack":
        sim.set_incident_angle(float(incidence[0]), float(incidence[1]))
        eps = _pattern_eps(case_name, lattice, device=device, dtype=dtype)
        sim.add_layer(thickness=50.0, eps=eps)
        sim.add_layer(thickness=40.0, eps=1.44)
        sim.add_layer(thickness=50.0, eps=eps)
    else:
        sim.set_incident_angle(float(incidence[0]), float(incidence[1]))
        sim.add_layer(thickness=70.0 if case_name == "s4_lossy_grating" else 80.0, eps=_pattern_eps(case_name, lattice, device=device, dtype=dtype))

    sim.solve_global_smatrix()
    return sim, _orders(order)


def _power_totals(sim, orders, polarization):
    transmission = sim.S_parameters(orders, polarization=polarization, port="transmission", power_norm=True)
    reflection = sim.S_parameters(orders, polarization=polarization, port="reflection", power_norm=True)
    return float(torch.sum(torch.abs(transmission) ** 2)), float(torch.sum(torch.abs(reflection) ** 2))


def _fixture_or_skip(case):
    path = S4_DIR / case["fixture"]
    if not path.exists():
        pytest.skip(f"S4 fixture {path.name} is pending; run tools/generate_s4_fixtures.py with official Stanford S4")
    return np.load(path, allow_pickle=False)


def test_s4_manifest_declares_core_gate_cases():
    manifest = _load_manifest()
    names = {case["name"] for case in manifest["cases"]}
    assert {
        "s4_uniform_slab",
        "s4_oblique_interface",
        "s4_1d_binary_grating",
        "s4_2d_rect_metasurface",
        "s4_lossy_grating",
        "s4_multilayer_stack",
    } <= names
    assert manifest["source"] == "Stanford S4"


@pytest.mark.parametrize("case", _load_manifest()["cases"], ids=lambda case: case["name"])
def test_committed_s4_fixture_matches_torcwa(case):
    fixture = _fixture_or_skip(case)
    sim, orders = _build_torcwa(case["name"], fixture)
    manifest = _load_manifest()
    tol = manifest["tolerances"][case["kind"]]

    pol_map = {"s": "ss", "p": "pp"}
    for s4_pol, torcwa_pol in pol_map.items():
        t_total, r_total = _power_totals(sim, orders, torcwa_pol)
        if case["kind"] == "homogeneous":
            assert np.allclose(t_total, float(fixture[f"{s4_pol}_T_total"]), rtol=tol["rtol"], atol=tol["atol"])
            assert np.allclose(r_total, float(fixture[f"{s4_pol}_R_total"]), rtol=tol["rtol"], atol=tol["atol"])
        else:
            power_atol = tol.get("power_atol", tol["atol"])
            assert np.allclose(t_total, float(fixture[f"{s4_pol}_T_total"]), rtol=tol["power_rtol"], atol=power_atol)
            assert np.allclose(r_total, float(fixture[f"{s4_pol}_R_total"]), rtol=tol["power_rtol"], atol=power_atol)


def _has_official_s4():
    try:
        module = importlib.import_module("S4")
    except Exception:
        return False
    return hasattr(module, "NewSimulation") or hasattr(module, "New")


@pytest.mark.s4_live
def test_live_s4_generator_smoke(tmp_path):
    if not _has_official_s4():
        pytest.skip("official Stanford S4 Python extension is not importable")
    command = [
        sys.executable,
        str(ROOT / "tools" / "generate_s4_fixtures.py"),
        "--output-dir",
        str(tmp_path),
        "--case",
        "s4_uniform_slab",
        "--overwrite",
    ]
    subprocess.run(command, check=True)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "s4_uniform_slab.npz").exists()

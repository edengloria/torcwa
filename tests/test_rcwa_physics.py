import pytest

torch = pytest.importorskip("torch")

import torcwa
from torcwa.v2 import MaterialGrid, RCWAConfig, RCWASolver, SolverOptions
from torcwa.v2.physics import fresnel_amplitudes


def _cpu_solver(*, order=(0, 0), dtype=torch.complex128):
    sim = torcwa.rcwa(freq=1 / 500, order=list(order), L=[300.0, 300.0], dtype=dtype, device=torch.device("cpu"))
    sim.set_incident_angle(0.0, 0.0)
    return sim


def test_empty_stack_is_identity_for_zero_order():
    sim = _cpu_solver()
    sim.solve_global_smatrix()

    txx = sim.S_parameters([0, 0], polarization="xx", port="transmission", power_norm=False)
    rxx = sim.S_parameters([0, 0], polarization="xx", port="reflection", power_norm=False)

    assert torch.allclose(txx, torch.ones_like(txx), atol=1e-12, rtol=1e-12)
    assert torch.allclose(rxx, torch.zeros_like(rxx), atol=1e-12, rtol=1e-12)


def test_single_interface_matches_fresnel_normal_incidence_and_conserves_power():
    n1 = torch.tensor(1.0, dtype=torch.complex128)
    n2 = torch.tensor(1.5, dtype=torch.complex128)
    sim = _cpu_solver()
    sim.add_output_layer(eps=n2**2, mu=1.0)
    sim.set_incident_angle(0.0, 0.0)
    sim.solve_global_smatrix()

    fresnel = fresnel_amplitudes(n1, n2, torch.tensor(0.0, dtype=torch.complex128))
    t_raw = sim.S_parameters([0, 0], polarization="ss", port="transmission", power_norm=False)
    r_raw = sim.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=False)
    t_power = sim.S_parameters([0, 0], polarization="ss", port="transmission", power_norm=True)
    r_power = sim.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=True)

    assert torch.allclose(t_raw, fresnel["t_te"].reshape_as(t_raw), atol=1e-12, rtol=1e-12)
    assert torch.allclose(r_raw, fresnel["r_te"].reshape_as(r_raw), atol=1e-12, rtol=1e-12)
    assert torch.allclose(torch.abs(t_power) ** 2 + torch.abs(r_power) ** 2, torch.ones_like(t_power.real), atol=1e-12, rtol=1e-12)


def test_evanescent_typo_alias_matches_canonical_parameter():
    sim = _cpu_solver()
    sim.add_output_layer(eps=2.25, mu=1.0)
    sim.set_incident_angle(0.0, 0.0)
    sim.solve_global_smatrix()

    canonical = sim.S_parameters([0, 0], polarization="xx", evanescent=1e-4)
    with pytest.warns(DeprecationWarning):
        alias = sim.S_parameters([0, 0], polarization="xx", evanscent=1e-4)

    assert torch.allclose(alias, canonical)


def test_v2_facade_matches_legacy_for_patterned_layer_and_fields():
    dtype = torch.complex64
    device = torch.device("cpu")
    lattice = [300.0, 300.0]
    geo = torcwa.geometry(Lx=lattice[0], Ly=lattice[1], nx=32, ny=32, edge_sharpness=30.0, dtype=torch.float32, device=device)
    mask = geo.rectangle(Wx=120.0, Wy=90.0, Cx=150.0, Cy=150.0)
    eps = mask * 2.25 + (1.0 - mask)

    legacy = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=lattice, dtype=dtype, device=device)
    legacy.set_incident_angle(0.0, 0.0)
    legacy.add_layer(thickness=80.0, eps=eps)
    legacy.solve_global_smatrix()

    config = RCWAConfig(freq=1 / 500, order=(1, 1), lattice=tuple(lattice), options=SolverOptions(dtype=dtype, device=device))
    v2 = RCWASolver(config).add_layer(80.0, eps=eps).solve()

    legacy_txx = legacy.S_parameters([0, 0], polarization="xx")
    v2_txx = v2.s_parameter([0, 0], polarization="xx")
    assert torch.allclose(v2_txx, legacy_txx)

    v2.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="ps")
    x_axis = torch.linspace(0.0, lattice[0], 8)
    z_axis = torch.linspace(-10.0, 90.0, 5)
    electric, magnetic = v2.field_plane(plane="xz", axis0=x_axis, axis1=z_axis, offset=lattice[1] / 2)
    electric_chunked, magnetic_chunked = v2.field_plane(plane="xz", axis0=x_axis, axis1=z_axis, offset=lattice[1] / 2, chunk_size=2)

    assert [tuple(component.shape) for component in electric] == [(8, 5), (8, 5), (8, 5)]
    assert [tuple(component.shape) for component in magnetic] == [(8, 5), (8, 5), (8, 5)]
    assert torch.allclose(electric_chunked[0], electric[0])
    assert torch.allclose(magnetic_chunked[0], magnetic[0])
    assert torch.isfinite(torch.real(electric[0])).all()

    y_axis = torch.linspace(0.0, lattice[1], 7)
    yz_electric, yz_magnetic = v2.field_plane(plane="yz", axis0=y_axis, axis1=z_axis, offset=lattice[0] / 2, chunk_size=2)
    assert [tuple(component.shape) for component in yz_electric] == [(7, 5), (7, 5), (7, 5)]
    assert [tuple(component.shape) for component in yz_magnetic] == [(7, 5), (7, 5), (7, 5)]
    assert torch.isfinite(torch.real(yz_electric[0])).all()

    xy_electric, xy_magnetic = v2.field_plane(plane="xy", axis0=x_axis, axis1=y_axis, layer_num=0, offset=20.0)
    xy_electric_chunked, xy_magnetic_chunked = v2.field_plane(plane="xy", axis0=x_axis, axis1=y_axis, layer_num=0, offset=20.0, chunk_size=2)
    assert [tuple(component.shape) for component in xy_electric] == [(8, 7), (8, 7), (8, 7)]
    assert [tuple(component.shape) for component in xy_magnetic] == [(8, 7), (8, 7), (8, 7)]
    assert torch.allclose(xy_electric_chunked[0], xy_electric[0])
    assert torch.allclose(xy_magnetic_chunked[0], xy_magnetic[0])


def test_small_geometry_gradient_is_finite():
    radius = torch.tensor(65.0, dtype=torch.float64, requires_grad=True)
    geo = torcwa.geometry(Lx=300.0, Ly=300.0, nx=28, ny=28, edge_sharpness=20.0, dtype=torch.float64, device=torch.device("cpu"))
    mask = geo.circle(R=radius, Cx=150.0, Cy=150.0)
    eps = mask * 2.25 + (1.0 - mask)

    sim = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=[300.0, 300.0], dtype=torch.complex128, device=torch.device("cpu"))
    sim.set_incident_angle(0.0, 0.0)
    sim.add_layer(thickness=50.0, eps=eps)
    sim.solve_global_smatrix()
    transmission = torch.abs(sim.S_parameters([0, 0], polarization="xx")) ** 2
    transmission.backward()

    assert radius.grad is not None
    assert torch.isfinite(radius.grad)


def test_material_convolution_cache_reuses_nongrad_tensor_only():
    torcwa.rcwa.clear_material_cache()
    device = torch.device("cpu")
    geo = torcwa.geometry(Lx=300.0, Ly=300.0, nx=24, ny=24, edge_sharpness=20.0, dtype=torch.float32, device=device)
    mask = geo.rectangle(Wx=100.0, Wy=80.0, Cx=150.0, Cy=150.0)
    eps = mask * 2.25 + (1.0 - mask)

    sim1 = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=[300.0, 300.0], dtype=torch.complex64, device=device)
    sim1.set_incident_angle(0.0, 0.0)
    sim1.add_layer(thickness=40.0, eps=eps)
    assert len(torcwa.rcwa._material_conv_cache) == 1

    sim2 = torcwa.rcwa(freq=1 / 550, order=[1, 1], L=[300.0, 300.0], dtype=torch.complex64, device=device)
    sim2.set_incident_angle(0.0, 0.0)
    sim2.add_layer(thickness=40.0, eps=eps)
    assert len(torcwa.rcwa._material_conv_cache) == 1

    torcwa.rcwa.clear_material_cache()
    eps_grad = eps.detach().clone().requires_grad_(True)
    sim3 = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=[300.0, 300.0], dtype=torch.complex64, device=device)
    sim3.set_incident_angle(0.0, 0.0)
    sim3.add_layer(thickness=40.0, eps=eps_grad)
    assert len(torcwa.rcwa._material_conv_cache) == 0

    torcwa.rcwa.clear_material_cache()
    no_cache = MaterialGrid(eps, (300.0, 300.0), cache_key=("rect", 1), cache=False)
    config = RCWAConfig(freq=1 / 500, order=(1, 1), lattice=(300.0, 300.0), options=SolverOptions(dtype=torch.complex64, device=device))
    RCWASolver(config).add_layer(40.0, eps=no_cache).solve()
    assert len(torcwa.rcwa._material_conv_cache) == 0


def test_v2_memory_mode_reaches_legacy_solver_and_validates():
    device = torch.device("cpu")
    config = RCWAConfig(
        freq=1 / 500,
        order=(0, 0),
        lattice=(300.0, 300.0),
        options=SolverOptions(dtype=torch.complex128, device=device, memory_mode="memory"),
    )
    solver = RCWASolver(config).solve()
    assert solver.legacy_solver().memory_mode == "memory"

    with pytest.raises(ValueError):
        SolverOptions(memory_mode="large-workspace")


def test_v2_solve_sweep_matches_manual_fixed_geometry_loop():
    torcwa.rcwa.clear_material_cache()
    device = torch.device("cpu")
    dtype = torch.complex64
    lattice = (300.0, 300.0)
    geo = torcwa.geometry(Lx=lattice[0], Ly=lattice[1], nx=28, ny=28, edge_sharpness=30.0, dtype=torch.float32, device=device)
    mask = geo.rectangle(Wx=110.0, Wy=90.0, Cx=150.0, Cy=150.0)
    eps = MaterialGrid(mask * 2.25 + (1.0 - mask), lattice)
    freqs = torch.tensor([1 / 450, 1 / 500, 1 / 550], dtype=torch.float32, device=device)

    config = RCWAConfig(freq=freqs[0], order=(1, 1), lattice=lattice, options=SolverOptions(dtype=dtype, device=device))
    solver = RCWASolver(config).add_layer(70.0, eps=eps)
    sweep = solver.solve_sweep(
        freqs,
        requests=[
            {"name": "txx", "orders": [0, 0], "polarization": "xx"},
            {"name": "tyy", "orders": [0, 0], "polarization": "yy"},
        ],
    )

    manual_txx, manual_tyy = [], []
    for freq in freqs:
        manual = RCWASolver(RCWAConfig(freq=freq, order=(1, 1), lattice=lattice, options=SolverOptions(dtype=dtype, device=device)))
        manual.add_layer(70.0, eps=eps).solve()
        manual_txx.append(manual.s_parameter([0, 0], polarization="xx"))
        manual_tyy.append(manual.s_parameter([0, 0], polarization="yy"))

    assert torch.allclose(sweep["txx"], torch.stack(manual_txx, dim=0))
    assert torch.allclose(sweep["tyy"], torch.stack(manual_tyy, dim=0))
    assert len(torcwa.rcwa._material_conv_cache) == 1


def test_legacy_s_parameter_normalization_cache_is_equivalent():
    sim = _cpu_solver(order=(1, 1), dtype=torch.complex128)
    sim.add_layer(thickness=80.0, eps=2.25, mu=1.0)
    sim.solve_global_smatrix()
    orders = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)

    first = sim.S_parameters(orders, polarization="xx")
    cache_size = len(sim._s_parameter_cache)
    second = sim.S_parameters(orders, polarization="xx")

    assert cache_size > 0
    assert len(sim._s_parameter_cache) == cache_size
    assert torch.allclose(first, second, atol=1e-12, rtol=1e-12)


def test_memory_mode_homogeneous_structured_path_matches_balanced():
    device = torch.device("cpu")
    dtype = torch.complex128
    balanced = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=[300.0, 300.0], dtype=dtype, device=device)
    balanced.memory_mode = "balanced"
    balanced.set_incident_angle(0.0, 0.0)
    balanced.add_layer(thickness=80.0, eps=2.25, mu=1.0)
    balanced.solve_global_smatrix()
    balanced.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")

    memory = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=[300.0, 300.0], dtype=dtype, device=device)
    memory.memory_mode = "memory"
    memory.set_incident_angle(0.0, 0.0)
    memory.add_layer(thickness=80.0, eps=2.25, mu=1.0)
    memory.solve_global_smatrix()
    memory.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")

    assert memory.P[-1] is None
    assert torch.allclose(
        memory.S_parameters([0, 0], polarization="xx"),
        balanced.S_parameters([0, 0], polarization="xx"),
        atol=1e-10,
        rtol=1e-10,
    )

    x_axis = torch.linspace(0.0, 300.0, 6)
    z_axis = torch.linspace(-10.0, 90.0, 5)
    mem_electric, mem_magnetic = memory.field_xz(x_axis, z_axis, 150.0)
    bal_electric, bal_magnetic = balanced.field_xz(x_axis, z_axis, 150.0)
    assert torch.allclose(mem_electric[0], bal_electric[0], atol=1e-9, rtol=1e-9)
    assert torch.allclose(mem_magnetic[1], bal_magnetic[1], atol=1e-9, rtol=1e-9)


def test_memory_mode_patterned_field_streaming_matches_balanced():
    device = torch.device("cpu")
    dtype = torch.complex64
    lattice = [300.0, 300.0]
    geo = torcwa.geometry(Lx=lattice[0], Ly=lattice[1], nx=28, ny=28, edge_sharpness=30.0, dtype=torch.float32, device=device)
    mask = geo.rectangle(Wx=110.0, Wy=90.0, Cx=150.0, Cy=150.0)
    eps = mask * 2.25 + (1.0 - mask)

    def build(mode):
        sim = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=lattice, dtype=dtype, device=device)
        sim.memory_mode = mode
        sim.set_incident_angle(0.0, 0.0)
        sim.add_layer(thickness=70.0, eps=eps)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")
        return sim

    balanced = build("balanced")
    memory = build("memory")
    x_axis = torch.linspace(0.0, lattice[0], 6)
    y_axis = torch.linspace(0.0, lattice[1], 5)
    z_axis = torch.linspace(-10.0, 80.0, 4)

    mem_xz, _ = memory.field_xz(x_axis, z_axis, lattice[1] / 2)
    bal_xz, _ = balanced.field_xz(x_axis, z_axis, lattice[1] / 2)
    mem_xy, _ = memory.field_xy(0, x_axis, y_axis, z_prop=20.0)
    bal_xy, _ = balanced.field_xy(0, x_axis, y_axis, z_prop=20.0)

    assert torch.allclose(mem_xz[0], bal_xz[0], atol=5e-5, rtol=5e-4)
    assert torch.allclose(mem_xy[0], bal_xy[0], atol=5e-5, rtol=5e-4)

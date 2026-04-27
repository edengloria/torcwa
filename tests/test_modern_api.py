import pytest

torch = pytest.importorskip("torch")

import torcwa


def test_modern_api_matches_legacy_interface_and_power_balance():
    device = torch.device("cpu")
    dtype = torch.complex128

    legacy = torcwa.rcwa(freq=1 / 500, order=[0, 0], L=[300.0, 300.0], dtype=dtype, device=device)
    legacy.add_output_layer(eps=2.25, mu=1.0)
    legacy.set_incident_angle(0.0, 0.0)
    legacy.solve_global_smatrix()

    stack = torcwa.Stack(period=(300.0, 300.0), output_eps=2.25)
    result = torcwa.RCWA(wavelength=500.0, orders=(0, 0), dtype=dtype, device=device).solve(
        stack,
        torcwa.PlaneWave(polarization="s"),
    )

    assert torch.allclose(result.transmission(polarization="s", power=False), legacy.S_parameters([0, 0], polarization="ss", power_norm=False))
    assert torch.allclose(result.reflection(polarization="s", power=False), legacy.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=False))

    balance = result.power_balance(input_polarization="s")
    assert torch.allclose(balance["T"] + balance["R"] + balance["A"], torch.ones_like(balance["T"]), atol=1e-12, rtol=1e-12)
    assert balance["A"].abs().item() < 1e-12


def test_modern_api_patterned_layer_fields_match_legacy():
    device = torch.device("cpu")
    dtype = torch.complex64
    period = (300.0, 300.0)
    cell = torcwa.UnitCell(period=period, grid=(32, 32), edge_sharpness=30.0, dtype=torch.float32, device=device)
    mask = cell.rectangle(size=(120.0, 90.0), center=(150.0, 150.0))
    eps = torcwa.material.mix(1.0, 2.25, mask)

    legacy = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=list(period), dtype=dtype, device=device)
    legacy.set_incident_angle(0.0, 0.0)
    legacy.add_layer(thickness=80.0, eps=eps)
    legacy.solve_global_smatrix()
    legacy.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")

    stack = torcwa.Stack(period=period)
    stack.add_layer(thickness=80.0, eps=torcwa.MaterialGrid(eps, period))
    result = torcwa.RCWA(wavelength=500.0, orders=(1, 1), dtype=dtype, device=device).solve(
        stack,
        torcwa.PlaneWave(polarization="x"),
    )

    assert torch.allclose(result.transmission(polarization="x"), legacy.S_parameters([0, 0], polarization="xx"))

    x_axis = torch.linspace(0.0, period[0], 8)
    z_axis = torch.linspace(-10.0, 90.0, 5)
    modern_field = result.fields.plane("xz", x=x_axis, z=z_axis, y=period[1] / 2)
    legacy_electric, legacy_magnetic = legacy.field_xz(x_axis, z_axis, period[1] / 2)

    assert tuple(modern_field.Ex.shape) == (8, 5)
    assert torch.allclose(modern_field.Ex, legacy_electric[0])
    assert torch.allclose(modern_field.Hy, legacy_magnetic[1])


def test_modern_api_sweep_matches_manual_loop():
    device = torch.device("cpu")
    dtype = torch.complex64
    stack = torcwa.Stack(period=(300.0, 300.0))
    stack.add_layer(thickness=80.0, eps=2.25)
    solver = torcwa.RCWA(wavelength=500.0, orders=(0, 0), dtype=dtype, device=device)
    wavelengths = torch.tensor([450.0, 500.0, 550.0], dtype=torch.float32)

    sweep = solver.sweep(
        stack,
        source=torcwa.PlaneWave(polarization="x"),
        wavelength=wavelengths,
        outputs=[
            torcwa.Output.transmission(order=(0, 0), polarization="x", name="txx"),
            torcwa.Output.reflection(order=(0, 0), polarization="x", name="rxx"),
        ],
    )

    manual_txx, manual_rxx = [], []
    for wavelength in wavelengths:
        result = torcwa.RCWA(wavelength=wavelength, orders=(0, 0), dtype=dtype, device=device).solve(
            stack,
            torcwa.PlaneWave(polarization="x"),
        )
        manual_txx.append(result.transmission(polarization="x"))
        manual_rxx.append(result.reflection(polarization="x"))

    assert torch.allclose(sweep["txx"], torch.stack(manual_txx, dim=0))
    assert torch.allclose(sweep["rxx"], torch.stack(manual_rxx, dim=0))


def test_modern_api_field_chunk_option_and_s_only_mode():
    device = torch.device("cpu")
    dtype = torch.complex64
    period = (300.0, 300.0)
    stack = torcwa.Stack(period=period)
    stack.add_layer(thickness=50.0, eps=2.25)
    options = torcwa.v2.SolverOptions(dtype=dtype, device=device, field_chunk_size=2)
    solver = torcwa.RCWA(wavelength=500.0, orders=(1, 1), options=options)
    result = solver.solve(stack, torcwa.PlaneWave(polarization="x"))

    x_axis = torch.linspace(0.0, period[0], 6)
    z_axis = torch.linspace(-10.0, 70.0, 5)
    option_chunked = result.fields.plane("xz", x=x_axis, z=z_axis, y=period[1] / 2)
    explicit_chunked = result.fields.plane("xz", x=x_axis, z=z_axis, y=period[1] / 2, chunk_size=2)
    assert torch.allclose(option_chunked.Ex, explicit_chunked.Ex)

    s_only = solver.solve(stack, torcwa.PlaneWave(polarization="x"), store_fields=False)
    assert torch.allclose(s_only.transmission(polarization="x"), result.transmission(polarization="x"))
    with pytest.raises(ValueError):
        s_only.fields.plane("xz", x=x_axis, z=z_axis, y=period[1] / 2)


def test_result_power_helpers_are_cached_and_consistent():
    device = torch.device("cpu")
    stack = torcwa.Stack(period=(300.0, 300.0))
    stack.add_layer(thickness=80.0, eps=2.25)
    result = torcwa.RCWA(wavelength=500.0, orders=(1, 1), dtype=torch.complex64, device=device).solve(stack)

    first = result.diffraction_table(input_polarization="x")
    second = result.diffraction_table(input_polarization="x")
    balance = result.power_balance(input_polarization="x")

    assert first["T"].data_ptr() == second["T"].data_ptr()
    assert torch.allclose(torch.sum(first["T"]), balance["T"])
    assert torch.allclose(torch.sum(first["R"]), balance["R"])


def test_modern_api_gradient_is_finite():
    device = torch.device("cpu")
    radius = torch.tensor(65.0, dtype=torch.float64, requires_grad=True)
    cell = torcwa.UnitCell(period=(300.0, 300.0), grid=(24, 24), edge_sharpness=20.0, dtype=torch.float64, device=device)
    mask = cell.circle(radius=radius, center=(150.0, 150.0))
    eps = torcwa.material.mix(1.0, 2.25, mask)

    stack = torcwa.Stack(period=(300.0, 300.0))
    stack.add_layer(thickness=50.0, eps=eps)
    result = torcwa.RCWA(wavelength=500.0, orders=(1, 1), dtype=torch.complex128, device=device).solve(stack)
    objective = torch.abs(result.transmission(polarization="x"))[0] ** 2
    objective.backward()

    assert radius.grad is not None
    assert torch.isfinite(radius.grad)


def test_modern_api_strict_validation():
    with pytest.raises(ValueError):
        torcwa.PlaneWave(polarization="left-circular")

    stack = torcwa.Stack(period=(300.0, 300.0))
    result = torcwa.RCWA(wavelength=500.0, orders=(0, 0), dtype=torch.complex128, device=torch.device("cpu")).solve(stack)

    with pytest.raises(ValueError):
        result.transmission(polarization="bad")

    with pytest.raises(ValueError):
        result.fields.plane("rt", x=torch.tensor([0.0]), z=torch.tensor([0.0]), y=0.0)

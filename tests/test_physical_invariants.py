import math

import pytest

torch = pytest.importorskip("torch")

import torcwa
from torcwa.v2.physics import kz_branch, propagating_mask


def _solver(*, eps_out=1.0, order=(0, 0), angle=0.0, dtype=torch.complex128):
    sim = torcwa.rcwa(freq=1 / 500, order=list(order), L=[300.0, 300.0], dtype=dtype, device=torch.device("cpu"))
    if eps_out != 1.0:
        sim.add_output_layer(eps=eps_out, mu=1.0)
    sim.set_incident_angle(angle, 0.0)
    return sim


def _slab_reference(n1, n2, n3, thickness, freq):
    omega = 2 * 3.141592652589793 * freq
    delta = omega * n2 * thickness
    phase = complex(math.cos(delta), math.sin(delta))
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    denom = 1 + r12 * r23 * phase**2
    return t12 * t23 * phase / denom, (r12 + r23 * phase**2) / denom


def test_uniform_slab_matches_fabry_perot_phase():
    thickness = 120.0
    sim = _solver()
    sim.add_layer(thickness=thickness, eps=2.25, mu=1.0)
    sim.solve_global_smatrix()

    t_ref, r_ref = _slab_reference(1.0, 1.5, 1.0, thickness, 1 / 500)
    t = sim.S_parameters([0, 0], polarization="ss", power_norm=False)
    r = sim.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=False)

    assert torch.allclose(t, torch.as_tensor([t_ref], dtype=torch.complex128), atol=1e-9, rtol=1e-9)
    assert torch.allclose(r, torch.as_tensor([r_ref], dtype=torch.complex128), atol=1e-9, rtol=1e-9)


def test_brewster_angle_suppresses_tm_reflection():
    angle = math.atan(1.5)
    sim = _solver(eps_out=2.25, angle=angle)
    sim.solve_global_smatrix()

    r_pp = sim.S_parameters([0, 0], polarization="pp", port="reflection", power_norm=False)
    r_ss = sim.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=False)

    assert torch.abs(r_pp).item() < 1e-12
    assert torch.abs(r_ss).item() > 0.1


def test_near_critical_angle_branch_classifies_evanescent_transmission():
    n_glass = math.sqrt(2.25)
    angle = math.asin(1 / n_glass) + 0.05
    sim = torcwa.rcwa(freq=1 / 500, order=[0, 0], L=[300.0, 300.0], dtype=torch.complex128, device=torch.device("cpu"))
    sim.add_input_layer(eps=2.25, mu=1.0)
    sim.set_incident_angle(angle, 0.0, angle_layer="input")
    sim.solve_global_smatrix()

    kz_out = kz_branch(torch.tensor(1.0, dtype=torch.complex128), sim.Kx_norm_dn, sim.Ky_norm_dn)
    assert not bool(propagating_mask(kz_out)[0])
    transmitted_power = sim.S_parameters([0, 0], polarization="ss", port="transmission", power_norm=True)
    assert torch.allclose(transmitted_power, torch.zeros_like(transmitted_power))


def test_lossless_slab_conserves_power():
    lossless = _solver()
    lossless.add_layer(thickness=120.0, eps=2.25, mu=1.0)
    lossless.solve_global_smatrix()
    t = lossless.S_parameters([0, 0], polarization="ss", power_norm=True)
    r = lossless.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=True)
    assert torch.allclose(torch.abs(t) ** 2 + torch.abs(r) ** 2, torch.ones_like(t.real), atol=1e-12, rtol=1e-12)


def test_lossy_slab_absorption_is_nonnegative():
    lossy = _solver()
    lossy.add_layer(thickness=120.0, eps=2.25 + 0.08j, mu=1.0)
    lossy.solve_global_smatrix()
    t_loss = lossy.S_parameters([0, 0], polarization="ss", power_norm=True)
    r_loss = lossy.S_parameters([0, 0], polarization="ss", port="reflection", power_norm=True)
    absorption = 1 - torch.abs(t_loss) ** 2 - torch.abs(r_loss) ** 2
    assert absorption.item() >= -1e-10
    assert absorption.item() > 1e-4


def test_reciprocal_symmetric_slab_has_equal_forward_backward_transmission():
    sim = _solver()
    sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
    sim.solve_global_smatrix()

    forward = sim.S_parameters([0, 0], direction="forward", polarization="ss", port="transmission", power_norm=False)
    backward = sim.S_parameters([0, 0], direction="backward", polarization="ss", port="transmission", power_norm=False)
    assert torch.allclose(forward, backward, atol=1e-12, rtol=1e-12)


def test_tangential_fields_are_continuous_at_single_interface():
    sim = _solver(eps_out=2.25)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")

    x_axis = torch.tensor([0.0], dtype=torch.float64)
    z_axis = torch.tensor([-1e-7, 1e-7], dtype=torch.float64)
    electric, magnetic = sim.field_xz(x_axis, z_axis, 0.0)

    assert torch.allclose(electric[0][0, 0], electric[0][0, 1], atol=1e-7, rtol=1e-7)
    assert torch.allclose(magnetic[1][0, 0], magnetic[1][0, 1], atol=1e-7, rtol=1e-7)


def test_small_geometry_gradient_matches_finite_difference():
    def objective(radius_value):
        radius = torch.as_tensor(radius_value, dtype=torch.float64)
        geo = torcwa.geometry(Lx=300.0, Ly=300.0, nx=24, ny=24, edge_sharpness=20.0, dtype=torch.float64, device=torch.device("cpu"))
        mask = geo.circle(R=radius, Cx=150.0, Cy=150.0)
        eps = mask * 2.25 + (1.0 - mask)
        sim = torcwa.rcwa(freq=1 / 500, order=[1, 1], L=[300.0, 300.0], dtype=torch.complex128, device=torch.device("cpu"))
        sim.set_incident_angle(0.0, 0.0)
        sim.add_layer(thickness=50.0, eps=eps)
        sim.solve_global_smatrix()
        return torch.abs(sim.S_parameters([0, 0], polarization="xx"))[0] ** 2

    radius = torch.tensor(65.0, dtype=torch.float64, requires_grad=True)
    value = objective(radius)
    value.backward()

    step = 0.1
    finite_difference = (objective(65.0 + step) - objective(65.0 - step)) / (2 * step)
    assert torch.isfinite(radius.grad)
    assert torch.allclose(radius.grad, finite_difference, rtol=1e-3, atol=1e-7)

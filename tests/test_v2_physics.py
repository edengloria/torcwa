import math

import pytest

torch = pytest.importorskip("torch")

from torcwa.v2.config import FourierBasis, RCWAConfig
from torcwa.v2.physics import diffraction_order_indices, fresnel_amplitudes, kz_branch, propagating_mask


def test_config_builds_expected_fourier_basis():
    config = RCWAConfig(freq=1 / 500, order=(1, 2), lattice=(300.0, 400.0))
    basis = config.basis()
    tensors = basis.tensors()

    assert isinstance(basis, FourierBasis)
    assert basis.order_count == 15
    assert tensors["order_x"].tolist() == [-1, 0, 1]
    assert tensors["order_y"].tolist() == [-2, -1, 0, 1, 2]


def test_diffraction_order_indices_strict_and_clamped_modes():
    orders = torch.tensor([[0, 0], [1, -1]])

    assert diffraction_order_indices(orders, 1, 1).tolist() == [4, 6]
    with pytest.raises(ValueError):
        diffraction_order_indices(torch.tensor([[2, 0]]), 1, 1)
    assert diffraction_order_indices(torch.tensor([[2, 0]]), 1, 1, clamp=True).tolist() == [7]


def test_kz_branch_and_propagating_mask():
    kx = torch.tensor([0.0, 2.0], dtype=torch.complex128)
    ky = torch.tensor([0.0, 0.0], dtype=torch.complex128)
    kz = kz_branch(1.0, kx, ky)

    assert torch.real(kz[0]) > 0
    assert torch.imag(kz[1]) >= 0
    assert propagating_mask(kz).tolist() == [True, False]


def test_fresnel_normal_incidence_equal_te_tm_reflection():
    n1 = torch.tensor(1.0, dtype=torch.complex128)
    n2 = torch.tensor(1.5, dtype=torch.complex128)
    theta = torch.tensor(0.0, dtype=torch.complex128)
    fresnel = fresnel_amplitudes(n1, n2, theta)
    expected_r = (n1 - n2) / (n1 + n2)

    assert torch.allclose(fresnel["r_te"], expected_r)
    assert torch.allclose(fresnel["r_tm"], expected_r)
    assert abs(float(torch.real(fresnel["r_te"])) + 0.2) < math.sqrt(torch.finfo(torch.float64).eps)

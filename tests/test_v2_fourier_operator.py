import pytest

torch = pytest.importorskip("torch")

from torcwa.v2.fourier import material_convolution_apply, material_convolution_dense


def test_fourier_convolution_operator_matches_dense_matrix():
    dtype = torch.complex128
    device = torch.device("cpu")
    order = (2, 1)
    generator = torch.Generator(device=device).manual_seed(77)
    material = torch.complex(
        torch.randn((12, 10), dtype=torch.float64, device=device, generator=generator),
        0.1 * torch.randn((12, 10), dtype=torch.float64, device=device, generator=generator),
    )
    vector = torch.complex(
        torch.randn(((2 * order[0] + 1) * (2 * order[1] + 1), 2), dtype=torch.float64, device=device, generator=generator),
        torch.randn(((2 * order[0] + 1) * (2 * order[1] + 1), 2), dtype=torch.float64, device=device, generator=generator),
    ).to(dtype)

    dense_result = material_convolution_dense(material, order) @ vector
    operator_result = material_convolution_apply(material, order, vector)

    assert torch.allclose(operator_result, dense_result, atol=1e-12, rtol=1e-12)


def test_fourier_convolution_operator_has_finite_material_gradient():
    dtype = torch.complex128
    device = torch.device("cpu")
    order = (1, 1)
    generator = torch.Generator(device=device).manual_seed(78)
    material = torch.randn((8, 8), dtype=torch.float64, device=device, generator=generator).to(dtype).requires_grad_(True)
    vector = torch.complex(
        torch.randn((9, 1), dtype=torch.float64, device=device, generator=generator),
        torch.randn((9, 1), dtype=torch.float64, device=device, generator=generator),
    )

    output = material_convolution_apply(material, order, vector)
    loss = torch.sum(torch.abs(output) ** 2)
    loss.backward()

    assert material.grad is not None
    assert torch.isfinite(torch.real(material.grad)).all()
    assert torch.isfinite(torch.imag(material.grad)).all()

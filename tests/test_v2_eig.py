import pytest

torch = pytest.importorskip("torch")

from torcwa.v2.eig import stabilized_eig


def test_stabilized_eig_backward_stays_on_device_and_is_finite():
    matrix = torch.tensor(
        [[2.0 + 0.0j, 0.1 + 0.2j], [0.2 - 0.1j, 1.0 + 0.0j]],
        dtype=torch.complex128,
        requires_grad=True,
    )

    eigval, eigvec = stabilized_eig(matrix)
    loss = torch.real(torch.sum(torch.abs(eigval) ** 2) + 0.1 * torch.sum(torch.abs(eigvec) ** 2))
    loss.backward()

    assert matrix.grad is not None
    assert matrix.grad.device == matrix.device
    assert torch.isfinite(torch.real(matrix.grad)).all()
    assert torch.isfinite(torch.imag(matrix.grad)).all()


def test_stabilized_eig_supports_vmap_forward():
    torch_func = pytest.importorskip("torch.func")

    base = torch.tensor(
        [[2.0 + 0.0j, 0.1 + 0.2j], [0.2 - 0.1j, 1.0 + 0.0j]],
        dtype=torch.complex128,
    )
    batch = torch.stack((base, base + 0.1 * torch.eye(2, dtype=base.dtype)))

    def eigvals(matrix):
        return stabilized_eig(matrix)[0]

    result = torch_func.vmap(eigvals)(batch)

    assert result.shape == (2, 2)
    assert torch.isfinite(torch.real(result)).all()
    assert torch.isfinite(torch.imag(result)).all()

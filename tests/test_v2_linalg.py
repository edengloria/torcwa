import pytest

torch = pytest.importorskip("torch")

from torcwa.v2.linalg import diag_post_multiply, diag_pre_multiply, solve_left, solve_right


def test_solve_left_matches_inverse_multiply():
    A = torch.tensor([[2.0, 0.5], [0.25, 3.0]], dtype=torch.complex128)
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)

    assert torch.allclose(solve_left(A, B), torch.linalg.inv(A) @ B)


def test_solve_right_matches_right_inverse_multiply():
    A = torch.tensor([[2.0, 0.5], [0.25, 3.0]], dtype=torch.complex128)
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)

    assert torch.allclose(solve_right(A, B), B @ torch.linalg.inv(A))


def test_diag_helpers_match_explicit_diag_products():
    diagonal = torch.tensor([1.0 + 1.0j, 2.0 - 0.5j], dtype=torch.complex128)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)

    assert torch.allclose(diag_pre_multiply(diagonal, matrix), torch.diag(diagonal) @ matrix)
    assert torch.allclose(diag_post_multiply(matrix, diagonal), matrix @ torch.diag(diagonal))

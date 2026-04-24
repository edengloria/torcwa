import pytest

torch = pytest.importorskip("torch")

from torcwa.v2.linalg import diag_post_multiply, diag_pre_multiply, lu_factor_left, lu_solve_left, solve_left, solve_left_many, solve_right


def test_solve_left_matches_inverse_multiply():
    A = torch.tensor([[2.0, 0.5], [0.25, 3.0]], dtype=torch.complex128)
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)

    assert torch.allclose(solve_left(A, B), torch.linalg.inv(A) @ B)


def test_solve_right_matches_right_inverse_multiply():
    A = torch.tensor([[2.0, 0.5], [0.25, 3.0]], dtype=torch.complex128)
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)

    assert torch.allclose(solve_right(A, B), B @ torch.linalg.inv(A))


def test_lu_solve_helpers_reuse_factorization():
    A = torch.tensor([[2.0, 0.5], [0.25, 3.0]], dtype=torch.complex128)
    B1 = torch.tensor([[1.0], [3.0]], dtype=torch.complex128)
    B2 = torch.tensor([[2.0, 4.0], [5.0, 7.0]], dtype=torch.complex128)

    lu, pivots = lu_factor_left(A)
    assert torch.allclose(lu_solve_left(lu, pivots, B1), solve_left(A, B1))

    X1, X2 = solve_left_many(A, [B1, B2])
    assert torch.allclose(X1, solve_left(A, B1))
    assert torch.allclose(X2, solve_left(A, B2))


def test_diag_helpers_match_explicit_diag_products():
    diagonal = torch.tensor([1.0 + 1.0j, 2.0 - 0.5j], dtype=torch.complex128)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)

    assert torch.allclose(diag_pre_multiply(diagonal, matrix), torch.diag(diagonal) @ matrix)
    assert torch.allclose(diag_post_multiply(matrix, diagonal), matrix @ torch.diag(diagonal))

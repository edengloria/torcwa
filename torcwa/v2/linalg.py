from __future__ import annotations

import torch


def identity_like(matrix: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Return a square identity matrix matching dtype/device of ``matrix``."""

    size = int(n if n is not None else matrix.shape[-1])
    return torch.eye(size, dtype=matrix.dtype, device=matrix.device)


def solve_left(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve ``A @ X = B`` without forming ``A.inverse()``."""

    return torch.linalg.solve(A, B)


def lu_factor_left(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor a matrix once for repeated left solves."""

    lu, pivots, info = torch.linalg.lu_factor_ex(A)
    if bool(torch.any(info != 0)):
        raise RuntimeError("LU factorization failed for a solve matrix")
    return lu, pivots


def lu_solve_left(lu: torch.Tensor, pivots: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve ``A @ X = B`` from a precomputed LU factorization."""

    return torch.linalg.lu_solve(lu, pivots, B)


def solve_left_many(A: torch.Tensor, rhs_list: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    """Solve several RHS matrices with one LU factorization."""

    if len(rhs_list) == 0:
        return []
    widths = [rhs.shape[-1] for rhs in rhs_list]
    rhs = torch.cat(tuple(rhs_list), dim=-1)
    lu, pivots = lu_factor_left(A)
    solution = lu_solve_left(lu, pivots, rhs)
    return list(torch.split(solution, widths, dim=-1))


def solve_right(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve ``X @ A = B`` without forming ``A.inverse()``."""

    return torch.linalg.solve(torch.swapaxes(A, -1, -2), torch.swapaxes(B, -1, -2)).swapaxes(-1, -2)


def diag_pre_multiply(diagonal: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Compute ``diag(diagonal) @ matrix`` using broadcasting."""

    return diagonal.unsqueeze(-1) * matrix


def diag_post_multiply(matrix: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    """Compute ``matrix @ diag(diagonal)`` using broadcasting."""

    return matrix * diagonal.unsqueeze(-2)


def block_2x2(
    top_left: torch.Tensor,
    top_right: torch.Tensor,
    bottom_left: torch.Tensor,
    bottom_right: torch.Tensor,
) -> torch.Tensor:
    """Assemble a 2x2 block matrix."""

    return torch.cat(
        (
            torch.cat((top_left, top_right), dim=-1),
            torch.cat((bottom_left, bottom_right), dim=-1),
        ),
        dim=-2,
    )

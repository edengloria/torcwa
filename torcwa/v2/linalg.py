from __future__ import annotations

import torch


def identity_like(matrix: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Return a square identity matrix matching dtype/device of ``matrix``."""

    size = int(n if n is not None else matrix.shape[-1])
    return torch.eye(size, dtype=matrix.dtype, device=matrix.device)


def solve_left(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve ``A @ X = B`` without forming ``A.inverse()``."""

    return torch.linalg.solve(A, B)


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

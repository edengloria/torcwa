from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .v2.config import MaterialGrid as _V2MaterialGrid


def _as_period(period) -> Tuple[float, float]:
    if len(period) != 2:
        raise ValueError("period must contain exactly two values")
    px, py = float(period[0]), float(period[1])
    if px <= 0 or py <= 0:
        raise ValueError("period values must be positive")
    return px, py


@dataclass(frozen=True)
class MaterialGrid:
    """Sampled material distribution on one periodic unit cell.

    ``values`` is a 2D PyTorch tensor so gradients can flow through geometry
    construction and material interpolation.
    """

    values: torch.Tensor
    period: Tuple[float, float]
    cache_key: object | None = None
    cache: bool = True

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.values):
            raise TypeError("MaterialGrid.values must be a torch.Tensor")
        if self.values.ndim != 2:
            raise ValueError("MaterialGrid.values must be a 2D tensor")
        object.__setattr__(self, "period", _as_period(self.period))

    @property
    def lattice(self) -> Tuple[float, float]:
        return self.period

    def to_v2(self) -> _V2MaterialGrid:
        return _V2MaterialGrid(
            self.values,
            self.period,
            cache_key=self.cache_key,
            cache=self.cache,
        )


def mix(background, inclusion, mask: torch.Tensor) -> torch.Tensor:
    """Blend two material values with a differentiable mask."""

    if not torch.is_tensor(mask):
        raise TypeError("mask must be a torch.Tensor")
    return background * (1 - mask) + inclusion * mask


def constant(value, *, shape, dtype=None, device=None) -> torch.Tensor:
    """Create a constant 2D material grid."""

    if len(shape) != 2:
        raise ValueError("shape must contain exactly two integers")
    return torch.full(tuple(int(v) for v in shape), value, dtype=dtype, device=device)


__all__ = ["MaterialGrid", "constant", "mix"]

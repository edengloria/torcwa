from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

import torch

FourierFactorization = Literal["legacy-compatible", "direct", "inverse"]
MemoryMode = Literal["balanced", "memory", "speed"]
AngleLayer = Literal["input", "output"]


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_pair_int(value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError("order must contain exactly two integers")
    ox, oy = int(value[0]), int(value[1])
    if ox < 0 or oy < 0:
        raise ValueError("Fourier orders must be non-negative")
    return ox, oy


def _as_pair_float(value: Sequence[float]) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError("lattice must contain exactly two values")
    lx, ly = float(value[0]), float(value[1])
    if lx <= 0 or ly <= 0:
        raise ValueError("lattice constants must be positive")
    return lx, ly


@dataclass(frozen=True)
class SolverOptions:
    """Numerical policy for v2 simulations."""

    dtype: torch.dtype = torch.complex64
    device: torch.device = field(default_factory=_default_device)
    stable_eig_grad: bool = True
    eig_broadening: Optional[float] = 1e-10
    avoid_pinv_instability: bool = False
    max_pinv_instability: float = 0.005
    fourier_factorization: FourierFactorization = "legacy-compatible"
    compile: bool = False
    field_chunk_size: Optional[int] = None
    memory_mode: MemoryMode = "balanced"

    def __post_init__(self) -> None:
        if self.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128")
        if self.fourier_factorization not in ("legacy-compatible", "direct", "inverse"):
            raise ValueError("unknown Fourier factorization mode")
        if self.memory_mode not in ("balanced", "memory", "speed"):
            raise ValueError("memory_mode must be 'balanced', 'memory', or 'speed'")
        if self.field_chunk_size is not None and self.field_chunk_size <= 0:
            raise ValueError("field_chunk_size must be positive when provided")

    @property
    def real_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == torch.complex64 else torch.float64


@dataclass(frozen=True)
class PortSpec:
    """Homogeneous input/output medium and incidence convention."""

    eps: complex = 1.0
    mu: complex = 1.0
    incident_angle: float = 0.0
    azimuth_angle: float = 0.0
    angle_layer: AngleLayer = "input"

    def __post_init__(self) -> None:
        if self.angle_layer not in ("input", "output"):
            raise ValueError("angle_layer must be 'input' or 'output'")


@dataclass(frozen=True)
class LayerSpec:
    """One internal RCWA layer."""

    thickness: float
    eps: object = 1.0
    mu: object = 1.0

    def __post_init__(self) -> None:
        if float(self.thickness) < 0:
            raise ValueError("layer thickness must be non-negative")


@dataclass(frozen=True)
class MaterialGrid:
    """A sampled material distribution on one periodic unit cell."""

    values: torch.Tensor
    lattice: Tuple[float, float]
    cache_key: object | None = None
    cache: bool = True

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("MaterialGrid.values must be a 2D tensor")
        _as_pair_float(self.lattice)


@dataclass(frozen=True)
class FourierBasis:
    """Fourier order bookkeeping and normalized transverse wavevectors."""

    order: Tuple[int, int]
    lattice: Tuple[float, float]
    frequency: object
    dtype: torch.dtype = torch.complex64
    device: torch.device = field(default_factory=_default_device)

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", _as_pair_int(self.order))
        object.__setattr__(self, "lattice", _as_pair_float(self.lattice))
        if self.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128")

    @property
    def order_count(self) -> int:
        ox, oy = self.order
        return (2 * ox + 1) * (2 * oy + 1)

    def tensors(self) -> dict[str, torch.Tensor]:
        ox, oy = self.order
        lx, ly = self.lattice
        freq = torch.as_tensor(self.frequency, dtype=self.dtype, device=self.device)
        order_x = torch.arange(-ox, ox + 1, dtype=torch.int64, device=self.device)
        order_y = torch.arange(-oy, oy + 1, dtype=torch.int64, device=self.device)
        gx_norm = 1 / (lx * freq)
        gy_norm = 1 / (ly * freq)
        return {
            "frequency": freq,
            "order_x": order_x,
            "order_y": order_y,
            "gx_norm": gx_norm,
            "gy_norm": gy_norm,
        }


@dataclass(frozen=True)
class RCWAConfig:
    """Top-level v2 solver configuration."""

    freq: object
    order: Tuple[int, int]
    lattice: Tuple[float, float]
    input_layer: PortSpec = field(default_factory=PortSpec)
    output_layer: PortSpec = field(default_factory=PortSpec)
    options: SolverOptions = field(default_factory=SolverOptions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", _as_pair_int(self.order))
        object.__setattr__(self, "lattice", _as_pair_float(self.lattice))

    def basis(self) -> FourierBasis:
        return FourierBasis(
            order=self.order,
            lattice=self.lattice,
            frequency=self.freq,
            dtype=self.options.dtype,
            device=self.options.device,
        )

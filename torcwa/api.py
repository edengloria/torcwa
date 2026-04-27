from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import torch

from .geometry import geometry
from .materials import MaterialGrid
from .results import Output, Result, canonical_s_polarization
from .sources import PlaneWave
from .v2 import MaterialGrid as V2MaterialGrid
from .v2 import PortSpec, RCWAConfig, RCWASolver, SolverOptions


def _default_device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_pair_float(value, name: str) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    left, right = float(value[0]), float(value[1])
    if left <= 0 or right <= 0:
        raise ValueError(f"{name} values must be positive")
    return left, right


def _as_pair_int(value, name: str) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    left, right = int(value[0]), int(value[1])
    if left < 0 or right < 0:
        raise ValueError(f"{name} values must be non-negative")
    return left, right


@dataclass(frozen=True)
class Layer:
    thickness: float
    eps: object = 1.0
    mu: object = 1.0

    def __post_init__(self) -> None:
        if float(self.thickness) < 0:
            raise ValueError("layer thickness must be non-negative")


class Stack:
    """Ordered periodic layer stack for the high-level TORCWA API."""

    def __init__(self, *, period: Sequence[float], input_eps=1.0, output_eps=1.0, input_mu=1.0, output_mu=1.0):
        self.period = _as_pair_float(period, "period")
        self.input_eps = input_eps
        self.output_eps = output_eps
        self.input_mu = input_mu
        self.output_mu = output_mu
        self.layers: list[Layer] = []

    def set_ports(self, *, input_eps=None, output_eps=None, input_mu=None, output_mu=None) -> "Stack":
        if input_eps is not None:
            self.input_eps = input_eps
        if output_eps is not None:
            self.output_eps = output_eps
        if input_mu is not None:
            self.input_mu = input_mu
        if output_mu is not None:
            self.output_mu = output_mu
        return self

    def add_layer(self, *, thickness: float, eps=1.0, mu=1.0) -> "Stack":
        self._validate_material_period(eps)
        self._validate_material_period(mu)
        self.layers.append(Layer(thickness=thickness, eps=eps, mu=mu))
        return self

    def copy(self) -> "Stack":
        copied = Stack(
            period=self.period,
            input_eps=self.input_eps,
            output_eps=self.output_eps,
            input_mu=self.input_mu,
            output_mu=self.output_mu,
        )
        copied.layers = list(self.layers)
        return copied

    def _validate_material_period(self, material) -> None:
        if isinstance(material, MaterialGrid) and material.period != self.period:
            raise ValueError("MaterialGrid.period must match Stack.period")
        if isinstance(material, V2MaterialGrid) and material.lattice != self.period:
            raise ValueError("MaterialGrid lattice must match Stack.period")


class UnitCell:
    """Convenience differentiable geometry builder for one periodic cell."""

    def __init__(
        self,
        *,
        period: Sequence[float],
        grid: Sequence[int] = (100, 100),
        edge_sharpness: float = 1000.0,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        self.period = _as_pair_float(period, "period")
        self.grid_shape = _as_pair_int(grid, "grid")
        self.geometry = geometry(
            Lx=self.period[0],
            Ly=self.period[1],
            nx=self.grid_shape[0],
            ny=self.grid_shape[1],
            edge_sharpness=edge_sharpness,
            dtype=dtype,
            device=_default_device(device),
        )

    def circle(self, *, radius, center):
        return self.geometry.circle(R=radius, Cx=center[0], Cy=center[1])

    def rectangle(self, *, size, center, angle=0.0):
        return self.geometry.rectangle(Wx=size[0], Wy=size[1], Cx=center[0], Cy=center[1], theta=angle)

    def ellipse(self, *, radius, center, angle=0.0):
        return self.geometry.ellipse(Rx=radius[0], Ry=radius[1], Cx=center[0], Cy=center[1], theta=angle)


class RCWA:
    """High-level PyTorch-style RCWA solver."""

    def __init__(
        self,
        *,
        wavelength,
        orders: Sequence[int],
        dtype: torch.dtype = torch.complex64,
        device=None,
        options: SolverOptions | None = None,
        memory_mode: str | None = None,
        store_fields: bool | None = None,
    ):
        self.wavelength = wavelength
        self.orders = _as_pair_int(orders, "orders")
        if options is None:
            options = SolverOptions(dtype=dtype, device=_default_device(device))
        elif device is not None:
            options = replace(options, device=_default_device(device))
        if memory_mode is not None:
            options = replace(options, memory_mode=memory_mode)
        if store_fields is not None:
            options = replace(options, store_fields=bool(store_fields))
        self.options = options

    @classmethod
    def from_frequency(cls, *, frequency, orders: Sequence[int], **kwargs) -> "RCWA":
        solver = cls(wavelength=1 / frequency, orders=orders, **kwargs)
        solver._frequency = frequency
        return solver

    @property
    def frequency(self):
        if hasattr(self, "_frequency"):
            return self._frequency
        return 1 / torch.as_tensor(self.wavelength, dtype=self.options.real_dtype, device=self.options.device)

    def solve(self, stack: Stack, source: PlaneWave | None = None, *, store_fields: bool | None = None) -> Result:
        if not isinstance(stack, Stack):
            raise TypeError("solve expects a torcwa.Stack instance")
        source = source or PlaneWave()
        options = replace(self.options, store_fields=bool(store_fields)) if store_fields is not None else self.options
        solver = self._build_solver(stack, source, options=options)
        solver.solve()
        legacy = solver.legacy_solver()
        source.apply(legacy)
        return Result(legacy_solver=legacy, source=source, options=options)

    def sweep(
        self,
        stack: Stack,
        *,
        source: PlaneWave | None = None,
        wavelength,
        outputs: Sequence[Output] | None = None,
    ) -> dict[str, torch.Tensor]:
        source = source or PlaneWave()
        outputs = list(outputs or [Output.transmission(order=(0, 0), polarization=source.polarization, name="T00")])
        wavelengths = torch.as_tensor(wavelength, dtype=self.options.real_dtype, device=self.options.device).reshape([-1])
        freqs = 1 / wavelengths
        sweep_options = replace(self.options, store_fields=False)
        base = RCWA(wavelength=wavelengths[0], orders=self.orders, options=sweep_options)
        v2_solver = base._build_solver(stack, source, options=sweep_options)
        requests = [
            {
                "name": output.name,
                "orders": list(output.order),
                "direction": output.direction,
                "port": output.port,
                "polarization": canonical_s_polarization(output.polarization),
                "power_norm": output.power,
            }
            for output in outputs
        ]
        return v2_solver.solve_sweep(
            freqs,
            incident_angles=source.incident_angle,
            azimuth_angles=source.azimuth_angle,
            requests=requests,
        )

    def _build_solver(self, stack: Stack, source: PlaneWave, *, options: SolverOptions | None = None) -> RCWASolver:
        options = options or self.options
        if source.direction == "forward":
            input_layer = PortSpec(
                eps=stack.input_eps,
                mu=stack.input_mu,
                incident_angle=source.incident_angle,
                azimuth_angle=source.azimuth_angle,
                angle_layer="input",
            )
            output_layer = PortSpec(eps=stack.output_eps, mu=stack.output_mu)
        else:
            input_layer = PortSpec(eps=stack.input_eps, mu=stack.input_mu, angle_layer="output")
            output_layer = PortSpec(
                eps=stack.output_eps,
                mu=stack.output_mu,
                incident_angle=source.incident_angle,
                azimuth_angle=source.azimuth_angle,
                angle_layer="output",
            )
        config = RCWAConfig(
            freq=self.frequency,
            order=self.orders,
            lattice=stack.period,
            input_layer=input_layer,
            output_layer=output_layer,
            options=options,
        )
        solver = RCWASolver(config)
        for layer in stack.layers:
            solver.add_layer(layer.thickness, eps=self._material_for_v2(layer.eps), mu=self._material_for_v2(layer.mu))
        return solver

    @staticmethod
    def _material_for_v2(material):
        if isinstance(material, MaterialGrid):
            return material.to_v2()
        return material


__all__ = ["Layer", "Output", "PlaneWave", "RCWA", "Stack", "UnitCell"]

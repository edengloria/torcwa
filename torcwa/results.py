from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


_S_POLARIZATIONS = {"xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"}


def _canonical_s_polarization(polarization: str, *, default_output: bool = True) -> str:
    mapping = {
        "x": "xx",
        "y": "yy",
        "p": "pp",
        "s": "ss",
        "xx": "xx",
        "yx": "yx",
        "xy": "xy",
        "yy": "yy",
        "pp": "pp",
        "sp": "sp",
        "ps": "ps",
        "ss": "ss",
    }
    try:
        return mapping[polarization]
    except KeyError as exc:
        raise ValueError("polarization must be one of x, y, p, s, xx, yx, xy, yy, pp, sp, ps, ss") from exc


def canonical_s_polarization(polarization: str) -> str:
    return _canonical_s_polarization(polarization)


def _input_power_channels(input_polarization: str) -> tuple[str, str]:
    pol = _canonical_s_polarization(input_polarization)
    if pol[-1] == "x":
        return "xx", "yx"
    if pol[-1] == "y":
        return "xy", "yy"
    if pol[-1] == "p":
        return "pp", "sp"
    return "ps", "ss"


@dataclass(frozen=True)
class FieldPlane:
    electric: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    magnetic: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def Ex(self):
        return self.electric[0]

    @property
    def Ey(self):
        return self.electric[1]

    @property
    def Ez(self):
        return self.electric[2]

    @property
    def Hx(self):
        return self.magnetic[0]

    @property
    def Hy(self):
        return self.magnetic[1]

    @property
    def Hz(self):
        return self.magnetic[2]

    def as_dict(self, components: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        values = {
            "Ex": self.Ex,
            "Ey": self.Ey,
            "Ez": self.Ez,
            "Hx": self.Hx,
            "Hy": self.Hy,
            "Hz": self.Hz,
        }
        if components is None:
            return values
        return {name: values[name] for name in components}


class FieldAccessor:
    def __init__(self, result: "Result"):
        self._result = result

    def plane(self, plane: str, *, components: Sequence[str] | None = None, chunk_size: int | None = None, **axes):
        if not self._result.has_source:
            raise ValueError("field reconstruction requires a source; pass source to RCWA.solve(...)")
        if not getattr(self._result.legacy_solver, "store_fields", True):
            raise ValueError("field reconstruction is unavailable because this result was solved with store_fields=False")
        plane = str(plane).lower()
        legacy = self._result.legacy_solver
        previous_chunk_size = getattr(legacy, "field_chunk_size", None)
        default_chunk = getattr(self._result.options, "field_chunk_size", None)
        legacy.field_chunk_size = chunk_size if chunk_size is not None else default_chunk
        try:
            if plane == "xz":
                electric, magnetic = legacy.field_xz(axes["x"], axes["z"], axes["y"])
            elif plane == "yz":
                electric, magnetic = legacy.field_yz(axes["y"], axes["z"], axes["x"])
            elif plane == "xy":
                if "layer_num" not in axes:
                    raise ValueError("xy field planes require layer_num")
                electric, magnetic = legacy.field_xy(axes["layer_num"], axes["x"], axes["y"], z_prop=axes.get("z", 0.0))
            else:
                raise ValueError("plane must be one of 'xz', 'yz', or 'xy'")
        finally:
            legacy.field_chunk_size = previous_chunk_size

        field = FieldPlane(tuple(electric), tuple(magnetic))
        return field if components is None else field.as_dict(components)


@dataclass(frozen=True)
class Result:
    legacy_solver: object
    source: object | None = None
    options: object | None = None
    _orders_cache: dict = field(default_factory=dict, init=False, repr=False)
    _s_cache: dict = field(default_factory=dict, init=False, repr=False)
    _diffraction_cache: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def has_source(self) -> bool:
        return self.source is not None

    @property
    def fields(self) -> FieldAccessor:
        return FieldAccessor(self)

    def s_parameter(
        self,
        order=(0, 0),
        *,
        direction: str = "forward",
        port: str = "transmission",
        polarization: str = "xx",
        ref_order=(0, 0),
        power: bool = True,
        evanescent: float = 1e-3,
    ) -> torch.Tensor:
        polarization = _canonical_s_polarization(polarization)
        cache_key = self._s_cache_key(order, direction, port, polarization, ref_order, power, evanescent)
        cached = self._s_cache.get(cache_key)
        if cached is not None:
            return cached
        value = self.legacy_solver.S_parameters(
            order,
            direction=direction,
            port=port,
            polarization=polarization,
            ref_order=ref_order,
            power_norm=power,
            evanescent=evanescent,
        )
        self._s_cache[cache_key] = value
        return value

    def transmission(self, order=(0, 0), *, polarization: str = "x", direction: str = "forward", power: bool = True) -> torch.Tensor:
        return self.s_parameter(order, direction=direction, port="transmission", polarization=polarization, power=power)

    def reflection(self, order=(0, 0), *, polarization: str = "x", direction: str = "forward", power: bool = True) -> torch.Tensor:
        return self.s_parameter(order, direction=direction, port="reflection", polarization=polarization, power=power)

    def orders(self) -> torch.Tensor:
        cached = self._orders_cache.get("all")
        if cached is not None:
            return cached
        sim = self.legacy_solver
        ox, oy = torch.meshgrid(sim.order_x, sim.order_y, indexing="ij")
        orders = torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=1)
        self._orders_cache["all"] = orders
        return orders

    def power_balance(self, *, input_polarization: str = "x", direction: str = "forward", orders=None) -> dict[str, torch.Tensor]:
        powers = self._diffraction_powers(input_polarization=input_polarization, direction=direction, orders=orders)
        transmission = torch.sum(powers["T"])
        reflection = torch.sum(powers["R"])
        absorption = 1 - transmission - reflection
        return {"T": transmission, "R": reflection, "A": absorption}

    def diffraction_table(self, *, input_polarization: str = "x", direction: str = "forward") -> dict[str, torch.Tensor]:
        powers = self._diffraction_powers(input_polarization=input_polarization, direction=direction, orders=None)
        return {"orders": powers["orders"], "T": powers["T"], "R": powers["R"]}

    def _diffraction_powers(self, *, input_polarization: str, direction: str, orders=None) -> dict[str, torch.Tensor]:
        orders_tensor = self.orders() if orders is None else torch.as_tensor(orders, dtype=torch.int64, device=self.legacy_solver._device).reshape([-1, 2])
        cache_key = None
        if orders is None:
            cache_key = (input_polarization, direction)
            cached = self._diffraction_cache.get(cache_key)
            if cached is not None:
                return cached

        channels = _input_power_channels(input_polarization)
        transmission_parts = [
            torch.abs(self.s_parameter(orders_tensor, direction=direction, port="transmission", polarization=channel)) ** 2
            for channel in channels
        ]
        reflection_parts = [
            torch.abs(self.s_parameter(orders_tensor, direction=direction, port="reflection", polarization=channel)) ** 2
            for channel in channels
        ]
        powers = {
            "orders": orders_tensor,
            "T": torch.stack(transmission_parts, dim=0).sum(dim=0),
            "R": torch.stack(reflection_parts, dim=0).sum(dim=0),
        }
        if cache_key is not None:
            self._diffraction_cache[cache_key] = powers
        return powers

    @staticmethod
    def _tensor_cache_key(value):
        if torch.is_tensor(value):
            return ("tensor", int(value.data_ptr()), tuple(value.shape), str(value.dtype), str(value.device))
        tensor = torch.as_tensor(value).reshape([-1])
        return tuple(tensor.detach().cpu().tolist())

    def _s_cache_key(self, order, direction, port, polarization, ref_order, power, evanescent):
        return (
            self._tensor_cache_key(order),
            direction,
            port,
            polarization,
            self._tensor_cache_key(ref_order),
            bool(power),
            float(evanescent),
        )


@dataclass(frozen=True)
class Output:
    name: str
    port: str
    order: tuple[int, int] = (0, 0)
    polarization: str = "x"
    direction: str = "forward"
    power: bool = True

    @classmethod
    def transmission(cls, *, order=(0, 0), polarization: str = "x", direction: str = "forward", power: bool = True, name: str | None = None):
        return cls(name or f"T{polarization}_{order[0]}_{order[1]}", "transmission", tuple(order), polarization, direction, power)

    @classmethod
    def reflection(cls, *, order=(0, 0), polarization: str = "x", direction: str = "forward", power: bool = True, name: str | None = None):
        return cls(name or f"R{polarization}_{order[0]}_{order[1]}", "reflection", tuple(order), polarization, direction, power)

    def evaluate(self, result: Result) -> torch.Tensor:
        if self.port == "transmission":
            return result.transmission(self.order, polarization=self.polarization, direction=self.direction, power=self.power)
        if self.port == "reflection":
            return result.reflection(self.order, polarization=self.polarization, direction=self.direction, power=self.power)
        raise ValueError("Output.port must be 'transmission' or 'reflection'")


__all__ = ["FieldAccessor", "FieldPlane", "Output", "Result"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple


Direction = Literal["forward", "backward"]
Notation = Literal["auto", "xy", "ps"]


def _as_angle(angle) -> Tuple[object, object]:
    if isinstance(angle, (int, float)):
        return angle, 0.0
    if len(angle) != 2:
        raise ValueError("angle must be a scalar or a pair (incident, azimuth)")
    return angle[0], angle[1]


def _canonical_direction(direction: str) -> Direction:
    if direction in ("f", "forward"):
        return "forward"
    if direction in ("b", "backward"):
        return "backward"
    raise ValueError("direction must be 'forward' or 'backward'")


def _canonical_source_polarization(polarization: str) -> str:
    mapping = {
        "x": "x",
        "xx": "x",
        "y": "y",
        "yy": "y",
        "p": "p",
        "pp": "p",
        "s": "s",
        "ss": "s",
    }
    try:
        return mapping[polarization]
    except KeyError as exc:
        raise ValueError("PlaneWave polarization must be one of 'x', 'y', 'p', or 's'") from exc


@dataclass(frozen=True)
class PlaneWave:
    """Single-order plane-wave source.

    ``angle`` is ``(incident_angle, azimuth_angle)`` in radians.
    """

    angle: object = (0.0, 0.0)
    polarization: str = "x"
    direction: Direction | str = "forward"
    amplitude: Sequence[complex] | None = None
    notation: Notation = "auto"
    order: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "angle", _as_angle(self.angle))
        object.__setattr__(self, "polarization", _canonical_source_polarization(self.polarization))
        object.__setattr__(self, "direction", _canonical_direction(str(self.direction)))
        if self.notation not in ("auto", "xy", "ps"):
            raise ValueError("notation must be 'auto', 'xy', or 'ps'")
        if len(self.order) != 2:
            raise ValueError("order must contain exactly two integers")
        object.__setattr__(self, "order", (int(self.order[0]), int(self.order[1])))

    @property
    def incident_angle(self):
        return self.angle[0]

    @property
    def azimuth_angle(self):
        return self.angle[1]

    def legacy_notation(self) -> str:
        if self.notation != "auto":
            return self.notation
        return "xy" if self.polarization in ("x", "y") else "ps"

    def legacy_amplitude(self):
        if self.amplitude is not None:
            if len(self.amplitude) != 2:
                raise ValueError("PlaneWave.amplitude must contain exactly two values")
            return list(self.amplitude)
        if self.polarization in ("x", "p"):
            return [1.0, 0.0]
        return [0.0, 1.0]

    def apply(self, legacy_solver) -> None:
        kwargs = {
            "amplitude": self.legacy_amplitude(),
            "direction": self.direction,
            "notation": self.legacy_notation(),
        }
        if self.order == (0, 0):
            legacy_solver.source_planewave(**kwargs)
        else:
            legacy_solver.source_fourier(orders=list(self.order), **kwargs)


__all__ = ["PlaneWave"]

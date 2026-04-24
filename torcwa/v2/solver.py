from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from .config import LayerSpec, PortSpec, RCWAConfig, SolverOptions


class RCWASolver:
    """V2-facing solver facade with explicit configuration and validation hooks.

    This class is the public entrypoint for the v2 refactor.  It currently
    delegates the full RCWA solve to the legacy implementation while exposing a
    typed, stateless configuration surface and centralizing compatibility
    behavior.  Low-level kernels are being moved to ``torcwa.v2`` helper modules
    and are used by the legacy class where safe.
    """

    def __init__(self, config: RCWAConfig):
        self.config = config
        self.layers: list[LayerSpec] = []
        self._legacy = None

    @classmethod
    def from_legacy_args(
        cls,
        *,
        freq,
        order: Sequence[int],
        L: Sequence[float],
        dtype: torch.dtype = torch.complex64,
        device: torch.device | None = None,
        stable_eig_grad: bool = True,
        avoid_Pinv_instability: bool = False,
        max_Pinv_instability: float = 0.005,
    ) -> "RCWASolver":
        options = SolverOptions(
            dtype=dtype,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device,
            stable_eig_grad=stable_eig_grad,
            avoid_pinv_instability=avoid_Pinv_instability,
            max_pinv_instability=max_Pinv_instability,
        )
        return cls(RCWAConfig(freq=freq, order=(int(order[0]), int(order[1])), lattice=(float(L[0]), float(L[1])), options=options))

    def with_ports(
        self,
        *,
        input_layer: PortSpec | None = None,
        output_layer: PortSpec | None = None,
    ) -> "RCWASolver":
        self.config = replace(
            self.config,
            input_layer=input_layer or self.config.input_layer,
            output_layer=output_layer or self.config.output_layer,
        )
        self._legacy = None
        return self

    def add_layer(self, thickness: float, eps=1.0, mu=1.0) -> "RCWASolver":
        self.layers.append(LayerSpec(thickness=thickness, eps=eps, mu=mu))
        self._legacy = None
        return self

    def solve(self):
        legacy = self._build_legacy()
        legacy.solve_global_smatrix()
        return self

    def s_parameter(self, orders, **kwargs):
        legacy = self._require_solved()
        return legacy.S_parameters(orders=orders, **kwargs)

    def source_planewave(self, **kwargs):
        legacy = self._require_solved()
        legacy.source_planewave(**kwargs)
        return self

    def field_plane(self, *, plane: str, axis0, axis1, offset=0.0, layer_num: int | None = None):
        legacy = self._require_solved()
        if plane == "xz":
            return legacy.field_xz(axis0, axis1, offset)
        if plane == "yz":
            return legacy.field_yz(axis0, axis1, offset)
        if plane == "xy":
            if layer_num is None:
                raise ValueError("layer_num is required for xy field planes")
            return legacy.field_xy(layer_num, axis0, axis1, z_prop=offset)
        raise ValueError("plane must be one of 'xy', 'xz', or 'yz'")

    def legacy_solver(self):
        return self._require_solved()

    def _require_solved(self):
        if self._legacy is None or not hasattr(self._legacy, "S"):
            self.solve()
        return self._legacy

    def _build_legacy(self):
        from ..rcwa import rcwa

        cfg = self.config
        options = cfg.options
        sim = rcwa(
            freq=cfg.freq,
            order=list(cfg.order),
            L=list(cfg.lattice),
            dtype=options.dtype,
            device=options.device,
            stable_eig_grad=options.stable_eig_grad,
            avoid_Pinv_instability=options.avoid_pinv_instability,
            max_Pinv_instability=options.max_pinv_instability,
        )
        if cfg.input_layer.eps != 1.0 or cfg.input_layer.mu != 1.0:
            sim.add_input_layer(eps=cfg.input_layer.eps, mu=cfg.input_layer.mu)
        if cfg.output_layer.eps != 1.0 or cfg.output_layer.mu != 1.0:
            sim.add_output_layer(eps=cfg.output_layer.eps, mu=cfg.output_layer.mu)

        angle_ref = cfg.input_layer if cfg.input_layer.angle_layer == "input" else cfg.output_layer
        sim.set_incident_angle(
            inc_ang=angle_ref.incident_angle,
            azi_ang=angle_ref.azimuth_angle,
            angle_layer=angle_ref.angle_layer,
        )
        for layer in self.layers:
            sim.add_layer(thickness=layer.thickness, eps=layer.eps, mu=layer.mu)
        self._legacy = sim
        return sim

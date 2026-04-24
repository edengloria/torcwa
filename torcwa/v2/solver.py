from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from .config import LayerSpec, MaterialGrid, PortSpec, RCWAConfig, SolverOptions


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

    def field_plane(self, *, plane: str, axis0, axis1, offset=0.0, layer_num: int | None = None, chunk_size: int | None = None):
        legacy = self._require_solved()
        previous_chunk_size = getattr(legacy, "field_chunk_size", None)
        legacy.field_chunk_size = chunk_size if chunk_size is not None else self.config.options.field_chunk_size
        try:
            if plane == "xz":
                return legacy.field_xz(axis0, axis1, offset)
            if plane == "yz":
                return legacy.field_yz(axis0, axis1, offset)
            if plane == "xy":
                if layer_num is None:
                    raise ValueError("layer_num is required for xy field planes")
                return legacy.field_xy(layer_num, axis0, axis1, z_prop=offset)
            raise ValueError("plane must be one of 'xy', 'xz', or 'yz'")
        finally:
            legacy.field_chunk_size = previous_chunk_size

    def solve_sweep(
        self,
        freqs,
        incident_angles=0.0,
        azimuth_angles=0.0,
        requests: list[dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Loop-backed fixed-geometry frequency/angle sweep.

        The initial v2 sweep API intentionally reuses the validated legacy
        solve path and material convolution cache.  It supports fixed layer
        stacks and returns only requested S-parameters.
        """

        cfg = self.config
        options = cfg.options
        real_dtype = options.real_dtype
        freqs_tensor = torch.as_tensor(freqs, dtype=real_dtype, device=options.device).reshape([-1])
        incident_tensor = self._sweep_values(incident_angles, len(freqs_tensor), real_dtype, options.device)
        azimuth_tensor = self._sweep_values(azimuth_angles, len(freqs_tensor), real_dtype, options.device)
        requests = requests or [{"name": "txx", "orders": [0, 0], "polarization": "xx"}]

        outputs: dict[str, list[torch.Tensor]] = {}
        for idx in range(len(freqs_tensor)):
            input_layer = cfg.input_layer
            output_layer = cfg.output_layer
            if cfg.input_layer.angle_layer == "input":
                input_layer = replace(input_layer, incident_angle=incident_tensor[idx], azimuth_angle=azimuth_tensor[idx])
            else:
                output_layer = replace(output_layer, incident_angle=incident_tensor[idx], azimuth_angle=azimuth_tensor[idx])

            sweep_config = replace(cfg, freq=freqs_tensor[idx], input_layer=input_layer, output_layer=output_layer)
            sweep_solver = RCWASolver(sweep_config)
            sweep_solver.layers = list(self.layers)
            sweep_solver.solve()
            legacy = sweep_solver.legacy_solver()

            for request_index, request in enumerate(requests):
                request_kwargs = dict(request)
                name = str(request_kwargs.pop("name", f"request_{request_index}"))
                outputs.setdefault(name, []).append(legacy.S_parameters(**request_kwargs))

        return {name: torch.stack(values, dim=0) for name, values in outputs.items()}

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
            sim.add_layer(thickness=layer.thickness, eps=self._material_value(layer.eps), mu=self._material_value(layer.mu))
        self._legacy = sim
        return sim

    @staticmethod
    def _material_value(value):
        if isinstance(value, MaterialGrid):
            return value.values if value.cache else value.values.clone()
        return value

    @staticmethod
    def _sweep_values(value, count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape([-1])
        if tensor.numel() == 1:
            return tensor.expand(count)
        if tensor.numel() != count:
            raise ValueError("sweep angle inputs must be scalar or match freqs length")
        return tensor

from __future__ import annotations

from typing import Literal

import torch

KzDirection = Literal["positive", "negative"]


def kz_branch(
    k0_norm2: torch.Tensor | complex | float,
    kx_norm: torch.Tensor,
    ky_norm: torch.Tensor,
    *,
    direction: KzDirection = "positive",
) -> torch.Tensor:
    """Return the RCWA kz branch with outgoing/decaying evanescent convention."""

    kz = torch.sqrt(torch.as_tensor(k0_norm2, dtype=kx_norm.dtype, device=kx_norm.device) - kx_norm**2 - ky_norm**2)
    if direction == "positive":
        return torch.where(torch.imag(kz) < 0, torch.conj(kz), kz)
    if direction == "negative":
        kz = torch.where(torch.imag(kz) < 0, torch.conj(kz), kz)
        return -kz
    raise ValueError("direction must be 'positive' or 'negative'")


def propagating_mask(kz: torch.Tensor, *, evanescent: float = 1e-3) -> torch.Tensor:
    """Classify propagating orders using the legacy TORCWA evanescent ratio."""

    ratio = torch.abs(torch.real(kz) / torch.imag(kz))
    return ratio >= evanescent


def diffraction_order_indices(
    orders: torch.Tensor,
    order_x: int,
    order_y: int,
    *,
    clamp: bool = False,
) -> torch.Tensor:
    """Map diffraction order pairs to flattened Fourier indices."""

    orders = torch.as_tensor(orders, dtype=torch.int64).reshape([-1, 2])
    if clamp:
        orders = orders.clone()
        orders[:, 0] = torch.clamp(orders[:, 0], min=-order_x, max=order_x)
        orders[:, 1] = torch.clamp(orders[:, 1], min=-order_y, max=order_y)
    elif bool(torch.any((orders[:, 0] < -order_x) | (orders[:, 0] > order_x) | (orders[:, 1] < -order_y) | (orders[:, 1] > order_y))):
        raise ValueError("requested diffraction order is outside the truncated Fourier basis")

    return (2 * order_y + 1) * (orders[:, 0] + order_x) + orders[:, 1] + order_y


def fresnel_amplitudes(
    n1: torch.Tensor,
    n2: torch.Tensor,
    incident_angle: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Analytical single-interface Fresnel amplitudes for validation."""

    theta_t_cos = torch.sqrt(1 - (n1 / n2 * torch.sin(incident_angle)) ** 2)
    theta_i_cos = torch.cos(incident_angle)

    r_te = (n1 * theta_i_cos - n2 * theta_t_cos) / (n1 * theta_i_cos + n2 * theta_t_cos)
    t_te = 2 * n1 * theta_i_cos / (n1 * theta_i_cos + n2 * theta_t_cos)
    r_tm = (n1 * theta_t_cos - n2 * theta_i_cos) / (n1 * theta_t_cos + n2 * theta_i_cos)
    t_tm = 2 * n1 * theta_i_cos / (n1 * theta_t_cos + n2 * theta_i_cos)

    return {"r_te": r_te, "t_te": t_te, "r_tm": r_tm, "t_tm": t_tm}

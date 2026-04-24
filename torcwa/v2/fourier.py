from __future__ import annotations

from collections.abc import Sequence

import torch


def order_vectors(order: Sequence[int], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened Fourier order coordinates in TORCWA ordering."""

    order_x = torch.arange(-int(order[0]), int(order[0]) + 1, dtype=torch.int64, device=device)
    order_y = torch.arange(-int(order[1]), int(order[1]) + 1, dtype=torch.int64, device=device)
    order_x_grid, order_y_grid = torch.meshgrid(order_x, order_y, indexing="ij")
    return order_x_grid.reshape([-1]), order_y_grid.reshape([-1])


def material_convolution_dense(material: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    """Build the dense truncated Fourier convolution matrix.

    This mirrors the legacy `_material_conv` indexing convention and is intended
    for validation of alternative operator forms, not as a separate solver path.
    """

    material_n = material.shape[0] * material.shape[1]
    order_x, order_y = order_vectors(order, device=material.device)
    indices = torch.arange(order_x.numel(), dtype=torch.int64, device=material.device)
    row, col = torch.meshgrid(indices, indices, indexing="ij")
    material_fft = torch.fft.fft2(material) / material_n
    return material_fft[order_x[row] - order_x[col], order_y[row] - order_y[col]]


def material_convolution_apply(material: torch.Tensor, order: Sequence[int], vector: torch.Tensor) -> torch.Tensor:
    """Apply the truncated Fourier convolution without storing the dense matrix."""

    material_n = material.shape[0] * material.shape[1]
    order_x, order_y = order_vectors(order, device=material.device)
    mode_count = order_x.numel()
    original_shape = vector.shape
    rhs = vector.reshape([mode_count, -1])
    material_fft = torch.fft.fft2(material) / material_n
    output = torch.empty_like(rhs)

    for row in range(mode_count):
        coeff = material_fft[order_x[row] - order_x, order_y[row] - order_y]
        output[row] = torch.sum(coeff.reshape([-1, 1]) * rhs, dim=0)

    return output.reshape(original_shape)

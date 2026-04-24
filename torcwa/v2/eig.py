from __future__ import annotations

import torch


class StabilizedEig(torch.autograd.Function):
    """GPU-resident eigendecomposition with Lorentzian-broadened backward."""

    @staticmethod
    def forward(x: torch.Tensor, broadening: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eig(x)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        x, broadening = inputs
        eigval, eigvec = output
        ctx.save_for_backward(x, eigval, eigvec, broadening)

    @staticmethod
    def backward(ctx, grad_eigval: torch.Tensor, grad_eigvec: torch.Tensor):
        x, eigval, eigvec, broadening = ctx.saved_tensors
        grad_eigval_matrix = torch.diag_embed(grad_eigval)
        gaps = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        tiny = torch.finfo(torch.float32 if gaps.dtype == torch.complex64 else torch.float64).tiny
        broadening_value = torch.clamp(torch.real(broadening).to(torch.float32 if gaps.dtype == torch.complex64 else torch.float64), min=tiny)
        while broadening_value.ndim < gaps.ndim:
            broadening_value = broadening_value.unsqueeze(-1)
        F = torch.conj(gaps) / (torch.abs(gaps) ** 2 + broadening_value)
        eye = torch.eye(F.shape[-1], dtype=torch.bool, device=F.device)
        F = torch.where(eye, torch.zeros_like(F), F)

        eigvec_h = torch.swapaxes(torch.conj(eigvec), -1, -2)
        rhs = grad_eigval_matrix + torch.conj(F) * torch.matmul(eigvec_h, grad_eigvec)
        grad = torch.matmul(torch.linalg.solve(eigvec_h, rhs), eigvec_h)
        if not torch.is_complex(x):
            grad = torch.real(grad)
        return grad, None

    @staticmethod
    def vmap(info, in_dims, x: torch.Tensor, broadening: torch.Tensor):
        x_bdim, broadening_bdim = in_dims
        if x_bdim is None:
            x = x.expand((info.batch_size,) + tuple(x.shape))
        else:
            x = torch.movedim(x, x_bdim, 0)

        if broadening_bdim is None:
            broadening = broadening.expand(info.batch_size)
        else:
            broadening = torch.movedim(broadening, broadening_bdim, 0)

        output = StabilizedEig.apply(x, broadening)
        return output, (0, 0)


def stabilized_eig(
    x: torch.Tensor,
    *,
    broadening: float | None = 1e-10,
    stable: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select native or stabilized eigendecomposition."""

    if not stable:
        return torch.linalg.eig(x)
    if broadening is None:
        broadening = torch.finfo(torch.float32 if x.dtype == torch.complex64 else torch.float64).tiny
    broadening_tensor = torch.as_tensor(broadening, dtype=torch.float32 if x.dtype == torch.complex64 else torch.float64, device=x.device)
    return StabilizedEig.apply(x, broadening_tensor)

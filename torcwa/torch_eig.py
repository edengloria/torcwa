import torch

'''
PyTorch 2.x
Complex domain eigen-decomposition with numerical stability
'''

class Eig(torch.autograd.Function):
    broadening_parameter = 1e-10

    @staticmethod
    def forward(ctx,x):
        eigval, eigvec = torch.linalg.eig(x)
        ctx.save_for_backward(x,eigval,eigvec)
        return eigval, eigvec

    @staticmethod
    def backward(ctx,grad_eigval,grad_eigvec):
        x, eigval, eigvec = ctx.saved_tensors

        grad_eigval = torch.diag_embed(grad_eigval)
        s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        # Lorentzian broadening: get small error but stabilizing the gradient calculation
        if Eig.broadening_parameter is not None:
            F = torch.conj(s)/(torch.abs(s)**2 + Eig.broadening_parameter)
        elif s.dtype == torch.complex64:
            F = torch.conj(s)/(torch.abs(s)**2 + 1.4e-45)
        elif s.dtype == torch.complex128:
            F = torch.conj(s)/(torch.abs(s)**2 + 4.9e-324)

        diag_indices = torch.arange(F.shape[-1],dtype=torch.int64,device=F.device)
        F[diag_indices,diag_indices] = 0.
        XH = torch.transpose(torch.conj(eigvec),-2,-1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)

        grad = torch.matmul(torch.linalg.solve(XH, grad_eigval + tmp), XH)
        if not torch.is_complex(x):
            grad = torch.real(grad)

        return grad

import torch

import torcwa as tw


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
period = (300.0, 300.0)
radius = torch.tensor(65.0, dtype=torch.float64, device=device, requires_grad=True)

cell = tw.UnitCell(period=period, grid=(48, 48), edge_sharpness=25.0, dtype=torch.float64, device=device)
mask = cell.circle(radius=radius, center=(150.0, 150.0))
eps = tw.material.mix(1.0, 2.25, mask)

stack = tw.Stack(period=period)
stack.add_layer(thickness=50.0, eps=eps)

result = tw.RCWA(wavelength=500.0, orders=(1, 1), dtype=torch.complex128, device=device).solve(
    stack,
    tw.PlaneWave(polarization="x"),
)

objective = -torch.abs(result.transmission(polarization="x"))[0] ** 2
objective.backward()

print("objective:", objective.detach().cpu())
print("d objective / d radius:", radius.grad.detach().cpu())

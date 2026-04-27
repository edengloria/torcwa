import torch

import torcwa as tw


device = "cuda" if torch.cuda.is_available() else "cpu"
period = (300.0, 300.0)

cell = tw.UnitCell(period=period, grid=(128, 128), edge_sharpness=50.0, device=device)
mask = cell.rectangle(size=(120.0, 90.0), center=(150.0, 150.0))
eps = tw.material.mix(1.0, 12.25, mask)

stack = tw.Stack(period=period)
stack.add_layer(thickness=80.0, eps=tw.MaterialGrid(eps, period))

result = tw.RCWA(wavelength=500.0, orders=(3, 3), device=device).solve(
    stack,
    tw.PlaneWave(polarization="x"),
)

fields = result.fields.plane(
    "xz",
    x=torch.linspace(0.0, period[0], 96, device=device),
    z=torch.linspace(-80.0, 160.0, 120, device=device),
    y=period[1] / 2,
)

print("Ex shape:", tuple(fields.Ex.shape))
print("max |Ex|:", torch.max(torch.abs(fields.Ex)).detach().cpu())

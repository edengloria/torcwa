import torch

import torcwa as tw


device = "cuda" if torch.cuda.is_available() else "cpu"
period = (500.0, 500.0)

cell = tw.UnitCell(period=period, grid=(192, 48), edge_sharpness=60.0, device=device)
mask = cell.rectangle(size=(220.0, 500.0), center=(250.0, 250.0))
eps = tw.material.mix(1.0, 12.25, mask)

stack = tw.Stack(period=period)
stack.add_layer(thickness=220.0, eps=tw.MaterialGrid(eps, period))

result = tw.RCWA(wavelength=700.0, orders=(5, 0), device=device).solve(
    stack,
    tw.PlaneWave(polarization="x"),
)

table = result.diffraction_table(input_polarization="x")
print("orders:", table["orders"].cpu())
print("T:", table["T"].detach().cpu())
print("R:", table["R"].detach().cpu())
print("balance:", result.power_balance(input_polarization="x"))

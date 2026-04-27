import torch

import torcwa as tw


device = "cuda" if torch.cuda.is_available() else "cpu"

stack = tw.Stack(period=(300.0, 300.0))
stack.add_layer(thickness=120.0, eps=2.25)

solver = tw.RCWA(wavelength=500.0, orders=(0, 0), device=device)
wavelength = torch.linspace(450.0, 700.0, 16, device=device)

sweep = solver.sweep(
    stack,
    source=tw.PlaneWave(polarization="s"),
    wavelength=wavelength,
    outputs=[
        tw.Output.transmission(order=(0, 0), polarization="s", name="T00"),
        tw.Output.reflection(order=(0, 0), polarization="s", name="R00"),
    ],
)

print("wavelength:", wavelength.detach().cpu())
print("T00:", torch.abs(sweep["T00"].reshape(-1)).detach().cpu() ** 2)
print("R00:", torch.abs(sweep["R00"].reshape(-1)).detach().cpu() ** 2)

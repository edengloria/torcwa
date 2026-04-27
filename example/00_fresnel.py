import torch

import torcwa as tw


device = "cuda" if torch.cuda.is_available() else "cpu"

stack = tw.Stack(period=(300.0, 300.0), output_eps=2.25)
solver = tw.RCWA(wavelength=500.0, orders=(0, 0), dtype=torch.complex128, device=device)
source = tw.PlaneWave(angle=(0.0, 0.0), polarization="s")

result = solver.solve(stack, source)

print("t00:", result.transmission(polarization="s", power=False))
print("r00:", result.reflection(polarization="s", power=False))
print("power:", result.power_balance(input_polarization="s"))

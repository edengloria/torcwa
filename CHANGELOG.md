# Changelog

## 0.2.0.dev0

- Added the `torcwa.v2` developer-preview API with dataclass configuration,
  reusable linalg/physics/eig helpers, and an `RCWASolver` facade.
- Raised the target runtime to Python 3.10+ and PyTorch 2.11+.
- Made `S_parameters(..., evanescent=...)` the canonical spelling while keeping
  the legacy `evanscent` alias with a deprecation warning.
- Replaced key explicit inverse products in the legacy numerical path with
  `torch.linalg.solve`-based helpers.
- Replaced selected explicit diagonal products with broadcasting in source and
  field reconstruction paths.
- Kept eig backward on the active device and added a v2 stabilized eig helper
  with finite-gradient and `torch.func.vmap` smoke coverage.
- Fixed empty-stack reflection S-parameter queries by initializing zero
  reflection blocks as matrices.
- Added pytest coverage for analytical Fresnel checks, lossless power
  conservation, typo alias compatibility, v2 facade compatibility, finite field
  reconstruction, and finite geometry gradients.
- Added `benchmarks/v2_microbench.py` and v2 QA/migration documentation.

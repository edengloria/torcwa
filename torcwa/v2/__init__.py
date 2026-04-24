"""Accuracy-first TORCWA v2 API.

The v2 package is intentionally additive.  The legacy ``torcwa.rcwa`` class
remains available while the new stateless pieces are validated against
analytical, legacy, and external-solver references.
"""

from .config import (
    FourierBasis,
    LayerSpec,
    MaterialGrid,
    PortSpec,
    RCWAConfig,
    SolverOptions,
)
from .eig import stabilized_eig
from .linalg import (
    block_2x2,
    diag_post_multiply,
    diag_pre_multiply,
    identity_like,
    solve_left,
    solve_right,
)
from .physics import (
    diffraction_order_indices,
    fresnel_amplitudes,
    kz_branch,
    propagating_mask,
)
from .solver import RCWASolver

__all__ = [
    "FourierBasis",
    "LayerSpec",
    "MaterialGrid",
    "PortSpec",
    "RCWAConfig",
    "RCWASolver",
    "SolverOptions",
    "block_2x2",
    "diag_post_multiply",
    "diag_pre_multiply",
    "diffraction_order_indices",
    "fresnel_amplitudes",
    "identity_like",
    "kz_branch",
    "propagating_mask",
    "solve_left",
    "solve_right",
    "stabilized_eig",
]

"""Internal numerical building blocks for the modern TORCWA API.

The public ``torcwa.RCWA`` API uses these module boundaries while the validated
legacy solver remains available for compatibility.
"""

from .basis import FourierBasis, order_vectors
from .eig import stabilized_eig
from .fourier import material_convolution_apply, material_convolution_dense
from .linalg import (
    block_2x2,
    diag_post_multiply,
    diag_pre_multiply,
    identity_like,
    lu_factor_left,
    lu_solve_left,
    solve_left,
    solve_left_many,
    solve_right,
)
from .physics import diffraction_order_indices, fresnel_amplitudes, kz_branch, propagating_mask

__all__ = [
    "FourierBasis",
    "block_2x2",
    "diag_post_multiply",
    "diag_pre_multiply",
    "diffraction_order_indices",
    "fresnel_amplitudes",
    "identity_like",
    "kz_branch",
    "lu_factor_left",
    "lu_solve_left",
    "material_convolution_apply",
    "material_convolution_dense",
    "order_vectors",
    "propagating_mask",
    "solve_left",
    "solve_left_many",
    "solve_right",
    "stabilized_eig",
]

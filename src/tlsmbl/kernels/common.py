"""Operand sanity gate: INV-4 (ARCHITECTURE.md §3).

Every matricized compression operand is checked before a kernel consumes it: finite
entries always; if flagged as a norm object, Hermiticity after symmetrization.
"""

from __future__ import annotations

import torch

from tlsmbl.core.guards import NumericalCorruption

_HERMITICITY_RTOL = 1e-10


def check_operand(Wmat: torch.Tensor, *, is_norm_object: bool = False) -> torch.Tensor:
    """INV-4 gate. Returns the (possibly symmetrized) operand or raises."""
    if not torch.isfinite(Wmat.real).all() or (
        Wmat.is_complex() and not torch.isfinite(Wmat.imag).all()
    ):
        raise NumericalCorruption("INV-4: non-finite entries in compression operand")
    if is_norm_object:
        herm_gap = torch.linalg.norm(Wmat - Wmat.mH)
        if herm_gap > _HERMITICITY_RTOL * torch.linalg.norm(Wmat):
            raise NumericalCorruption(
                f"INV-4: operand flagged as norm object but ||rho - rho^H||_F "
                f"= {float(herm_gap):.3e} exceeds {_HERMITICITY_RTOL} * ||rho||_F"
            )
        Wmat = (Wmat + Wmat.mH) / 2
    return Wmat

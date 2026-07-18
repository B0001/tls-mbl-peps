"""Internal unit system (ARCHITECTURE.md §2).

Internal convention: hbar = 1, all energies in units of W, all lengths in units of the
coarse-graining constant a. Physical (SI) constants are converted exactly once, at
ingestion, into the two dimensionless numbers the solver consumes. No SI `Quantity`
may appear anywhere past that boundary -- the tensor layers (peps/, kernels/) must
never import this module.
"""

from __future__ import annotations

from typing import NewType

Quantity = NewType("Quantity", float)
"""A physical (SI-tagged) scalar. Distinct from a plain `float` so that a stray SI value
flowing into the dimensionless solver layers is a type error, not a silent unit bug."""


def dimensionless_coupling(P0: Quantity, U0: Quantity) -> float:
    """g_J = P0 * U0 (§2): the dimensionless dipolar coupling scale J(a)/W."""
    return float(P0) * float(U0)


def coarse_graining_length(P0: Quantity, t: Quantity, W: Quantity) -> float:
    """a = (P0 * t * W)^(-1/2) (§2): the internal length unit."""
    return float((float(P0) * float(t) * float(W)) ** -0.5)

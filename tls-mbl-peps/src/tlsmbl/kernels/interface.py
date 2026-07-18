"""Truncation-backend interface (ARCHITECTURE.md §8.3).

Every truncation kernel -- exact reference, sketched, and later Rust -- consumes the
matricized E-5 operand `Wmat` and returns a `TruncResult`. The reference (`ExactSVD`)
is the oracle; equivalence tests (§14.7) bind the others to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TruncResult:
    """U (m x chi'), S (chi',), Vh (chi' x n) with chi' <= chi; the discarded weight
    feeding INV-1; the posterior error estimate for sketched backends (INV-3), None
    for exact."""

    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    disc_weight: float
    posterior_err: float | None = None


class TruncationBackend(Protocol):
    def truncate(self, Wmat: torch.Tensor, chi: int) -> TruncResult: ...

"""Shared data model (ARCHITECTURE.md §5) and dtype threading (§4 dtype policy).

`complex128` everywhere on certification paths; `complex64` allowed only in `bench`
mode. Every tensor-producing call takes a `TensorSpec` and reads `.dtype` from it --
no module hardcodes `torch.complex128`/`torch.complex64` directly.

Conventions (§6, CLAUDE.md gotchas): a site is `(x, y)` with x = column, y = row,
both 0-based; the linear site index is `s = y*L + x` (row-major); site 0 is the
slowest bit / first Kronecker factor; Z = diag(1, -1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

Site = tuple[int, int]
"""(x, y): column, row -- 0-based."""


def site_index(site: Site, L: int) -> int:
    """Row-major linear index s = y*L + x (§6); site 0 = slowest bit in the ED basis."""
    x, y = site
    return y * L + x


class ModelParams(BaseModel):
    """Per-realization physics parameters (§5). `seed_realization` is spawned via
    core/rng.py from (master_seed, realization_index), never chosen by hand."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    L: int = Field(gt=0)
    W: float = 1.0  # always 1.0 internally; kept for clarity (§2)
    delta_min: float = Field(default=1e-3, gt=0, lt=1)
    g_J: float = Field(default=1e-3, gt=0)
    R_c: int = Field(default=3, gt=0)
    polaron_kappa: float = Field(default=0.0, ge=0)
    seed_realization: int


@dataclass(frozen=True)
class DisorderRealization:
    """One disorder draw (§5). Arrays are indexed [y, x] (row, column); `J` is keyed by
    canonically ordered (site_a, site_b) pairs with a < b in row-major order, and its
    iteration order is sorted (reproducibility)."""

    params: ModelParams
    eps: np.ndarray  # (L, L) float64, eps/W in [-1, 1]
    delta: np.ndarray  # (L, L) float64, polaron-renormalized Delta~/W
    J: dict[tuple[Site, Site], float]
    h_mf: np.ndarray  # (L, L) float64 Hartree tail field
    rng_fingerprint: str

    def __post_init__(self) -> None:
        L = self.params.L
        for name in ("eps", "delta", "h_mf"):
            arr = getattr(self, name)
            if arr.shape != (L, L):
                raise ValueError(f"{name} shape {arr.shape} != ({L}, {L})")


@dataclass(frozen=True)
class HamiltonianTerms:
    """Canonical term list (§5): the ONLY Hamiltonian representation. ED, PEPS energy,
    and SDRG all consume this."""

    L: int
    onsite: list[tuple[Site, Literal["z", "x"], float]]
    pair: list[tuple[Site, Site, float]]  # zz only in v1
    norm_local: float = field(init=False)  # Sum|coeff|, INV-8 threshold input

    def __post_init__(self) -> None:
        total = sum(abs(c) for _, _, c in self.onsite) + sum(abs(J) for _, _, J in self.pair)
        object.__setattr__(self, "norm_local", float(total))

Mode = Literal["certified", "bench"]

_DTYPE_BY_MODE: dict[Mode, torch.dtype] = {
    "certified": torch.complex128,
    "bench": torch.complex64,
}
_REAL_DTYPE_BY_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
}


@dataclass(frozen=True)
class TensorSpec:
    """The dtype/device a run's tensors are constructed with. One instance per run,
    threaded down through every constructor -- never rebuilt ad hoc mid-pipeline."""

    mode: Mode = "certified"
    device: str = "cpu"

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPE_BY_MODE[self.mode]

    @property
    def real_dtype(self) -> torch.dtype:
        return _REAL_DTYPE_BY_COMPLEX[self.dtype]

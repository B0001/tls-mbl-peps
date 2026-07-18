"""Hamiltonian assembly (ARCHITECTURE.md §7.2).

H = sum_i (eps_i + h_mf_i)/2 sigma^z_i + Delta~_i/2 sigma^x_i
    + sum_{(i,j): r<=R_c} J_ij sigma^z_i sigma^z_j

emitted as `HamiltonianTerms` -- the single source of truth consumed by ED, PEPS
energy assembly, and SDRG alike.
"""

from __future__ import annotations

from typing import Literal

from tlsmbl.core.types import DisorderRealization, HamiltonianTerms, Site, site_index


def build_terms(real: DisorderRealization) -> HamiltonianTerms:
    L = real.params.L
    onsite: list[tuple[Site, Literal["z", "x"], float]] = []
    for y in range(L):
        for x in range(L):
            onsite.append(((x, y), "z", (real.eps[y, x] + real.h_mf[y, x]) / 2.0))
            onsite.append(((x, y), "x", real.delta[y, x] / 2.0))
    # Canonical pair order = row-major site index (§5/§6), NOT lexicographic (x, y):
    # downstream floating-point sums must be order-stable across the codebase.
    pair = [
        (i, j, J)
        for (i, j), J in sorted(
            real.J.items(), key=lambda kv: (site_index(kv[0][0], L), site_index(kv[0][1], L))
        )
    ]
    return HamiltonianTerms(L=L, onsite=onsite, pair=pair)

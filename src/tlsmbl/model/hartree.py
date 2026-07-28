"""Hartree tail bound, INV-5 (ARCHITECTURE.md §3, §7.4).

v1 default is the pure-truncation baseline: h_mf = 0, K_max = 1, with the rigorous
tail bound reported regardless. The self-consistency loop itself lands in Phase 5
behind `config.model.hartree.enabled`.
"""

from __future__ import annotations

import math


def tail_bound(g_J: float, R_c: int) -> float:
    """Rigorous per-site bound on the neglected r > R_c dipolar tail (lattice form):
    delta_e_tail = g_J * sum_{r > R_c} r^-3 ~= 2*pi*g_J / R_c."""
    return 2.0 * math.pi * g_J / R_c


def tail_certified(g_J: float, R_c: int, e_per_site: float, tau_tail: float) -> bool:
    """INV-5 gate: the bound must be small relative to the reported energy, else the
    artifact is marked UNCERTIFIED."""
    return tail_bound(g_J, R_c) <= tau_tail * abs(e_per_site)

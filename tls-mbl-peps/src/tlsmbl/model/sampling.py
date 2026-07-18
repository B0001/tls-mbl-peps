"""Disorder sampling (ARCHITECTURE.md §7.1).

The RNG draw order is normative and matches the executed prototype
(`prototypes/golden_3x3.py::sample_realization`) bitwise for the same generator:
(1) eps as one (L, L) uniform block, (2) delta as one (L, L) log-uniform block,
(3) optional polaron block only if kappa > 0, (4) one uniform draw per qualifying
pair, iterating row-major sites with a < b. tests/golden/ enforces this parity;
do not reorder draws.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from tlsmbl.core.types import DisorderRealization, ModelParams, Site


def _qualifying_pairs(L: int, R_c: int) -> list[tuple[Site, Site, float]]:
    """Canonically ordered site pairs with 1 <= r <= R_c (Euclidean), plus r."""
    sites = [(x, y) for y in range(L) for x in range(L)]
    out: list[tuple[Site, Site, float]] = []
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            (x1, y1), (x2, y2) = sites[a], sites[b]
            r = math.hypot(x1 - x2, y1 - y2)
            if 1.0 <= r <= R_c:
                out.append((sites[a], sites[b], r))
    return out


def sample_realization(
    params: ModelParams, rng: np.random.Generator
) -> DisorderRealization:
    """One disorder draw. `rng` comes from core/rng.realization_streams (INV-6)."""
    L = params.L
    eps = rng.uniform(-1.0, 1.0, (L, L))
    delta = np.exp(rng.uniform(np.log(params.delta_min), 0.0, (L, L)))
    if params.polaron_kappa > 0:
        delta = delta * np.exp(-params.polaron_kappa * rng.uniform(0.0, 1.0, (L, L)))
    J: dict[tuple[Site, Site], float] = {}
    for site_a, site_b, r in _qualifying_pairs(L, params.R_c):
        J[(site_a, site_b)] = params.g_J * rng.uniform(-1.0, 1.0) / r**3
    fingerprint = hashlib.sha256(
        eps.tobytes() + delta.tobytes() + np.array(sorted(J.values())).tobytes()
    ).hexdigest()
    return DisorderRealization(
        params=params,
        eps=eps,
        delta=delta,
        J=J,
        h_mf=np.zeros((L, L)),
        rng_fingerprint=fingerprint,
    )

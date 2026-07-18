"""Initialization (ARCHITECTURE.md §10.1).

Product state from the local ground directions of the (possibly SDRG-transformed)
on-site terms, D-padded with gauge-breaking noise. Simple-update warmup is a later
option behind config (`su_steps`); §10.1's product init is empirically validated
(prototype: LBFGS from product init reached 7e-9 of ED at g_J=1e-3).
"""

from __future__ import annotations

import numpy as np

from tlsmbl.core.types import DisorderRealization, TensorSpec
from tlsmbl.peps.state import PEPSState


def product_init(
    real: DisorderRealization,
    D: int,
    spec: TensorSpec,
    seed_seq: np.random.SeedSequence,
    noise: float = 1e-2,
) -> PEPSState:
    L = real.params.L
    vectors: list[list[np.ndarray]] = []
    for y in range(L):
        row = []
        for x in range(L):
            eps = real.eps[y, x] + real.h_mf[y, x]
            delta = real.delta[y, x]
            h2 = np.array([[eps / 2, delta / 2], [delta / 2, -eps / 2]])
            row.append(np.linalg.eigh(h2)[1][:, 0].astype(np.complex128))
        vectors.append(row)
    return PEPSState.from_product(vectors, D, spec, seed_seq, noise=noise)

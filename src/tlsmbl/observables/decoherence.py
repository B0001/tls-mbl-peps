"""Tier-2 decoherence estimates (ARCHITECTURE.md §12) -- model-dependent.

Every model input is echoed verbatim into the output and the fixed DISCLAIMER
travels with it: the static solver cannot derive the TLS relaxation rates
gamma_i; they come from the declared phenomenological model
gamma_i = gamma0 * (E_i/W)^3 * coth(E_i / 2T). This is the only module (with
io/) allowed to convert back to physical units (§2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tlsmbl.core.types import DisorderRealization

DISCLAIMER = (
    "Tier-2 output: Gamma_1 depends on the declared phenomenological inputs "
    "(g0, gamma0, T, omega_q) echoed in this record; the static solver cannot "
    "derive TLS relaxation rates. Tier-1 observables are the rigorous outputs."
)


@dataclass(frozen=True)
class Tier2Inputs:
    omega_q: float  # qubit frequency, units of W
    g0: float  # coupling scale, units of W
    gamma0: float  # relaxation scale, units of W
    T: float  # temperature, units of W


@dataclass(frozen=True)
class Gamma1Estimate:
    gamma_1: float  # units of W
    inputs: Tier2Inputs  # echoed verbatim
    n_fluctuators: int
    disclaimer: str


def gamma_1(real: DisorderRealization, inputs: Tier2Inputs) -> Gamma1Estimate:
    """Gamma_1(omega_q) = sum_i 2 g_i^2 gamma_i / (gamma_i^2 + (E_i - omega_q)^2),
    g_i = g0 * (Delta~_i / E_i), gamma_i = gamma0 (E_i/W)^3 coth(E_i / 2T)."""
    E = np.sqrt((real.eps + real.h_mf) ** 2 + real.delta**2)
    W = real.params.W
    total = 0.0
    for Ei, Di in zip(E.ravel(), real.delta.ravel()):
        gi = inputs.g0 * (Di / Ei)
        gam = inputs.gamma0 * (Ei / W) ** 3 / math.tanh(Ei / (2 * inputs.T))
        total += 2 * gi**2 * gam / (gam**2 + (Ei - inputs.omega_q) ** 2)
    return Gamma1Estimate(
        gamma_1=float(total),
        inputs=inputs,
        n_fluctuators=E.size,
        disclaimer=DISCLAIMER,
    )

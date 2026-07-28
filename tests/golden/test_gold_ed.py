"""T-GOLD-ED (§14.1, blocking Phase 2): LBFGS-optimized PEPS energies certified
against the stored ED fixtures (ADR-014 seed convention).

Weak coupling (g_J = 1e-3, near-product state) is the fast blocking gate: if it
fails, wiring is wrong, not physics. The strong-coupling case runs under
TLSMBL_FULL_GOLD=1 (optimizer-floor limited, minutes)."""

import json
import os

import numpy as np
import pytest

from gen_fixtures import FIXTURES, MASTER, fixture_name
from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.optimize.init import product_init
from tlsmbl.optimize.lbfgs_driver import optimize_lbfgs

SPEC = TensorSpec()
BACKEND = ExactSVD(eps_F=1e-12)


def _setup(L: int, g_J: float, k: int):
    params = ModelParams(L=L, g_J=g_J, R_c=3, seed_realization=k)
    real = sample_realization(params, realization_streams(MASTER, k).disorder)
    e0 = json.loads((FIXTURES / fixture_name(L, g_J, k)).read_text())["energies"][0]
    return real, build_terms(real), e0


def test_gold_ed_weak_coupling_L3() -> None:
    """§14.1a: weak g_J, D=2, chi=D^2: optimized energy within 1e-7 of ED and
    variational-bound-respecting (prototype parity: 7.2e-9 at 203 iters)."""
    real, terms, e0 = _setup(3, 1e-3, 0)
    state = product_init(real, 2, SPEC, np.random.SeedSequence(1))
    res = optimize_lbfgs(
        state, terms, chi=4, backend=BACKEND, max_outer=30, inner_iters=20
    )
    rel = abs(res.energy - e0) / abs(e0)
    assert res.energy >= e0 - 1e-9  # variational bound
    assert rel < 1e-7, f"rel gap {rel:.3e} after {res.n_iters} iters"


@pytest.mark.skipif(
    os.environ.get("TLSMBL_FULL_GOLD") != "1", reason="strong-coupling gate is slow"
)
def test_gold_ed_strong_coupling_L3() -> None:
    """§14.1b regime at prototype-parity thresholds (2.6e-7 measured at D=2; gate
    at 1e-5 to absorb optimizer-floor variation within a bounded iteration budget)."""
    real, terms, e0 = _setup(3, 0.3, 0)
    state = product_init(real, 2, SPEC, np.random.SeedSequence(1))
    res = optimize_lbfgs(
        state, terms, chi=4, backend=BACKEND, max_outer=100, inner_iters=20
    )
    rel = abs(res.energy - e0) / abs(e0)
    assert res.energy >= e0 - 1e-9
    assert rel < 1e-5, f"rel gap {rel:.3e} after {res.n_iters} iters"

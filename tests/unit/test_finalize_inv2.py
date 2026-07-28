"""T-INV-2 (§14.8): the chi-stability audit stamps certified=False on an unstable
environment and preserves certification on a stable one -- without raising."""

import numpy as np

from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.optimize.finalize import chi_extrapolation_check
from tlsmbl.peps.energy import energy_certified
from tlsmbl.peps.state import PEPSState

SPEC = TensorSpec()


def _setup(seed: int = 21):
    params = ModelParams(L=3, g_J=0.3, R_c=3, seed_realization=seed)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(seed)))
    return build_terms(real), PEPSState.random(3, 2, SPEC, np.random.SeedSequence(seed))


def test_inv2_stable_at_lossless_chi() -> None:
    terms, state = _setup()
    report = energy_certified(
        state, terms, chi=4, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7
    )
    out = chi_extrapolation_check(
        state, terms, report, ExactSVD(), tau_chi=1e-6, eps_env=1e-8, eps_env_E=1e-7
    )
    assert out.certified
    e1, e2 = out.chi_stability
    assert abs(e1 - e2) <= 1e-6


def test_inv2_marks_uncertified_on_instability() -> None:
    """Random truncating state: E(chi) vs E(2chi) differ at O(disc) scale; with the
    INV-1 gates loosened to let chi=2 through, INV-2 must catch it and mark the
    artifact uncertified rather than raise."""
    terms, state = _setup()
    report = energy_certified(
        state, terms, chi=2, backend=ExactSVD(), eps_env=1.0, eps_env_E=10.0
    )
    out = chi_extrapolation_check(
        state, terms, report, ExactSVD(), tau_chi=1e-10, eps_env=1.0, eps_env_E=10.0
    )
    assert not out.certified
    e1, e2 = out.chi_stability
    assert abs(e1 - e2) > 1e-10

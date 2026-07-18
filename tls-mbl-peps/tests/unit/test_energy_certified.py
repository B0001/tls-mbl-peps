"""T-INV-1 / T-INV-7 (§14.8): the certified-energy factory's gates fire, and the
report is impossible to mint without them."""

import numpy as np
import pytest
import torch

from tlsmbl.core.guards import NumericalCorruption
from tlsmbl.core.types import HamiltonianTerms, ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.energy import (
    EnergyReport,
    EnvironmentNotConverged,
    energy_certified,
)
from tlsmbl.peps.state import PEPSState

SPEC = TensorSpec()


def _setup(D: int = 2, L: int = 3, seed: int = 11):
    params = ModelParams(L=L, g_J=0.3, R_c=3, seed_realization=seed)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(seed)))
    terms = build_terms(real)
    state = PEPSState.random(L, D, SPEC, np.random.SeedSequence(seed))
    return state, terms


def test_certified_report_minted_at_lossless_chi() -> None:
    state, terms = _setup()
    report = energy_certified(
        state, terms, chi=4, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7
    )
    assert report.certified
    assert report.env.max_disc_weight <= 1e-8
    assert report.env.updown_gap <= 1e-7
    assert report.e_per_site == report.e_total / 9


def test_inv1_disc_weight_gate_fires_on_truncating_chi() -> None:
    """chi=1 on a random D=2 state discards real weight -> INV-1 must refuse."""
    state, terms = _setup()
    with pytest.raises(EnvironmentNotConverged, match="INV-1"):
        energy_certified(
            state, terms, chi=1, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7
        )


def test_report_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="energy_certified"):
        EnergyReport(
            e_total=0.0,
            e_per_site=0.0,
            env=None,  # type: ignore[arg-type]
            tail_bound=0.0,
            chi_stability=None,
            grad_norm=0.0,
            n_iters=0,
            wall_s=0.0,
            certified=True,
        )


def test_inv7_nan_in_state_raises_with_provenance() -> None:
    state, terms = _setup()
    bad = state.tensors[0][0].clone()
    bad[0, 0, 0, 0, 0] = torch.nan
    state.tensors[0][0] = bad
    with pytest.raises(NumericalCorruption):
        energy_certified(
            state, terms, chi=4, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7
        )


def test_free_case_certified_energy_is_variational() -> None:
    """Certified energy must sit above the ED ground energy (sanity of assembly)."""
    from tlsmbl.model.ed_reference import ed_ground

    state, terms = _setup()
    report = energy_certified(
        state, terms, chi=4, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7
    )
    assert report.e_total >= ed_ground(terms).energies[0] - 1e-9


def _dummy_terms() -> HamiltonianTerms:
    return HamiltonianTerms(L=2, onsite=[((0, 0), "z", 1.0)], pair=[])
"""P1 parity: the production model layer reproduces the executed prototype
(`prototypes/golden_3x3.py`) bitwise on sampling and to solver precision on ED.
The prototype is the oracle (CLAUDE.md: production reproduces its numbers through
tests/golden/)."""

import numpy as np
import pytest

import golden_3x3 as proto  # prototypes/, via conftest sys.path
from tlsmbl.core.types import ModelParams
from tlsmbl.model.ed_reference import build_H, ed_ground, free_energy_analytic
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization

SEEDS = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(proto.MASTER).spawn(2)]


def _production_realization(seed: int, g_J: float):
    params = ModelParams(L=3, g_J=g_J, R_c=3, delta_min=1e-3, seed_realization=seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return sample_realization(params, rng)


@pytest.mark.parametrize("g_J", [1e-3, 0.3])
@pytest.mark.parametrize("seed", SEEDS)
def test_sampler_bitwise_parity_with_prototype(seed: int, g_J: float) -> None:
    ours = _production_realization(seed, g_J)
    theirs = proto.sample_realization(seed, g_J)
    np.testing.assert_array_equal(ours.eps, theirs["eps"])
    np.testing.assert_array_equal(ours.delta, theirs["delta"])
    assert ours.J == theirs["pairs"]


@pytest.mark.parametrize("g_J", [1e-3, 0.3])
@pytest.mark.parametrize("seed", SEEDS)
def test_ed_ground_energy_parity_with_prototype(seed: int, g_J: float) -> None:
    terms = build_terms(_production_realization(seed, g_J))
    ours = ed_ground(terms).energies[0]
    onsite, pair = proto.build_terms(proto.sample_realization(seed, g_J))
    theirs = float(np.linalg.eigh(proto.build_H(onsite, pair).toarray())[0][0])
    assert abs(ours - theirs) < 1e-12


def test_hamiltonian_matrix_identical_to_prototype() -> None:
    seed, g_J = SEEDS[0], 0.3
    H_ours = build_H(build_terms(_production_realization(seed, g_J)))
    onsite, pair = proto.build_terms(proto.sample_realization(seed, g_J))
    H_theirs = proto.build_H(onsite, pair)
    assert abs(H_ours - H_theirs).max() == 0.0


def test_free_energy_analytic_matches_ed() -> None:
    real = _production_realization(SEEDS[0], 1e-3)
    zeroJ = build_terms(real)
    terms = type(zeroJ)(L=zeroJ.L, onsite=zeroJ.onsite, pair=[])
    e_ed = ed_ground(terms).energies[0]
    assert abs(e_ed - free_energy_analytic(real.eps, real.delta)) < 1e-10

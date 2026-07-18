"""P2 contraction parity: the torch PEPS engine reproduces the executed prototype
(golden_3x3.py) at lossless chi = D^2 on 3x3 -- norm, site observables, all pair
correlators, and total energy against the brute-force statevector oracle."""

import numpy as np
import pytest
import torch

import golden_3x3 as proto
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.boundary import build_env
from tlsmbl.peps.energy import Z, energy_differentiable, sandwich
from tlsmbl.peps.state import PEPSState

SEED = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(proto.MASTER).spawn(2)][0]
SPEC = TensorSpec()


def _state(D: int, seed: int) -> PEPSState:
    return PEPSState.random(3, D, SPEC, np.random.SeedSequence(seed))


def _terms(g_J: float):
    params = ModelParams(L=3, g_J=g_J, R_c=3, seed_realization=SEED)
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    return build_terms(sample_realization(params, rng))


@pytest.mark.parametrize("D", [2, 3])
def test_random_state_bitwise_matches_prototype(D: int) -> None:
    ours = _state(D, SEED + 7 * D)
    theirs = proto.random_peps(D, SEED + 7 * D)
    for y in range(3):
        for x in range(3):
            np.testing.assert_array_equal(ours.tensors[y][x].numpy(), theirs[y][x])


@pytest.mark.parametrize("g_J", [1e-3, 0.3])
@pytest.mark.parametrize("D", [2, 3])
def test_energy_matches_brute_force_oracle(D: int, g_J: float) -> None:
    state = _state(D, SEED + 7 * D)
    terms = _terms(g_J)
    E = float(energy_differentiable(state, terms, chi=D * D, backend=ExactSVD()))
    A = proto.random_peps(D, SEED + 7 * D)
    onsite, pair = proto.build_terms(proto.sample_realization(SEED, g_J))
    bf = proto.brute_observables(A, onsite, pair)
    assert abs(E - bf["E"]) / max(abs(bf["E"]), 1e-12) < 1e-9


@pytest.mark.parametrize("D", [2, 3])
def test_observables_match_brute_force(D: int) -> None:
    state = _state(D, SEED + 7 * D)
    backend = ExactSVD()
    chi = D * D
    env = build_env(state, chi, backend, want_grad=False)
    assert max(env.disc_weights) < 1e-24  # provably lossless at chi = D^2 on 3x3
    A = proto.random_peps(D, SEED + 7 * D)
    onsite, pair = proto.build_terms(proto.sample_realization(SEED, 0.3))
    bf = proto.brute_observables(A, onsite, pair)
    with torch.no_grad():
        for y in range(3):
            norm = sandwich(env.tops[y], state, y, env.bottoms[y + 1])
            for x in range(3):
                sz = sandwich(env.tops[y], state, y, env.bottoms[y + 1], {x: Z}) / norm
                assert abs(float(sz.real) - bf["sz"][(x, y)]) < 1e-9

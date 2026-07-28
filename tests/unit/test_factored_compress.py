"""ADR-015 equivalence gate: the factored (ADR-007 v1.1) compression must
reproduce the v1 explicit path -- same compressed states, same INV-1 discarded
weights, same energies and gradients -- on lossless AND genuinely truncating
configurations, both sweep orientations, and through the batched dressed path.
This test is the guard ARCHITECTURE.md §8.6 requires before activating v1.1."""

import numpy as np
import pytest
import torch

from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.factored import compress_factored
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.kernels.zipup import compress
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps import boundary as boundary_mod
from tlsmbl.peps.boundary import absorb_row_top, build_env, build_tops
from tlsmbl.peps.energy import energy_certified, energy_differentiable
from tlsmbl.peps.state import PEPSState

SEED = 20260726
SPEC = TensorSpec()


def _setup(L: int = 3, D: int = 2):
    params = ModelParams(L=L, g_J=0.3, R_c=3, seed_realization=SEED)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(SEED)))
    terms = build_terms(real)
    state = PEPSState.random(L, D, SPEC, np.random.SeedSequence(SEED))
    return state, terms


def _overlap(A: list[torch.Tensor], B: list[torch.Tensor]) -> complex:
    E = torch.ones(1, 1, dtype=A[0].dtype)
    for a, b in zip(A, B):
        E = torch.einsum("ab,avr,bvs->rs", E, a.conj(), b)
    return complex(E.reshape(()))


def _fidelity(A: list[torch.Tensor], B: list[torch.Tensor]) -> float:
    ov = _overlap(A, B)
    return abs(ov) ** 2 / (abs(_overlap(A, A)) * abs(_overlap(B, B)))


@pytest.mark.parametrize("chi,truncating", [(8, False), (3, True)])
def test_single_row_compress_matches_v1(chi: int, truncating: bool) -> None:
    """One row absorption+compression, factored vs materialized, on a mid-lattice
    boundary MPS (nontrivial bonds both sides)."""
    state, _ = _setup(L=4, D=2)
    tops, _ = build_tops(state, 4, ExactSVD(), want_grad=False)
    base = tops[2]  # two rows absorbed, bonds <= 4
    row = [boundary_mod.double_layer(state.tensors[2][x]) for x in range(4)]

    fat = absorb_row_top(base, row)
    out1, s1 = compress(fat, chi, ExactSVD(), want_grad=False)
    out2, s2 = compress_factored(base.tensors, row, chi, ExactSVD(), want_grad=False)

    if truncating:
        assert s1.max_disc_weight > 1e-8  # the config genuinely truncates
    assert _fidelity(out1, out2) == pytest.approx(1.0, abs=1e-9)
    assert abs(s1.log_norm - s2.log_norm) < 1e-8
    np.testing.assert_allclose(s2.disc_weights, s1.disc_weights, rtol=1e-5, atol=1e-11)


@pytest.mark.parametrize("chi", [8, 3])
def test_build_env_matches_v1_both_orientations(chi: int) -> None:
    """Full environment build (tops AND bottoms -- both absorption conventions),
    per-level state fidelity and INV-1 disc weights."""
    state, _ = _setup(L=4, D=2)
    env1 = build_env(state, chi, ExactSVD(), want_grad=False)
    env2 = build_env(state, chi, ExactSVD(), want_grad=False, factored=True)
    for m1, m2 in zip(env1.tops[1:] + env1.bottoms[:-1], env2.tops[1:] + env2.bottoms[:-1]):
        assert _fidelity(m1.tensors, m2.tensors) == pytest.approx(1.0, abs=1e-8)
        assert abs(m1.log_norm - m2.log_norm) < 1e-7
    np.testing.assert_allclose(
        env2.disc_weights, env1.disc_weights, rtol=1e-4, atol=1e-10
    )


@pytest.mark.parametrize("chi", [4, 2])
def test_energy_and_gradient_match_v1(chi: int) -> None:
    """End-to-end: full differentiable energy (onsite + same-row + batched
    cross-row dressing) and its gradient, factored vs v1, lossless (chi=4) and
    truncating (chi=2)."""
    state, terms = _setup(L=3, D=2)
    grads = []
    energies = []
    for factored in (False, True):
        for row in state.tensors:
            for t in row:
                t.grad = None
                t.requires_grad_(True)
        E = energy_differentiable(state, terms, chi, ExactSVD(), factored=factored)
        E.backward()
        energies.append(float(E.detach()))
        grads.append(
            torch.cat(
                [state.tensors[y][x].grad.flatten() for y in range(3) for x in range(3)]
            )
        )
    assert energies[1] == pytest.approx(energies[0], rel=1e-8, abs=1e-10)
    gap = float(torch.linalg.norm(grads[1] - grads[0]) / torch.linalg.norm(grads[0]))
    assert gap < 1e-6, f"gradient rel gap {gap:.3e}"


def test_factored_with_sketched_backend() -> None:
    """§8.6's actual target pairing: the sketched kernel consuming the factored
    operand W~ (T-EQ-BACKENDS tolerance, same as the v1 pipeline half)."""
    from tlsmbl.kernels.rsvd import SketchedSVD

    state, terms = _setup(L=3, D=3)
    e_exact = float(
        energy_differentiable(state, terms, 9, ExactSVD(), factored=True).detach()
    )
    sketched = SketchedSVD(seed=SEED)
    e_sketch = float(
        energy_differentiable(state, terms, 9, sketched, factored=True).detach()
    )
    assert abs(e_exact - e_sketch) <= 1e-5, f"{e_exact} vs {e_sketch}"
    assert sketched.call_count > 0


def test_energy_certified_factored_and_no_fat_materialization(monkeypatch) -> None:
    """The certified path (both dressing directions, batched extends) agrees, and
    the factored path never calls the fat-row absorption -- the structural
    guarantee that Theta(chi^2 D^6) tensors are gone."""
    state, terms = _setup(L=3, D=2)
    r1 = energy_certified(
        state, terms, 4, ExactSVD(), eps_env=1e-6, eps_env_E=1e-6
    )

    def _boom(*a, **k):
        raise AssertionError("fat-row absorption called on the factored path")

    monkeypatch.setattr(boundary_mod, "absorb_row_top", _boom)
    monkeypatch.setattr(boundary_mod, "absorb_row_bottom", _boom)
    r2 = energy_certified(
        state, terms, 4, ExactSVD(), eps_env=1e-6, eps_env_E=1e-6, factored=True
    )
    assert r2.e_total == pytest.approx(r1.e_total, rel=1e-9, abs=1e-11)
    assert r2.env.updown_gap < 1e-6

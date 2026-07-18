"""T-AD-FD (§14.4, blocking Phase 2): central finite differences vs autograd through
the full differentiable energy graph on L=3, D=2 -- at lossless chi=4 AND a genuinely
truncating chi=2 (parity contract, docs/HANDOFF.md)."""

import numpy as np
import pytest
import torch

from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.boundary import build_env
from tlsmbl.peps.energy import energy_differentiable
from tlsmbl.peps.state import PEPSState

SEED = 20260717
SPEC = TensorSpec()


def _setup():
    params = ModelParams(L=3, g_J=0.3, R_c=3, seed_realization=SEED)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(SEED)))
    terms = build_terms(real)
    state = PEPSState.random(3, 2, SPEC, np.random.SeedSequence(SEED))
    return state, terms


def _energy_fn(state: PEPSState, terms, chi: int):
    def f() -> torch.Tensor:
        return energy_differentiable(state, terms, chi, ExactSVD())

    return f


@pytest.mark.parametrize("chi,expect_truncating", [(4, False), (2, True)])
def test_ad_matches_central_fd(chi: int, expect_truncating: bool) -> None:
    state, terms = _setup()
    disc = max(build_env(state, chi, ExactSVD(), want_grad=False).disc_weights)
    assert (disc > 1e-6) == expect_truncating

    for row in state.tensors:
        for t in row:
            t.requires_grad_(True)
    E = _energy_fn(state, terms, chi)()
    E.backward()

    rng = np.random.default_rng(5)
    h = 1e-5
    checked = 0
    rel_errs = []
    while checked < 12:
        y, x = rng.integers(0, 3), rng.integers(0, 3)
        t = state.tensors[y][x]
        idx = tuple(rng.integers(0, s) for s in t.shape)
        for part in (1.0, 1.0j):  # real and imaginary coordinate
            with torch.no_grad():
                orig = t[idx].item()
                t[idx] = orig + h * part
                Ep = float(_energy_fn(state, terms, chi)())
                t[idx] = orig - h * part
                Em = float(_energy_fn(state, terms, chi)())
                t[idx] = orig
            fd = (Ep - Em) / (2 * h)
            # d/dRe -> Re(grad_conj); d/dIm -> Im: torch stores grad w.r.t. conj
            g = t.grad[idx].item()
            ad = g.real if part == 1.0 else g.imag
            denom = max(abs(fd), abs(ad), 1e-10)
            rel_errs.append(abs(fd - ad) / denom)
        checked += 1
    assert max(rel_errs) < 1e-6, f"max rel err {max(rel_errs):.3e}"

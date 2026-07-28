"""Regression test (found running the L=8 pilot, 2026-07-20): an oversized
LBFGS line-search trial step can overflow the tensor network into non-finite
entries, which INV-7 now catches at canonicalization (kernels/zipup.py) with a
clear NumericalCorruption instead of a cryptic LAPACK convergence failure. The
optimizer closure must treat that as a rejected trial step, not crash the run."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tlsmbl.core.guards import NumericalCorruption
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.optimize.lbfgs_driver import optimize_lbfgs
from tlsmbl.peps.state import PEPSState

SPEC = TensorSpec()


def test_canonicalization_raises_numerical_corruption_on_nan() -> None:
    from tlsmbl.kernels.zipup import compress

    fat = [torch.ones(1, 4, 2, dtype=torch.complex128) for _ in range(3)]
    fat[1] = fat[1].clone()
    fat[1][0, 0, 0] = float("nan")
    with pytest.raises(NumericalCorruption, match="INV-7"):
        compress(fat, chi=2, backend=ExactSVD(), want_grad=True)


def test_optimize_lbfgs_survives_forced_closure_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force an occasional closure evaluation to look like a blown-up trial step
    (raising NumericalCorruption) -- a realistic isolated-fault rate, not every
    line-search trial in a row. The optimizer must reject those steps and still
    converge to a sane, finite result rather than crashing or reporting the
    rejection sentinel as the final energy."""
    import tlsmbl.optimize.lbfgs_driver as driver_mod

    real_fn = driver_mod.energy_differentiable
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] % 11 == 0:
            raise NumericalCorruption("INV-7: simulated blown-up trial step")
        return real_fn(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(driver_mod, "energy_differentiable", flaky)

    seed = 20260720
    params = ModelParams(L=3, g_J=0.3, R_c=3, seed_realization=seed)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(seed)))
    terms = build_terms(real)
    state = PEPSState.random(3, 2, SPEC, np.random.SeedSequence(seed))

    res = optimize_lbfgs(state, terms, chi=4, backend=ExactSVD(), max_outer=10, inner_iters=10)
    assert calls["n"] > 0  # the flaky path was actually exercised
    assert np.isfinite(res.energy)
    assert res.energy < 1e29  # never returns the rejection sentinel as final energy

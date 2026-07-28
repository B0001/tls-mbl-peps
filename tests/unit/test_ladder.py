"""§10.3 ladder: rungs improve (or hold) the energy monotonically within tolerance,
grow() warm-starts correctly, and the 1/D extrapolation is produced."""

import numpy as np

from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.optimize.ladder import run_ladder

SPEC = TensorSpec()


def test_ladder_two_rungs_weak_coupling() -> None:
    params = ModelParams(L=3, g_J=1e-3, R_c=3, seed_realization=0)
    real = sample_realization(params, realization_streams(20260716, 0).disorder)
    terms = build_terms(real)
    result = run_ladder(
        real,
        terms,
        [2, 3],
        SPEC,
        np.random.SeedSequence(2),
        ExactSVD(eps_F=1e-12),
        max_outer=12,
        inner_iters=20,
    )
    assert [r.D for r in result.rungs] == [2, 3]
    assert result.state.D == 3
    # Rung energies: D=3 must not be meaningfully worse than D=2 (warm start).
    assert result.rungs[1].energy <= result.rungs[0].energy + 1e-8
    assert result.extrapolated_E is not None
    assert result.fit_residual is not None


def test_grow_preserves_state_content() -> None:
    from tlsmbl.peps.state import PEPSState

    state = PEPSState.random(3, 2, SPEC, np.random.SeedSequence(9))
    grown = state.grow(3, SPEC, np.random.SeedSequence(10), noise=1e-3)
    assert grown.D == 3
    for y in range(3):
        for x in range(3):
            old = state.tensors[y][x]
            new = grown.tensors[y][x]
            s = old.shape
            # old block preserved exactly; new slices small
            assert (new[:, : s[1], : s[2], : s[3], : s[4]] == old).all()
            assert float(new.abs().max()) >= float(old.abs().max())

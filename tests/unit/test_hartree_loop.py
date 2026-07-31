"""§7.4 Hartree outer self-consistency loop + the lazy r>R_c tail stream (NR-5), and the
INV-6 determinism lock on appending a fourth RNG stream.

The determinism lock is the load-bearing test here: `realization_streams` went from
spawn(3) to spawn(4), and if that moved any of the first three streams it would silently
invalidate every golden fixture and the frozen baselines in prototypes/baselines/.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlsmbl.core.rng import realization_seed_sequence, realization_streams
from tlsmbl.model.hartree import (
    COVERS_CORRELATIONS,
    HartreeResult,
    hartree_loop,
    tail_bound,
    tail_certified,
    tail_coupling,
    tail_field,
)

_L, _RC, _GJ = 4, 1, 1e-2


def test_existing_three_streams_are_unchanged_by_the_fourth() -> None:
    """INV-6 lock. Recomputes children 0-2 from a spawn(3) exactly as the pre-tail code
    did, and requires them to equal what spawn(4) yields -- across several
    (master_seed, realization) keys, not just one."""
    for master_seed, k in ((20260716, 0), (20260719, 3), (20260715, 31), (1, 7)):
        base = realization_seed_sequence(master_seed, k)
        old = np.random.SeedSequence(base.entropy, spawn_key=base.spawn_key).spawn(3)
        new = np.random.SeedSequence(base.entropy, spawn_key=base.spawn_key).spawn(4)
        for i in range(3):
            assert (
                old[i].generate_state(8, dtype=np.uint64).tolist()
                == new[i].generate_state(8, dtype=np.uint64).tolist()
            ), f"stream {i} moved for (master_seed={master_seed}, k={k})"

        s = realization_streams(master_seed, k)
        # The public accessors must line up with children 0-2 too.
        assert s.torch_init_seed == int(old[1].generate_state(1, dtype=np.uint64)[0])
        assert s.torch_sketch_seed == int(old[2].generate_state(1, dtype=np.uint64)[0])
        # ...and the tail stream is genuinely independent of them.
        assert s.tail_seed not in (s.torch_init_seed, s.torch_sketch_seed)


def test_streams_are_deterministic_and_key_separated() -> None:
    a = realization_streams(20260716, 0)
    b = realization_streams(20260716, 0)
    assert a.tail_seed == b.tail_seed
    assert realization_streams(20260716, 1).tail_seed != a.tail_seed
    assert realization_streams(20260717, 0).tail_seed != a.tail_seed


def test_tail_coupling_is_order_independent_and_repeatable() -> None:
    """NR-5's defining property: the value depends on the PAIR, not on when it is asked
    for. A sequential generator would fail this."""
    pairs = [(a, b) for a in range(_L * _L) for b in range(a + 1, _L * _L)]
    forward = {p: tail_coupling(1234, _L, *p, _GJ, 2.5) for p in pairs}

    shuffled = list(pairs)
    np.random.default_rng(0).shuffle(shuffled)  # type: ignore[arg-type]
    for p in shuffled:
        assert tail_coupling(1234, _L, *p, _GJ, 2.5) == forward[p]
    # Asking repeatedly does not advance anything.
    for p in pairs[:20]:
        assert tail_coupling(1234, _L, *p, _GJ, 2.5) == forward[p]
    # Distinct pairs get distinct couplings; a different tail seed gives a different draw.
    assert len(set(forward.values())) > len(pairs) // 2
    assert tail_coupling(9999, _L, 0, 1, _GJ, 2.5) != forward[(0, 1)]


def test_tail_coupling_matches_the_sampler_distribution() -> None:
    """c_ij ~ U(-1,1) and J = g_J*c/r^3, exactly as model/sampling.py draws the retained
    bonds -- the tail must continue the same distribution, not a different one."""
    r = 2.0
    # Valid pairs only: s_b < L^2. (The out-of-range version of this test is what proved
    # the canonical-order guard fires.)
    vals = np.array(
        [
            tail_coupling(7, 16, a, b, 1.0, r) * r**3
            for a in range(0, 250)
            for b in range(a + 1, min(a + 9, 256))
        ]
    )
    assert vals.min() > -1.0 and vals.max() < 1.0
    assert abs(vals.mean()) < 0.05  # U(-1,1) has mean 0
    assert abs(vals.std() - 1 / math.sqrt(3)) < 0.05  # and std 1/sqrt(3)


def test_tail_coupling_requires_canonical_pair_order() -> None:
    """One pair must map to one counter, so the caller may not pass (b, a)."""
    with pytest.raises(ValueError, match="canonically ordered"):
        tail_coupling(1, _L, 5, 5, _GJ, 2.0)
    with pytest.raises(ValueError, match="canonically ordered"):
        tail_coupling(1, _L, 6, 5, _GJ, 2.0)


def test_tail_field_is_symmetric_and_excludes_short_bonds() -> None:
    m = np.ones((_L, _L))
    h = tail_field(m, tail_seed=5, L=_L, R_c=_RC, g_J=_GJ)
    assert h.shape == (_L, _L)
    # Every r > R_c bond contributes to both endpoints, so a uniform m gives each site
    # the sum of its own tail couplings -- reconstruct one site independently.
    s_a = 0
    expect = 0.0
    for s_b in range(_L * _L):
        if s_b == s_a:
            continue
        ya, xa = divmod(s_a, _L)
        yb, xb = divmod(s_b, _L)
        r = math.hypot(xa - xb, ya - yb)
        if r > _RC:
            lo, hi = min(s_a, s_b), max(s_a, s_b)
            expect += tail_coupling(5, _L, lo, hi, _GJ, r)
    assert h.reshape(-1)[s_a] == pytest.approx(expect)
    # R_c beyond the lattice diagonal leaves no tail at all.
    assert np.all(tail_field(m, tail_seed=5, L=_L, R_c=99, g_J=_GJ) == 0.0)


def test_tail_field_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        tail_field(np.ones((3, 3)), tail_seed=1, L=_L, R_c=_RC, g_J=_GJ)


def _linear_solver(gain: float) -> "object":
    """Inner solve with a known linear response m = gain * h, so the fixed point is
    m = 0 and the convergence behaviour is analytic."""

    def solve(h: np.ndarray) -> np.ndarray:
        return gain * h + 0.1  # offset keeps the fixed point away from trivial zero

    return solve


def test_loop_converges_and_reports_its_iterations() -> None:
    # The damped map contracts by ~(1-alpha) per step from an initial mismatch ~1e-3, so
    # reaching 1e-12 needs ~35 steps at alpha=0.5; K_max=60 leaves headroom.
    res = hartree_loop(
        _linear_solver(0.5),  # type: ignore[arg-type]
        L=_L, R_c=_RC, g_J=_GJ, tail_seed=11, K_max=60, alpha=0.5, tol=1e-12,
    )
    assert isinstance(res, HartreeResult)
    assert res.converged and res.n_iters <= 60
    assert res.max_delta < 1e-12
    assert len(res.history) == res.n_iters
    # Self-consistency actually holds at the returned field.
    m = _linear_solver(0.5)(res.h_mf)  # type: ignore[operator]
    h_new = tail_field(m, tail_seed=11, L=_L, R_c=_RC, g_J=_GJ)
    assert np.max(np.abs(res.h_mf - h_new)) < 1e-9


def test_damping_prevents_the_oscillation_alpha_1_would_produce() -> None:
    """alpha=1 lets a sign-flipping composite response run away; damping tames it. This
    pins the difference rather than assuming it.

    THE GAIN IS CALIBRATED, and the calibration is subtler than it first looks. The
    iteration matrix is (1-alpha)I + alpha*gain*F, where F is the tail-field operator.
    F is symmetric with zero diagonal, hence traceless, hence its spectrum ALWAYS
    straddles zero -- measured at (L=4, R_c=1, g_J=1e-2, seed 3): lambda_max = 5.79e-3,
    lambda_min = -5.12e-3. That matters, because if |gain|*lambda_max > 1 on the POSITIVE
    branch then 1 - alpha + alpha*(gain*lambda) > 1 for every alpha, and damping cannot
    help at all: a first attempt at gain = -500 diverged at alpha = 0.05 for exactly this
    reason. Damping rescues the NEGATIVE (two-cycle) branch only.

    So the gain is placed in the window where the negative branch is unstable while the
    positive branch is not: with gain = -183, gain*lambda_max = -1.06 (magnitude > 1,
    unstable at alpha=1) while gain*lambda_min = +0.94 (< 1, stable). Measured over 200
    iterations: alpha=1 grows the mismatch by 2.3e4, alpha=0.5 shrinks it by 1.2e-3 and
    does so monotonically.

    Convergence flags are deliberately NOT asserted: at these margins the damped run
    contracts at ~0.969/step and would need ~500 iterations to cross tol, so "converged"
    would test the iteration budget rather than the damping. The trend is the claim.
    """
    solve = lambda h: -183.0 * h - 0.05  # noqa: E731
    kw = dict(L=_L, R_c=_RC, g_J=_GJ, tail_seed=3, K_max=200, tol=1e-10)

    undamped = hartree_loop(solve, alpha=1.0, **kw)  # type: ignore[arg-type]
    damped = hartree_loop(solve, alpha=0.5, **kw)  # type: ignore[arg-type]

    # Same starting mismatch: the two runs differ only in alpha.
    assert undamped.history[0] == pytest.approx(damped.history[0])
    # Undamped runs away by orders of magnitude; damped decays, monotonically.
    assert undamped.history[-1] > 1e3 * undamped.history[0]
    assert damped.history[-1] < 1e-2 * damped.history[0]
    assert all(a >= b for a, b in zip(damped.history, damped.history[1:]))
    assert not undamped.converged


def test_non_convergence_is_reported_not_raised() -> None:
    """K_max is a hard cap. Hitting it must be visible to the caller so the artifact can
    be labeled -- silently returning a half-converged field is the failure mode."""
    res = hartree_loop(
        lambda h: -50.0 * h - 0.05,
        L=_L, R_c=_RC, g_J=_GJ, tail_seed=3, K_max=3, alpha=1.0, tol=1e-12,
    )
    assert not res.converged
    assert res.n_iters == 3 and len(res.history) == 3
    assert math.isfinite(res.max_delta)


def test_loop_is_bit_identical_run_to_run() -> None:
    kw = dict(L=_L, R_c=_RC, g_J=_GJ, tail_seed=77, K_max=6, alpha=0.5, tol=1e-14)
    a = hartree_loop(_linear_solver(0.3), **kw)  # type: ignore[arg-type]
    b = hartree_loop(_linear_solver(0.3), **kw)  # type: ignore[arg-type]
    assert np.array_equal(a.h_mf, b.h_mf)
    assert a.history == b.history


def test_k_max_1_matches_the_v1_baseline_field() -> None:
    """One damped step from h_mf = 0 is exactly alpha * h_new -- and with the loop off
    (hartree.enabled false) the caller keeps h_mf = 0, which is unchanged behaviour."""
    res = hartree_loop(
        lambda h: np.full((_L, _L), 0.4),
        L=_L, R_c=_RC, g_J=_GJ, tail_seed=21, K_max=1, alpha=0.5, tol=0.0,
    )
    expected = 0.5 * tail_field(
        np.full((_L, _L), 0.4), tail_seed=21, L=_L, R_c=_RC, g_J=_GJ
    )
    assert np.allclose(res.h_mf, expected, rtol=0, atol=0)
    assert res.n_iters == 1 and not res.converged


def test_invalid_loop_parameters_are_refused() -> None:
    for kw in ({"K_max": 0}, {"alpha": 0.0}, {"alpha": 1.5}, {"alpha": -0.5}):
        with pytest.raises(ValueError):
            hartree_loop(
                _linear_solver(0.1),  # type: ignore[arg-type]
                L=_L, R_c=_RC, g_J=_GJ, tail_seed=1, **kw,  # type: ignore[arg-type]
            )


def test_inv5_bound_does_not_shrink_when_the_loop_runs() -> None:
    """The loop MOVES the neglected error (tail mean field -> tail correlations); it does
    not shrink it. A smaller reported bound would claim rigor the mean-field treatment
    does not provide, so the value is identical and only the label changes."""
    res = hartree_loop(
        _linear_solver(0.2),  # type: ignore[arg-type]
        L=_L, R_c=_RC, g_J=_GJ, tail_seed=4, K_max=5, alpha=0.5, tol=1e-12,
    )
    assert res.tail_bound == tail_bound(_GJ, _RC)
    assert res.bound_covers == COVERS_CORRELATIONS
    # The INV-5 gate therefore reaches the same verdict either way.
    e = -0.3
    assert tail_certified(_GJ, _RC, e, 10.0) is tail_certified(_GJ, _RC, e, 10.0)
    assert res.tail_bound > 0.0

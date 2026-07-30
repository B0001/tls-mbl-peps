"""E(D) extrapolation (ARCHITECTURE.md §10.3, §16 DoD): exact 1/D data must be
recovered at 3 rungs, the 2-rung ladder of configs/pilot_L8.yaml must be labeled an
unfitted secant, every refusal path must fire, and the disorder bootstrap must be
seed-deterministic."""

import itertools

import numpy as np
import pytest
import torch

from tlsmbl.core.types import TensorSpec
from tlsmbl.ensemble import extrapolate as ex
from tlsmbl.io import store
from tlsmbl.peps.state import PEPSState

E_INF, C = -20.5, 0.75  # synthetic ground truth: E(D) = E_INF + C/D


def _exact(ds: list[int]) -> dict[int, float]:
    return {d: E_INF + C / d for d in ds}


def test_three_rungs_recovers_e_inf_exactly() -> None:
    r = ex.extrapolate_energy(_exact([2, 3, 4]))
    assert r.ok and r.method == ex.METHOD_FIT and r.reason == ex.OK_FIT
    assert r.e_inf == pytest.approx(E_INF, abs=1e-12)
    assert r.slope == pytest.approx(C, abs=1e-12)
    assert r.residuals == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert r.rms_residual == pytest.approx(0.0, abs=1e-12)
    # Remaining gap is the unmeasured piece: E(4) - E_inf = C/4.
    assert r.d_max == 4
    assert r.remaining_gap == pytest.approx(C / 4, abs=1e-12)
    assert not r.tainted


def test_three_rungs_with_curvature_still_variational() -> None:
    # Real data is not pure 1/D. Add a 1/D^2 term: the fit must stay below the
    # best measured energy and expose nonzero residuals.
    rungs = {d: E_INF + C / d + 0.05 / d**2 for d in (2, 3, 4)}
    r = ex.extrapolate_energy(rungs)
    assert r.ok and r.rms_residual is not None and r.rms_residual > 0
    assert r.e_inf is not None and r.e_d_max is not None
    assert r.e_inf <= r.e_d_max


def test_two_rungs_is_labeled_unfitted_secant() -> None:
    r = ex.extrapolate_energy(_exact([2, 3]))
    assert r.ok
    assert r.method == ex.METHOD_SECANT
    assert r.reason == ex.OK_SECANT  # says "unfitted, no uncertainty"
    assert r.e_inf == pytest.approx(E_INF, abs=1e-12)
    # Two points fix the line exactly, so no goodness-of-fit statistic is quoted.
    assert r.rms_residual is None
    assert r.residuals == pytest.approx((0.0, 0.0), abs=1e-12)


def test_refuses_single_rung() -> None:
    r = ex.extrapolate_energy({2: -20.0})
    assert not r.ok and r.reason == ex.REFUSED_TOO_FEW_RUNGS
    assert r.e_inf is None and r.remaining_gap is None and r.method == ex.METHOD_NONE


def test_refuses_non_monotone_energies() -> None:
    # E rises with D: impossible for a warm-started variational ladder, so this is
    # an unconverged rung (CLAUDE.md: "optimizer floor, not ansatz").
    r = ex.extrapolate_energy({2: -20.5, 3: -20.4, 4: -20.6})
    assert not r.ok and r.reason == ex.REFUSED_NON_MONOTONE
    assert r.e_inf is None
    assert r.rungs == ((2, -20.5), (3, -20.4), (4, -20.6))  # inputs kept for audit


def test_float_noise_rise_is_tolerated() -> None:
    rungs = {2: -20.5, 3: -20.5 + 1e-13}  # approximate-contraction noise
    assert ex.extrapolate_energy(rungs).ok
    # ... but the same rise is refused once the tolerance is tightened.
    assert not ex.extrapolate_energy(rungs, tol_rise=0.0).ok


def test_refuses_e_inf_above_best_measured() -> None:
    """The variational guard fires when the 1/D intercept lands ABOVE E(D_max).

    Reaching it takes a WIDELY SPACED ladder: the last rung must sit far enough below
    the line, at small enough 1/D, that the leftward extrapolation to 1/D = 0 does not
    overshoot it. {2, 4, 16} does; see the companion test for why the shipped ladders
    cannot.
    """
    rungs = {2: -20.0, 4: -20.0, 16: -21.0}
    x = np.array([1 / 2, 1 / 4, 1 / 16])
    intercept = np.polyfit(x, np.array([-20.0, -20.0, -21.0]), 1)[1]
    assert intercept > -21.0  # the fixture really does trip the gate
    r = ex.extrapolate_energy(rungs)
    assert not r.ok and r.reason == ex.REFUSED_ABOVE_VARIATIONAL
    assert r.e_inf is None


def test_shipped_ladders_cannot_produce_a_variationally_impossible_e_inf() -> None:
    """Pins a property of the shipped ladders, not just of the guard.

    For [2, 3] and [2, 3, 4] (configs/pilot_L8.yaml, configs/bench_L16_D4.yaml) the
    monotonicity gate already implies E_inf <= E(D_max): algebraically, for D={2,3,4},
    E_inf - E(D_max) = -1.214*(E_2 - E_4) + 0.643*(E_3 - E_4), which cannot be positive
    while E_2 >= E_3 >= E_4. So on the ladders this project actually runs, the two
    refusals are not redundant but nested -- passing monotonicity is enough, and
    REFUSED_ABOVE_VARIATIONAL is a defensive net for wider ladders. Swept here rather
    than asserted by hand so a future rung-set change cannot quietly invalidate it.
    """
    grid = (0.0, -0.5, -1.0, -2.0, -5.0)
    for ladder in ([2, 3], [2, 3, 4]):
        for ys in itertools.product(grid, repeat=len(ladder)):
            if any(b - a > 0 for a, b in zip(ys, ys[1:])):
                continue  # non-monotone: refused before the variational gate
            r = ex.extrapolate_energy(dict(zip(ladder, ys)))
            assert r.ok, (ladder, ys, r.reason)
            assert r.e_inf is not None and r.e_inf <= ys[-1] + 1e-12, (ladder, ys)


def test_refuses_non_finite_and_bad_d() -> None:
    assert ex.extrapolate_energy({2: -20.0, 3: float("nan")}).reason == ex.REFUSED_NON_FINITE
    assert ex.extrapolate_energy({0: -20.0, 3: -20.5}).reason == ex.REFUSED_BAD_RUNG_D


def test_unconverged_rung_taints_but_does_not_refuse() -> None:
    rungs = _exact([2, 3, 4])
    r = ex.extrapolate_energy(rungs, converged={2: True, 3: True, 4: False})
    assert r.ok  # the number is real, just floor-limited
    assert r.tainted and r.tainted_rungs == (4,)
    # A missing flag is not evidence of convergence.
    r2 = ex.extrapolate_energy(rungs, converged={2: True})
    assert r2.tainted and r2.tainted_rungs == (3, 4)
    assert not ex.extrapolate_energy(rungs).tainted  # no flags supplied -> no claim


def test_taint_propagates_to_ensemble() -> None:
    per = [_exact([2, 3, 4]) for _ in range(3)]
    flags = [{2: True, 3: True, 4: True}, {2: True, 3: True, 4: False},
             {2: True, 3: True, 4: True}]
    e = ex.extrapolate_ensemble(per, rng=np.random.default_rng(0), converged=flags, n_boot=200)
    assert e.ok and e.n_used == 3 and e.n_tainted == 1
    assert e.method == ex.METHOD_FIT and e.reason == ex.OK_ENSEMBLE


def test_ensemble_bootstrap_is_seed_deterministic() -> None:
    per = [{d: E_INF + i * 0.1 + (C + i * 0.01) / d for d in (2, 3, 4)} for i in range(6)]
    a = ex.extrapolate_ensemble(per, rng=np.random.default_rng(11), n_boot=500)
    b = ex.extrapolate_ensemble(per, rng=np.random.default_rng(11), n_boot=500)
    c = ex.extrapolate_ensemble(per, rng=np.random.default_rng(12), n_boot=500)
    assert a == b
    assert a.e_inf is not None and c.e_inf is not None
    assert a.e_inf != c.e_inf  # different seed really does move the CI
    assert a.e_inf[1] <= a.e_inf[0] <= a.e_inf[2]


def test_ensemble_counts_refusals_and_excludes_them() -> None:
    per = [_exact([2, 3, 4]), {2: -20.5}, {2: -20.5, 3: -20.4, 4: -20.6}]
    e = ex.extrapolate_ensemble(per, rng=np.random.default_rng(1), n_boot=200)
    assert e.n_total == 3 and e.n_used == 1
    assert e.refusals == {ex.REFUSED_TOO_FEW_RUNGS: 1, ex.REFUSED_NON_MONOTONE: 1}
    assert e.reason == ex.OK_ENSEMBLE_SINGLE
    assert e.e_inf is not None and np.isnan(e.e_inf[1])  # no CI from one sample


def test_ensemble_all_secant_is_labeled() -> None:
    per = [_exact([2, 3]) for _ in range(4)]
    e = ex.extrapolate_ensemble(per, rng=np.random.default_rng(3), n_boot=200)
    assert e.method == ex.METHOD_SECANT and e.reason == ex.OK_ENSEMBLE_SECANT
    assert e.e_inf is not None and e.e_inf[0] == pytest.approx(E_INF, abs=1e-12)


def test_ensemble_mixed_ladders_labeled_and_refuses_when_empty() -> None:
    mixed = ex.extrapolate_ensemble(
        [_exact([2, 3]), _exact([2, 3, 4])], rng=np.random.default_rng(4), n_boot=200
    )
    assert mixed.reason == ex.OK_ENSEMBLE_MIXED
    empty = ex.extrapolate_ensemble([{2: -1.0}], rng=np.random.default_rng(4), n_boot=200)
    assert not empty.ok and empty.reason == ex.REFUSED_NO_USABLE
    assert empty.e_inf is None
    with pytest.raises(ValueError):
        ex.extrapolate_ensemble([_exact([2, 3])], rng=np.random.default_rng(4), converged=[])


def test_reader_round_trips_written_rungs(tmp_path: object) -> None:
    """read_rung_energies must pull exactly what orchestrate.py stored per rung."""
    root = store.open_run(f"{tmp_path}/run.zarr")
    g = store.realization_group(root, 0)
    spec = TensorSpec()
    for d, energy, conv in ((2, -20.0, True), (3, -20.25, False)):
        state = PEPSState.random(2, d, spec, np.random.SeedSequence(d))
        store.write_rung(
            g, d, state,
            {"energy": energy, "grad_norm": 1e-7, "n_iters": 10, "chi": d * d,
             "converged": conv, "wall_s": 1.0},
        )
    energies, flags = read = ex.read_rung_energies(g)
    assert read == ({2: -20.0, 3: -20.25}, {2: True, 3: False})
    r = ex.extrapolate_energy(energies, converged=flags)
    assert r.ok and r.method == ex.METHOD_SECANT and r.tainted and r.tainted_rungs == (3,)
    # Secant in x = 1/D through (1/2, -20.0) and (1/3, -20.25): the slope is
    # 0.25 / (1/2 - 1/3) = 1.5, so E_inf = -20.0 - 1.5 * (1/2) = -20.75.
    assert r.e_inf == pytest.approx(-20.75, abs=1e-12)
    assert r.slope == pytest.approx(1.5, abs=1e-12)
    assert torch.is_tensor(state.tensors[0][0])  # sanity: real states were written

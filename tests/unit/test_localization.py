"""Localization length xi (ARCHITECTURE.md §12): recovery on synthetic exponentials,
bootstrap coverage, determinism, and every refusal path -- including the measured
pilot numbers (runs/pilot_L8.zarr/REPORT.md), which must come back not-ok."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlsmbl.observables.localization import DEFAULT_NOISE_FLOOR, fit_xi

# Measured pilot (L=8, g_J=1e-3, 4 certified realizations): note the sign alternation
# and that r=2,3 sit far below env.eps_env = 1e-8.
PILOT_CZZ = {1: -3.494e-07, 2: +1.262e-09, 3: -2.273e-10}


def _exp_real(xi: float, amp: float, r_max: int, sign: float = -1.0) -> dict[int, float]:
    return {r: sign * amp * math.exp(-r / xi) for r in range(1, r_max + 1)}


def test_exact_exponential_recovers_xi() -> None:
    # Amplitude-only disorder: the disorder mean of |Czz| is <amp> exp(-r/xi), so xi is
    # recoverable to machine precision and R^2 is 1.
    xi_true = 2.5
    data = [_exp_real(xi_true, amp, 6) for amp in (1.0e-2, 3.0e-2, 7.0e-3)]
    fit = fit_xi(data, rng=np.random.default_rng(0), n_boot=200)
    assert fit.ok, fit.detail
    assert fit.reason == "ok"
    assert fit.xi is not None and abs(fit.xi - xi_true) < 1e-10
    assert fit.r2 is not None and fit.r2 > 1 - 1e-12
    assert fit.residual_rms is not None and fit.residual_rms < 1e-10
    assert fit.r_used == (1, 2, 3, 4, 5, 6)
    assert fit.boot_frac_ok == 1.0
    assert "xi: 2.500" in fit.summary_line()


def _scattered(n: int, seed: int, xi_true: float = 3.0) -> list[dict[int, float]]:
    """n realizations with scattered rate AND amplitude -- the realistic case."""
    rng_data = np.random.default_rng(seed)
    return [
        _exp_real(
            float(xi_true * (1.0 + 0.08 * rng_data.standard_normal())),
            float(1.0e-2 * math.exp(0.3 * rng_data.standard_normal())),
            5,
        )
        for _ in range(n)
    ]


def test_bootstrap_ci_covers_the_estimators_own_target() -> None:
    """Coverage is asserted against the ENSEMBLE ENVELOPE rate, not against xi_true.

    The module fits log<|Czz|>_disorder, so its target is the envelope's decay rate.
    When per-realization rates are scattered, that is NOT the mean of per-realization
    xi: E[exp(-r/xi)] > exp(-r E[1/xi]) by Jensen, so the envelope decays slower than
    the typical realization and its rate is a genuinely different population parameter
    (the module docstring says so). Asserting `ci` brackets xi_true would therefore test
    a claim the estimator does not make -- and at 64 realizations the CI is tight enough
    (~1%) that the Jensen offset alone can push xi_true outside it.

    So: take the large-sample envelope rate as the target, check it is close to xi_true
    (the bias is small in this regime, worth pinning), and require the 64-realization CI
    to cover it across seeds. One miss in five is allowed -- a 95% CI that never missed
    would mean the bootstrap was overcovering.
    """
    xi_true = 3.0
    target = fit_xi(_scattered(4000, 999, xi_true), rng=np.random.default_rng(0), n_boot=2)
    assert target.ok and target.xi is not None
    assert abs(target.xi - xi_true) / xi_true < 0.05  # Jensen offset is small here

    covered = 0
    for seed in range(5):
        fit = fit_xi(_scattered(64, seed), rng=np.random.default_rng(seed), n_boot=2000)
        assert fit.ok, fit.detail
        assert fit.ci is not None and fit.xi is not None
        assert fit.ci[0] < fit.ci[1]  # rate scatter => non-degenerate CI
        assert abs(fit.xi - xi_true) / xi_true < 0.1
        covered += fit.ci[0] <= target.xi <= fit.ci[1]
    assert covered >= 4, f"bootstrap CI covered the envelope rate in only {covered}/5 draws"


def test_amplitude_only_scatter_gives_a_degenerate_ci() -> None:
    """The flip side, and why the test above needs rate scatter: with amplitude-only
    disorder every realization has the SAME shape, so the mean magnitude is exp(-r/xi)
    for every resample and the bootstrap has nothing to vary. A zero-width CI here is
    correct -- the realization sample carries no information about the rate -- and it is
    reported as such rather than being widened to look respectable."""
    data = [_exp_real(2.5, amp, 6) for amp in (1.0e-2, 3.0e-2, 7.0e-3)]
    fit = fit_xi(data, rng=np.random.default_rng(0), n_boot=500)
    assert fit.ok and fit.ci is not None and fit.xi is not None
    # Degenerate to float noise, not bitwise: resamples draw the same three amplitudes
    # in different multiplicities, so the mean rounds differently while the slope is
    # analytically identical.
    assert fit.ci[1] - fit.ci[0] < 1e-12
    assert fit.ci[0] == pytest.approx(fit.xi, abs=1e-12)


def test_deterministic_under_fixed_seed() -> None:
    data = [_exp_real(2.0, amp, 5) for amp in (1.0e-2, 2.0e-2, 5.0e-3, 1.5e-2)]
    a = fit_xi(data, rng=np.random.default_rng(1234), n_boot=500)
    b = fit_xi(data, rng=np.random.default_rng(1234), n_boot=500)
    assert a == b
    c = fit_xi(data, rng=np.random.default_rng(4321), n_boot=500)
    assert c.xi == a.xi  # point estimate is rng-independent


def test_pilot_numbers_refused_at_noise_floor() -> None:
    fit = fit_xi([dict(PILOT_CZZ) for _ in range(4)], rng=np.random.default_rng(0))
    assert not fit.ok
    assert fit.reason == "at_noise_floor"
    assert fit.xi is None and fit.ci is None
    assert f"{DEFAULT_NOISE_FLOOR:.1e}" in fit.detail
    assert "unresolved" in fit.summary_line()


def test_pilot_numbers_refused_for_sign_alternation_below_floor() -> None:
    # Even if a caller declares a far more permissive noise floor, the sign alternation
    # of the disorder-mean correlator still blocks the fit.
    fit = fit_xi(
        [dict(PILOT_CZZ) for _ in range(4)],
        rng=np.random.default_rng(0),
        noise_floor=1.0e-12,
    )
    assert not fit.ok
    assert fit.reason == "sign_inconsistent"
    assert fit.xi is None


def test_refuse_too_few_bins() -> None:
    data = [{1: -1.0e-2, 2: -1.0e-3} for _ in range(3)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "too_few_bins"


def test_refuse_all_bins_at_noise_floor() -> None:
    data = [{1: -1.0e-12, 2: -3.0e-13, 3: -1.0e-13} for _ in range(3)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "at_noise_floor"
    assert fit.r_used == ()


def test_refuse_xi_larger_than_r_max() -> None:
    data = [_exp_real(10.0, amp, 3) for amp in (1.0e-2, 2.0e-2)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "xi_unresolved_large"
    assert fit.xi is not None and abs(fit.xi - 10.0) < 1e-9  # audit trail kept
    assert fit.ci is None


def test_refuse_xi_below_lattice_spacing() -> None:
    data = [_exp_real(0.3, amp, 3) for amp in (1.0e-2, 2.0e-2)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "xi_below_lattice"


def test_refuse_non_monotone() -> None:
    data = [{1: -1.0e-4, 2: -1.0e-3, 3: -1.0e-5} for _ in range(3)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "non_monotone"


def test_refuse_flat_envelope() -> None:
    data = [{1: -1.0e-4, 2: -1.0e-4, 3: -1.0e-4} for _ in range(3)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "no_decay"


def test_refuse_poor_log_linear_fit() -> None:
    # Plateau then cliff: monotone, single-signed, above the floor, but nowhere near
    # log-linear.
    data = [{1: -1.0e-2, 2: -9.9e-3, 3: -9.8e-3, 4: -1.0e-4} for _ in range(3)]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "poor_fit"
    assert fit.r2 is not None and fit.r2 < 0.9


def test_refuse_empty_and_single_realization() -> None:
    assert fit_xi([], rng=np.random.default_rng(0)).reason == "no_data"
    one = fit_xi([_exp_real(2.0, 1e-2, 5)], rng=np.random.default_rng(0))
    assert not one.ok and one.reason == "too_few_realizations"


def test_refuse_no_shared_r_bins() -> None:
    data = [{1: -1.0e-2, 2: -1.0e-3}, {3: -1.0e-4, 4: -1.0e-5}]
    fit = fit_xi(data, rng=np.random.default_rng(0))
    assert not fit.ok and fit.reason == "no_data"


def test_only_bins_shared_by_all_realizations_are_used() -> None:
    full = _exp_real(2.0, 1.0e-2, 5)
    short = {r: v for r, v in full.items() if r <= 4}
    fit = fit_xi([full, short], rng=np.random.default_rng(0), n_boot=200)
    assert fit.ok, fit.detail
    assert fit.r_available == (1, 2, 3, 4)
    assert fit.r_used == (1, 2, 3, 4)


def test_noise_floor_truncates_tail_but_keeps_fit() -> None:
    """An exponential whose TAIL sinks below the floor: the leading bins still resolve
    xi, and the dropped tail shows up as r_used shorter than r_available.

    The floor is passed explicitly. With mean amplitude 1.5e-2 and xi=1 the envelope is
    1.5e-2*exp(-r): r=7 gives 1.37e-5 (above) and r=8 gives 5.03e-6 (below), so a 1e-5
    floor cuts exactly the last bin. Under the DEFAULT 1e-8 floor nothing would be cut
    (the r=8 value is still 5e-6, three decades clear), so this test would silently
    assert nothing.
    """
    xi_true = 1.0
    data = [_exp_real(xi_true, amp, 8) for amp in (1.0e-2, 2.0e-2)]
    fit = fit_xi(data, rng=np.random.default_rng(0), n_boot=200, noise_floor=1.0e-5)
    assert fit.ok, fit.detail
    assert fit.r_available == (1, 2, 3, 4, 5, 6, 7, 8)
    assert fit.r_used == (1, 2, 3, 4, 5, 6, 7)  # r=8 dropped at the floor
    assert fit.xi is not None and abs(fit.xi - xi_true) < 1e-10
    # Sanity on the premise: the default floor would not have cut anything.
    wide = fit_xi(data, rng=np.random.default_rng(0), n_boot=200)
    assert wide.ok and wide.r_used == (1, 2, 3, 4, 5, 6, 7, 8)

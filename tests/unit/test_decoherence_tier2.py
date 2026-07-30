"""Tier-2 decoherence (ARCHITECTURE.md §12): the fluctuator table, Gamma_1 and the
spectral-diffusion proxy, plus the guards that keep declared-model arithmetic from
returning inf/nan.

The first test is a regression lock: `gamma_1` was refactored to share one E_i definition
with the new table, so its value on a fixed realization must not move.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tlsmbl.core.types import DisorderRealization, ModelParams
from tlsmbl.observables.decoherence import (
    DISCLAIMER,
    WEIGHT_BARE,
    WEIGHT_STATE,
    Tier2Inputs,
    fluctuator_table,
    gamma_1,
    spectral_diffusion,
    splittings,
    tier2_record,
)

_INPUTS = Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=0.05)


def _real(L: int = 3, seed: int = 0) -> DisorderRealization:
    """A fixed disorder draw. Built directly (not via sampling.py) so the numbers this
    file locks cannot drift with the sampler."""
    rng = np.random.default_rng(seed)
    return DisorderRealization(
        params=ModelParams(L=L, delta_min=1e-3, g_J=1e-3, R_c=3, seed_realization=seed),
        eps=rng.uniform(-1.0, 1.0, (L, L)),
        delta=rng.uniform(1e-3, 0.5, (L, L)),
        J={},
        h_mf=np.zeros((L, L)),
        rng_fingerprint="test",
    )


def test_gamma_1_value_is_unchanged_by_the_shared_splitting_refactor() -> None:
    """Regression lock. The reference is recomputed here from the §12 formula with the
    ORIGINAL inlined arithmetic (explicit python loop over raveled arrays), so it is an
    independent check of the vectorized implementation, not a copy of it."""
    real = _real()
    E = np.sqrt((real.eps + real.h_mf) ** 2 + real.delta**2)
    W = real.params.W
    expected = 0.0
    for Ei, Di in zip(E.ravel(), real.delta.ravel()):
        gi = _INPUTS.g0 * (Di / Ei)
        gam = _INPUTS.gamma0 * (Ei / W) ** 3 / math.tanh(Ei / (2 * _INPUTS.T))
        expected += 2 * gi**2 * gam / (gam**2 + (Ei - _INPUTS.omega_q) ** 2)

    got = gamma_1(real, _INPUTS)
    assert got.gamma_1 == pytest.approx(expected, rel=1e-14)
    assert got.n_fluctuators == 9
    assert got.disclaimer == DISCLAIMER
    assert got.inputs is _INPUTS  # echoed verbatim, not rebuilt


def test_gamma_1_on_resonance_is_the_lorentzian_peak() -> None:
    """Single fluctuator exactly on resonance: E_i = omega_q collapses the Lorentzian to
    2 g_i^2 / gamma_i, the analytic peak."""
    L = 1
    delta = 0.3
    eps = 0.4
    E = math.hypot(eps, delta)
    real = DisorderRealization(
        params=ModelParams(L=L, delta_min=1e-3, g_J=1e-3, R_c=3, seed_realization=0),
        eps=np.full((L, L), eps),
        delta=np.full((L, L), delta),
        J={},
        h_mf=np.zeros((L, L)),
        rng_fingerprint="test",
    )
    inputs = Tier2Inputs(omega_q=E, g0=1e-3, gamma0=1e-4, T=0.05)
    gam = inputs.gamma0 * E**3 / math.tanh(E / (2 * inputs.T))
    gi = inputs.g0 * delta / E
    assert gamma_1(real, inputs).gamma_1 == pytest.approx(2 * gi**2 / gam, rel=1e-12)


def test_splittings_are_shared_and_respect_the_yx_convention() -> None:
    real = _real()
    E = splittings(real)
    assert E.shape == (3, 3)
    # E is indexed [y, x] like every DisorderRealization array, and the table keys are
    # (x, y) like static.py's per-site dicts. The two must line up.
    table = fluctuator_table(real, _INPUTS)
    for f in table:
        x, y = f.site
        assert f.E == pytest.approx(float(E[y, x]), rel=0, abs=0)
    # Row-major site order, s = y*L + x (§6): site 0 first.
    assert [f.site for f in table][:4] == [(0, 0), (1, 0), (2, 0), (0, 1)]


def test_splittings_are_bounded_below_by_delta_min() -> None:
    """E_i >= |Delta~_i| >= delta_min > 0 structurally, so the Delta~/E weight can never
    divide by zero -- the property the module's guard documents."""
    real = _real()
    assert np.all(splittings(real) >= real.delta)
    assert np.all(splittings(real) >= real.params.delta_min)


def test_splittings_reject_corrupt_fields() -> None:
    real = _real()
    bad = DisorderRealization(
        params=real.params,
        eps=real.eps,
        delta=np.zeros_like(real.delta),  # E = |eps| and eps can be ~0
        J={},
        h_mf=-real.eps,  # cancels eps exactly => E = 0
        rng_fingerprint="test",
    )
    with pytest.raises(ValueError, match="non-positive splitting"):
        splittings(bad)


def test_bare_and_state_weights_are_distinguishable_and_labeled() -> None:
    real = _real()
    bare = fluctuator_table(real, _INPUTS)
    assert {f.weight_kind for f in bare} == {WEIGHT_BARE}
    for f in bare:
        x, y = f.site
        E = math.hypot(real.eps[y, x], real.delta[y, x])
        assert f.transverse_weight == pytest.approx(real.delta[y, x] / E)

    # A deliberately different local state: fully transverse polarization must give
    # weight 1, which the bare fields do not.
    sites = [f.site for f in bare]
    sx = {s: 1.0 for s in sites}
    sz = {s: 0.0 for s in sites}
    stated = fluctuator_table(real, _INPUTS, sx=sx, sz=sz)
    assert {f.weight_kind for f in stated} == {WEIGHT_STATE}
    assert all(f.transverse_weight == pytest.approx(1.0) for f in stated)
    assert stated[0].transverse_weight != pytest.approx(bare[0].transverse_weight)


def test_depolarized_state_reports_zero_weight_not_nan() -> None:
    real = _real()
    sites = [f.site for f in fluctuator_table(real, _INPUTS)]
    zero = {s: 0.0 for s in sites}
    table = fluctuator_table(real, _INPUTS, sx=zero, sz=zero)
    assert all(f.transverse_weight == 0.0 for f in table)  # not 0/0


def test_half_supplied_state_is_refused() -> None:
    real = _real()
    sites = [f.site for f in fluctuator_table(real, _INPUTS)]
    with pytest.raises(ValueError, match="both sx and sz"):
        fluctuator_table(real, _INPUTS, sx={s: 0.0 for s in sites})


def test_spectral_diffusion_vanishes_at_low_t_and_grows_with_t() -> None:
    """No thermally active fluctuator => no switching => no spectral diffusion.

    At T=1e-4 the activity factor is ~1e-217 rather than bitwise zero (E/2T is still
    inside the cosh clamp), so "vanishes" is asserted as physically-zero, not ==0.0. It
    underflows to exactly 0.0 by T=1e-8, which the guard test pins.
    """
    real = _real()
    cold = spectral_diffusion(real, Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=1e-4))
    assert cold.variance < 1e-200

    prev = 0.0
    for T in (0.01, 0.05, 0.2, 1.0):
        v = spectral_diffusion(
            real, Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=T)
        ).variance
        assert v > prev
        prev = v


def test_spectral_diffusion_on_resonance_contributes_nothing() -> None:
    """chi_i is the DISPERSIVE shift: exactly on resonance it vanishes (the weight is
    absorptive and lands in Gamma_1). This is the documented regularization, so it is
    pinned rather than left to an epsilon."""
    L = 1
    eps, delta = 0.4, 0.3
    E = math.hypot(eps, delta)
    real = DisorderRealization(
        params=ModelParams(L=L, delta_min=1e-3, g_J=1e-3, R_c=3, seed_realization=0),
        eps=np.full((L, L), eps),
        delta=np.full((L, L), delta),
        J={},
        h_mf=np.zeros((L, L)),
        rng_fingerprint="test",
    )
    on = spectral_diffusion(real, Tier2Inputs(omega_q=E, g0=1e-3, gamma0=1e-4, T=0.5))
    assert on.variance == pytest.approx(0.0, abs=1e-300)
    off = spectral_diffusion(real, Tier2Inputs(omega_q=E + 0.1, g0=1e-3, gamma0=1e-4, T=0.5))
    assert off.variance > 0.0


def test_guards_fire_instead_of_producing_inf_or_nan() -> None:
    real = _real()
    # T <= 0 and gamma0 <= 0 are refused at construction, not deep in a sum.
    for override in ({"T": 0.0}, {"T": -1.0}, {"gamma0": 0.0}, {"gamma0": -1.0}):
        kwargs: dict[str, float] = {
            "omega_q": 0.5, "g0": 1e-3, "gamma0": 1e-4, "T": 0.05, **override
        }
        with pytest.raises(ValueError):
            Tier2Inputs(**kwargs)
    with pytest.raises(ValueError, match="finite"):
        Tier2Inputs(omega_q=float("inf"), g0=1e-3, gamma0=1e-4, T=0.05)

    # Extreme E/2T (the overflow path that a naive 1/(1+exp(E/T)) would hit) stays finite.
    tiny_T = Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=1e-8)
    sd = spectral_diffusion(real, tiny_T)
    assert math.isfinite(sd.variance) and sd.variance == 0.0
    g1 = gamma_1(real, tiny_T)
    assert math.isfinite(g1.gamma_1)
    huge_T = Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=1e12)
    assert math.isfinite(spectral_diffusion(real, huge_T).variance)
    assert math.isfinite(gamma_1(real, huge_T).gamma_1)
    assert all(math.isfinite(f.gamma) for f in fluctuator_table(real, tiny_T))

    # Regression lock on the activity factor's form. An earlier implementation clamped the
    # cosh ARGUMENT to dodge overflow, which floored the activity at a spurious nonzero
    # constant (~1e-261) instead of letting it decay -- T=1e-8 and T=1e-4 then returned
    # the identical variance despite four orders of magnitude between them.
    #
    # Two properties kill any such saturation. (1) Deep in the frozen regime the answer is
    # EXACTLY zero, not merely tiny: the clamped form returned ~1e-270 here.
    assert spectral_diffusion(real, tiny_T).variance == 0.0
    # (2) Where the values are actually resolvable, they are strictly ordered in T. (Both
    # 1e-8 and 1e-4 underflow to a true 0 for this draw -- min E/T > 745 -- so comparing
    # those two could not distinguish decay from saturation in either direction.)
    warm = [
        spectral_diffusion(real, Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=t)).variance
        for t in (0.02, 0.05, 0.1)
    ]
    assert 0.0 < warm[0] < warm[1] < warm[2]


def test_record_round_trips_through_json_and_is_tier_flagged() -> None:
    real = _real()
    rec = tier2_record(real, _INPUTS)
    d = rec.to_json_dict()
    assert json.loads(json.dumps(d)) == d  # zarr attrs are JSON
    assert d["tier"] == 2  # cannot be mistaken for a Tier-1 observable
    assert d["disclaimer"] == DISCLAIMER
    assert d["n_fluctuators"] == 9
    assert d["weight_kind"] == WEIGHT_BARE
    assert d["gamma_1"] == pytest.approx(gamma_1(real, _INPUTS).gamma_1)
    assert d["inputs"]["T"] == _INPUTS.T


def test_record_echoes_raw_unit_tagged_inputs_verbatim() -> None:
    """§12: every model input echoed verbatim. The config carries unit-tagged strings,
    which this layer refuses to reinterpret -- it just carries them through."""
    raw = {"omega_q": "5.0GHz", "g0": "1.0MHz", "gamma0": "0.1MHz", "T": "50mK"}
    inputs = Tier2Inputs(omega_q=0.5, g0=1e-3, gamma0=1e-4, T=0.05, raw=raw)
    rec = tier2_record(_real(), inputs)
    assert rec.inputs["raw"] == raw
    lines = "\n".join(rec.report_lines())
    for k, v in raw.items():
        assert f"{k}={v}" in lines
    assert DISCLAIMER in lines
    assert "Gamma_1" in lines and "spectral-diffusion" in lines


def test_report_lines_fall_back_to_floats_when_no_raw_strings() -> None:
    lines = "\n".join(tier2_record(_real(), _INPUTS).report_lines())
    assert "omega_q=0.5" in lines
    assert DISCLAIMER in lines

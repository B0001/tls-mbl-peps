"""T-INV-3, second failure action (§3 INV-3, §14.8, ADR-016): "if fallback_rate > 20%
over a sweep, disable sketching for the realization and log".

The gate-level fallback itself is covered by tests/unit/test_rsvd.py; this file covers
the *disable*, which was a dead config knob (`kernels.fallback_disable_rate` was
validated by pydantic, echoed in all four configs and cited in docs/HANDOFF.md as the
reason D=6 "would switch itself off anyway", but nothing read it).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from tlsmbl.kernels.rsvd import SketchedSVD

_CHI, _M = 8, 96


def _operand(spectrum: np.ndarray, seed: int, m: int = _M) -> torch.Tensor:
    """Same construction as test_rsvd.py: an m x m matrix with a prescribed spectrum.
    `m` matters here -- the structural (`k <= chi`) path needs a genuinely small
    operand, not a large one of low rank."""
    rng = np.random.default_rng(seed)

    def crandn(*s: int) -> np.ndarray:
        out: np.ndarray = (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        return out

    k = len(spectrum)
    U, _ = np.linalg.qr(crandn(m, k))
    V, _ = np.linalg.qr(crandn(m, k))
    return torch.from_numpy((U * spectrum) @ V.conj().T).to(torch.complex128)


def _slow() -> torch.Tensor:
    """sigma_k ~ 1/k: the probe estimator is structurally pessimistic here, so the
    two-sided gate falls back on every call (ADR-009's accepted conservative case)."""
    return _operand(1.0 / np.arange(1, _M + 1), 2)


def _fast() -> torch.Tensor:
    """sigma_k = e^(-0.5k): the gate accepts (ADR-009's defining instance)."""
    return _operand(np.exp(-0.5 * np.arange(1, _M + 1)), 1)


def test_disable_fires_and_latches_on_slow_spectra() -> None:
    W = _slow()
    b = SketchedSVD(seed=7, disable_rate=0.20, min_gate_calls=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            b.truncate(W, _CHI)
        assert b.sketching_disabled, "INV-3 disable did not fire at a 100% gate-fallback rate"
        assert b.disabled_at_call == 4
        assert len(caught) == 1 and "INV-3" in str(caught[0].message)
        # Latched: further calls go exact WITHOUT attempting a sketch, so they are not
        # counted as fallbacks (there was no sketch to fall back from) and the warning
        # is not repeated.
        before = b.fallback_count
        for _ in range(3):
            assert b.truncate(W, _CHI).posterior_err is None
        assert b.fallback_count == before
        assert b.gate_call_count == 4
        assert b.call_count == 7
        assert len(caught) == 1

    # Monotonic: even an operand the gate would happily accept stays on the exact path.
    assert b.truncate(_fast(), _CHI).posterior_err is None
    assert b.sketching_disabled


def test_warmup_prevents_disable_on_one_unlucky_call() -> None:
    """ADR-016(b): without a warmup the first fallback reads as rate 1.0 and would
    disable sketching permanently on call one."""
    b = SketchedSVD(seed=7, disable_rate=0.20, min_gate_calls=32)
    b.truncate(_slow(), _CHI)
    assert b.gate_fallback_count == 1
    assert b.gate_fallback_rate == 1.0  # the rate is tripped ...
    assert not b.sketching_disabled  # ... but the warmup has not elapsed


def test_structural_fallback_does_not_count_toward_the_disable() -> None:
    """ADR-016(a): `k <= chi` (no oversampling headroom) is operand geometry, not gate
    thrashing. Counting it would disable sketching on exactly the small early-ladder
    rungs where falling back to exact costs nothing."""
    small = _operand(np.exp(-0.5 * np.arange(1, 9)), 5, m=8)  # 8x8, chi=8 => no headroom
    b = SketchedSVD(seed=13, disable_rate=0.20, min_gate_calls=1)
    for _ in range(10):
        b.truncate(small, chi=8)
    assert b.structural_fallback_count == 10
    assert b.gate_fallback_count == 0
    assert b.fallback_count == 10  # still reported in the §11 audit ...
    assert b.fallback_rate == 1.0
    assert b.gate_fallback_rate == 0.0  # ... but the INV-3 signal is clean
    assert not b.sketching_disabled


def test_accepting_gate_never_disables() -> None:
    b = SketchedSVD(seed=9, disable_rate=0.20, min_gate_calls=2)
    W = _fast()
    for _ in range(6):
        assert b.truncate(W, _CHI).posterior_err is not None
    assert b.gate_fallback_count == 0 and not b.sketching_disabled


def test_disable_is_opt_in() -> None:
    """disable_rate=None (the dataclass default, used by the kernel unit tests) keeps
    sketching on no matter what the counters say; orchestrate.py always passes the
    config value."""
    b = SketchedSVD(seed=7, min_gate_calls=1)
    for _ in range(5):
        b.truncate(_slow(), _CHI)
    assert b.gate_fallback_rate == 1.0
    assert not b.sketching_disabled and b.disabled_at_call is None


def test_stats_record_is_json_able_and_complete() -> None:
    """The §11 audit trail: orchestrate.py persists this dict into zarr attrs."""
    import json

    b = SketchedSVD(seed=7, disable_rate=0.20, min_gate_calls=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            b.truncate(_slow(), _CHI)
    s = b.stats()
    assert json.loads(json.dumps(s)) == s
    assert s["sketching_disabled"] is True
    assert s["gate_fallback_rate"] == 1.0
    assert set(s) == {
        "call_count",
        "gate_call_count",
        "fallback_count",
        "gate_fallback_count",
        "structural_fallback_count",
        "fallback_rate",
        "gate_fallback_rate",
        "sketching_disabled",
        "disabled_at_call",
    }

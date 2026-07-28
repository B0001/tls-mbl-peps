"""T-PERF (§14.9, blocking Phase 3): fitted D-scaling exponent gap between exact
and sketched kernels >= 1.6 (theory 2.0; prototype measured 3.30 over D=4..8).

Timing-based, so the full gate runs under TLSMBL_FULL_GOLD=1 (D up to 6, ~1 min);
the fast suite keeps a smoke assertion that sketching wins at D=4."""

import os

import pytest

from tlsmbl.kernels.bench import run_kernel_bench


def test_sketched_faster_at_D4_smoke() -> None:
    res = run_kernel_bench([4], reps=3)
    (p,) = res.points
    assert p.gate_passed
    assert p.sketched_s < p.exact_s, f"sketched {p.sketched_s:.4f}s vs exact {p.exact_s:.4f}s"


@pytest.mark.skipif(
    os.environ.get("TLSMBL_FULL_GOLD") != "1", reason="timing gate is slow (D up to 6)"
)
def test_exponent_gap_gate() -> None:
    res = run_kernel_bench([3, 4, 6], reps=3)
    assert all(p.gate_passed for p in res.points)
    assert res.exponent_gap >= 1.6, (
        f"exponent gap {res.exponent_gap:.2f} "
        f"(exact {res.exact_exponent:.2f}, sketched {res.sketched_exponent:.2f})"
    )

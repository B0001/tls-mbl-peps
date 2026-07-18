"""A/B harness smoke (§16 P4): both arms run, produce finite ED gaps, and the
report aggregates. The full >= 16-realization measurement is the CLI run
(`tlsmbl ab-test configs/ab_sdrg.yaml`); which arm wins is an empirical output,
not a test assertion (a negative result is a valid exit)."""

from tlsmbl.sdrg.ab import run_ab


def test_ab_two_realizations_L3() -> None:
    report = run_ab(
        master_seed=20260716,
        n_realizations=2,
        L=3,
        g_J=0.3,
        D=2,
        max_outer=15,
        inner_iters=20,
    )
    assert len(report.records) == 2
    for r in report.records:
        assert r.gap_with_sdrg >= 0.0 and r.gap_without_sdrg >= 0.0
        assert r.gap_with_sdrg < 0.1 and r.gap_without_sdrg < 0.1  # both arms sane
        assert not r.sdrg_bypassed
        assert r.n_decimations > 0
    assert "gap ratio" in report.verdict

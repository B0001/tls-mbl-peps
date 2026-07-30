"""P5 exit tests (§16): end-to-end pipeline into zarr, resume-after-kill,
ensemble-level T-DET, INV-5 wiring, aggregation."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from tlsmbl.core.config import Config
from tlsmbl.ensemble.aggregate import aggregate_run
from tlsmbl.ensemble.orchestrate import run_ensemble, run_realization
from tlsmbl.io import store

_BASE = {
    "run": {"name": "t", "master_seed": 20260716, "n_realizations": 2, "workers": 1,
            "out": "PLACEHOLDER"},
    "model": {"L": 3, "delta_min": 1.0e-3, "g_J": 0.3, "R_c": 3, "polaron_kappa": 0.0,
              "hartree": {"enabled": False}},
    "sdrg": {"enabled": True, "omega_stop": 0.3, "f_max": 0.4,
             "keep_first_order": True, "tau_sdrg": 0.05},
    "peps": {"ladder": [2], "chi_factor": 1, "dtype": "complex128"},
    "env": {"eps_env": 1.0e-8, "eps_env_E": 1.0e-6, "polish": True,
            "checkpoint_rows": True, "retry_max": 3, "dchi": "auto"},
    "kernels": {"backend": "exact", "oversample": 8, "power_iters": 1, "eta": 1.0e-6,
                "c_gate": 10.0, "probes": 6, "fallback_disable_rate": 0.2,
                "eps_F": 1.0e-12},
    "optimize": {"su_steps": 0, "inner_iters": 20, "max_outer": 8, "tol_E": 1.0e-8,
                 "tol_g_scale": 1.0e-6},
    # tau_tail is deliberately huge: at g_J = 0.3 the rigorous INV-5 bound
    # 2*pi*g_J/R_c ~ 0.63 rightly dwarfs |e_per_site| -- these tests exercise the
    # wiring, not the physics verdict (test_inv5_tail_gate covers the refusal).
    "invariants": {"tau_chi": 1.0e-4, "tau_tail": 10.0, "allow_uncertified": False},
    "observables": {"tier2": {"enabled": False}},
}


def _cfg(tmp_path: Path, **overrides) -> Config:
    data = yaml.safe_load(yaml.safe_dump(_BASE))
    data["run"]["out"] = str(tmp_path / "run.zarr")
    for dotted, v in overrides.items():
        section, key = dotted.split(".")
        data[section][key] = v
    return Config.model_validate(data)


def test_pipeline_end_to_end_and_aggregate(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out = run_ensemble(cfg)
    root = store.open_run(out)
    assert "manifest" in root.attrs
    for k in range(2):
        g = store.realization_group(root, k)
        assert store.get_stage(g) == "finalized"
        rep = dict(g.attrs["report"])
        assert rep["certified"]  # exact backend, lossless chi, loose tau_tail
        assert rep["tail_bound"] > 0  # INV-5 value present
        obs = dict(g.attrs["observables"])
        assert 0.0 <= obs["q_ea"] <= 1.0
    agg = aggregate_run(out)
    assert agg.n_used == 2
    assert (out / "REPORT.md").exists() or Path(str(out), "REPORT.md").exists()


def test_report_carries_the_definition_of_done_sections(tmp_path: Path) -> None:
    """§18: REPORT.md must contain E(D) extrapolation, q_EA, xi, n_res(r) and the full
    invariant audit. The two verdict-carrying entries (xi, E(D)) must render whichever
    way they resolve, so this asserts the sections are present and self-labeling rather
    than pinning numbers that depend on the optimizer floor."""
    cfg = _cfg(tmp_path, **{"peps.ladder": [2, 3]})
    out = run_ensemble(cfg)
    agg = aggregate_run(out)
    text = Path(str(out), "REPORT.md").read_text()

    assert "q_EA:" in text and "n_res(r):" in text
    assert "## E(D) extrapolation (1/D)" in text
    # A 2-rung ladder is a secant, and the report has to say so rather than dressing it
    # up as a fit (the pilot config ships exactly this ladder).
    assert agg.e_of_d.method == "secant_2pt"
    assert "secant_2pt" in text and "no fit residual" in text
    # xi either resolves with a CI or states why not -- never a bare number.
    assert "xi: " in text
    assert agg.xi.ok == (agg.xi.xi is not None)
    if not agg.xi.ok:
        assert "unresolved" in text and agg.xi.reason in text
    # Full audit, including the entries that were missing before ADR-016.
    for line in (
        "worst discarded weight (INV-1)",
        "worst up/down gap (INV-1)",
        "uncertified excluded",
        "worst sketch gate-fallback rate (INV-3)",
        "sketching auto-disabled (INV-3)",
        "SDRG bypasses (INV-8)",
    ):
        assert line in text, f"missing audit line: {line}"
    # sdrg.enabled is True in _BASE, so the bypass audit must be a number, not "n/a".
    assert agg.n_sdrg_bypassed is not None
    assert "n/a (Stage A off)" not in text


def test_sketched_run_records_the_inv3_audit_trail(tmp_path: Path) -> None:
    """ADR-016: the sketched backend's counters must reach the stored report and the
    REPORT.md audit -- `EnvCertificate.fallback_count` used to be hardcoded to 0."""
    cfg = _cfg(tmp_path, **{"kernels.backend": "sketched"})
    out = run_ensemble(cfg)
    root = store.open_run(out)
    stats = dict(store.realization_group(root, 0).attrs["report"])["sketch_stats"]
    assert stats is not None, "sketched run recorded no INV-3 stats"
    assert stats["call_count"] > 0
    assert stats["sketching_disabled"] is False  # tiny lossless operands: no thrashing
    assert 0.0 <= stats["gate_fallback_rate"] <= 1.0

    agg = aggregate_run(out)
    assert agg.worst_gate_fallback_rate is not None
    assert agg.n_sketch_disabled == 0
    text = Path(str(out), "REPORT.md").read_text()
    assert "not recorded" not in text  # a sketched run must report a real rate


def test_resume_after_kill_loses_at_most_one_rung(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"peps.ladder": [2, 3], "optimize.max_outer": 4})
    out = Path(cfg.run.out)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_realization(cfg, 0, out, _fail_after_stage="rung")
    def fresh_group():  # zarr caches attrs on open handles; re-open to observe writes
        return store.realization_group(store.open_run(out), 0)

    g = fresh_group()
    assert store.get_stage(g) == "rung"
    assert store.rungs_done(g) == [2]  # first rung checkpointed before the crash
    status = run_realization(cfg, 0, out)  # resume completes remaining work
    assert status == "finalized"
    assert store.rungs_done(fresh_group()) == [2, 3]
    # rerun is a no-op
    assert run_realization(cfg, 0, out) == "finalized"


def test_ensemble_t_det_same_key_identical_report(tmp_path: Path) -> None:
    cfg_a = _cfg(tmp_path / "a")
    cfg_b = _cfg(tmp_path / "b")
    run_realization(cfg_a, 0, cfg_a.run.out)
    run_realization(cfg_b, 0, cfg_b.run.out)
    ra = dict(store.realization_group(store.open_run(cfg_a.run.out), 0).attrs["report"])
    rb = dict(store.realization_group(store.open_run(cfg_b.run.out), 0).attrs["report"])
    assert ra["e_total"] == rb["e_total"]  # bitwise (T-DET)
    assert ra["chi_stability"] == rb["chi_stability"]


def test_inv5_tail_gate_marks_uncertified(tmp_path: Path) -> None:
    """tau_tail tiny: the INV-5 bound cannot be met -> artifact stored uncertified
    and excluded from aggregation."""
    cfg = _cfg(tmp_path, **{"invariants.tau_tail": 1.0e-9, "run.n_realizations": 1})
    run_realization(cfg, 0, cfg.run.out)
    g = store.realization_group(store.open_run(cfg.run.out), 0)
    rep = dict(g.attrs["report"])
    assert not rep["inv5_ok"]
    assert not rep["certified"]
    with pytest.raises(RuntimeError, match="no usable"):
        aggregate_run(cfg.run.out)
    agg = aggregate_run(cfg.run.out, allow_uncertified=True)
    assert agg.n_certified == 0 and agg.n_used == 1


def test_disorder_roundtrip(tmp_path: Path) -> None:
    import numpy as np

    from tlsmbl.core.rng import realization_streams
    from tlsmbl.core.types import ModelParams
    from tlsmbl.model.sampling import sample_realization

    params = ModelParams(L=4, g_J=0.3, R_c=3, seed_realization=0)
    real = sample_realization(params, realization_streams(1, 0).disorder)
    root = store.open_run(tmp_path / "x.zarr")
    g = store.realization_group(root, 0)
    store.write_disorder(g, real)
    back = store.read_disorder(g)
    np.testing.assert_array_equal(back.eps, real.eps)
    np.testing.assert_array_equal(back.delta, real.delta)
    assert back.J == real.J
    assert back.params == real.params
    assert dataclasses.asdict(back)["rng_fingerprint"] == real.rng_fingerprint

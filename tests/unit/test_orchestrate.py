"""P5 exit tests (§16): end-to-end pipeline into zarr, resume-after-kill,
ensemble-level T-DET, INV-5 wiring, aggregation."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from tlsmbl.core.config import Config
from tlsmbl.ensemble.aggregate import aggregate_run
from tlsmbl.observables.decoherence import Tier2InputsUnavailable, inputs_from_config
from tlsmbl.ensemble.orchestrate import run_ensemble, run_realization
from tlsmbl.io import store
from tlsmbl.model.hartree import tail_bound

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


def test_tier2_disabled_says_so_in_the_report(tmp_path: Path) -> None:
    """§12: Tier-2's status must never be ambiguous. Off is stated, not omitted."""
    out = run_ensemble(_cfg(tmp_path))
    agg = aggregate_run(out)
    assert agg.tier2 is None
    text = Path(str(out), "REPORT.md").read_text()
    assert "## Tier 2" in text and "not computed" in text
    assert "observables.tier2.enabled = false" in text


def test_tier2_enabled_reaches_the_report_with_echoed_inputs(tmp_path: Path) -> None:
    """§18: REPORT.md carries Tier-2 Gamma_1 with echoed model inputs. Inputs are given
    in units of W (bare numerics) -- unit-tagged strings are refused, see
    test_tier2_unit_tagged_inputs_are_refused."""
    cfg = _cfg(
        tmp_path,
        **{
            "observables.tier2": {
                "enabled": True, "omega_q": "0.5", "g0": "1e-3",
                "gamma0": "1e-4", "T": "0.05",
            }
        },
    )
    out = run_ensemble(cfg)
    root = store.open_run(out)
    rec = dict(store.realization_group(root, 0).attrs["observables"])["tier2"]
    assert rec is not None and rec["tier"] == 2
    # Weights come from the certified state, not the bare fields, because orchestrate
    # passes the measured polarization.
    assert rec["weight_kind"] == "measured_state"
    assert rec["gamma_1"] > 0.0
    assert rec["inputs"]["raw"] == {
        "omega_q": "0.5", "g0": "1e-3", "gamma0": "1e-4", "T": "0.05"
    }

    agg = aggregate_run(out)
    assert agg.tier2 is not None and agg.tier2["ok"] is True
    text = Path(str(out), "REPORT.md").read_text()
    assert "## Tier 2" in text and "NOT certified" in text
    assert "Gamma_1(omega_q):" in text and "spectral-diffusion rms" in text
    assert "omega_q=0.5" in text  # echoed verbatim
    assert "static solver cannot" in text  # DISCLAIMER travels into the report


def test_tier2_unit_tagged_inputs_are_refused(tmp_path: Path) -> None:
    """Converting "5.0GHz" into units of W needs W in that unit, which no layer carries.
    Refusing beats inventing a W: a Tier-2 number looks identical either way."""
    cfg = _cfg(
        tmp_path,
        **{
            "observables.tier2": {
                "enabled": True, "omega_q": "5.0GHz", "g0": "1e-3",
                "gamma0": "1e-4", "T": "0.05",
            }
        },
    )
    # workers=1 propagates directly; the pool path would wrap this in a RuntimeError.
    with pytest.raises(Tier2InputsUnavailable, match="carries a physical unit"):
        run_ensemble(cfg)
    # ...and it names the offending field rather than failing generically.
    with pytest.raises(Tier2InputsUnavailable, match="requires T"):
        inputs_from_config("0.5", "1e-3", "1e-4", None)


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


def test_exact_run_audits_itself_as_exact_rather_than_unrecorded(tmp_path: Path) -> None:
    """ADR-017: an exact backend has no sketch path, and the audit must say that instead
    of the old sentence that also covered artifacts predating the INV-3 audit."""
    cfg = _cfg(tmp_path, **{"kernels.backend": "exact"})
    out = run_ensemble(cfg)
    root = store.open_run(out)
    assert dict(root.attrs["manifest"])["kernel_backend"] == "exact"

    agg = aggregate_run(out)
    assert agg.kernel_backend == "exact"
    assert agg.worst_gate_fallback_rate is None  # nothing to fall back from
    text = Path(str(out), "REPORT.md").read_text()
    assert "n/a (exact backend" in text
    assert "predates the INV-3 audit" not in text


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


def _hartree_cfg(tmp_path: Path, *, sdrg: bool = False, **hartree: object) -> Config:
    h = {"enabled": True, "K_max": 3, "alpha": 0.5, "tol": 1e-12, **hartree}
    return _cfg(
        tmp_path,
        **{
            "model.hartree": h,
            # R_c=1 leaves a real r>R_c tail on the 3x3 lattice for the loop to feed on.
            "model.R_c": 1,
            "sdrg.enabled": sdrg,
        },
    )


def test_hartree_loop_runs_and_reaches_a_self_consistent_field(tmp_path: Path) -> None:
    """§16 P5's last exit item: model.hartree.enabled is actually driven, not ignored."""
    cfg = _hartree_cfg(tmp_path)
    out = run_ensemble(cfg)
    root = store.open_run(out)
    g = store.realization_group(root, 0)
    assert store.get_stage(g) == "finalized"

    rep = dict(g.attrs["report"])
    hr = rep["hartree"]
    assert hr is not None, "hartree.enabled produced no record -- silently no-opped?"
    assert 1 <= hr["n_iters"] <= 3
    assert len(hr["history"]) == hr["n_iters"]
    assert "CORRELATIONS" in hr["bound_covers"]  # INV-5 label, not a shrunken bound
    # The mean field actually moved: h_mf = 0 would leave nothing to converge.
    assert hr["history"][0] > 0.0

    # The stored artifact must describe the H whose energy it reports: the disorder group
    # is written at sample time with h_mf = 0, so finalize re-persists the converged
    # field. Without that, `read_disorder` would hand downstream analysis a different
    # Hamiltonian from the one that was certified.
    real = store.read_disorder(g)
    assert not np.array_equal(real.h_mf, np.zeros_like(real.h_mf))
    # ...and it is the field the FINAL inner solve used (the loop damps after solving,
    # so the last checkpointed field is the one the certified state was optimized in).
    assert np.allclose(real.h_mf, np.array(store.read_hartree(g)["h_mf"]), rtol=0, atol=0)
    # INV-5's reported bound is the same value the baseline reports.
    assert rep["tail_bound"] == pytest.approx(tail_bound(cfg.model.g_J, cfg.model.R_c))
    assert rep["certified"]


def test_hartree_off_records_no_loop_and_is_unchanged(tmp_path: Path) -> None:
    """The v1 baseline path must be untouched by the wiring: no record, h_mf stays 0."""
    out = run_ensemble(_cfg(tmp_path, **{"model.R_c": 1}))
    g = store.realization_group(store.open_run(out), 0)
    assert dict(g.attrs["report"])["hartree"] is None
    real = store.read_disorder(g)
    assert np.array_equal(real.h_mf, np.zeros_like(real.h_mf))


def test_hartree_resume_after_kill_still_loses_at_most_one_rung(tmp_path: Path) -> None:
    """The resume contract must survive the outer loop. Killing mid-ladder must resume
    into the SAME outer iteration -- the rungs on disk belong to the checkpointed field,
    which is why write_hartree happens before the ladder and reset_rungs after it.

    T-DET applies here: a resumed run must be BITWISE identical to an uninterrupted one.

    That assertion caught a real bug during development. `PEPSState.from_product` calls
    `seed_seq.spawn(L*L)`, which is stateful, and the ladder's SeedSequences were
    originally hoisted out of the per-outer-iteration path -- so an uninterrupted run
    drew spawn children L^2..2L^2-1 at outer iteration 2 while a resumed run (which
    skips outer 1's init) drew 0..L^2-1, giving different product inits and a 1.05e-6
    relative energy difference. This is exactly the hazard core/rng.py's docstring warns
    about ("reusing a live object across calls would silently break bit-reproducibility").
    The fix builds a fresh root SeedSequence per inner solve; these assertions are what
    keep it fixed, so do not weaken them to a tolerance.
    """
    cfg = _hartree_cfg(tmp_path, K_max=3)
    out = Path(cfg.run.out)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_realization(cfg, 0, out, _fail_after_stage="rung")

    g = store.realization_group(store.open_run(out), 0)
    mid = store.read_hartree(g)
    assert mid is not None and mid["outer"] == 1
    assert store.rungs_done(g), "the completed rung should have survived the kill"

    assert run_realization(cfg, 0, out) == "finalized"
    resumed = dict(store.realization_group(store.open_run(out), 0).attrs["report"])
    assert resumed["certified"]

    clean_cfg = _hartree_cfg(tmp_path / "clean", K_max=3)
    Path(clean_cfg.run.out).parent.mkdir(parents=True, exist_ok=True)
    run_realization(clean_cfg, 0, clean_cfg.run.out)
    clean = dict(
        store.realization_group(store.open_run(clean_cfg.run.out), 0).attrs["report"]
    )
    assert resumed["hartree"]["n_iters"] == clean["hartree"]["n_iters"]
    # Bitwise, every outer iteration -- not just the one the checkpoint covered.
    assert resumed["hartree"]["history"] == clean["hartree"]["history"]
    assert resumed["e_total"] == clean["e_total"]
    assert resumed["max_disc_weight"] == clean["max_disc_weight"]


def test_hartree_reports_non_convergence_rather_than_hiding_it(tmp_path: Path) -> None:
    """K_max=1 cannot satisfy a tight tol. The artifact must say so."""
    cfg = _hartree_cfg(tmp_path, K_max=1, tol=1e-30)
    out = run_ensemble(cfg)
    hr = dict(store.realization_group(store.open_run(out), 0).attrs["report"])["hartree"]
    assert hr["converged"] is False and hr["n_iters"] == 1


@pytest.mark.parametrize("stage", ["sampled", "sdrg", "hartree", "rung"])
def test_hartree_resume_is_bitwise_from_every_checkpoint_stage(
    tmp_path: Path, stage: str
) -> None:
    """The bitwise resume contract must hold from EVERY stage the pipeline checkpoints,
    not just the mid-ladder one. "hartree" and "sdrg" are the stages the outer loop
    added, and they are the ones where a stale or prematurely-cleared rung would show up:
    a kill at "hartree" lands between the field checkpoint and the ladder that belongs to
    it, which is exactly the window the write-before-ladder ordering exists to make safe.
    """
    # Stage A is off by default in this fixture, so the "sdrg" stage would never fire
    # and the kill would silently no-op -- turn it on for that case specifically.
    use_sdrg = stage == "sdrg"
    cfg = _hartree_cfg(tmp_path, K_max=3, sdrg=use_sdrg)
    out = Path(cfg.run.out)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_realization(cfg, 0, out, _fail_after_stage=stage)
    assert run_realization(cfg, 0, out) == "finalized"
    resumed = dict(store.realization_group(store.open_run(out), 0).attrs["report"])

    clean_cfg = _hartree_cfg(tmp_path / f"clean_{stage}", K_max=3, sdrg=use_sdrg)
    Path(clean_cfg.run.out).parent.mkdir(parents=True, exist_ok=True)
    run_realization(clean_cfg, 0, clean_cfg.run.out)
    clean = dict(
        store.realization_group(store.open_run(clean_cfg.run.out), 0).attrs["report"]
    )
    assert resumed["e_total"] == clean["e_total"], f"resume from {stage!r} not bitwise"
    assert resumed["hartree"]["history"] == clean["hartree"]["history"]
    assert resumed["certified"] and clean["certified"]

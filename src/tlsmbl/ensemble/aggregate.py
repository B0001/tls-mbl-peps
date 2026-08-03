"""Disorder aggregation (ARCHITECTURE.md §11): bootstrap CIs over certified
realizations (uncertified excluded unless allow_uncertified, then labeled), plus a
run-level REPORT.md echoing every invariant statistic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tlsmbl.ensemble.extrapolate import (
    METHOD_SECANT,
    EnsembleExtrapolation,
    extrapolate_ensemble,
    read_rung_energies,
)
from tlsmbl.io import store
from tlsmbl.observables.localization import XiFit, fit_xi

_BOOT = 10_000


@dataclass
class Aggregate:
    n_total: int
    n_certified: int
    n_used: int
    e_per_site: tuple[float, float, float]  # mean, ci_lo, ci_hi
    q_ea: tuple[float, float, float]
    czz_r: dict[int, tuple[float, float, float]]
    n_res_r: dict[int, float]
    # §18 definition of done: REPORT.md carries the E(D) extrapolation and xi. Both are
    # verdict-carrying -- they report "unresolved" with a reason rather than a number
    # they cannot stand behind (see their modules).
    xi: XiFit
    e_of_d: EnsembleExtrapolation
    # Tier-2 (§12), None unless the run enabled it. Disorder means with bootstrap CIs
    # over the same certified set; the declared inputs and DISCLAIMER ride along from the
    # per-realization records, which are the authority on what was declared.
    tier2: dict[str, object] | None
    max_disc_weight: float
    max_updown_gap: float
    # §11 audit. INV-3: worst per-realization gate-fallback rate and how many
    # realizations tripped the auto-disable. INV-8: SDRG bypass count. None where the
    # run did not exercise the mechanism (exact backend / sdrg disabled) -- reported as
    # "n/a" rather than a misleading 0.
    worst_gate_fallback_rate: float | None
    n_sketch_disabled: int | None
    n_sdrg_bypassed: int | None
    worst_sdrg_ledger: float | None
    # ADR-017. The manifest's recorded truncation backend, or None for artifacts written
    # before the field existed. This is what disambiguates "no sketch stats because the
    # backend never sketches" from "no sketch stats because nothing recorded them".
    kernel_backend: str | None


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    means = rng.choice(values, size=(_BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def aggregate_run(path: str | Path, *, allow_uncertified: bool = False) -> Aggregate:
    root = store.open_run(path)
    reports, observables = [], []
    ladders: list[dict[int, float]] = []
    ladder_flags: list[dict[int, bool]] = []
    n_total = n_cert = 0
    for name in store.list_realizations(root):
        g = store.subgroup(root, f"realizations/{name}")
        if store.get_stage(g) != "finalized":
            continue
        n_total += 1
        rep = store.read_attr_dict(g, "report")
        cert = bool(rep["certified"])
        n_cert += cert
        if cert or allow_uncertified:
            rep["_uncertified"] = not cert
            reports.append(rep)
            observables.append(store.read_attr_dict(g, "observables"))
            # Per-rung ladder energies for the E(D) extrapolation. Read from the rung
            # records the ladder already checkpointed, so this never re-optimizes.
            energies, flags = read_rung_energies(g)
            ladders.append(energies)
            ladder_flags.append(flags)
    if not reports:
        raise RuntimeError("no usable realizations (all uncertified or unfinished)")

    rng = np.random.default_rng(0)
    e = np.array([r["e_per_site"] for r in reports])
    q = np.array([o["q_ea"] for o in observables])
    czz: dict[int, list[float]] = {}
    nres: dict[int, list[float]] = {}
    for o in observables:
        for r, v in o["czz_r"].items():
            czz.setdefault(int(r), []).append(v)
        for r, v in o["n_res_r"].items():
            nres.setdefault(int(r), []).append(v)
    # §11 audit rollup. `sketch_stats` is absent for exact-backend runs and for
    # artifacts written before the INV-3 audit landed, hence the None-vs-0 care.
    sk = [r["sketch_stats"] for r in reports if r.get("sketch_stats")]
    byp = [r["sdrg_bypassed"] for r in reports if r.get("sdrg_bypassed") is not None]
    led = [
        r["sdrg_ledger_total"] for r in reports if r.get("sdrg_ledger_total") is not None
    ]
    agg = Aggregate(
        n_total=n_total,
        n_certified=n_cert,
        n_used=len(reports),
        e_per_site=_boot_ci(e, rng),
        q_ea=_boot_ci(q, rng),
        czz_r={r: _boot_ci(np.array(v), rng) for r, v in sorted(czz.items())},
        n_res_r={r: float(np.mean(v)) for r, v in sorted(nres.items())},
        # Fresh generators, not the shared `rng`: threading one generator through
        # three estimators would make each one's numbers depend on the order the
        # others were called in.
        xi=fit_xi(
            [{int(r): float(v) for r, v in o["czz_r"].items()} for o in observables],
            rng=np.random.default_rng(1),
        ),
        e_of_d=extrapolate_ensemble(
            ladders, rng=np.random.default_rng(2), converged=ladder_flags
        ),
        tier2=_aggregate_tier2(observables, np.random.default_rng(3)),
        max_disc_weight=max(r["max_disc_weight"] for r in reports),
        max_updown_gap=max(r["updown_gap"] for r in reports),
        worst_gate_fallback_rate=(
            max(float(s["gate_fallback_rate"]) for s in sk) if sk else None
        ),
        n_sketch_disabled=(
            sum(bool(s["sketching_disabled"]) for s in sk) if sk else None
        ),
        n_sdrg_bypassed=sum(bool(b) for b in byp) if byp else None,
        worst_sdrg_ledger=max(float(x) for x in led) if led else None,
        kernel_backend=(
            store.read_attr_dict(root, "manifest").get("kernel_backend")
            if "manifest" in root.attrs
            else None
        ),
    )
    _write_outputs(root, path, agg)
    return agg


def _aggregate_tier2(
    observables: list[dict[str, Any]], rng: np.random.Generator
) -> dict[str, Any] | None:
    """Disorder-averages the per-realization Tier-2 records, or None if the run had none.

    Refuses to average across *different* declared inputs: Gamma_1 at two different
    omega_q are two different quantities, and a mean of them means nothing. Mixed inputs
    are reported as such instead of being silently pooled.
    """
    recs = [o["tier2"] for o in observables if o.get("tier2")]
    if not recs:
        return None
    declared = {json.dumps(r["inputs"], sort_keys=True) for r in recs}
    if len(declared) > 1:
        return {
            "ok": False,
            "reason": "mixed_declared_inputs",
            "n_realizations": len(recs),
            "disclaimer": recs[0]["disclaimer"],
        }
    out: dict[str, Any] = {
        "ok": True,
        "reason": "ok",
        "n_realizations": len(recs),
        "inputs": recs[0]["inputs"],
        "weight_kind": recs[0]["weight_kind"],
        "disclaimer": recs[0]["disclaimer"],
        "tier": 2,
    }
    for key in ("gamma_1", "spectral_diffusion_rms", "mean_transverse_weight"):
        out[key] = _boot_ci(np.array([float(r[key]) for r in recs]), rng)
    out["min_abs_detuning"] = float(min(float(r["min_abs_detuning"]) for r in recs))
    return out


def _inv3_rate_phrase(rate: float | None, backend: str | None) -> str:
    """The §11 INV-3 audit line (ADR-017).

    A missing rate has three distinct causes and they are not interchangeable: the
    backend had no sketch path, the backend had one but nothing recorded statistics,
    or the artifact predates the audit entirely. Before the manifest recorded the
    backend, all three collapsed into one evasive sentence; each now gets its own.
    """
    if rate is not None:
        return f"{rate:.1%}"
    if backend is None:
        return "not recorded (artifact predates the INV-3 audit)"
    if backend == "exact":
        return "n/a (exact backend -- no sketch path to fall back from)"
    # Backend recorded as sketching, yet no realization reported statistics. That is a
    # wiring anomaly, not a clean bill of health, so the report says so rather than
    # implying the gate was never exercised.
    return (
        f"not recorded, though the manifest records the '{backend}' backend "
        "-- no realization reported sketch statistics"
    )


def _tier2_lines(t: dict[str, Any] | None) -> list[str]:
    """Tier-2 block for REPORT.md. Absent-by-default is stated explicitly rather than
    left as a silent omission -- a missing section reads as an oversight, and §12's whole
    point is that Tier-2's status is never ambiguous."""
    if t is None:
        return [
            "- not computed (observables.tier2.enabled = false). Tier-1 observables "
            "above are the rigorous outputs."
        ]
    if not t.get("ok"):
        return [
            f"- not aggregated ({t['reason']}): {t['n_realizations']} realizations do "
            "not share one set of declared inputs, and Gamma_1 at different declared "
            "inputs are different quantities.",
            f"- {t['disclaimer']}",
        ]
    g, sd, w = t["gamma_1"], t["spectral_diffusion_rms"], t["mean_transverse_weight"]
    raw = (t["inputs"] or {}).get("raw") or {}
    echoed = ", ".join(f"{k}={v}" for k, v in sorted(raw.items())) or ", ".join(
        f"{k}={t['inputs'][k]:.6g}" for k in ("omega_q", "g0", "gamma0", "T")
    )
    return [
        f"- Gamma_1(omega_q): {g[0]:.6e} [{g[1]:.6e}, {g[2]:.6e}] (units of W)",
        f"- spectral-diffusion rms shift: {sd[0]:.6e} [{sd[1]:.6e}, {sd[2]:.6e}]",
        f"- transverse weight [{t['weight_kind']}]: {w[0]:.4f} [{w[1]:.4f}, {w[2]:.4f}]",
        f"- closest resonance |E_i - omega_q| over the ensemble: {t['min_abs_detuning']:.6e}",
        f"- declared model inputs (echoed verbatim): {echoed}",
        f"- {t['disclaimer']}",
    ]


def _e_of_d_lines(e: EnsembleExtrapolation) -> list[str]:
    """Renders the E(D) block. A secant-only ensemble (the 2-rung ladder of
    configs/pilot_L8.yaml) is labeled as such: it carries no fit uncertainty, so
    presenting it like a 3-rung fit would overstate what the ladder measured."""
    if not e.ok or e.e_inf is None or e.remaining_gap is None:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(e.refusals.items()))
        return [
            f"- unresolved ({e.reason}); {e.n_used}/{e.n_total} realizations usable"
            + (f" [{detail}]" if detail else "")
        ]
    lines = [
        f"- method: {e.method} ({e.n_used}/{e.n_total} realizations"
        + (f", {e.n_tainted} tainted by an unconverged rung" if e.n_tainted else "")
        + ")",
        f"- E_inf (total): {e.e_inf[0]:+.9f} [{e.e_inf[1]:+.9f}, {e.e_inf[2]:+.9f}]",
        f"- remaining gap |E(D_max) - E_inf|: {e.remaining_gap[0]:.3e} "
        f"[{e.remaining_gap[1]:.3e}, {e.remaining_gap[2]:.3e}]"
        "  <- unmeasured ansatz truncation, not a bound",
    ]
    if e.method == METHOD_SECANT:
        lines.append(
            "- NOTE: 2-rung ladder -> exact 2-point secant, no fit residual and no "
            "per-realization fit uncertainty. Add a third rung for a fitted E(D)."
        )
    if e.refusals:
        lines.append(
            "- excluded: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(e.refusals.items()))
        )
    return lines


def _write_outputs(root, path: str | Path, agg: Aggregate) -> None:  # type: ignore[no-untyped-def]
    a = root.require_group("aggregate")
    a.attrs["summary"] = json.loads(json.dumps(agg, default=lambda o: o.__dict__))
    lines = [
        "# Run report",
        "",
        f"Realizations: {agg.n_total} finalized, {agg.n_certified} certified, "
        f"{agg.n_used} aggregated.",
        "",
        "## Observables (mean [95% bootstrap CI])",
        f"- e_per_site: {agg.e_per_site[0]:+.9f} [{agg.e_per_site[1]:+.9f}, {agg.e_per_site[2]:+.9f}]",
        f"- q_EA: {agg.q_ea[0]:.6f} [{agg.q_ea[1]:.6f}, {agg.q_ea[2]:.6f}]",
        "- Czz(r): "
        + ", ".join(f"r={r}: {v[0]:+.3e}" for r, v in agg.czz_r.items()),
        "- n_res(r): " + ", ".join(f"r={r}: {v:.2f}" for r, v in agg.n_res_r.items()),
        f"- {agg.xi.summary_line()}",
        "",
        "## E(D) extrapolation (1/D)",
        *_e_of_d_lines(agg.e_of_d),
        "",
        "## Tier 2 — model-dependent decoherence estimates (NOT certified)",
        *_tier2_lines(agg.tier2),
        "",
        "## Invariant audit",
        f"- worst discarded weight (INV-1): {agg.max_disc_weight:.3e}",
        f"- worst up/down gap (INV-1): {agg.max_updown_gap:.3e}",
        f"- uncertified excluded: {agg.n_total - agg.n_certified}",
        "- worst sketch gate-fallback rate (INV-3): "
        + _inv3_rate_phrase(agg.worst_gate_fallback_rate, agg.kernel_backend),
        "- realizations with sketching auto-disabled (INV-3): "
        + ("n/a" if agg.n_sketch_disabled is None else str(agg.n_sketch_disabled)),
        "- SDRG bypasses (INV-8): "
        + (
            "n/a (Stage A off)"
            if agg.n_sdrg_bypassed is None
            else f"{agg.n_sdrg_bypassed}"
            + (
                ""
                if agg.worst_sdrg_ledger is None
                else f", worst ledger {agg.worst_sdrg_ledger:.3f}"
            )
        ),
    ]
    Path(path, "REPORT.md").write_text("\n".join(lines) + "\n")

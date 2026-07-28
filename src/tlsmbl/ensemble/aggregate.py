"""Disorder aggregation (ARCHITECTURE.md §11): bootstrap CIs over certified
realizations (uncertified excluded unless allow_uncertified, then labeled), plus a
run-level REPORT.md echoing every invariant statistic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tlsmbl.io import store

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
    max_disc_weight: float
    max_updown_gap: float


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    means = rng.choice(values, size=(_BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def aggregate_run(path: str | Path, *, allow_uncertified: bool = False) -> Aggregate:
    root = store.open_run(path)
    reports, observables = [], []
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
    agg = Aggregate(
        n_total=n_total,
        n_certified=n_cert,
        n_used=len(reports),
        e_per_site=_boot_ci(e, rng),
        q_ea=_boot_ci(q, rng),
        czz_r={r: _boot_ci(np.array(v), rng) for r, v in sorted(czz.items())},
        n_res_r={r: float(np.mean(v)) for r, v in sorted(nres.items())},
        max_disc_weight=max(r["max_disc_weight"] for r in reports),
        max_updown_gap=max(r["updown_gap"] for r in reports),
    )
    _write_outputs(root, path, agg)
    return agg


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
        "",
        "## Invariant audit",
        f"- worst discarded weight (INV-1): {agg.max_disc_weight:.3e}",
        f"- worst up/down gap (INV-1): {agg.max_updown_gap:.3e}",
        f"- uncertified excluded: {agg.n_total - agg.n_certified}",
    ]
    Path(path, "REPORT.md").write_text("\n".join(lines) + "\n")

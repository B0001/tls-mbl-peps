"""tlsmbl CLI (ARCHITECTURE.md §20).

    tlsmbl run configs/benchmark.yaml    # full pipeline, resumable
    tlsmbl verify runs/bench.zarr        # offline invariant re-check (INV audit)
    tlsmbl bench kernels --D 2 3 4 6 8   # D-scaling microbenchmark -> D3 deliverable
    tlsmbl ab-test configs/ab_sdrg.yaml  # Stage-A value measurement -> D4 deliverable
    tlsmbl aggregate runs/bench.zarr     # D2 deliverable + REPORT.md

P0 wires the command surface and the config/manifest path (`run` validates a config
and prints its manifest). The pipeline bodies land with the phases that implement
them (§16); until then they exit with a clear "not yet implemented" status rather
than pretending to do work.
"""

from __future__ import annotations

from pathlib import Path

import typer

from tlsmbl.core.config import Config
from tlsmbl.io.manifest import build_manifest

app = typer.Typer(no_args_is_help=True, add_completion=False)

_NOT_YET_IMPLEMENTED = 2


def _stub(command: str, lands_in: str) -> None:
    typer.echo(f"tlsmbl {command}: not yet implemented (lands in {lands_in})", err=True)
    raise typer.Exit(code=_NOT_YET_IMPLEMENTED)


@app.command()
def run(config: Path = typer.Argument(..., exists=True, help="Run config YAML.")) -> None:
    """Full pipeline: sample -> SDRG -> ladder -> finalize -> store. Resumable."""
    from tlsmbl.ensemble.orchestrate import run_ensemble

    cfg = Config.from_yaml(config)
    manifest = build_manifest(cfg)
    typer.echo(f"config_hash={manifest.config_hash}")
    typer.echo(f"master_seed={manifest.master_seed}")
    typer.echo(f"git_sha={manifest.git_sha} dirty={manifest.git_dirty}")
    out = run_ensemble(cfg, progress=lambda k, s: typer.echo(f"realization {k}: {s}"))
    typer.echo(f"run complete: {out}")


@app.command()
def verify(artifact: Path = typer.Argument(..., exists=True, help="run.zarr to re-check.")) -> None:
    """Offline invariant re-check (INV audit) on stored reports."""
    from tlsmbl.io import store as _store

    root = _store.open_run(artifact)
    manifest = _store.read_attr_dict(root, "manifest") if "manifest" in root.attrs else {}
    thresholds = dict(manifest.get("invariant_thresholds", {}))
    tau_chi = float(thresholds.get("tau_chi", 1e-6))
    tau_tail = float(thresholds.get("tau_tail", 0.02))
    failures = 0
    for name in _store.list_realizations(root):
        g = _store.subgroup(root, f"realizations/{name}")
        if _store.get_stage(g) != "finalized":
            typer.echo(f"{name}: INCOMPLETE (stage={_store.get_stage(g)})")
            failures += 1
            continue
        rep = _store.read_attr_dict(g, "report")
        checks = {
            "INV-2": (
                len(rep["chi_stability"]) == 2
                and abs(rep["chi_stability"][0] - rep["chi_stability"][1]) <= tau_chi
            ),
            "INV-5": rep["tail_bound"] <= tau_tail * abs(rep["e_per_site"]),
            "certified-flag": bool(rep["certified"])
            or not thresholds.get("allow_uncertified", False),
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad and rep["certified"]:
            typer.echo(f"{name}: FAIL {bad} but stored as certified")
            failures += 1
        else:
            typer.echo(
                f"{name}: ok (certified={rep['certified']}, "
                f"disc={rep['max_disc_weight']:.2e}, gap={rep['updown_gap']:.2e})"
            )
    raise typer.Exit(code=1 if failures else 0)


@app.command()
def bench(
    component: str = typer.Argument(..., help="e.g. 'kernels'"),
    D: list[int] = typer.Option([2, 3, 4, 6], "--D", help="Bond dimensions to sweep."),
    reps: int = typer.Option(5, help="Repetitions per point (median reported)."),
) -> None:
    """D-scaling microbenchmark (D3 deliverable): exact vs sketched kernel."""
    if component != "kernels":
        typer.echo(f"unknown bench component {component!r}", err=True)
        raise typer.Exit(code=1)
    from tlsmbl.kernels.bench import run_kernel_bench

    res = run_kernel_bench(list(D), reps=reps)
    for p in res.points:
        speedup = p.exact_s / p.sketched_s
        typer.echo(
            f"D={p.D} n={p.n:5d}  exact={p.exact_s:.4g}s  sketched={p.sketched_s:.4g}s"
            f"  speedup={speedup:5.1f}x  gate_passed={p.gate_passed}"
        )
    typer.echo(
        f"exponents: exact {res.exact_exponent:.2f}, sketched {res.sketched_exponent:.2f}, "
        f"gap {res.exponent_gap:.2f} (T-PERF gate >= 1.6)"
    )


@app.command()
def aggregate(
    artifact: Path = typer.Argument(..., exists=True, help="run.zarr to aggregate."),
    allow_uncertified: bool = typer.Option(False, help="Include uncertified, labeled."),
) -> None:
    """Disorder-averaged observables + REPORT.md (D2 deliverable)."""
    from tlsmbl.ensemble.aggregate import aggregate_run

    agg = aggregate_run(artifact, allow_uncertified=allow_uncertified)
    typer.echo(
        f"aggregated {agg.n_used}/{agg.n_total} realizations "
        f"({agg.n_certified} certified)"
    )
    typer.echo(f"e_per_site = {agg.e_per_site[0]:+.9f} "
               f"[{agg.e_per_site[1]:+.9f}, {agg.e_per_site[2]:+.9f}]")
    typer.echo(f"q_EA = {agg.q_ea[0]:.6f}; REPORT.md written next to the store")


@app.command(name="ab-test")
def ab_test(config: Path = typer.Argument(..., exists=True, help="A/B config YAML.")) -> None:
    """SDRG preconditioning value measurement (D4 deliverable). A negative result
    is a valid outcome -- Stage A is quarantined by design."""
    import yaml

    from tlsmbl.sdrg.ab import run_ab

    cfg = yaml.safe_load(config.read_text())
    report = run_ab(**cfg)
    for r in report.records:
        typer.echo(
            f"k={r.k}  E_ED={r.e_ed:+.9f}  gap A(sdrg)={r.gap_with_sdrg:.3e}  "
            f"B(off)={r.gap_without_sdrg:.3e}  decimations={r.n_decimations}"
            f"{'  BYPASSED' if r.sdrg_bypassed else ''}  ledger={r.ledger_total:.2e}"
        )
    typer.echo(
        f"mean gap: with SDRG {report.mean_gap_with:.3e}, "
        f"without {report.mean_gap_without:.3e} -> {report.verdict}"
    )


if __name__ == "__main__":
    app()

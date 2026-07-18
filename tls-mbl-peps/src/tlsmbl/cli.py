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
    """Validate CONFIG and print its manifest. Full pipeline lands in Phase 5."""
    cfg = Config.from_yaml(config)
    manifest = build_manifest(cfg)
    typer.echo(f"config_hash={manifest.config_hash}")
    typer.echo(f"master_seed={manifest.master_seed}")
    typer.echo(f"git_sha={manifest.git_sha} dirty={manifest.git_dirty}")
    _stub("run", "Phase 5 (ensemble orchestration)")


@app.command()
def verify(artifact: Path = typer.Argument(..., help="run.zarr to re-check.")) -> None:
    """Offline invariant re-check (INV audit)."""
    _stub("verify", "Phase 5 (io/store.py)")


@app.command()
def bench(
    component: str = typer.Argument(..., help="e.g. 'kernels'"),
    D: list[int] = typer.Option(None, "--D", help="Bond dimensions to sweep."),
) -> None:
    """D-scaling microbenchmark (D3 deliverable)."""
    _stub("bench", "Phase 3 (sketched backend + microbench)")


@app.command()
def aggregate(artifact: Path = typer.Argument(..., help="run.zarr to aggregate.")) -> None:
    """Disorder-averaged observables + REPORT.md (D2 deliverable)."""
    _stub("aggregate", "Phase 5 (ensemble/aggregate.py)")


@app.command(name="ab-test")
def ab_test(config: Path = typer.Argument(..., help="A/B config YAML.")) -> None:
    """SDRG preconditioning value measurement (D4 deliverable)."""
    _stub("ab-test", "Phase 4 (SDRG A/B harness)")


if __name__ == "__main__":
    app()

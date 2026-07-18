from pathlib import Path

from typer.testing import CliRunner

from tlsmbl.cli import app

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"

runner = CliRunner()


def test_run_prints_manifest_then_stubs() -> None:
    result = runner.invoke(app, ["run", str(SMOKE)])
    assert "config_hash=" in result.stdout
    assert "master_seed=20260716" in result.stdout
    assert result.exit_code == 2  # stubbed pipeline: "not yet implemented"


def test_verify_is_stubbed() -> None:
    result = runner.invoke(app, ["verify", "runs/bench.zarr"])
    assert result.exit_code == 2

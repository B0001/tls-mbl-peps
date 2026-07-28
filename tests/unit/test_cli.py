"""CLI surface tests. The heavy pipeline paths (run/aggregate/ab-test bodies) are
covered by tests/unit/test_orchestrate.py and test_ab_harness.py against the
library API; here we check wiring, help, and the verify audit on a real store."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from tlsmbl.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "verify", "bench", "aggregate", "ab-test"):
        assert cmd in result.stdout


def test_run_rejects_missing_config() -> None:
    result = runner.invoke(app, ["run", "does/not/exist.yaml"])
    assert result.exit_code != 0


def test_verify_audits_stored_run(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_orchestrate import _cfg

    from tlsmbl.ensemble.orchestrate import run_ensemble

    cfg = _cfg(tmp_path, **{"run.n_realizations": 1})
    run_ensemble(cfg)  # writes the manifest verify reads thresholds from
    result = runner.invoke(app, ["verify", str(cfg.run.out)])
    assert result.exit_code == 0
    assert "00000: ok" in result.stdout


def test_ab_test_rejects_bad_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "ab.yaml"
    bad.write_text(yaml.safe_dump({"not_a_param": 1}))
    result = runner.invoke(app, ["ab-test", str(bad)])
    assert result.exit_code != 0

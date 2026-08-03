from pathlib import Path

import pytest

from tlsmbl.core.config import Config
from tlsmbl.core.rng import MissingSeedError
from tlsmbl.io.manifest import build_manifest

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"


def test_manifest_records_seed_and_hash() -> None:
    cfg = Config.from_yaml(SMOKE)
    manifest = build_manifest(cfg)
    assert manifest.master_seed == cfg.run.master_seed
    assert manifest.config_hash == cfg.config_hash()
    assert manifest.invariant_thresholds["tau_chi"] == cfg.invariants.tau_chi


def test_manifest_records_package_versions() -> None:
    manifest = build_manifest(Config.from_yaml(SMOKE))
    assert "numpy" in manifest.package_versions
    assert "pydantic" in manifest.package_versions


def test_manifest_records_the_kernel_backend() -> None:
    """ADR-017: the INV-3 audit cannot interpret absent sketch statistics unless the
    artifact says whether a sketch path existed at all."""
    cfg = Config.from_yaml(SMOKE)
    assert build_manifest(cfg).kernel_backend == cfg.kernels.backend


def test_kernel_backend_does_not_perturb_the_config_hash() -> None:
    """T-DET: `kernel_backend` is derived from the config, so recording it must not
    change what `config_hash` commits to."""
    cfg = Config.from_yaml(SMOKE)
    assert build_manifest(cfg).config_hash == cfg.config_hash()
    assert build_manifest(cfg).config_hash == build_manifest(cfg).config_hash


def test_manifest_refuses_without_seed() -> None:
    """INV-6 refusal test: even if the pydantic-level guarantee is bypassed, the
    io/manifest.py gate independently refuses to build a manifest without a seed."""
    cfg = Config.from_yaml(SMOKE)
    object.__setattr__(cfg.run, "master_seed", None)  # frozen model; bypass deliberately
    with pytest.raises(MissingSeedError):
        build_manifest(cfg)

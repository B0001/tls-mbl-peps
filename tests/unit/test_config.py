from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tlsmbl.core.config import Config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SMOKE = CONFIGS / "smoke.yaml"
BENCHMARK = CONFIGS / "benchmark.yaml"


def test_config_loads_smoke_and_benchmark() -> None:
    assert Config.from_yaml(SMOKE).model.L == 4
    assert Config.from_yaml(BENCHMARK).model.L == 16


def test_config_hash_deterministic_same_config() -> None:
    """T-DET (config hashing): same config -> identical hash, every time."""
    h1 = Config.from_yaml(SMOKE).config_hash()
    h2 = Config.from_yaml(SMOKE).config_hash()
    assert h1 == h2


def test_config_hash_differs_for_different_config() -> None:
    assert Config.from_yaml(SMOKE).config_hash() != Config.from_yaml(BENCHMARK).config_hash()


def test_config_hash_changes_with_any_field() -> None:
    data = yaml.safe_load(SMOKE.read_text())
    base = Config.model_validate(data).config_hash()
    data["model"]["R_c"] = data["model"]["R_c"] + 1
    changed = Config.model_validate(data).config_hash()
    assert base != changed


def test_config_rejects_unknown_key() -> None:
    data = yaml.safe_load(SMOKE.read_text())
    data["not_a_real_section"] = {"x": 1}
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_config_requires_master_seed() -> None:
    """INV-6: the config schema itself refuses to build without a seed."""
    data = yaml.safe_load(SMOKE.read_text())
    del data["run"]["master_seed"]
    with pytest.raises(ValidationError):
        Config.model_validate(data)

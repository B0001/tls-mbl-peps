"""Configuration schema (ARCHITECTURE.md §13; pydantic v2, single YAML).

Every field is validated with ranges; unknown keys are a hard error (`extra="forbid"`).
`Config.config_hash()` is the canonical-JSON SHA256 that `io/manifest.py` records
(INV-6) and that `T-DET` checks for stability under re-serialization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    name: str
    master_seed: int
    n_realizations: int = Field(gt=0)
    workers: int = Field(gt=0)
    out: str


class HartreeConfig(StrictModel):
    enabled: bool = False
    K_max: int = Field(default=8, gt=0)
    alpha: float = Field(default=0.5, gt=0, le=1)
    tol: float = Field(default=1.0e-4, gt=0)


class ModelConfig(StrictModel):
    L: int = Field(gt=0)
    delta_min: float = Field(default=1.0e-3, gt=0, lt=1)
    g_J: float = Field(default=1.0e-3, gt=0)
    R_c: int = Field(default=3, gt=0)
    polaron_kappa: float = Field(default=0.0, ge=0)
    hartree: HartreeConfig = Field(default_factory=HartreeConfig)


class SdrgConfig(StrictModel):
    enabled: bool = True
    omega_stop: float = Field(gt=0)
    f_max: float = Field(gt=0, le=1)
    keep_first_order: bool = True
    tau_sdrg: float = Field(gt=0)


class PepsConfig(StrictModel):
    ladder: list[int] = Field(min_length=1)
    chi_factor: int = Field(default=1, gt=0)
    dtype: Literal["complex128", "complex64"] = "complex128"


class EnvConfig(StrictModel):
    eps_env: float = Field(gt=0)
    eps_env_E: float = Field(gt=0)
    polish: bool = True
    checkpoint_rows: bool = True
    retry_max: int = Field(default=3, ge=0)
    dchi: str | int = "auto"
    # ADR-007 v1.1 / ADR-015: compress from the factored (M, a) representation,
    # never materializing Theta(chi^2 D^6) fat tensors. The D >= 6 ladder rungs
    # require this; default off pending nothing -- it is equivalence-tested and
    # strictly lighter, but v1 remains the reference path.
    factored: bool = False


class KernelsConfig(StrictModel):
    backend: Literal["exact", "sketched"] = "sketched"
    oversample: int = Field(default=8, ge=0)
    power_iters: int = Field(default=1, ge=0)
    eta: float = Field(gt=0)
    c_gate: float = Field(gt=0)
    probes: int = Field(default=6, gt=0)
    fallback_disable_rate: float = Field(gt=0, le=1)
    eps_F: float = Field(gt=0)


class OptimizeConfig(StrictModel):
    su_steps: int = Field(ge=0)
    inner_iters: int = Field(gt=0)
    max_outer: int = Field(gt=0)
    tol_E: float = Field(gt=0)
    tol_g_scale: float = Field(gt=0)


class InvariantsConfig(StrictModel):
    tau_chi: float = Field(gt=0)
    tau_tail: float = Field(gt=0)
    allow_uncertified: bool = False


class Tier2Config(StrictModel):
    # omega_q/g0/gamma0/T carry explicit units (e.g. "5.0GHz") and are echoed
    # verbatim into reports (§12) -- not parsed as floats at the config layer.
    enabled: bool = False
    omega_q: str | None = None
    g0: str | None = None
    gamma0: str | None = None
    T: str | None = None


class ObservablesConfig(StrictModel):
    tier2: Tier2Config = Field(default_factory=Tier2Config)


class Config(StrictModel):
    run: RunConfig
    model: ModelConfig
    sdrg: SdrgConfig
    peps: PepsConfig
    env: EnvConfig
    kernels: KernelsConfig
    optimize: OptimizeConfig
    invariants: InvariantsConfig
    observables: ObservablesConfig = Field(default_factory=ObservablesConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def canonical_json(self) -> str:
        """Stable serialization: sorted keys, no whitespace -- the T-DET input."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

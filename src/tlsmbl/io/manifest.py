"""INV-6 manifest (ARCHITECTURE.md §11).

Records everything needed to reproduce or audit a run: config hash, git SHA + dirty
flag, package versions, master seed, invariant thresholds, and the truncation kernel
backend. `tlsmbl verify run.zarr` (Phase 5+) re-derives this hash offline against the
stored artifact.

`kernel_backend` (ADR-017) is the backend the run was *configured* with. It is
redundant with `config_hash` in the cryptographic sense and deliberately so: the hash
proves which config ran but cannot be read, and the §11 INV-3 audit has to interpret
the *absence* of sketch statistics, which is only meaningful once the artifact says
whether a sketch path existed at all. The backend a given realization *ended up*
running is per-realization state, not run state -- the manifest is written before any
realization starts, so it cannot honestly carry an outcome; ADR-016's auto-disable is
recorded per realization in `report.sketch_stats` instead.

The field is not part of `Config.config_hash()`'s input (that hash is over the config
alone), so adding it leaves T-DET untouched.
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tlsmbl.core.config import Config
from tlsmbl.core.rng import require_master_seed

_TRACKED_PACKAGES = ("numpy", "scipy", "torch", "pydantic", "zarr", "typer")


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_hash: str
    git_sha: str | None
    git_dirty: bool
    package_versions: dict[str, str]
    master_seed: int
    invariant_thresholds: dict[str, float | bool]
    # ADR-017. Typed `str`, not the config's Literal: a future backend (the ADR-008
    # Rust kernel) must be recordable in an artifact without a schema bump, and readers
    # of old artifacts must tolerate values this build has never heard of.
    kernel_backend: str


def _git_info(repo_root: Path | None = None) -> tuple[str | None, bool]:
    """Best-effort git SHA + dirty flag; (None, False) outside a git checkout."""
    cwd = repo_root or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_manifest(config: Config, *, repo_root: Path | None = None) -> Manifest:
    """INV-6 gate: refuses to build a manifest without a master seed."""
    master_seed = require_master_seed(config.run.master_seed)
    git_sha, git_dirty = _git_info(repo_root)
    return Manifest(
        config_hash=config.config_hash(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        package_versions=_package_versions(),
        master_seed=master_seed,
        invariant_thresholds=dict(config.invariants.model_dump()),
        kernel_backend=config.kernels.backend,
    )

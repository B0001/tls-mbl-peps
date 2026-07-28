"""INV-6 manifest (ARCHITECTURE.md §11).

Records everything needed to reproduce or audit a run: config hash, git SHA + dirty
flag, package versions, master seed, and invariant thresholds. `tlsmbl verify
run.zarr` (Phase 5+) re-derives this hash offline against the stored artifact.
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
    )

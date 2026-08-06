#!/usr/bin/env python3
"""Drive the tlsmbl solver: environment gate, end-to-end CLI run, internal API.

Run it through uv from the repo root -- never bare `python`, the venv is the
only place torch and tlsmbl exist:

    uv run python .claude/skills/run-tlsmbl/driver.py all

Stages (each is also a standalone subcommand):

    env    torch importable + `make verify` tier-1 gate (51/51, 9/9)
    run    tlsmbl run -> verify -> aggregate on a scratch copy of smoke.yaml
    resume re-run the same config; asserts the stage markers short-circuit it
    bench  kernel D-scaling, asserts the T-PERF exponent gap >= 1.6
    api    direct invocation: mint an EnergyReport, prove the INV-1 gate fires

Artifacts land in a scratch dir (default: .driver-out/, override with
--out) so nothing here ever touches the tracked runs/ stores.

ponytail: subprocess + asserts, no pytest. This drives the app, it is not
part of the suite -- `uv run pytest tests` is still the real test gate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sh(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Run in the repo root, stream nothing, return (exit code, combined output)."""
    t0 = time.time()
    shown = " ".join("<inline script>" if "\n" in c else c for c in cmd)
    print(f"  $ {shown}", flush=True)
    p = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout
    )
    out = p.stdout + p.stderr
    print(f"    ({time.time() - t0:.1f}s, exit={p.returncode})", flush=True)
    return p.returncode, out


def tail(out: str, n: int = 12) -> str:
    return "\n".join("    | " + ln for ln in out.strip().splitlines()[-n:])


# --- stages ---------------------------------------------------------------


def stage_env(_out: Path) -> None:
    """The gate CLAUDE.md makes mandatory. torch first: `uv sync` drops it."""
    rc, o = sh(["uv", "run", "python", "-c", "import torch;print(torch.__version__)"])
    assert rc == 0, (
        "torch missing from the venv. `uv sync` UNINSTALLS it -- torch is "
        "deliberately outside uv.lock. Fix:\n"
        "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    print(f"    torch {o.strip()}")

    rc, o = sh(["make", "verify"], timeout=900)
    assert rc == 0, f"make verify failed:\n{tail(o, 30)}"
    assert "ALL PASS (51/51)" in o, f"contraction oracle regressed:\n{tail(o, 30)}"
    assert "ALL PASS (9/9)" in o, f"SDRG oracle regressed:\n{tail(o, 30)}"
    print("    tier-1 gate: 51/51 and 9/9")


def _smoke_config(out: Path) -> Path:
    """Copy configs/smoke.yaml with its output redirected into the scratch dir."""
    store = out / "smoke.zarr"
    cfg = out / "smoke.yaml"
    cfg.write_text(
        (REPO / "configs" / "smoke.yaml").read_text().replace("runs/smoke.zarr", str(store))
    )
    return cfg


def stage_run(out: Path) -> None:
    """The actual app, end to end: sample -> SDRG -> ladder -> finalize -> store."""
    cfg = _smoke_config(out)
    store = out / "smoke.zarr"
    shutil.rmtree(store, ignore_errors=True)

    rc, o = sh(["uv", "run", "tlsmbl", "run", str(cfg)], timeout=1800)
    assert rc == 0, f"run failed:\n{tail(o, 25)}"
    assert "run complete" in o, f"no completion line:\n{tail(o)}"
    # Both realizations must reach the terminal stage, not just exit 0.
    assert o.count("finalized") == 2, f"expected 2 finalized realizations:\n{tail(o)}"
    print(tail(o, 6))

    # Offline invariant re-check on what was just written.
    rc, o = sh(["uv", "run", "tlsmbl", "verify", str(store)])
    assert rc == 0, f"verify failed:\n{tail(o)}"
    assert "FAIL" not in o and "INCOMPLETE" not in o, f"invariant audit unhappy:\n{tail(o)}"
    assert o.count("certified=True") == 2, f"realizations not certified:\n{tail(o)}"
    print(tail(o, 4))

    rc, o = sh(["uv", "run", "tlsmbl", "aggregate", str(store)])
    assert rc == 0, f"aggregate failed:\n{tail(o)}"
    assert "e_per_site" in o, f"no energy in aggregate output:\n{tail(o)}"
    # "written next to the store" is a lie -- it goes INSIDE the .zarr.
    report = store / "REPORT.md"
    assert report.is_file(), f"REPORT.md not at {report}"
    print(tail(o, 4))
    print(f"    REPORT.md: {report} ({report.stat().st_size} bytes)")


def stage_resume(out: Path) -> None:
    """Zarr stage markers make a completed run a near-no-op. Guards regressions."""
    cfg = _smoke_config(out)
    assert (out / "smoke.zarr").exists(), "run stage must come first"
    t0 = time.time()
    rc, o = sh(["uv", "run", "tlsmbl", "run", str(cfg)], timeout=1800)
    dt = time.time() - t0
    assert rc == 0, f"resume failed:\n{tail(o)}"
    # A cold run is ~35s; resume is ~2s. 15s is slack, not a timing assertion.
    assert dt < 15, f"resume took {dt:.1f}s -- stage markers are not short-circuiting"
    print(f"    resumed in {dt:.1f}s (cold run is ~35s)")


def stage_bench(out: Path) -> None:
    """Kernel D-scaling. The sketched backend must out-scale exact by >= 1.6."""
    rc, o = sh(
        ["uv", "run", "tlsmbl", "bench", "kernels", "--D", "2", "--D", "3", "--D", "4"],
        timeout=900,
    )
    assert rc == 0, f"bench failed:\n{tail(o)}"
    gap_line = next((ln for ln in o.splitlines() if "gap" in ln), "")
    assert gap_line, f"no exponent line:\n{tail(o)}"
    gap = float(gap_line.split("gap")[1].split("(")[0].strip())
    assert gap >= 1.6, f"T-PERF gate: gap {gap} < 1.6\n{tail(o)}"
    print(tail(o, 5))
    (out / "bench.txt").write_text(o)


def stage_api(_out: Path) -> None:
    """Direct invocation -- the layer most changes touch. No CLI, no zarr."""
    code = r"""
import numpy as np
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.energy import EnvironmentNotConverged, energy_certified
from tlsmbl.peps.state import PEPSState

L, D, seed = 3, 2, 11
params = ModelParams(L=L, g_J=0.3, R_c=3, seed_realization=seed)
real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(seed)))
terms = build_terms(real)
state = PEPSState.random(L, D, TensorSpec(), np.random.SeedSequence(seed))

# chi=4 is lossless for D=2: the report mints and carries its certificate.
rep = energy_certified(state, terms, chi=4, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7)
assert rep.certified, "expected a certified report at lossless chi"
assert rep.e_per_site == rep.e_total / (L * L)
print(f"e_total={rep.e_total:.12f}  e_per_site={rep.e_per_site:.12f}")
print(f"certified={rep.certified}  disc={rep.env.max_disc_weight:.3e}"
      f"  updown_gap={rep.env.updown_gap:.3e}")

# INV-1 is a constructor gate, not a test assert: chi=1 discards real weight,
# so the report must be impossible to mint at all.
try:
    energy_certified(state, terms, chi=1, backend=ExactSVD(), eps_env=1e-8, eps_env_E=1e-7)
except EnvironmentNotConverged as e:
    assert "INV-1" in str(e), f"wrong gate fired: {e}"
    print(f"INV-1 refused chi=1 as designed: {str(e)[:80]}")
else:
    raise SystemExit("INV-1 did NOT fire at chi=1 -- certification is broken")
"""
    rc, o = sh(["uv", "run", "python", "-c", code], timeout=900)
    assert rc == 0, f"internal API smoke failed:\n{tail(o, 25)}"
    print(tail(o, 6))


STAGES = {
    "env": stage_env,
    "run": stage_run,
    "resume": stage_resume,
    "bench": stage_bench,
    "api": stage_api,
}
ORDER = ["env", "run", "resume", "bench", "api"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="*", default=["all"], help=f"{' '.join(ORDER)} | all")
    ap.add_argument("--out", default=str(REPO / ".driver-out"), help="scratch dir")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ORDER if "all" in args.stages else args.stages
    bad = [n for n in names if n not in STAGES]
    if bad:
        print(f"unknown stage(s): {bad}; pick from {ORDER}", file=sys.stderr)
        return 2

    print(f"repo={REPO}\nout={out}\nstages={names}\n")
    failed = []
    for name in names:
        print(f"[{name}]", flush=True)
        try:
            STAGES[name](out)
            print(f"[{name}] PASS\n", flush=True)
        except (AssertionError, subprocess.TimeoutExpired) as e:
            print(f"[{name}] FAIL: {e}\n", file=sys.stderr, flush=True)
            failed.append(name)

    print("=" * 60)
    print(f"FAILED: {failed}" if failed else f"ALL PASS ({len(names)} stages)")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())

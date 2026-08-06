---
name: run-tlsmbl
description: Build, run, verify, and benchmark the tlsmbl PEPS solver — the `tlsmbl` CLI, its end-to-end pipeline on a config, the certified-energy internal API, the Docker/Kubernetes batch image, and the test suite. Use when asked to run, start, launch, smoke-test, benchmark, verify, containerize, or reproduce numbers from this solver, or to confirm a change works in the real app rather than only in tests.
---

# Running tlsmbl

`tlsmbl` is a batch scientific solver, not a service — no window, no port, no UI.
`tlsmbl run <config.yaml>` is run-to-completion and writes a `.zarr` store. So the
agent path is a driver script that exercises the CLI end to end, plus a direct
`import`-and-call path for the internals most changes actually touch.

All paths below are relative to the repo root. Everything goes through `uv`;
never `pip install`, never `source .venv/bin/activate`.

## Prerequisites

`uv` and (for the container path only) Docker. No system packages needed —
verified on macOS/arm64, Python 3.14.6, uv 0.11.29.

```bash
uv sync --extra dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**The order matters and the second line is not optional.** torch is deliberately
outside `uv.lock`, so `uv sync` *uninstalls* it — see Gotchas.

`jax` is not installed and is not needed; `make verify-jax` is tier-2 prototype
parity only. Skip it.

## Run (agent path)

```bash
uv run python .claude/skills/run-tlsmbl/driver.py all
```

~50 s total. Prints a per-stage PASS/FAIL and exits non-zero if any stage fails.
Stages, runnable individually or in any subset (`driver.py api`, `driver.py run bench`):

| Stage | What it proves | Time |
|---|---|---|
| `env` | torch importable; `make verify` prints `ALL PASS (51/51)` and `(9/9)` | ~3 s |
| `run` | `tlsmbl run` → `verify` → `aggregate` on a scratch smoke config; both realizations `finalized` and `certified=True`; `REPORT.md` written | ~40 s |
| `resume` | re-running the same config short-circuits on zarr stage markers (asserts < 15 s) | ~2 s |
| `bench` | kernel D-scaling; asserts the T-PERF exponent gap ≥ 1.6 | ~1 s |
| `api` | direct invocation: mints an `EnergyReport`, then proves the INV-1 gate *refuses* to mint one at χ=1 | ~1 s |

Artifacts land in `.driver-out/` (gitignored, `--out` to change). The driver never
writes to the tracked `runs/` stores.

Expected tail:

```
[api] PASS

============================================================
ALL PASS (5 stages)
```

### Direct invocation (no CLI, no zarr)

The layer most changes touch. Import and call — this is the pattern `driver.py api`
runs, and the fastest way to check a change to the energy/contraction path:

```bash
uv run python -c "
import numpy as np
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.energy import energy_certified
from tlsmbl.peps.state import PEPSState

params = ModelParams(L=3, g_J=0.3, R_c=3, seed_realization=11)
real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(11)))
state = PEPSState.random(3, 2, TensorSpec(), np.random.SeedSequence(11))
rep = energy_certified(state, build_terms(real), chi=4, backend=ExactSVD(),
                       eps_env=1e-8, eps_env_E=1e-7)
print(rep.e_total, rep.certified, rep.env.max_disc_weight)
"
```

Verified output, bit-stable across repeat runs (L=3, D=2, seed 11, χ=4):

```
0.10954031554793286 True 0.0
```

## Run (human path)

```bash
uv run tlsmbl run configs/smoke.yaml     # ~35 s cold, ~2 s on re-run (resumes)
uv run tlsmbl verify runs/smoke.zarr     # offline invariant re-check
uv run tlsmbl aggregate runs/smoke.zarr  # bootstrap CIs + REPORT.md
uv run tlsmbl bench kernels --D 2 --D 3 --D 4
```

Note `--D` repeats per value (`--D 2 --D 3`), it is not variadic — `--D 2 3 4` fails.

Configs in `configs/`: `smoke` (L=4, 2 realizations, ~35 s) → `hartree_L4` →
`ab_sdrg` → `pilot_L8` → `benchmark` / `bench_L16_D4`. Start with `smoke`.

## Test

```bash
uv run pytest tests -q                       # 228 passed, 2 skipped, ~6 min
uv run ruff check .                          # All checks passed!
uv run mypy --strict src                     # Success: no issues found in 48 source files
TLSMBL_FULL_GOLD=1 uv run pytest tests -q    # 240 passed, 5:28
```

Run the suite detached — it takes ~6 min and `-q` prints nothing at all until it
finishes, so it looks hung the whole time.

The README says "138 passed"; that is stale, 228 is current. `TLSMBL_FULL_GOLD=1`
turns the 2 skips into real tests and adds the slow golden/perf gates (L=4 golden
grid, strong-coupling ED, T-PERF exponent) for 240 total — it is not meaningfully
slower, so prefer it when validating a change to the energy or kernel path.

A `UserWarning: torch.linalg.svd failed to converge (gesdd); falling back to
scipy gesvd` in `test_svd_convergence_fallback` is the test doing its job, not a
failure.

## Container

Builds and runs; 588 MB, ~26 s incremental rebuild.

```bash
docker build -t tls-mbl-peps:0.0.1 .
mkdir -p .driver-out/dockerruns
docker run --rm -v "$PWD/.driver-out/dockerruns:/app/runs" \
    tls-mbl-peps:0.0.1 run configs/smoke.yaml
```

Verified: writes `.driver-out/dockerruns/smoke.zarr`, ~60 s (slower than host —
the image pins `OMP/OPENBLAS/MKL_NUM_THREADS=1`). `ENTRYPOINT` is `tlsmbl`, so
pass the subcommand only (`run configs/smoke.yaml`, not `tlsmbl run ...`).

`k8s/` deploys this as a batch Job (`kubectl apply -k k8s/overlays/dev`). Not
verified — no cluster in this environment.

## Gotchas

- **`uv sync` uninstalls torch.** It is outside `uv.lock` on purpose, so any
  `uv sync` / `uv sync --extra dev` prunes it and the next `tlsmbl run` dies at
  `import torch` in `ensemble/orchestrate.py`. Re-run the CPU-index install after
  *every* sync. `driver.py env` checks this first for exactly this reason.
- **The same trap bit the Dockerfile** — its `uv pip install torch` sat *before*
  the final `uv sync --frozen --no-dev`, which pruned it. The image built green
  and failed at runtime. Fixed by moving the torch layer last; if you reorder
  those layers, `docker run ... run configs/smoke.yaml` is what catches it, not
  `docker build`.
- **`aggregate` says "REPORT.md written next to the store" — it isn't.** It lands
  *inside* the store: `runs/smoke.zarr/REPORT.md`.
- **`run.out` feeds `config_hash`.** Redirecting output to a scratch path changes
  the hash (`configs/smoke.yaml` → `67e9cc5e…`, same config with a different `out`
  → `d71d4e90…`), so driver runs are not hash-comparable with `runs/*.zarr`.
- **Re-running a finished config looks like it re-ran but didn't.** It reprints
  every `realization N: finalized` line in ~2 s off zarr stage markers. To force
  real work, delete the store.
- **`tlsmbl ab-test configs/ab_sdrg.yaml` is far slower than it looks.** 16
  realizations × two arms × an ED oracle, single-process (~1 core), and it
  buffers *all* output — so it prints nothing and looks hung. It ran 55 min wall
  here without finishing and was killed; **not verified end to end.** Run it
  detached and budget hours, or cut `n_realizations` in the config first.
- **`git_sha=None dirty=False` inside the container is expected**, not a bug —
  `.dockerignore` excludes `.git`. Host runs report the real SHA.
- Never relax a tolerance to make `make verify` pass (CLAUDE.md §0). A failure
  there is an environment problem.
- `prototypes/` is a frozen reference oracle. Never import it from `src/`, never
  overwrite `prototypes/baselines/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` (host) | You ran `uv sync`. Re-run `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`. |
| `ModuleNotFoundError: No module named 'torch'` (container) | The torch layer is before the last `uv sync` in the Dockerfile. Move it after. |
| `Got unexpected extra argument(s) (3 4)` from `bench kernels` | `--D` is not variadic: use `--D 2 --D 3 --D 4`. The README's `--D 2 3 4 6 8` is wrong. |
| `tlsmbl run` returns in ~2 s and claims success | It resumed. Delete the `.zarr` to force a cold run. |
| `aggregate` succeeded but you can't find `REPORT.md` | It's inside the `.zarr` directory. |
| `driver.py` can't find `uv`/`tlsmbl` | Run it via `uv run python ...` from the repo root, not bare `python`. |

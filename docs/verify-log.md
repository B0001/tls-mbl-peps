# verify-log — fresh-session environment gate

Run 2026-07-17. Python 3.12.0 via `uv venv` at `tls-mbl-peps/.venv`; deps installed with
`uv pip install -e ".[dev]"`, `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`,
`uv pip install jax`. torch==2.13.0 (CPU wheel, confirmed no CUDA link issue), jax==0.11.0.

## `make verify`

```
golden_3x3.py  -> ALL PASS (51/51)
sdrg_3site.py  -> ALL PASS (9/9)
bench_kernel.py timing 2 3 4 -> ran clean (no gate assertions in `timing` subcommand)
```

All 51/9 checks green, matching `docs/HANDOFF.md`'s validated-numbers table (contraction
err ≤ 2.4e-15, disc ≡ 0 at χ=D²; SDRG rule↔Schur identity ≤ 3.5e-16, convergence order → 3.00).

## `make verify-jax`

```
consistency D=2: rel 2.032e-15   (HANDOFF: 2.8e-15)
consistency D=3: rel 2.218e-14   (HANDOFF: 1.4e-14)
T-AD-FD chi=4 (lossless, disc=0):     max rel err 1.337e-09  (< 1e-6 required)
T-AD-FD chi=2 (truncating, disc=1.93e-02): max rel err 4.763e-09  (< 1e-6 required)
```

Both χ configurations pass, including the genuinely truncating χ=2 case per the P2 parity
contract in CLAUDE.md. Numbers are the same order of magnitude as HANDOFF's reference run
(8.2e-10 / 4.7e-9) — expected BLAS/platform-level variation, well inside tolerance.

**Verdict: environment gate PASSED. Proceeding to P0.**

## Production port session (2026-07-17/18)

P0-P5 implemented and committed phase-by-phase (see git log). Final state:
`uv run pytest tests` -> 124 passed; `TLSMBL_FULL_GOLD=1` adds the slow gates
(L=4 golden grid, strong-coupling T-GOLD-ED, T-PERF exponent gap: measured 4.16).
ruff + mypy --strict clean. `make verify` / `make verify-jax` (prototype tier)
unchanged and green.

## Python 3.14 migration + repo-root promotion (2026-07-28)

The project moved out of the `tls-mbl-peps/` subdirectory to the repo root, and the
interpreter moved 3.12.0 -> 3.14.6. Re-ran every gate at the new root on standard
(GIL) CPython 3.14.6, torch 2.13.0 CPU, jax 0.11.0, quimb 1.14.0, numpy 2.4.6:

```
uv run pytest tests            -> 138 passed, 2 skipped
make verify                    -> ALL PASS (51/51), ALL PASS (9/9)
make verify-jax                -> consistency D=2 2.032e-15,  D=3 2.218e-14
                                  T-AD-FD chi=4 (disc=0)       1.337e-09
                                  T-AD-FD chi=2 (disc=1.93e-2) 4.763e-09
uv run mypy                    -> Success: no issues found in 46 source files
uv run ruff check .            -> All checks passed!
tlsmbl run configs/smoke.yaml  -> 42 s, both realizations certified (disc <= 1.2e-22)
tlsmbl verify runs/smoke.zarr  -> clean
```

The jax-tier numbers are **bit-identical** to the 3.12 run recorded above, so the
interpreter bump moved no numerics.

**3.14 is the ceiling.** torch 2.13.0 publishes `cp310`-`cp314` wheels only; 3.15 has
none. `requires-python` is now `>=3.11,<3.15` so this fails at resolve time with a clear
message instead of at `import torch`. Free-threaded 3.14 has a `cp314t` torch wheel and
the suite does pass on it, but certification runs are not validated there — use a
standard build.

**Two environment defects fixed while re-verifying:**
- The tracked `uv.lock` pinned numba 0.53.1 (via quimb), which cannot build on Python
  >=3.10, so `uv sync --extra dev` failed outright. Regenerated: numba is now 0.66.0 and
  `uv sync --extra dev` works.
- `ruff` is unpinned in the dev extra and its *default* rule set has widened across
  releases; under 0.16.0 the previously-clean tree reported 53 findings, so "ruff clean"
  had quietly stopped being reproducible. `[tool.ruff.lint] select` is now explicit and
  `prototypes/` is excluded. None of the findings were real bugs — see the pyproject
  comments for the audit (B008 is the typer idiom; B023/F821 are same-iteration lambdas).

## Smoke budget gate (2026-07-19)

`tlsmbl run configs/smoke.yaml` end-to-end on an 8-core/16GB M-series laptop:
**33 s wall** (§16 budget: < 5 min on 4 cores). Both realizations certified
(disc <= 1.2e-22, up/down gap <= 8.9e-16); `verify` clean; `aggregate` wrote
REPORT.md (q_EA = 0.907 at g_J = 1e-3 — deep localized regime).

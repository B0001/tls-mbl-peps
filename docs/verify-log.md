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

## Smoke budget gate (2026-07-19)

`tlsmbl run configs/smoke.yaml` end-to-end on an 8-core/16GB M-series laptop:
**33 s wall** (§16 budget: < 5 min on 4 cores). Both realizations certified
(disc <= 1.2e-22, up/down gap <= 8.9e-16); `verify` clean; `aggregate` wrote
REPORT.md (q_EA = 0.907 at g_J = 1e-3 — deep localized regime).

# CLAUDE.md — tlsmbl

Variational PEPS solver for many-body localization of TLS defects in amorphous
Al₂O₃ (superconducting-qubit decoherence), with SDRG preconditioning, sketch-gated
truncation, and certified-by-construction energy reporting.
**ARCHITECTURE.md is the normative spec. Its §0 rules bind you.**

## Read order — before writing any code
1. `ARCHITECTURE.md` §0 (rules for the implementing agent), then §3 (invariants), §6 (index conventions).
2. `docs/HANDOFF.md` — what is already validated and the exact numbers your code must reproduce.
3. The Gotchas section below.

## Step 0 — environment gate (always, before anything else)
**Everything goes through `uv`.** Never `pip install`, never `source .venv/bin/activate`.
```
uv sync --extra dev                 # Python 3.11-3.14; .python-version pins 3.14
# torch (needed from Phase 2):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
#   Do NOT install torch from default PyPI: that linux wheel is CUDA-linked
#   (observed failure in prototyping: missing libcublasLt on CPU-only hosts).
# jax (optional, tier-2 prototype parity only):  uv pip install jax
make verify        # tier-1 baselines: must show "ALL PASS (51/51)", "ALL PASS (9/9)"
make verify-jax    # if jax present: consistency ~1e-15; T-AD-FD ≤1e-6 at χ=4 AND χ=2
uv run pytest tests   # production suite: 228 passed, 2 skipped (~11 min)
```
If `make verify` fails: **stop and fix the environment.** Never touch a tolerance to pass.

**Python ceiling is 3.14, and it is a hard one:** torch 2.13.0 publishes `cp310`–`cp314`
wheels only, so 3.15 cannot work. `requires-python = ">=3.11,<3.15"` makes that fail at
resolve time. Use a standard CPython build, not free-threaded: the suite does pass on
no-GIL 3.14, but certification runs are not validated there.

## Non-negotiables (distilled; full text in ARCHITECTURE.md §0/§3)
- Phase order (§16). Phase N+1 does not start before Phase N's exit tests are green in CI.
- Invariants INV-1..9 are **constructors/gates on the hot path**, not test-only asserts.
  `EnergyReport` must be impossible to instantiate without passing its gates.
- Einsum strings E-1..E-5 and the §6 index conventions are law. If you believe one is
  wrong: write a failing shape test first, then fix doc and code together.
- `complex128` on certification paths; dtype threaded via `TensorSpec`, never hardcoded.
- `prototypes/` is a **reference oracle, not production code**. Never import it from
  `src/`. Production must reproduce its numbers through `tests/golden/`.
- Anything underspecified: pick the simplest invariant-consistent option, record an ADR
  in `docs/adr/` (next number: **ADR-017**), continue. Do not stall.

## Executed ADRs — do not re-litigate (each records a defect found by running code)
| ADR | Operational consequence for you |
|---|---|
| 009 | INV-3 gate is two-sided: `est ≤ max(η·σ₁, c_gate·σ̂_{χ+1})`. Fixed-η alone rejects optimal sketches. |
| 010 | Compression = exact right-canonicalization sweep **then** truncating sweep. Naive zip-up truncates gauge artifacts and silently corrupts INV-1 (measured O(10⁻¹) errors at provably lossless χ). |
| 011 | Canonicalization **inside the AD graph** uses full-rank SVD, not QR/LQ: torch and JAX QR-VJPs reject wide matrices, and the right-edge operand is wide. |
| 012 | Truncation gradient = full economy SVD + slice — exact for the compression graph (outputs recombine bilinearly), including kept↔discarded coupling. The projector formula is the *sketched-backend fallback only*. |
| 013 | SDRG PT₂ coefficients: use §9's **current** formulas (original prose had two factor-2 errors). `sdrg_3site.py` Tier-I identity is the arbiter for any change. |
| 015 | Factored compression (`env.factored`, needed for D≥6) = bond-Gram canonicalization, NOT the original uncanonicalized LinearOp sketch (that violates ADR-010). Equivalence gate: `tests/unit/test_factored_compress.py`. |
| 016 | INV-3's *second* failure action (auto-disable at >20% fallback) was a dead config knob; now enforced **in `SketchedSVD` on the hot path**, driven by `gate_fallback_rate` only (structural `k≤χ` fallbacks excluded), warmup-gated at 32 sketchable calls, monotonic. `fallback_count` no longer hardcoded to 0 in `EnvCertificate`. |

## Status → task queue
**DONE, validated in prototypes (math framework-portable):** kernel scaling + INV-3 gate
(`bench_kernel.py`); contraction engine E-1..E-5 incl. dressed correlators (`golden_3x3.py`,
51/51); differentiable energy, exact truncation gradients, LBFGS→ED (`ad_phase2.py`); L=4
golden incl. INV-2 in anger (`phase3_4x4.py`); SDRG decimation rules (`sdrg_3site.py`, 9/9).

**P0–P5 COMPLETE (2026-07-31).** All §16 exit criteria green; §18's definition of done is
met: `REPORT.md` carries E(D) extrapolation, q_EA, ξ, n_res(r), Tier-2 Γ₁ with echoed
inputs, and the full invariant audit. Suite 228 passed / 2 skipped, mypy + ruff clean,
`make verify` 51/51 and 9/9, `make verify-jax` bit-identical across four recorded runs.

**Remaining, all conditional or study work:**
1. **P6 (cond.)** Rust zip-up kernel via pyo3 — the only path to D=6 at production L
   (measured: D=6 is wall-clock-blocked, ~1.7 h/gradient step; ADR-008 scopes the FFI to
   the whole row loop, not the SVD, because ADR-010's canonicalization not the SVD is the cost).
2. **P7 (cond.)** L2/L3 sharding — only if the 1/D extrapolation demands D ≥ 10.
3. **Physics study plan (§12):** the D-ladder extrapolation, R_c and L sweeps, and the
   resonance census now all have machinery; what is missing is *runs*. The L=8 pilot is
   deep-localized (ξ correctly unresolved, n_res ≡ 0) — a g_J sweep is what would show a
   crossover. Note 3 of 4 pilot realizations are tainted by an unconverged D=3 rung.
4. ~~Record the kernel backend in the manifest.~~ **DONE (ADR-017, 2026-08-02.)** The
   audit line now distinguishes three causes of an absent fallback rate. Established while
   fixing it: `runs/pilot_L8.zarr` was written at `b973d23`, eight commits before ADR-016
   wired `sketch_stats`, so its "not recorded" line is a genuine pre-audit artifact and
   *not* a wiring bug — re-running it under the current build is what gets a real rate.

## Gotchas (paid for in prototyping — do not pay again)
- Prototypes reuse 3×3 modules for 4×4 via a **module-global `L` monkey-patch**
  (`phase3_4x4.py`). Production parametrizes `L` explicitly. Never copy the global.
- Under truncation, **cross-engine agreement is not a certificate** (flat spectra make the
  kept subspace gauge-ambiguous at √disc scale). INV-2 stability is the certificate. On
  optimized physical states disc collapses (measured 2×10⁻¹⁵ at nominally truncating χ).
- Optimization gaps are usually **optimizer floor, not ansatz**: 4×4 D=2 went
  4.4e-6 → 8.9e-8 with more iterations. Check the iteration budget before blaming the ansatz.
- Conventions: site `s = y·L + x` (row-major); site 0 = slowest bit = first Kronecker
  factor; `Z = diag(1, −1)` so bit 1 ↔ σᶻ = −1.
- `quimb` is allowed **only** inside `tests/golden/` as an independent oracle (§4).
- Baselines (frozen reference JSONs) live in `prototypes/baselines/`; re-runs write
  `prototypes/results/` (gitignored). Never overwrite baselines.

## First actions in a fresh session
1. `git init && git add -A && git commit -m "handoff baseline: validated prototypes + normative spec"`
2. `make verify` (and `make verify-jax` if jax installed); save the output to `docs/verify-log.md` and commit.
3. Begin P0. Keep ARCHITECTURE.md's §14 provenance updated as each phase exits.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

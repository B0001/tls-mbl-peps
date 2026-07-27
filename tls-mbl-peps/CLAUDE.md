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
```
python3.11+;  pip install -e ".[dev]"
# torch (needed from Phase 2):
pip install torch --index-url https://download.pytorch.org/whl/cpu
#   Do NOT plain `pip install torch`: the default PyPI linux wheel is CUDA-linked
#   (observed failure in prototyping: missing libcublasLt on CPU-only hosts).
# jax (optional, tier-2 prototype parity only):  pip install jax
make verify        # tier-1 baselines: must show "ALL PASS (51/51)", "ALL PASS (9/9)"
make verify-jax    # if jax present: consistency ~1e-15; T-AD-FD ≤1e-6 at χ=4 AND χ=2
```
If `make verify` fails: **stop and fix the environment.** Never touch a tolerance to pass.

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
  in `docs/adr/` (next number: **ADR-016**), continue. Do not stall.

## Executed ADRs — do not re-litigate (each records a defect found by running code)
| ADR | Operational consequence for you |
|---|---|
| 009 | INV-3 gate is two-sided: `est ≤ max(η·σ₁, c_gate·σ̂_{χ+1})`. Fixed-η alone rejects optimal sketches. |
| 010 | Compression = exact right-canonicalization sweep **then** truncating sweep. Naive zip-up truncates gauge artifacts and silently corrupts INV-1 (measured O(10⁻¹) errors at provably lossless χ). |
| 011 | Canonicalization **inside the AD graph** uses full-rank SVD, not QR/LQ: torch and JAX QR-VJPs reject wide matrices, and the right-edge operand is wide. |
| 012 | Truncation gradient = full economy SVD + slice — exact for the compression graph (outputs recombine bilinearly), including kept↔discarded coupling. The projector formula is the *sketched-backend fallback only*. |
| 013 | SDRG PT₂ coefficients: use §9's **current** formulas (original prose had two factor-2 errors). `sdrg_3site.py` Tier-I identity is the arbiter for any change. |
| 015 | Factored compression (`env.factored`, needed for D≥6) = bond-Gram canonicalization, NOT the original uncanonicalized LinearOp sketch (that violates ADR-010). Equivalence gate: `tests/unit/test_factored_compress.py`. |

## Status → task queue
**DONE, validated in prototypes (math framework-portable):** kernel scaling + INV-3 gate
(`bench_kernel.py`); contraction engine E-1..E-5 incl. dressed correlators (`golden_3x3.py`,
51/51); differentiable energy, exact truncation gradients, LBFGS→ED (`ad_phase2.py`); L=4
golden incl. INV-2 in anger (`phase3_4x4.py`); SDRG decimation rules (`sdrg_3site.py`, 9/9).

**TODO, in order (exit criteria = ARCHITECTURE.md §16 rows):**
1. **P0** `src/tlsmbl/core`: units, rng (INV-6), guards (INV-7), pydantic config (§13), manifest.
2. **P1** port model + ED oracle from prototypes into `src/tlsmbl/model` + golden fixtures.
3. **P2** torch port of peps/kernels/autodiff/optimize. Parity contract: pass the SAME gates
   at the SAME thresholds as the prototypes (table in `docs/HANDOFF.md`); T-AD-FD must
   include a genuinely truncating χ configuration, not only the lossless one.
4. **P3** sketched backend + two-sided gate + T-PERF (exponent gap ≥1.6; prototype measured 3.30).
5. **P4** SDRG circuit/transform/ledger + A/B harness (a negative A/B result is a valid exit).
6. **P5** ensemble/zarr/resume/aggregate + INV-5. P6/P7 conditional per §16.

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

# HANDOFF — prototyping session record (July 2026)

State of the world for the implementing session. Everything below was **executed**, not
reviewed: five prototype programs run against exact references (finite differences,
brute-force statevectors, sparse ED, Schur complements), producing five spec amendments
(ADR-009..013) now baked into ARCHITECTURE.md.

## Provenance chain
Physics derivation → architecture v1 → **D3 kernel benchmark** (found ADR-009: two-sided
INV-3 gate) → **3×3 golden battery** (found ADR-010: canonicalize-then-truncate) →
**Phase-2 AD prototype** (found ADR-011: SVD canonicalization in AD graph; ADR-012: exact
slice-through-full-SVD backward) → **4×4 golden battery** (flat-spectrum certificate
finding; disc-collapse measurement) → **T-SDRG-3SITE** (found ADR-013: two factor-2
coefficient errors in §9 prose, corrected in place).

## Validated numbers = parity targets

| Suite (prototype / baseline JSON) | Configuration | Result | Gate |
|---|---|---|---|
| Kernel scaling (`bench_kernel.py`) | exact SVD, D=4–7, χ=D², complex128, 1 core | slope **11.57** (theory 12) | — |
| | sketched RSVD, D=4–8 | slope **8.27** (theory 10) | — |
| | exponent gap | **3.30** | ≥ 1.6 (T-PERF) ✓ |
| | speedup at D=6 | **33.4×** (theory 36×) | ≥ 20× ✓ |
| Gate v2 validation | σₖ=e^(−0.5k), D=4/D=6 | sketch error = optimal to 14 digits; principal angle ~9e-5° | pass ✓ |
| | σₖ=1/k (slow) | conservative fallback | correct ✓ |
| 3×3 golden (`golden_3x3.py`) | 2 seeds × 2 g_J × D∈{2,3}, χ=D² | **51/51**; contraction err ≤ 2.4e-15; disc ≡ 0 | all ✓ |
| | ED sparse ≡ dense | ~1e-15; J=0 analytic ~1e-16 | ✓ |
| Phase-2 AD (`ad_phase2.py`) | jax↔numpy consistency | 2.8e-15 (D=2), 1.4e-14 (D=3) | <1e-11 ✓ |
| | T-AD-FD, χ=4 lossless | **8.2e-10** | <1e-6 ✓ |
| | T-AD-FD, χ=2, disc=1.9e-2 | **4.7e-9** | <1e-6 ✓ |
| | LBFGS→ED, g=1e-3, D=2 | rel gap **7.2e-9** (203 iters, product init) | ✓ |
| | g=0.3, D=2 / D=3 | **2.6e-7** / **7.2e-8** (monotone in D) | ✓ |
| 4×4 golden (`phase3_4x4.py`) | sparse ED at 2¹⁶, 98 pair terms | residual ~1e-14, ~2 s | ✓ |
| | consistency at provably lossless χ=16 | **2.9e-14**, disc ≡ 0 | ✓ |
| | random-state χ=8 engine discrepancy | 1.2e-3 = √disc — *expected*, see finding below | n/a |
| | T-AD-FD, χ=8, disc=3.8e-5 | **2.9e-9** | <1e-6 ✓ |
| | opt g=0.3 D=2 χ=16 | rel gap **8.9e-8** (4000 iters; 4.4e-6 at 800 → optimizer floor) | ✓ |
| | opt g=1e-3 D=2 χ=16 | **6.4e-10** | <1e-8 ✓ |
| | opt χ=6 truncated path | 2.0e-7; INV-2 stability **1.4e-11**; disc collapsed to **2.2e-15** | ✓ |
| T-SDRG-3SITE (`sdrg_3site.py`) | site / AF-bond / F-bond × 3 draws | **9/9**; rule↔Schur identity ≤ 3.5e-16; convergence order → **3.00** | ✓ |

## Two findings worth internalizing
1. **Certificate design.** Under truncation, two correct engines legitimately disagree at
   √(disc) scale on flat-spectrum states (kept subspace is gauge-ambiguous). Therefore
   cross-engine agreement is *not* a validity certificate; INV-2 χ-stability is. This is
   why ε_F broadening exists (INV-7) and why the certificate stack is built the way it is.
2. **The physics premise is measured.** On *optimized* (physical) states, the entanglement
   spectrum decays so fast that a nominally truncating χ becomes effectively exact
   (disc 1e-4 → 2e-15 through optimization). This is the localized-phase spectral-decay
   premise that Stage-B sketching and INV-3 tightness rest on — now an observation.
3. **The kernel speedup above is *not* a pipeline speedup (measured 2026-07-26).** The
   33.4× row is the truncation kernel in isolation. In the full row, ADR-010's mandatory
   exact canonicalization sweep is Θ(χ³D⁸) — a factor D² *above* the Θ(χ³D⁶) truncation
   that sketching accelerates to Θ(χ³D⁴) — and it is not sketchable. Amdahl therefore caps
   the row speedup: measured at D=6, χ=36, row 42.1 s exact → 29.5 s sketched (**1.4×**,
   not 33×). Consequences: (a) D=6 is wall-clock-blocked, not memory-blocked — end-to-end
   energy+gradient at L=8 measured 2.4 / 14.8 / 182 s for D=2/3/4 (exponent **8.73**),
   extrapolating D=6 to ~1.7 h *per gradient step*, ~2 orders over the §15 budget;
   (b) ADR-008's choice to scope the Rust kernel to the **whole row loop** rather than the
   SVD is retroactively the right call, and is the only path to D=6 at production L;
   (c) INV-3 fallback measured 25% at D=6 on a product-like state — above the 20%
   auto-disable, so sketching would switch itself off there anyway.

## Port-parity contract (Phase 2, torch)
- Same gates, same thresholds as the table above. Optimizer parity = passing the same
  gates (not bitwise energies). T-AD-FD must include a genuinely truncating χ.
- The exact-backend TruncSVD Function: full economy SVD in forward, sliced cotangents in
  backward (ADR-012), wrapped with ε_F-broadened hardening + INV-9 gauge fix for
  near-degenerate spectra. The projector-term formula is implemented only inside the
  sketched backend (Phase 3), where its O(σ_{χ+1}/σ_χ) error is bounded by INV-3.

## Open items (deliberately deferred)
- Torch port itself (Phases 0–2 engineering); INV wiring as constructors; ensemble layer.
- Hardened broadened backward around the full-SVD path (formula in §8.5; test against the
  jax prototype gradients).
- Hartree tail loop (spec'd, default off, INV-5 bound reported regardless).
- Converged D≥3 runs at L=4 → production CI (per-iteration cost was beyond the
  prototyping container; exactly the regime the sketched kernel + multicore address).
- The physics study plan: D-ladder extrapolation, R_c and L sweeps, resonance census
  n_res(r), Tier-2 Γ₁ with declared model inputs (§12).
- L2/L3 sharding: stubs only, opened only if extrapolation demands D ≥ 10 (§15/§16 P7).

## Novelty register status (→ owner's invention-spec pipeline, §19)
NR-1 posterior-gated randomized truncation inside an AD graph (strengthened by ADR-012's
exact/fallback split); NR-2 SDRG-pinned-lattice composition with fidelity ledger (now
empirically grounded via the disc-collapse measurement); NR-3 gate-constructible certified
artifacts; NR-4 operator-dressed boundary-MPS correlator caching; NR-5 reproducible lazy
tail-coupling streams. Candidate addition: the two-tier identity+convergence-order test
pattern of `sdrg_3site.py`.

## Environment notes from prototyping
1-core / 3 GB container; `/bin/sh` (no brace expansion) for tooling; PyPI `torch` wheel is
CUDA-linked (libcublasLt failure on CPU-only) → use the pytorch.org CPU index; jax 0.11
used as AD substrate for prototypes only; numpy/scipy-openblas64 for kernels (same
LAPACK/BLAS class as torch — exponents transfer, constants differ).

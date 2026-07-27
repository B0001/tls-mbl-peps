# ADR-015: Factored compression realizes ADR-007 v1.1 via bond-Gram canonicalization

Status: accepted (executed; equivalence-gated by `tests/unit/test_factored_compress.py`).

## Context

ADR-007 deferred a "factored `LinearOp`" v1.1 for when the D-ladder hits memory
limits. That limit was hit: the L=16 production launch capped the ladder at D=4
because a single fat-MPS column tensor at D=6, chi=36 is Theta(chi^2 D^6) ~ 1 GB
(recorded in `configs/bench_L16_D4.yaml`), and the batched dressed-environment
path multiplies that by the batch size.

The v1.1 sentence in §8.6 as originally written — evaluate `Wop @ v` as einsums
against `(carry, M, a)`, peak Theta(chi D^4), never materialize `Wmat` — predates
ADR-010 and is **incompatible with it**: it runs the truncating sweep on the
uncanonicalized fat MPS, which is exactly the naive zip-up ADR-010 measured as
O(1e-1) silent corruption. Any ADR-010-honoring compression needs the Schmidt
spectrum across each bond, which requires a bond-sized (chi D^2)^2 object
somewhere; a Theta(chi D^4) peak is therefore unattainable. The attainable and
sufficient target is eliminating the Theta(chi^2 D^6) fat tensors — a D^2 (= 36x
at D=6) reduction; the remaining Theta(chi^2 D^4) objects are ~27 MB at the
acceptance regime and are not the binding constraint.

## Decision

`kernels/factored.py` compresses the fat MPS directly from its factored per-site
form `(M, a)` (boundary tensor + double layer), never materializing fat tensors:

1. **Canonical gauge by bond Grams, not an LQ sweep.** With `R_x` the right part
   of the fat MPS at bond x, transfer `G_x = R_x R_x^H` right-to-left through
   the factored tensors in D^2 physical-leg slices (largest intermediate:
   one (chi D^2)^2 slice). Truncation at site x SVDs `W~ = W @ L_x` with
   `G_x = L_x L_x^H` (Cholesky, relative jitter 1e-12 on the mean-diag-normalized
   Gram, escalated x100 on failure). `W~` has the same Schmidt spectrum as
   ADR-010's LQ-swept operand, at the same flop class (2 chi^3 D^8 per site),
   with D^2 less memory. The carry is `U^H W` (no `L` inverse anywhere).
2. **Precision consequence, accepted.** Gram formation squares conditioning:
   singular values below ~sqrt(eps_mach) sigma_1 are noise, i.e. discarded-weight
   resolution floors at ~1e-16 relative — two decades below the tightest gate in
   use (eps_env = 1e-8). Measured equivalence vs v1: state fidelity 1 - O(1e-10),
   disc weights to rtol 1e-5, energies to 1e-9, gradients to 1e-6 relative,
   on lossless AND genuinely truncating configurations, both orientations,
   through the batched dressed path.
3. **AD.** Einsums + matmuls + Cholesky + the backend SVD only — no QR, so
   ADR-011's wide-QR constraint is moot on this path; gradients flow through the
   Gram and Cholesky (detached rescales are exact by scale-invariance of U).
4. **Activation.** `env.factored` config flag (default False; v1 stays the
   reference oracle). Threaded through build/extend/batched-extend, energy,
   LBFGS, finalize, observables, orchestration. Required for D >= 6 rungs.

The sketched backend consumes `W~` as an explicit matrix — with the fat tensor
gone there is nothing left worth abstracting behind a `LinearOp`, and `U^H W`
needs `W` materialized regardless. §8.6's v1.1 text is updated together with
this ADR (§0 rule: doc and code fixed together).

## Consequences

Measured, single row absorb+compress, exact backend: D=4/chi=16/L=16 peak RSS
0.86 -> 0.31 GB and 1.7 -> 0.7 s; D=6/chi=36/L=8 runs in 1.17 GB peak where v1's
fat row alone is ~8 GB before its LQ sweep allocates anything. The D=6 ladder
rung is now memory-feasible; its wall-clock remains the sketched/Rust phases'
concern (flop class unchanged by design). If chi^2 D^4 ever becomes the binding
constraint (D >= 10, §15), that is Phase 7 sharding, not an extension of this
decision.

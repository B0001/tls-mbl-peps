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

## §18 definition-of-done gaps closed + ADR-016 (2026-07-30)

Four items §18 requires in REPORT.md were missing; two of the four are now in, plus a
dead invariant knob was found and wired. Re-verified on the same 3.14.6 / torch 2.13.0
root:

```
uv run pytest tests            -> 179 passed, 2 skipped   (was 138 passed, 2 skipped)
uv run mypy                    -> Success: no issues found in 48 source files
uv run ruff check .            -> All checks passed!
make verify                    -> ALL PASS (51/51), ALL PASS (9/9)
make verify-jax                -> consistency D=2 2.032e-15,  D=3 2.218e-14
                                  T-AD-FD chi=4 (disc=0)       1.337e-09
                                  T-AD-FD chi=2 (disc=1.93e-2) 4.763e-09
```

The jax-tier numbers are again **bit-identical** to both prior runs: nothing in this
change touches the numerics.

**Landed.** (a) `observables/localization.py` — xi from a log-linear fit of the
disorder-mean |Czz(r)| with a bootstrap CI over realizations; (b)
`ensemble/extrapolate.py` — E(D) = E_inf + c/D with the remaining gap
|E(D_max) - E_inf| as the honest ansatz-truncation statement; both wired into
`aggregate.py` and REPORT.md. Both are verdict-carrying: they report "unresolved" with a
machine-readable reason instead of a number they cannot support.

**ADR-016 — a dead invariant.** INV-3's second failure action ("fallback_rate > 20% =>
disable sketching for the realization and log") was never implemented:
`kernels.fallback_disable_rate` was validated by pydantic, echoed in all four configs and
cited in HANDOFF as the reason D=6 "would switch itself off anyway", but nothing read it.
Two adjacent gaps in the same trail: `EnvCertificate.fallback_count` was hardcoded to `0`
in `energy_certified`, and `zipup.py` held a `fallbacks += 0` placeholder — so even the
*first* failure action's counter never reached `EnergyReport`, and §11's "REPORT.md echoes
fallback rates, bypass counts" was unmeetable. All now enforced on the hot path in
`SketchedSVD` (gate fallbacks only, warmup-gated, monotonic — see ADR-016 for why each),
with `tests/unit/test_inv3_auto_disable.py` proving the action fires, latches, and does
*not* fire on structural `k<=chi` fallbacks.

**Measured on the existing L=8 pilot** (`tlsmbl aggregate runs/pilot_L8.zarr`):
- `xi: unresolved (at_noise_floor)` — only r=1 of 3 bins sits above the 1e-8 floor
  (6.66e-07, 6.84e-09, 3.06e-10), and the signed disorder mean alternates sign. This is
  the correct verdict, not a defect: at L=8, g_J=1e-3 the correlator is at the
  contraction noise floor, so no length is resolvable. A naive log-linear fit on those
  three numbers returns a confident-looking xi ~ 0.27, which is why the refusal exists.
- `E(D)`: secant_2pt over the [2, 3] ladder, E_inf = -18.182377238
  [-19.889269649, -16.475484827], remaining gap 3.59e-05. Labeled a secant because two
  rungs fix the line exactly — no fit residual, no per-realization fit uncertainty.
- **3 of 4 realizations are tainted by an unconverged rung** (D=3 hit `max_outer: 60`).
  Per CLAUDE.md's "optimizer floor, not ansatz" gotcha this bounds how much the pilot's
  E_inf is worth; a re-run with a larger iteration budget is the fix, not a bigger D.
- The audit reports the INV-3 rate as **"not recorded"**, not "exact backend": the pilot
  ran `backend: sketched`, but its artifacts predate the plumbing, and the manifest does
  not record which backend ran. Recording the kernel config in the manifest is follow-up.

**Still open from §18** (both attempted this session, neither landed — no partial code
committed): Tier-2 Γ₁ with echoed model inputs in REPORT.md (the `gamma_1` function
exists in `observables/decoherence.py`; the §12 fluctuator table, the spectral-diffusion
proxy and the report wiring do not), and the §7.4 Hartree self-consistency loop with its
lazy r>R_c tail stream (P5's last exit item; `hartree.py` is still bound-only).

## Tier-2 (§12) + Hartree loop (§7.4) — 2026-07-30, same session

```
uv run pytest tests            -> 211 passed, 2 skipped   (was 179 after the xi/E(D) work)
uv run mypy                    -> Success: no issues found in 48 source files
uv run ruff check .            -> All checks passed!
make verify                    -> ALL PASS (51/51), ALL PASS (9/9)
make verify-jax                -> consistency D=2 2.032e-15,  D=3 2.218e-14
                                  T-AD-FD chi=4 1.337e-09,  chi=2 4.763e-09
```

Jax-tier numbers bit-identical for the third recorded run. **This matters more than
usual here: `core/rng.py` went from `spawn(3)` to `spawn(4)` to add the Hartree tail
stream.** `SeedSequence` derives children by index, so appending leaves children 0-2
untouched — verified directly across four (master_seed, realization) keys in
`test_existing_three_streams_are_unchanged_by_the_fourth`, and confirmed end-to-end by
the golden fixtures and both baselines still passing. Stream ORDER is now documented as
frozen: new streams append, never insert.

**Tier-2 (§12) complete.** All three outputs now exist (only `gamma_1` did): the
fluctuator table, `gamma_1` refactored onto one shared `splittings` definition, and the
spectral-diffusion proxy. Wired through orchestrate -> zarr attrs -> REPORT.md, default
off, and REPORT.md now *states* "not computed" when off instead of omitting the section.
Two decisions worth knowing:
- Transverse weights are LABELED `bare_field` vs `measured_state` rather than one being
  silently substituted; orchestrate passes the certified state's polarization, so
  production runs report `measured_state`.
- `inputs_from_config` REFUSES unit-tagged strings ("5.0GHz"). Converting one to units of
  W needs W in that unit and no layer carries it (W == 1.0 internally, §2). A Tier-2
  number looks identical whether the conversion was right or invented, which is exactly
  why guessing is unacceptable. Declare Tier-2 inputs in units of W.

A real bug surfaced while testing the thermal activity factor: clamping the cosh
*argument* to dodge overflow floors the factor at a spurious ~1e-261 constant instead of
letting it decay, so T=1e-8 and T=1e-4 returned identical variances. Now u/(1+u)^2 with
u = exp(-E/T) — exact, stable at both ends, underflows to a true 0.0 for E >> T.
Regression-locked.

**Hartree (§7.4): loop implemented and tested, NOT yet driven by the orchestrator.**
`hartree_loop`, `tail_field` and the counter-based lazy tail stream (NR-5) are done, with
15 tests. The tail couplings are drawn from the same U(-1,1) distribution as
`sampling.py`'s retained bonds, and are keyed by the PAIR (Philox counter = s_a*N + s_b),
so the same value comes back regardless of request order, repetition, or outer iteration
— the property that makes the loop reproducible without an O(N^2) table.

Driving it from `orchestrate.py` means re-entering the D-ladder once per outer iteration
with per-iteration checkpoint stages, and the resume-after-kill contract has not been
extended to cover that. Rather than accept `hartree.enabled: true` and silently run the
h_mf = 0 baseline, `run_realization` now **raises** — a knob that is read, validated and
ignored is precisely the defect ADR-016 was written about. This is the one remaining
piece of §16 P5.

**INV-5 semantics under the loop, decided and tested:** the bound does NOT shrink. The
loop *moves* the neglected error (the tail's mean-field part is treated; its correlations
are not), and 2*pi*g_J/R_c still bounds what is left. Reporting a smaller bound because
the loop ran would claim rigor the mean-field treatment does not provide. Only the label
(`HartreeResult.bound_covers`) changes; any future tightening needs its own ADR and proof.

**Calibration note for whoever touches `test_damping_prevents_the_oscillation`:** the
tail-field operator is symmetric with zero diagonal, hence traceless, hence its spectrum
always straddles zero (measured lambda_max = 5.79e-3, lambda_min = -5.12e-3 at L=4,
R_c=1, g_J=1e-2, seed 3). If the gain pushes the POSITIVE branch above 1, damping cannot
help for any alpha — a first attempt at gain = -500 diverged even at alpha = 0.05. The
gain is placed in the window where only the negative (two-cycle) branch is unstable.

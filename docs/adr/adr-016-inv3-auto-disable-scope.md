# ADR-016 — INV-3's auto-disable: what counts as a fallback, and where the gate lives

**Status:** accepted, implemented (2026-07-30)
**Supersedes:** nothing. **Amends:** §3 INV-3 failure action, §8.6.

## Context

INV-3 has two failure actions. The first — "failing the posterior gate falls back to the
exact backend for that call and increments `fallback_count`" — shipped with Phase 3 and
is covered by `tests/unit/test_rsvd.py`. The second did not ship:

> if `fallback_rate > 20%` over a sweep, disable sketching for the realization and log

`kernels.fallback_disable_rate` was validated by pydantic, echoed in all four configs,
listed in §17's risk table as the mitigation for "sketch gate thrashing (slow spectral
decay)", and cited in `docs/HANDOFF.md` as the reason D=6 "would switch itself off
anyway" (measured 25% fallback there) — but **nothing read it**. The claim that the
solver self-limits under slow spectral decay was unbacked by code. `rsvd.py`'s docstring
deferred enforcement to "the orchestration layer reading these", and the orchestration
layer did not.

Two further gaps in the same audit trail: `EnvCertificate.fallback_count` was hardcoded
to `0` in `energy_certified`, and `zipup.py` carried a `fallbacks += 0` placeholder
commented "sketched backend counts its own fallbacks (Phase 3)" — so even the *first*
failure action's counter never reached `EnergyReport`, and §11's requirement that
REPORT.md echo "fallback rates, bypass counts" could not be met.

The spec sentence leaves three things open, and each has a wrong answer that looks right.

## Decision

**(a) Only *gate* fallbacks count toward the disable rate.** `SketchedSVD.truncate` has
two exits to the exact backend: the INV-3 gate rejecting a sketch, and the structural
`k = min(chi + p, min(m, n)) <= chi` case where there is no oversampling headroom, the
gate cannot see `sigma_{chi+1}`, and exact SVD is cheaper anyway. Only the first is
"thrashing". Counting the second would disable sketching on precisely the small
early-ladder operands where falling back costs nothing — and, because the disable is
monotonic, would then keep it off for the expensive rungs that follow. The counters are
therefore split (`gate_fallback_count`, `structural_fallback_count`), with
`gate_fallback_rate = gate_fallback_count / gate_call_count` driving the disable and the
combined `fallback_count` / `fallback_rate` retained for the §11 audit. This is why the
public `fallback_count` remains the total: `tests/unit/test_rsvd.py` asserts it for both
exits, and both really are fallbacks — they just are not both *symptoms*.

**(b) The rate is consulted only after a warmup** (`min_gate_calls`, default 32). Without
one, a single unlucky first call reads as rate 1.0 > 0.2 and disables sketching
permanently on call one — turning a per-realization performance guard into a coin flip.

**(c) The gate lives on the hot path, in the backend, not in the caller.** §3's action
says "disable sketching for the realization", and the backend instance *is* the
per-realization scope: `orchestrate.py::_backend` builds exactly one per realization from
that realization's spawned sketch stream. Putting the check where the counters already
live makes it a constructor/gate on the hot path as CLAUDE.md requires, rather than
something a future caller can forget to poll. `rsvd.py`'s docstring is corrected
accordingly. The disable is **monotonic** — once off it stays off for the realization —
because re-enabling would make the kernel path depend on the order operands happen to
arrive within a sweep, which is a reproducibility hazard (INV-6) for no benefit.

"Over a sweep" is thus read as "over the realization's accumulated sketchable calls",
which is the coarsest reading and the only one that does not need a sweep-boundary
callback the architecture does not otherwise have.

## Consequences

- `kernels.fallback_disable_rate` is live. `configs/*.yaml` keep 0.20, so §17's risk
  mitigation and HANDOFF's D=6 self-disable claim are now backed by executed code
  (`tests/unit/test_inv3_auto_disable.py`, 6 tests: fires, latches, warms up, ignores
  structural fallbacks, never fires on an accepting gate, opt-in, JSON-able stats).
- `EnvCertificate` gains `sketch_stats`; `fallback_count` is read from the backend
  instead of hardcoded. `zipup.compress` and `factored.compress_factored` report a real
  per-compression count as a backend-counter delta (the delta, not
  `posterior_err is None`, because the latter cannot distinguish "sketch fell back" from
  "backend is exact and never sketches").
- **Certification is unaffected.** A disable changes which kernel computes a truncation,
  never whether the result is certified: INV-1's discarded-weight and up/down gates run
  identically on the exact path (which is the *more* accurate one). The disable is a
  performance signal, and the warning says so.
- REPORT.md's audit section now echoes the worst gate-fallback rate, the count of
  realizations that auto-disabled, and SDRG bypasses/worst ledger. Where a run recorded
  no sketch stats the report says "not recorded", **not** "exact backend": the manifest
  does not record which backend ran, so that inference would be a guess. Recording the
  kernel config in the manifest would remove the ambiguity and is left as follow-up.
- Backward compatibility: artifacts written before this ADR have no `sketch_stats` key;
  `aggregate.py` treats absence as unknown rather than as zero.

# ADR-017 — the manifest records the truncation backend, so the INV-3 audit can stop hedging

**Status:** accepted, implemented (2026-08-02)
**Supersedes:** nothing. **Amends:** §11 (manifest contents, REPORT.md audit rollup).
**Follows from:** ADR-016's consequences.

## Context

ADR-016 wired INV-3's audit trail end to end: `SketchedSVD.stats()` reports
`gate_fallback_rate` and `sketching_disabled`, `energy_certified` reads the backend's own
counters into `EnvCertificate`, and `aggregate_run` rolls the worst rate up into
REPORT.md. What it could not fix is what an *absent* rate means.

`aggregate.py` collected `sketch_stats` only from realizations that recorded it, and the
exact backend has no `stats()` method at all — correctly, since it has no gate to fall
back from. So a `None` rate had three causes with one printed sentence:

> worst sketch gate-fallback rate (INV-3): not recorded (exact backend, or artifact
> predates the INV-3 audit)

with a code comment conceding the ambiguity was deliberate, because "the manifest does not
record which backend ran". The audit therefore could not distinguish a run that provably
never sketched from a run whose instrumentation was missing. For an audit line whose whole
job is to say whether a certified number leaned on a sketch, that is the wrong failure
mode: it reads as reassurance in the case where reassurance is unwarranted.

`runs/pilot_L8.zarr` is the live example. Its config sets `kernels.backend: sketched`, its
REPORT.md prints the sentence above, and all four realizations store
`sketch_stats: None` — because the artifact was written at `b973d23`, eight commits before
ADR-016 landed. The reader had no way to tell that from the report.

## Decision

**1. `Manifest` carries `kernel_backend`.** Recorded from `config.kernels.backend` at
`build_manifest`, alongside the other reproduction fields.

**2. It records the *configured* backend, not the outcome.** The manifest is written once,
before any realization starts, so it cannot honestly report what a realization ended up
doing. ADR-016's auto-disable is per-realization state and stays where it already lives,
in `report.sketch_stats.sketching_disabled`, which the audit rolls up separately. A
manifest field that claimed to be an outcome would be a manifest field that lies whenever
the gate trips.

**3. Typed `str`, not the config's `Literal["exact", "sketched"]`.** ADR-008's Rust kernel
is a third backend; artifacts must be able to name a backend that the *reading* build has
never heard of. Readers treat unknown values as "some sketching backend" rather than
rejecting the artifact.

**4. The audit line splits into three, one per cause** (`_inv3_rate_phrase`):

| recorded backend | rate | printed |
|---|---|---|
| any | present | the rate |
| absent (legacy artifact) | None | `not recorded (artifact predates the INV-3 audit)` |
| `exact` | None | `n/a (exact backend -- no sketch path to fall back from)` |
| anything else | None | `not recorded, though the manifest records the '<b>' backend -- no realization reported sketch statistics` |

The last row is the one that matters: a sketching backend with no statistics is a wiring
anomaly, and the report now says so instead of filing it under the same benign sentence as
an exact run.

## Consequences

- New artifacts get a definite audit statement. Legacy artifacts keep the honest hedge —
  and it is now *narrower*, because it no longer has to cover the exact-backend case.
- `config_hash` is unchanged: it hashes the config, and `kernel_backend` is derived from
  it, so T-DET is untouched. The field is redundant with the hash cryptographically and
  deliberately so — the hash proves which config ran but cannot be *read*.
- CLAUDE.md's task-queue item 4 ("record the kernel backend in the manifest so the INV-3
  audit can say 'exact backend' instead of 'not recorded'") is closed.
- Re-running the pilot under the current build would move it from the legacy row to a real
  rate; until then its REPORT.md correctly identifies itself as pre-audit.

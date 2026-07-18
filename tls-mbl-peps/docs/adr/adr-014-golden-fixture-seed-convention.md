# ADR-014: Golden-fixture seed convention and canonical pair order

Status: accepted (Phase 1).

## Context
§14.1 (T-GOLD-ED) prescribes "seeds {0,1,2}, L in {3,4}" but does not define how those
seeds map to RNG streams, and §5 says SparsePairs iteration "is sorted" without naming
the sort key. Both must be pinned for bit-stable fixtures.

## Decision
1. **Fixture seeds.** T-GOLD fixtures use `master_seed = 20260716` (the prototype
   golden master) with realization index `k in {0,1,2}`; the disorder generator is
   `core/rng.realization_streams(20260716, k).disorder` — the same INV-6 spawn path
   production uses, so fixtures and pipeline share one seed discipline. Grid:
   L in {3,4} x g_J in {1e-3, 0.3} x k, 12 fixtures in `tests/golden/fixtures/`.
2. **Canonical pair order.** `SparsePairs` iteration sorts by row-major *site index*
   `(s_a, s_b)` with `s = y*L + x` — not lexicographic `(x, y)` tuples. This matches
   the executed prototype's insertion order bitwise (verified: sparse H matrices agree
   exactly; lexicographic ordering perturbed floating-point sums at 1e-15).
3. **Deterministic Lanczos.** `ed_reference.ed_ground` pins the eigsh start vector
   `v0 = 1/sqrt(dim)` so L=4 fixture regeneration is reproducible.

## Consequences
T-GOLD-ED (P2) certifies PEPS energies against these stored numbers; regeneration
drift is caught by `tests/golden/test_gold_fixtures.py`. Changing any of the three
choices above invalidates the fixtures and requires a new ADR.

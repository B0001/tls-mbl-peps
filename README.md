# tls-mbl-peps

**Variational PEPS solver for many-body localization of two-level-system (TLS) defects in
amorphous Al₂O₃** — the microscopic noise source behind decoherence in superconducting
qubits. Combines SDRG preconditioning, sketch-gated randomized truncation, and
certified-by-construction energy reporting.

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the normative spec; this README is the map.

---

## What it does

For disorder ensembles of the TLS Hamiltonian on finite `L×L` lattices, it computes
variationally certified ground states via finite PEPS, extracts disorder-averaged static
observables, and from them estimates qubit decoherence channels — with explicit two-tier
honesty about which outputs are rigorous and which are model-dependent.

The distinguishing idea is that **certification is structural, not procedural**. An
`EnergyReport` cannot be constructed without passing its invariant gates: nine invariants
(INV-1..9) live as constructors and guards on the hot path, not as test-only assertions.
A number that reaches you has already proven it earned the right to exist.

## Quickstart

```bash
uv sync                                                    # Python ≥3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Do **not** plain `pip install torch` — the default PyPI Linux wheel is CUDA-linked and
> fails on CPU-only hosts (missing `libcublasLt`, observed in prototyping).

```bash
make verify          # prototype tier-1 gate: must print ALL PASS (51/51) and ALL PASS (9/9)
uv run pytest tests  # production suite: 138 passed, 2 skipped
uv run tlsmbl run configs/smoke.yaml   # end-to-end, ~40 s on an 8-core laptop
uv run tlsmbl verify runs/smoke.zarr   # re-check the invariants offline
```

If `make verify` fails, **stop and fix the environment.** Never relax a tolerance to pass.

### CLI

```bash
tlsmbl run configs/benchmark.yaml    # full pipeline, resumable via zarr stage markers
tlsmbl verify runs/bench.zarr        # offline invariant re-check on stored reports
tlsmbl bench kernels --D 2 3 4 6 8   # D-scaling microbenchmark
tlsmbl ab-test configs/ab_sdrg.yaml  # SDRG value measurement
tlsmbl aggregate runs/bench.zarr     # bootstrap CIs + REPORT.md
```

## Layout

| Path | What lives there |
|---|---|
| `ARCHITECTURE.md` | Normative spec. §0 rules, §3 invariants, §6 index conventions, §16 phase gates. |
| `src/tlsmbl/` | Production package — `core`, `model`, `peps`, `kernels`, `sdrg`, `optimize`, `ensemble`, `observables`, `io`. |
| `prototypes/` | **Reference oracle, not production code.** Never imported from `src/`; production reproduces its numbers through `tests/golden/`. |
| `prototypes/baselines/` | Frozen reference JSONs. Re-runs write to `prototypes/results/` (gitignored). Never overwrite baselines. |
| `tests/` | `unit`, `golden` (cross-check oracles), `property` (hypothesis), `perf`. |
| `docs/HANDOFF.md` | The validated-numbers table — the parity targets any port must reproduce. |
| `docs/adr/` | Architecture decision records. Next number: **ADR-016**. |
| `configs/` | `smoke`, `pilot_L8`, `benchmark`, `bench_L16_D4`, `ab_sdrg`. |
| `tools/cloud_run.sh` | GCP provisioning + resumable launch for production-scale runs. |

## Status

Phases P0–P5 are implemented and green: `uv run pytest tests` → 138 passed, 2 skipped;
`ruff` and `mypy --strict` clean; `TLSMBL_FULL_GOLD=1` adds the slow gates (L=4 golden grid,
strong-coupling ED, T-PERF exponent gap — measured **4.16**, required ≥1.6). P6/P7 (Rust
kernel, distributed sharding) are conditional per §16. See
[`docs/verify-log.md`](docs/verify-log.md) for the environment gate record.

Remaining: production-scale runs (the L=16 D=4 config in `configs/bench_L16_D4.yaml`, via
`tools/cloud_run.sh`) and the physics study plan in §12.

## Results worth knowing

Everything below was **executed**, not reviewed. Five prototype programs run against exact
references produced five spec amendments (ADR-009..013) now baked into the architecture.
Full table in [`docs/HANDOFF.md`](docs/HANDOFF.md).

- **Cross-engine agreement is not a certificate.** Under truncation, two *correct* engines
  legitimately disagree at √(disc) scale on flat-spectrum states, because the kept subspace
  is gauge-ambiguous. INV-2 χ-stability is the certificate. This is why the certificate
  stack is built the way it is.
- **The physics premise is measured, not assumed.** On *optimized* (physical) states the
  entanglement spectrum decays fast enough that a nominally truncating χ becomes effectively
  exact — discarded weight collapsed 1e-4 → 2e-15 through optimization. That localized-phase
  spectral decay is what sketching and INV-3 tightness rest on.
- **The 33× kernel speedup is not a pipeline speedup.** ADR-010's mandatory exact
  canonicalization sweep is Θ(χ³D⁸), a factor D² *above* the truncation that sketching
  accelerates — and it is not sketchable. Amdahl caps the row speedup at a measured **1.4×**
  (D=6, χ=36). D=6 is therefore wall-clock-blocked, not memory-blocked.
- **SDRG preconditioning measurably hurt at the first tested configuration** (mean gap ratio
  6.49, L=4/g_J=0.3/D=2). §16 P4 admits a negative A/B result as a valid exit; Stage A is
  quarantined behind INV-8 auto-bypass precisely so this costs nothing. Reasoning and the
  regimes where it could still help: [`docs/ab-report.md`](docs/ab-report.md).

## Conventions that bite

- Site indexing `s = y·L + x` (row-major); site 0 is the slowest bit and the first Kronecker
  factor; `Z = diag(1, −1)`, so bit 1 ↔ σᶻ = −1.
- `complex128` on all certification paths, threaded via `TensorSpec` — never hardcoded.
- Optimization gaps are usually **optimizer floor, not ansatz**: 4×4 D=2 went 4.4e-6 → 8.9e-8
  with more iterations. Check the iteration budget before blaming the ansatz.
- `quimb` is permitted **only** inside `tests/golden/` as an independent oracle.

## License

[Big Time Public License 2.0.0](LICENSE.md) — free for noncommercial use and small business;
big businesses need a commercial license on fair, reasonable, and nondiscriminatory terms.

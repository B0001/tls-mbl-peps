# D4 deliverable: SDRG A/B report (first measurement)

Run 2026-07-18: `tlsmbl ab-test configs/ab_sdrg.yaml` — 16 realizations, L=4,
g_J=0.3, D=2, chi=4, exact backend, matched 60x20 LBFGS budget per arm,
master_seed 20260716. Wall: ~50 min single-worker.

## Verdict: **SDRG hurts at this configuration** (mean gap ratio 6.49)

| metric | Stage A on | Stage A off |
|---|---|---|
| mean relative ED gap | 1.47e-2 | 2.27e-3 |
| best / worst realization | 7.7e-6 / 6.5e-2 | 1.1e-7 / 3.6e-2 |

Per-realization data: 7 decimations each (omega_stop 0.3, f_max 0.4), ledger
totals 0.05–0.24 (all under the INV-8 bypass threshold; no bypasses fired).

## Reading (why this is the expected negative, not a defect)

1. **The PT2 ledger is an energy bias, not just a bound.** Arm A's gap measures
   |E_PEPS(H-tilde) + E0 − E_ED(H)|: it includes the fidelity loss of the
   transform itself. Ledger totals of 2–4% of ||H||_local put a floor under
   arm A that arm B does not pay.
2. **There is nothing to precondition here.** At L=4, g_J=0.3 the bare D=2
   ansatz already reaches ~1e-6 relative gaps (HANDOFF: this regime is deep in
   the localized phase, near-product states). Stage A's §9 claim — that the
   circuit carries strong-coupling entanglement so smaller D_eff suffices — is
   vacuous when D=2 is already sufficient.
3. §16 P4 explicitly admits a negative A/B result as a valid exit; Stage A is
   quarantined (INV-8 auto-bypass) precisely so this outcome costs nothing in
   production: run with `sdrg.enabled: false` in this regime.

## Follow-ups for a regime where Stage A could plausibly help

- Larger effective coupling / frustration (g_J >~ 1) or D-starved settings
  (L=16 at D=2 vs D=4 matched-accuracy D_eff comparison, the actual §16 P4
  metric) — requires the L=16 production run.
- Tighter omega_stop (fewer, higher-confidence decimations) to shrink the
  ledger bias; sweep omega_stop in {0.5, 0.7, 1.0}.
- keep_first_order=false variant for comparison.

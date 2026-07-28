"""SDRG A/B harness (ARCHITECTURE.md §16 Phase 4; D4 deliverable).

Per realization, two arms at the SAME bond dimension and iteration budget:
  A (Stage A on):  SDRG-transform H, optimize PEPS on H-tilde, energy + E0
  B (Stage A off): optimize PEPS on H directly
both compared against the ED oracle (mandatory L in {3, 4}). Stage A's §9 claim is
that the circuit carries strong-coupling entanglement so arm A reaches matched
accuracy at smaller D; at fixed D that shows up as a smaller ED gap. A negative
result is a valid exit (§16) -- the harness measures, it does not assume.

INV-8: a bypassed transform makes arm A identical to arm B and is recorded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.ed_reference import ed_ground
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.optimize.init import product_init
from tlsmbl.optimize.lbfgs_driver import optimize_lbfgs
from tlsmbl.sdrg.transform import sdrg_transform


@dataclass
class ABRecord:
    k: int
    e_ed: float
    gap_with_sdrg: float  # |E_A - E_ED| / |E_ED|
    gap_without_sdrg: float
    sdrg_bypassed: bool
    n_decimations: int
    ledger_total: float


@dataclass
class ABReport:
    records: list[ABRecord]

    @property
    def mean_gap_with(self) -> float:
        return float(np.mean([r.gap_with_sdrg for r in self.records]))

    @property
    def mean_gap_without(self) -> float:
        return float(np.mean([r.gap_without_sdrg for r in self.records]))

    @property
    def verdict(self) -> str:
        ratio = self.mean_gap_with / max(self.mean_gap_without, 1e-300)
        if ratio < 0.5:
            return f"SDRG helps (gap ratio {ratio:.2f})"
        if ratio > 2.0:
            return f"SDRG hurts (gap ratio {ratio:.2f})"
        return f"no clear effect (gap ratio {ratio:.2f})"


def run_ab(
    *,
    master_seed: int,
    n_realizations: int,
    L: int,
    g_J: float,
    D: int,
    omega_stop: float = 0.3,
    f_max: float = 0.4,
    keep_first_order: bool = True,
    tau_sdrg: float = 0.05,
    max_outer: int = 40,
    inner_iters: int = 20,
) -> ABReport:
    spec = TensorSpec()
    backend = ExactSVD(eps_F=1e-12)
    chi = D * D
    records = []
    for k in range(n_realizations):
        streams = realization_streams(master_seed, k)
        params = ModelParams(L=L, g_J=g_J, R_c=3, seed_realization=k)
        real = sample_realization(params, streams.disorder)
        terms = build_terms(real)
        e_ed = float(ed_ground(terms).energies[0])

        res_b = optimize_lbfgs(
            product_init(real, D, spec, np.random.SeedSequence(streams.torch_init_seed)),
            terms,
            chi,
            backend,
            max_outer=max_outer,
            inner_iters=inner_iters,
        )
        gap_b = abs(res_b.energy - e_ed) / abs(e_ed)

        sdrg = sdrg_transform(
            terms,
            omega_stop=omega_stop,
            f_max=f_max,
            keep_first_order=keep_first_order,
            tau_sdrg=tau_sdrg,
        )
        if sdrg.bypassed:
            gap_a = gap_b
            n_dec = 0
        else:
            # §10.1: arm A's product init comes from H-tilde's on-site terms.
            L_ = terms.L
            eps_t, dlt_t = np.zeros((L_, L_)), np.zeros((L_, L_))
            for (x, y), op, c in sdrg.terms.onsite:
                if op == "z":
                    eps_t[y, x] += 2 * c
                else:
                    dlt_t[y, x] += 2 * c
            real_t = dataclasses.replace(real, eps=eps_t, delta=dlt_t)
            res_a = optimize_lbfgs(
                product_init(
                    real_t, D, spec, np.random.SeedSequence(streams.torch_init_seed)
                ),
                sdrg.terms,
                chi,
                backend,
                max_outer=max_outer,
                inner_iters=inner_iters,
            )
            assert sdrg.circuit is not None
            gap_a = abs(res_a.energy + sdrg.E0 - e_ed) / abs(e_ed)
            n_dec = len(sdrg.circuit.ops)
        records.append(
            ABRecord(
                k=k,
                e_ed=e_ed,
                gap_with_sdrg=gap_a,
                gap_without_sdrg=gap_b,
                sdrg_bypassed=sdrg.bypassed,
                n_decimations=n_dec,
                ledger_total=sdrg.ledger.total,
            )
        )
    return ABReport(records=records)

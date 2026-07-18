"""SDRG decimation loop and Hamiltonian transform (ARCHITECTURE.md §9.1, §9.4).

`sdrg_transform` runs the scale loop on a `HamiltonianTerms`, producing the
transformed terms (pinned sites keep their dominant rotated local term, ADR-002),
the circuit, the ledger, and the scalar offset E0. INV-8: if the ledger exceeds
tau_sdrg * ||H||_local the transform reports `bypassed=True` and the caller runs
Stage-A-off -- Stage A can never make a run fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tlsmbl.core.types import HamiltonianTerms, Site
from tlsmbl.sdrg.circuit import SDRGCircuit
from tlsmbl.sdrg.ledger import Ledger
from tlsmbl.sdrg.rules import EffectiveModel, bond_decimate, site_decimate


@dataclass
class SDRGResult:
    terms: HamiltonianTerms  # transformed (or original if bypassed)
    circuit: SDRGCircuit | None  # None iff bypassed
    ledger: Ledger
    E0: float  # scalar offset accumulated by decimations
    omega_sequence: list[float]  # decimation scales, flow diagnostic (§12)
    bypassed: bool


def _emit_terms(m: EffectiveModel) -> HamiltonianTerms:
    onsite: list[tuple[Site, Literal["z", "x"], float]] = []
    for site in sorted(m.eps, key=lambda s: s[1] * m.L + s[0]):
        if site in m.pinned_fields:
            # pinned site: dominant rotated local term (E_i/2) sigma~z (ADR-002)
            onsite.append((site, "z", m.pinned_fields[site] / 2.0))
        else:
            if m.eps[site] != 0.0:
                onsite.append((site, "z", m.eps[site] / 2.0))
            if m.dlt[site] != 0.0:
                onsite.append((site, "x", m.dlt[site] / 2.0))
    pair = [
        (a, b, Jv)
        for (a, b), Jv in sorted(
            m.J.items(), key=lambda kv: (kv[0][0][1] * m.L + kv[0][0][0],
                                         kv[0][1][1] * m.L + kv[0][1][0])
        )
        if Jv != 0.0
    ]
    return HamiltonianTerms(L=m.L, onsite=onsite, pair=pair)


def sdrg_transform(
    terms: HamiltonianTerms,
    *,
    omega_stop: float,
    f_max: float,
    keep_first_order: bool,
    tau_sdrg: float,
) -> SDRGResult:
    m = EffectiveModel.from_terms(terms)
    ledger = Ledger()
    circuit = SDRGCircuit()
    N = len(m.active)
    omegas: list[float] = []

    while len(circuit.ops) < f_max * N:
        site_best: tuple[float, Site] | None = max(
            ((m.site_scale(s), s) for s in m.active), default=None
        )
        bond_best = max(
            (
                (abs(Jv), (a, b))
                for (a, b), Jv in m.J.items()
                if a in m.active and b in m.active
            ),
            default=None,
        )
        omega_site = site_best[0] if site_best else 0.0
        omega_bond = bond_best[0] if bond_best else 0.0
        omega = max(omega_site, omega_bond)
        if omega <= omega_stop:
            break
        omegas.append(omega)
        if omega_site >= omega_bond:
            assert site_best is not None
            circuit.ops.append(
                site_decimate(m, site_best[1], ledger, keep_first_order=keep_first_order)
            )
        else:
            assert bond_best is not None
            a, b = bond_best[1]
            circuit.ops.append(bond_decimate(m, a, b, ledger))
        if ledger.exceeds(tau_sdrg, terms.norm_local):
            return SDRGResult(
                terms=terms,
                circuit=None,
                ledger=ledger,
                E0=0.0,
                omega_sequence=omegas,
                bypassed=True,
            )

    circuit.dropped_norm = ledger.dropped_norm
    circuit.projection_error = ledger.projection_error
    return SDRGResult(
        terms=_emit_terms(m),
        circuit=circuit,
        ledger=ledger,
        E0=m.E0,
        omega_sequence=omegas,
        bypassed=False,
    )

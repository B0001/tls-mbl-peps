"""Finalization: INV-2 chi-stability audit (ARCHITECTURE.md §3, §10.4).

Strict monotonicity of E in chi is not guaranteed for approximate contraction, so
the enforceable certificate is stability: re-evaluate once at 2*chi and require
|E(chi) - E(2chi)| <= tau_chi. Failure marks the artifact UNCERTIFIED (excluded
from aggregation unless --allow-uncertified) -- it never raises.
"""

from __future__ import annotations

import dataclasses

from tlsmbl.core.types import HamiltonianTerms
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.peps.energy import EnergyReport, energy_certified
from tlsmbl.peps.state import PEPSState


def chi_extrapolation_check(
    state: PEPSState,
    terms: HamiltonianTerms,
    report: EnergyReport,
    backend: TruncationBackend,
    *,
    tau_chi: float,
    eps_env: float,
    eps_env_E: float,
) -> EnergyReport:
    """Returns the report with chi_stability stamped; certified goes False if the
    stability gate fails (INV-2 failure action, not an exception)."""
    chi = report.env.chi
    report_2chi = energy_certified(
        state, terms, 2 * chi, backend, eps_env=eps_env, eps_env_E=eps_env_E
    )
    gap = abs(report.e_total - report_2chi.e_total)
    return dataclasses.replace(
        report,
        chi_stability=(report.e_total, report_2chi.e_total),
        certified=report.certified and gap <= tau_chi,
    )

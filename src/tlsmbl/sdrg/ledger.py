"""SDRG fidelity ledger, INV-8 (ARCHITECTURE.md §3, §9.4).

Stage A tracks the operator-norm sum of everything it drops. If the total exceeds
tau_sdrg * ||H||_local the preconditioner is REJECTED for that realization and the
pipeline runs Stage-A-off -- Stage A can only fail to help, never fail a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ledger:
    dropped_norm: float = 0.0  # beyond-PT2 truncations (site decimation)
    projection_error: float = 0.0  # doublet leakage (bond decimation)
    events: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.dropped_norm + self.projection_error

    def add_dropped(self, amount: float, what: str) -> None:
        self.dropped_norm += amount
        self.events.append(f"drop {amount:.3e}: {what}")

    def add_projection(self, amount: float, what: str) -> None:
        self.projection_error += amount
        self.events.append(f"leak {amount:.3e}: {what}")

    def exceeds(self, tau_sdrg: float, norm_local: float) -> bool:
        """INV-8 gate: True means bypass Stage A for this realization."""
        return self.total > tau_sdrg * norm_local

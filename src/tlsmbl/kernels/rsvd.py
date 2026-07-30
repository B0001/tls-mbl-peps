"""Sketched truncation backend, Stage B (ARCHITECTURE.md §8.6; INV-3 per ADR-009).

Halko-Martinsson-Tropp randomized SVD with oversampling p and power iterations q,
plus the two-sided posterior gate: the sketch may return only if its estimated
spectral error is below max(eta * sigma_1, c_gate * sigma_hat_{chi+1}) -- sketch
quality judged against the best any rank-chi truncation could achieve; whether
rank chi itself is adequate is INV-1's job. Failing the gate silently falls back
to the exact backend and increments `fallback_count` (never an error).

INV-3's second failure action -- "if fallback_rate > 20% over a sweep, disable
sketching for the realization and log" -- is enforced here on the hot path (ADR-016),
not by the caller: the backend instance IS the per-realization scope (orchestrate.py
builds one per realization from its spawned sketch stream), so the gate belongs where
the counters live. Two refinements the bare spec sentence leaves open, both decided in
ADR-016: (a) only *gate* fallbacks count toward the disable rate, not the structural
`k <= chi` fallback -- the latter is "the operand is too small to sketch", not gate
thrashing, and exact is cheaper there anyway, so counting it would disable sketching on
exactly the small early-ladder rungs where it costs nothing; (b) the rate is only
consulted after `min_gate_calls` sketchable calls, otherwise a single unlucky first
fallback reads as rate 1.0 and disables sketching permanently. The disable is
monotonic: once off, it stays off for the realization (re-enabling would make the
kernel path depend on operand order within a sweep).

AD contract (§8.5): the Gaussian test matrix and probe vectors are no-grad
buffers; gradients flow through QR and the small SVD only.

Reproducibility (INV-6): randomness comes from a torch.Generator seeded from the
realization's spawned sketch stream, passed at construction.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch

from tlsmbl.kernels.common import check_operand
from tlsmbl.kernels.interface import TruncResult
from tlsmbl.kernels.svd import ExactSVD, svd_gauge_fixed


def _crandn(
    rows: int, cols: int, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    re = torch.randn(rows, cols, generator=gen, dtype=torch.float64)
    im = torch.randn(rows, cols, generator=gen, dtype=torch.float64)
    return ((re + 1j * im) / math.sqrt(2.0)).to(dtype)


@dataclass
class SketchedSVD:
    """Randomized backend. Mutable counters track gate behavior per instance, which is
    the per-realization scope; the INV-3 disable at gate_fallback_rate >
    `disable_rate` fires on the hot path (see module docstring, ADR-016)."""

    seed: int
    oversample: int = 8
    power_iters: int = 1
    eta: float = 1e-6
    c_gate: float = 10.0
    probes: int = 6
    eps_F: float | None = None
    # INV-3 failure action. None disables the auto-disable itself (the kernel unit
    # tests want a backend that keeps sketching no matter what it measures);
    # orchestrate.py always passes config.kernels.fallback_disable_rate.
    disable_rate: float | None = None
    min_gate_calls: int = 32
    call_count: int = field(default=0, init=False)
    # Split per ADR-016(a): only gate rejections signal thrashing.
    gate_fallback_count: int = field(default=0, init=False)
    structural_fallback_count: int = field(default=0, init=False)
    gate_call_count: int = field(default=0, init=False)
    sketching_disabled: bool = field(default=False, init=False)
    disabled_at_call: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._gen = torch.Generator().manual_seed(self.seed)
        self._exact = ExactSVD(eps_F=self.eps_F)

    @property
    def fallback_count(self) -> int:
        """Every call that returned an exact result, for whatever reason (INV-1/§11
        audit reporting)."""
        return self.gate_fallback_count + self.structural_fallback_count

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.call_count if self.call_count else 0.0

    @property
    def gate_fallback_rate(self) -> float:
        """The INV-3 thrashing signal: gate rejections over *sketchable* calls only."""
        return (
            self.gate_fallback_count / self.gate_call_count
            if self.gate_call_count
            else 0.0
        )

    def stats(self) -> dict[str, float | int | bool | None]:
        """JSON-able audit record (§11: REPORT.md echoes fallback rates)."""
        return {
            "call_count": self.call_count,
            "gate_call_count": self.gate_call_count,
            "fallback_count": self.fallback_count,
            "gate_fallback_count": self.gate_fallback_count,
            "structural_fallback_count": self.structural_fallback_count,
            "fallback_rate": self.fallback_rate,
            "gate_fallback_rate": self.gate_fallback_rate,
            "sketching_disabled": self.sketching_disabled,
            "disabled_at_call": self.disabled_at_call,
        }

    def _maybe_disable(self) -> None:
        """INV-3 failure action. Monotonic, warmup-gated; logs once when it fires."""
        if self.sketching_disabled or self.disable_rate is None:
            return
        if self.gate_call_count < self.min_gate_calls:
            return
        if self.gate_fallback_rate > self.disable_rate:
            self.sketching_disabled = True
            self.disabled_at_call = self.call_count
            warnings.warn(
                f"INV-3: sketch gate fallback rate "
                f"{self.gate_fallback_rate:.1%} > {self.disable_rate:.1%} over "
                f"{self.gate_call_count} sketchable calls; disabling sketching for "
                f"this realization (slow spectral decay -- exact backend from here). "
                f"Energies stay certified; this is a performance signal.",
                stacklevel=3,
            )

    def truncate(self, Wmat: torch.Tensor, chi: int) -> TruncResult:
        self.call_count += 1
        Wmat = check_operand(Wmat)
        if self.sketching_disabled:
            # INV-3 auto-disable latched: exact for the rest of the realization. Not
            # counted as a fallback -- there was no sketch attempt to fall back from.
            return self._exact.truncate(Wmat, chi)
        m, n = Wmat.shape
        k = min(chi + self.oversample, min(m, n))
        if k <= chi:
            # No oversampling headroom: the posterior gate cannot see sigma_{chi+1},
            # and at these sizes exact SVD is cheaper than sketching anyway. Counted
            # separately: this is operand geometry, not gate thrashing (ADR-016).
            self.structural_fallback_count += 1
            return self._exact.truncate(Wmat, chi)
        self.gate_call_count += 1

        with torch.no_grad():
            G = _crandn(n, k, Wmat.dtype, self._gen)
            Om = _crandn(n, self.probes, Wmat.dtype, self._gen)

        Y = Wmat @ G
        for _ in range(self.power_iters):  # subspace iteration with re-orth
            Q0, _ = torch.linalg.qr(Y)
            Y = Wmat @ (Wmat.mH @ Q0)
        Q, _ = torch.linalg.qr(Y)
        B = Q.mH @ Wmat
        Ub, S, Vh = svd_gauge_fixed(B, self.eps_F)
        U = Q @ Ub

        with torch.no_grad():
            WOm = Wmat @ Om
            R = WOm - Q @ (Q.mH @ WOm)
            est = float(
                10.0 * math.sqrt(2.0 / math.pi) * torch.linalg.norm(R, dim=0).max()
            )
            thresh = max(self.eta * float(S[0]), self.c_gate * float(S[chi]))
        if est > thresh:  # INV-3: silent fallback, counted
            self.gate_fallback_count += 1
            self._maybe_disable()
            return self._exact.truncate(Wmat, chi)

        with torch.no_grad():
            tot = float((S**2).sum())
            disc = float((S[chi:] ** 2).sum() / tot) if tot > 0 else 0.0
        return TruncResult(
            U=U[:, :chi], S=S[:chi], Vh=Vh[:chi], disc_weight=disc, posterior_err=est
        )

"""Sketched truncation backend, Stage B (ARCHITECTURE.md §8.6; INV-3 per ADR-009).

Halko-Martinsson-Tropp randomized SVD with oversampling p and power iterations q,
plus the two-sided posterior gate: the sketch may return only if its estimated
spectral error is below max(eta * sigma_1, c_gate * sigma_hat_{chi+1}) -- sketch
quality judged against the best any rank-chi truncation could achieve; whether
rank chi itself is adequate is INV-1's job. Failing the gate silently falls back
to the exact backend and increments `fallback_count` (never an error).

AD contract (§8.5): the Gaussian test matrix and probe vectors are no-grad
buffers; gradients flow through QR and the small SVD only.

Reproducibility (INV-6): randomness comes from a torch.Generator seeded from the
realization's spawned sketch stream, passed at construction.
"""

from __future__ import annotations

import math
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
    """Randomized backend. Mutable counters track gate behavior per instance; the
    per-realization disable at fallback_rate > 20% (INV-3 failure action) is
    enforced by the orchestration layer reading these."""

    seed: int
    oversample: int = 8
    power_iters: int = 1
    eta: float = 1e-6
    c_gate: float = 10.0
    probes: int = 6
    eps_F: float | None = None
    call_count: int = field(default=0, init=False)
    fallback_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._gen = torch.Generator().manual_seed(self.seed)
        self._exact = ExactSVD(eps_F=self.eps_F)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.call_count if self.call_count else 0.0

    def truncate(self, Wmat: torch.Tensor, chi: int) -> TruncResult:
        self.call_count += 1
        Wmat = check_operand(Wmat)
        m, n = Wmat.shape
        k = min(chi + self.oversample, min(m, n))
        if k <= chi:
            # No oversampling headroom: the posterior gate cannot see sigma_{chi+1},
            # and at these sizes exact SVD is cheaper than sketching anyway.
            self.fallback_count += 1
            return self._exact.truncate(Wmat, chi)

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
            self.fallback_count += 1
            return self._exact.truncate(Wmat, chi)

        with torch.no_grad():
            tot = float((S**2).sum())
            disc = float((S[chi:] ** 2).sum() / tot) if tot > 0 else 0.0
        return TruncResult(
            U=U[:, :chi], S=S[:chi], Vh=Vh[:chi], disc_weight=disc, posterior_err=est
        )

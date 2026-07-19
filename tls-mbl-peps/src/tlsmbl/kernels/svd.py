"""Exact SVD truncation backend (ARCHITECTURE.md §8.3) -- the reference oracle.

`svd_gauge_fixed` implements INV-9: the largest-modulus entry of each column of U is
phase-rotated to be real positive (inverse phase applied to Vh). Required for AD
stability of complex SVD; every forward SVD in the codebase goes through it.

`eps_F` hardening (§8.5, INV-7): with `eps_F` set, the SVD backward is the custom
Lorentzian-broadened vjp F_ij = (s_j^2 - s_i^2) / ((s_j^2 - s_i^2)^2 + eps_F^2)
instead of torch's native one, so exactly-degenerate spectra cannot emit NaN into
the gradient. Validated against the native backward on non-degenerate operands and
against finite differences on degenerate ones (tests/unit/test_svd_hardened.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tlsmbl.kernels.common import check_operand
from tlsmbl.kernels.interface import TruncResult

_PHASE_FLOOR = 1e-300


def _gauge_phase(U: torch.Tensor) -> torch.Tensor:
    """Per-column phase of the largest-modulus entry; batch-aware ((..., m, k))."""
    idx = U.abs().argmax(dim=-2)
    ph = torch.gather(U, -2, idx.unsqueeze(-2)).squeeze(-2)
    return ph / ph.abs().clamp_min(_PHASE_FLOOR)


class _HardenedSVD(torch.autograd.Function):
    """Full economy SVD with the §8.5 broadened backward. Gauge fixing (INV-9) is
    applied inside forward so saved U/Vh are the gauge-fixed ones the formula needs."""

    generate_vmap_rule = False

    @staticmethod
    def forward(
        ctx: Any, A: torch.Tensor, eps_F: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        ph = _gauge_phase(U)
        U = U * ph.conj().unsqueeze(-2)
        Vh = Vh * ph.unsqueeze(-1)
        ctx.save_for_backward(U, S, Vh)
        ctx.eps_F = eps_F
        return U, S, Vh

    @staticmethod
    def backward(
        ctx: Any,
        gU: torch.Tensor,
        gS: torch.Tensor,
        gVh: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        U, S, Vh = ctx.saved_tensors
        eps = ctx.eps_F
        V = Vh.mH
        gV = gVh.mH
        # §8.5 Lorentzian broadening. Contract: exact vjp when spectral gaps exceed
        # eps_F; bounded finite (sub)gradient below that. Sub-eps_F kept-block
        # splittings lose the finite intra-block piece -- unavoidable for ANY
        # filter over (gU, gS, gVh), since the degenerate basis is arbitrary
        # (measured: torch's native backward fails FD there too).
        # All index ops use the trailing two dims so the formula is batch-aware
        # ((..., m, n) operands from the batched dressed-environment path).
        s2 = S**2
        diff = s2.unsqueeze(-2) - s2.unsqueeze(-1)  # diff[..., i, j] = s_j^2 - s_i^2
        F = diff / (diff**2 + eps**2)
        torch.diagonal(F, dim1=-2, dim2=-1).zero_()
        F = F.to(U.dtype)
        Sd = S.to(U.dtype)
        # Regularized inverse (same broadening scale): tiny singular values carry
        # (near-)zero cotangent weight in the compression loss (ADR-012).
        Sinv = (S / (S**2 + eps**2)).to(U.dtype)

        UhgU = U.mH @ gU
        VhgV = V.mH @ gV
        J = F * (UhgU - UhgU.mH)  # skew projections, broadened
        K = F * (VhgV - VhgV.mH)
        core = J * Sd.unsqueeze(-2) + Sd.unsqueeze(-1) * K + torch.diag_embed(gS.to(U.dtype))
        if U.is_complex():
            # Imaginary-diagonal correction: residual U(1) phase freedom per column.
            imdiag = torch.diagonal(UhgU, dim1=-2, dim2=-1).imag.to(U.dtype)
            core = core + 1.0j * torch.diag_embed(imdiag * Sinv)
        dA = U @ core @ Vh
        dA = dA + (gU - U @ UhgU) * Sinv.unsqueeze(-2) @ Vh
        dA = dA + U @ (Sinv.unsqueeze(-1) * (gV.mH - VhgV.mH @ V.mH))
        return dA, None


def svd_gauge_fixed(
    A: torch.Tensor, eps_F: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Economy SVD with the INV-9 phase gauge fixed. Differentiable; with `eps_F`
    set (certification runs) the backward is the hardened §8.5 vjp."""
    if eps_F is not None and torch.is_grad_enabled() and A.requires_grad:
        U, S, Vh = _HardenedSVD.apply(A, eps_F)  # type: ignore[no-untyped-call]
        return U, S, Vh
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    ph = _gauge_phase(U)
    return U * ph.conj().unsqueeze(-2), S, Vh * ph.unsqueeze(-1)


@dataclass(frozen=True)
class ExactSVD:
    """Reference truncation backend: full economy SVD, keep the leading chi triplet.
    INV-4 and INV-9 run inside; INV-7 finiteness is INV-4's finite check here (the
    @finite decorator wraps the tensor-network kernels that call this)."""

    eps_F: float | None = None

    def truncate(self, Wmat: torch.Tensor, chi: int) -> TruncResult:
        Wmat = check_operand(Wmat)
        U, S, Vh = svd_gauge_fixed(Wmat, self.eps_F)
        with torch.no_grad():
            tot = float((S**2).sum())
            disc = float((S[chi:] ** 2).sum() / tot) if tot > 0 else 0.0
        return TruncResult(
            U=U[:, :chi], S=S[:chi], Vh=Vh[:chi], disc_weight=disc, posterior_err=None
        )

    def truncate_batched(
        self, Wmat: torch.Tensor, chi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched exact truncation of (B, m, n) operands -- one LAPACK-batched SVD
        for a whole batch of independent dressed-environment compressions.
        Returns (U (B,m,chi'), S (B,chi'), Vh (B,chi',n), disc (B,) detached)."""
        Wmat = check_operand(Wmat)
        U, S, Vh = svd_gauge_fixed(Wmat, self.eps_F)
        with torch.no_grad():
            tot = (S**2).sum(dim=-1).clamp_min(1e-300)
            disc = (S[..., chi:] ** 2).sum(dim=-1) / tot
        return U[..., :chi], S[..., :chi], Vh[..., :chi, :], disc

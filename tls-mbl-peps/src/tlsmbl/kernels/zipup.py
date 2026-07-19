"""Zip-up compression (ARCHITECTURE.md §8.3) -- the bottleneck, isolated.

Per ADR-010: exact right-canonicalization sweep FIRST (so the truncated spectra are
the state's Schmidt spectra and INV-1's discarded weights certify the right thing),
THEN the truncating left->right sweep through the backend kernel on the E-5 operand.

Canonicalization realization (ADR-011): inside the AD graph (`want_grad=True`) the
sweep uses full-rank SVD -- torch's QR backward rejects the wide right-edge operand.
Outside the graph a cheaper LQ (QR of the conjugate transpose) is used.

Normalization: each compressed MPS is rescaled by a *detached* scalar whose log is
tracked in `CompressStats`. Energies are ratios of sandwiches sharing environments,
so a detached scale cancels identically in value AND gradient; this keeps large-L
contractions in range (INV-7) without touching AD exactness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tlsmbl.kernels.interface import TruncationBackend


@dataclass
class CompressStats:
    max_disc_weight: float
    disc_weights: list[float]
    log_norm: float
    fallback_count: int


def _right_canonicalize(
    mps: list[torch.Tensor], *, want_grad: bool, eps_F: float | None
) -> list[torch.Tensor]:
    """Exact sweep x = L-1..1: leave orthonormal-row tensors behind, push the factor
    leftward. No truncation."""
    from tlsmbl.kernels.svd import svd_gauge_fixed

    mps = list(mps)
    for x in range(len(mps) - 1, 0, -1):
        k, w, m = mps[x].shape
        mat = mps[x].reshape(k, w * m)
        if want_grad:
            U, S, Vh = svd_gauge_fixed(mat, eps_F)  # ADR-011; hardened when eps_F set
            mps[x] = Vh.reshape(Vh.shape[0], w, m)
            mps[x - 1] = torch.einsum("awk,kr->awr", mps[x - 1], U * S.to(U.dtype)[None, :])
        else:
            Q, R = torch.linalg.qr(mat.mH)  # LQ via QR of the conj transpose
            mps[x] = Q.mH.reshape(Q.shape[1], w, m)
            mps[x - 1] = torch.einsum("awk,kr->awr", mps[x - 1], R.mH)
    return mps


def compress(
    fat_mps: list[torch.Tensor],
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
) -> tuple[list[torch.Tensor], CompressStats]:
    """ADR-010 compression of a fat boundary MPS (tensors (l, v, r)) to bond <= chi."""
    mps = _right_canonicalize(
        fat_mps, want_grad=want_grad, eps_F=getattr(backend, "eps_F", None)
    )
    carry = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)
    out: list[torch.Tensor] = []
    discs: list[float] = []
    fallbacks = 0
    for T in mps:
        W = torch.einsum("kK,Kwm->kwm", carry, T)
        k, w, m = W.shape
        res = backend.truncate(W.reshape(k * w, m), chi)  # E-5 operand
        discs.append(res.disc_weight)
        if res.posterior_err is not None:
            fallbacks += 0  # sketched backend counts its own fallbacks (Phase 3)
        kk = res.S.shape[0]
        out.append(res.U.reshape(k, w, kk))
        carry = res.S.to(res.Vh.dtype)[:, None] * res.Vh
    out[-1] = out[-1] * carry[0, 0]
    scale = torch.linalg.norm(out[-1]).detach()
    out[-1] = out[-1] / scale
    return out, CompressStats(
        max_disc_weight=max(discs),
        disc_weights=discs,
        log_norm=float(torch.log(scale)),
        fallback_count=fallbacks,
    )


@dataclass
class BatchedCompressStats:
    max_disc_weight: torch.Tensor  # (B,) detached
    log_norm: torch.Tensor  # (B,) detached


def compress_batched(
    fat_mps: list[torch.Tensor],  # per site: (B, k, w, m)
    chi: int,
    backend: "object",
    *,
    want_grad: bool,
) -> tuple[list[torch.Tensor], BatchedCompressStats]:
    """Batched ADR-010 compression of B independent fat boundary MPSs sharing
    shapes (the §8.4 dressed-environment batch). Exact backend only: the batch is
    one LAPACK-batched SVD per site instead of B python-level kernel calls.
    Backends without `truncate_batched` (sketched) fall back to element loops at
    the caller."""
    from tlsmbl.kernels.svd import ExactSVD, svd_gauge_fixed

    exact = backend if isinstance(backend, ExactSVD) else ExactSVD(
        eps_F=getattr(backend, "eps_F", None)
    )
    eps_F = exact.eps_F
    mps = list(fat_mps)
    B = mps[0].shape[0]
    for x in range(len(mps) - 1, 0, -1):  # exact right-canonicalization, batched
        _, k, w, m = mps[x].shape
        mat = mps[x].reshape(B, k, w * m)
        if want_grad:
            U, S, Vh = svd_gauge_fixed(mat, eps_F)  # ADR-011
            mps[x] = Vh.reshape(B, Vh.shape[-2], w, m)
            mps[x - 1] = torch.einsum(
                "bawk,bkr->bawr", mps[x - 1], U * S.to(U.dtype).unsqueeze(-2)
            )
        else:
            Q, R = torch.linalg.qr(mat.mH)
            mps[x] = Q.mH.reshape(B, Q.shape[-1], w, m)
            mps[x - 1] = torch.einsum("bawk,bkr->bawr", mps[x - 1], R.mH)
    carry = torch.ones(B, 1, 1, dtype=mps[0].dtype, device=mps[0].device)
    out: list[torch.Tensor] = []
    max_disc = torch.zeros(B, dtype=torch.float64)
    for T in mps:
        W = torch.einsum("bkK,bKwm->bkwm", carry, T)
        _, k, w, m = W.shape
        U, S, Vh, disc = exact.truncate_batched(W.reshape(B, k * w, m), chi)
        max_disc = torch.maximum(max_disc, disc)
        kk = S.shape[-1]
        out.append(U.reshape(B, k, w, kk))
        carry = S.to(Vh.dtype).unsqueeze(-1) * Vh
    out[-1] = out[-1] * carry[:, 0, 0].reshape(B, 1, 1, 1)
    scale = (
        torch.linalg.norm(out[-1].reshape(B, -1), dim=-1).detach().clamp_min(1e-300)
    )
    out[-1] = out[-1] / scale.reshape(B, 1, 1, 1).to(out[-1].dtype)
    return out, BatchedCompressStats(
        max_disc_weight=max_disc, log_norm=torch.log(scale)
    )

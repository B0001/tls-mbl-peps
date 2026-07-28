"""Factored compression -- ADR-007 v1.1 realized under ADR-010 (see ADR-015).

Consumes a fat boundary MPS in FACTORED form -- per site the pair (M, a) with
fat = einsum('lpr,apbw->lawrb', M, a) -- and compresses to bond <= chi without
ever materializing the Theta(chi^2 D^6) fat column tensors (the ~1 GB objects
that capped the D-ladder at D=4; configs/bench_L16_D4.yaml). Peak per-site
memory is Theta(chi^2 D^4): the E-5 kernel operand itself plus one bond Gram
factor, each (chi D^2) x (chi D^2).

ADR-010's requirement (truncate the state's Schmidt spectrum, not gauge
artifacts) is met without the explicit LQ pre-sweep, which would materialize
fat-sized Q tensors. Instead the exact right-canonical gauge is carried by bond
GRAM matrices: with R_x the map from the bond right of site x to the physical
space of sites x+1..L-1, G_x = R_x R_x^H is transferred right-to-left through
the factored tensors in D^2 physical-leg slices (never exceeding chi^2 D^4 per
intermediate), and the truncation at site x SVDs W~ = W @ L_x with
G_x = L_x L_x^H (Cholesky). W~ has exactly the Schmidt spectrum the LQ-swept
operand has -- same flop class, D^2 x less memory. The price is Gram
conditioning: singular values below ~sqrt(eps_mach) of sigma_1 are noise, i.e.
discarded-weight resolution floors at ~1e-16 relative -- below every
certification threshold in use (eps_env = 1e-8). Equivalence with the v1 path
is pinned by tests/unit/test_factored_compress.py.

Gram scale control (INV-7): each transferred Gram is rescaled by a detached
scalar (mean diagonal). Only the column space and relative spectrum of L_x
enter the outputs -- U is scale-invariant, disc weights are relative, the
carry is U^H W without L -- so the rescale is exact in value AND gradient,
like the detached compression log-norms.

AD: everything here is einsums, matmuls, Cholesky, and the backend's SVD --
all with well-defined torch VJPs (no wide-QR, so ADR-011's constraint is moot
on this path). The Cholesky jitter (relative 1e-12, escalated x100 on failure)
bounds the backward's L^-1 factors.
"""

from __future__ import annotations

import torch

from tlsmbl.core.guards import NumericalCorruption
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.kernels.zipup import BatchedCompressStats, CompressStats

_CHOL_JITTER = 1e-12
_CHOL_RETRIES = 2

_LinAlgError: type[Exception] = getattr(torch.linalg, "LinAlgError")  # noqa: B009 -- not in stubs


def _chol_jittered(G: torch.Tensor) -> torch.Tensor:
    """Cholesky of a (batched) PSD Gram normalized to mean-diagonal ~1, with
    escalating relative jitter (rank-deficient bonds are routine: the fat bond
    generically exceeds the Schmidt rank)."""
    n = G.shape[-1]
    eye = torch.eye(n, dtype=G.dtype, device=G.device)
    eps = _CHOL_JITTER
    for attempt in range(_CHOL_RETRIES + 1):
        try:
            out: torch.Tensor = torch.linalg.cholesky(G + eps * eye)
            return out
        except _LinAlgError:
            if attempt == _CHOL_RETRIES:
                raise NumericalCorruption(
                    f"INV-7: bond Gram not factorizable at jitter {eps:.1e} "
                    f"(factored canonicalization sweep)"
                ) from None
            eps *= 100.0
    raise AssertionError("unreachable")


def _bond_chols(
    Ms: list[torch.Tensor], As: list[torch.Tensor]
) -> list[torch.Tensor | None]:
    """Right-to-left Gram sweep (the ADR-010 gauge, factored). Returns per site
    the Cholesky factor L_x of the Gram on the bond to its RIGHT (None at the
    last site: empty right part, Gram = I). Batch-first: Ms (B,l,p,r),
    As (B,al,p,be,w)."""
    L = len(Ms)
    chols: list[torch.Tensor | None] = [None] * L
    Lfac: torch.Tensor | None = None  # factor of the Gram right of site x
    for x in range(L - 1, 0, -1):
        M, a = Ms[x], As[x]
        if not torch.isfinite(M.real).all() or (
            M.is_complex() and not torch.isfinite(M.imag).all()
        ):
            # INV-7 provenanced here, not at the LAPACK call site (same contract
            # as zipup._right_canonicalize: an oversized LBFGS trial step lands
            # as a rejected step, not a cryptic Cholesky/SVD failure).
            raise NumericalCorruption(
                f"INV-7: non-finite operand entering factored Gram sweep "
                f"(site index {x}, shape {tuple(M.shape)})"
            )
        B, ld, p, r = M.shape
        al, be, w = a.shape[1], a.shape[3], a.shape[4]
        G = torch.zeros(B, ld * al, ld * al, dtype=M.dtype, device=M.device)
        for wi in range(w):
            # fat_w: the (chi D^2) x (chi D^2) physical-leg slice of the fat
            # tensor -- the largest object this sweep ever materializes.
            fat_w = torch.einsum("blpr,bapc->blarc", M, a[..., wi]).reshape(
                B, ld * al, r * be
            )
            T = fat_w if Lfac is None else fat_w @ Lfac
            G = G + T @ T.mH
        diag = torch.diagonal(G, dim1=-2, dim2=-1).real
        scale = diag.mean(dim=-1).detach().clamp_min(1e-300)
        G = G / scale.reshape(B, 1, 1).to(G.dtype)
        Lfac = _chol_jittered(G)
        chols[x - 1] = Lfac
    return chols


def _sweep_factored(
    Ms: list[torch.Tensor],
    As: list[torch.Tensor],
    chi: int,
    backend: TruncationBackend,
    *,
    batched_kernel: bool,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Truncating left->right sweep on the factored fat MPS in the Gram-carried
    canonical gauge. Batch-first; returns (out tensors (B,k,w,chi'), disc
    weights (B, L) detached, log_norm (B,) detached). With `batched_kernel` the
    backend must be an ExactSVD (truncate_batched); otherwise B must be 1 and
    any TruncationBackend (including sketched) works."""
    chols = _bond_chols(Ms, As)
    B = Ms[0].shape[0]
    dtype, device = Ms[0].dtype, Ms[0].device
    carry = torch.ones(B, 1, 1, dtype=dtype, device=device)
    out: list[torch.Tensor] = []
    discs: list[torch.Tensor] = []
    for x in range(len(Ms)):
        M, a = Ms[x], As[x]
        _, ld, p, r = M.shape
        al, be, w = a.shape[1], a.shape[3], a.shape[4]
        k = carry.shape[1]
        # E-5 operand, factored build: never touches a chi^2 D^6 intermediate.
        t1 = torch.einsum("bkla,blpr->bkapr", carry.reshape(B, k, ld, al), M)
        W = torch.einsum("bkapr,bapcw->bkwrc", t1, a).reshape(B, k * w, r * be)
        Lx = chols[x]
        Wt = W if Lx is None else W @ Lx
        if batched_kernel:
            U, _S, _Vh, disc = backend.truncate_batched(Wt, chi)  # type: ignore[attr-defined]
        else:
            res = backend.truncate(Wt[0], chi)
            U = res.U.unsqueeze(0)
            disc = torch.tensor([res.disc_weight], dtype=torch.float64)
        kk = U.shape[-1]
        out.append(U.reshape(B, k, w, kk))
        carry = U.mH @ W
        discs.append(disc)
    out[-1] = out[-1] * carry[:, 0, 0].reshape(B, 1, 1, 1)
    scale = (
        torch.linalg.norm(out[-1].reshape(B, -1), dim=-1).detach().clamp_min(1e-300)
    )
    out[-1] = out[-1] / scale.reshape(B, 1, 1, 1).to(dtype)
    return out, torch.stack(discs, dim=-1), torch.log(scale)


def compress_factored(
    Ms: list[torch.Tensor],
    As: list[torch.Tensor],
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
) -> tuple[list[torch.Tensor], CompressStats]:
    """Factored counterpart of zipup.compress: same contract, but the fat MPS
    arrives as (M (l,p,r), a (al,p,be,w)) pairs in top-absorption convention
    (bottom callers permute `a`; boundary.py owns the conventions). `want_grad`
    is accepted for interface symmetry -- unlike v1 this path has no
    QR-vs-SVD split (no QR at all)."""
    del want_grad
    out_b, discs, log_norm = _sweep_factored(
        [m.unsqueeze(0) for m in Ms],
        [a.unsqueeze(0) for a in As],
        chi,
        backend,
        batched_kernel=False,
    )
    disc_list = [float(d) for d in discs[0]]
    return [t[0] for t in out_b], CompressStats(
        max_disc_weight=max(disc_list),
        disc_weights=disc_list,
        log_norm=float(log_norm[0]),
        fallback_count=0,
    )


def compress_factored_batched(
    Ms: list[torch.Tensor],
    As: list[torch.Tensor],
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
) -> tuple[list[torch.Tensor], BatchedCompressStats]:
    """Factored counterpart of zipup.compress_batched (B independent fat MPSs,
    tensors (B,l,p,r) / (B,al,p,be,w)). Exact backend only, like
    compress_batched: one LAPACK-batched SVD per site."""
    del want_grad
    exact = backend if isinstance(backend, ExactSVD) else ExactSVD(
        eps_F=getattr(backend, "eps_F", None)
    )
    out, discs, log_norm = _sweep_factored(Ms, As, chi, exact, batched_kernel=True)
    return out, BatchedCompressStats(
        max_disc_weight=discs.max(dim=-1).values, log_norm=log_norm
    )

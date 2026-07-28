"""§8.5 hardened SVD backward validation (§14.4 requires FD validation before
anything depends on it, including a degenerate-singular-value construction)."""

import numpy as np
import pytest
import torch

from tlsmbl.kernels.svd import svd_gauge_fixed

EPS_F = 1e-12


def _crandn(rng: np.random.Generator, *shape: int) -> torch.Tensor:
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    return torch.from_numpy(a).to(torch.complex128)


def _loss_through_svd(A: torch.Tensor, chi: int, eps_F: float | None) -> torch.Tensor:
    """A bilinearly-recombining truncation loss (the ADR-012 class): the sliced
    triplet recombines into the rank-chi approximation, probed against a fixed
    matrix so U/S/Vh cotangents are all exercised."""
    U, S, Vh = svd_gauge_fixed(A, eps_F)
    approx = U[:, :chi] @ torch.diag(S[:chi].to(A.dtype)) @ Vh[:chi]
    probe = torch.arange(approx.numel(), dtype=torch.float64).reshape(approx.shape)
    return (approx.real * probe).sum() + (approx.imag * probe).sum() * 0.5


def test_hardened_matches_native_backward_nondegenerate() -> None:
    rng = np.random.default_rng(3)
    A0 = _crandn(rng, 10, 8) * torch.exp(-0.5 * torch.arange(8))[None, :]
    grads = []
    for eps_F in (None, EPS_F):
        A = A0.clone().requires_grad_(True)
        _loss_through_svd(A, 4, eps_F).backward()
        grads.append(A.grad.clone())
    assert torch.linalg.norm(grads[0] - grads[1]) / torch.linalg.norm(grads[0]) < 1e-8


def _degenerate_operand(rng: np.random.Generator, split: float = 0.0) -> torch.Tensor:
    """(Near-)degenerate spectrum s = (2, 1+split, 1, 1-split, 0.5, 0.25): at
    split=0 the un-broadened F_ij is singular within the triple and native backward
    can NaN."""
    Uq, _ = torch.linalg.qr(_crandn(rng, 10, 6))
    Vq, _ = torch.linalg.qr(_crandn(rng, 8, 6))
    s = torch.tensor([2.0, 1.0 + split, 1.0, 1.0 - split, 0.5, 0.25], dtype=torch.float64)
    return Uq @ torch.diag(s.to(torch.complex128)) @ Vq.mH


def _fd_check(A0: torch.Tensor, chi: int, eps_F: float, tol: float) -> None:
    rng = np.random.default_rng(44)
    A = A0.clone().requires_grad_(True)
    _loss_through_svd(A, chi, eps_F).backward()
    assert torch.isfinite(A.grad.real).all() and torch.isfinite(A.grad.imag).all()
    h = 1e-6
    rel = []
    for _ in range(8):
        i, j = rng.integers(0, A0.shape[0]), rng.integers(0, A0.shape[1])
        for part in (1.0, 1.0j):
            Ap, Am = A0.clone(), A0.clone()
            Ap[i, j] += h * part
            Am[i, j] -= h * part
            fd = (
                float(_loss_through_svd(Ap, chi, eps_F))
                - float(_loss_through_svd(Am, chi, eps_F))
            ) / (2 * h)
            g = A.grad[i, j]
            ad = float(g.real) if part == 1.0 else float(g.imag)
            rel.append(abs(fd - ad) / max(abs(fd), abs(ad), 1e-8))
    assert max(rel) < tol, f"max rel err {max(rel):.3e}"


def test_backward_matches_fd_nondegenerate() -> None:
    _fd_check(_crandn(np.random.default_rng(4), 10, 8), 3, EPS_F, 5e-6)


def test_backward_matches_fd_near_degenerate() -> None:
    """Splitting 1e-4 across a kept near-degenerate triple, eps_F = 1e-12 well below
    it: broadening must be inactive and FD must agree. This is the regime the
    certification default (kernels.eps_F = 1e-12) actually operates in."""
    _fd_check(_degenerate_operand(np.random.default_rng(5), split=1e-4), 4, 1e-12, 1e-5)


@pytest.mark.parametrize("chi", [4, 3, 1])
def test_exact_degeneracy_finite_bounded_gradient(chi: int) -> None:
    """Exact degeneracy, every cut position (kept whole, straddled, discarded whole):
    the SVD basis inside the block is arbitrary, so no backward computable from
    (gU, gS, gVh) can match FD there (measured: torch's native backward fails FD
    too). The hardened contract is INV-7's: a finite, bounded (sub)gradient, never
    NaN. At chi=1 the block carries zero cotangent and FD agreement does hold."""
    A = _degenerate_operand(np.random.default_rng(6)).requires_grad_(True)
    _loss_through_svd(A, chi, 1e-6).backward()
    assert torch.isfinite(A.grad.real).all() and torch.isfinite(A.grad.imag).all()
    assert float(torch.linalg.norm(A.grad)) < 1e4


def test_exact_degeneracy_discarded_block_matches_fd() -> None:
    """Degenerate triple entirely discarded (chi=1): its cotangents are zero, so FD
    agreement survives even exact degeneracy."""
    _fd_check(_degenerate_operand(np.random.default_rng(7)), 1, 1e-6, 5e-6)


def test_energy_graph_with_hardening_matches_unhardened() -> None:
    """Same energy gradient with and without eps_F on a healthy spectrum."""
    from tlsmbl.core.types import ModelParams, TensorSpec
    from tlsmbl.kernels.svd import ExactSVD
    from tlsmbl.model.hamiltonian import build_terms
    from tlsmbl.model.sampling import sample_realization
    from tlsmbl.peps.energy import energy_differentiable
    from tlsmbl.peps.state import PEPSState

    seed = 20260718
    params = ModelParams(L=3, g_J=0.3, R_c=3, seed_realization=seed)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(seed)))
    terms = build_terms(real)
    grads = []
    for eps_F in (None, EPS_F):
        state = PEPSState.random(3, 2, TensorSpec(), np.random.SeedSequence(seed))
        for row in state.tensors:
            for t in row:
                t.requires_grad_(True)
        energy_differentiable(state, terms, 2, ExactSVD(eps_F=eps_F)).backward()
        grads.append(torch.cat([t.grad.flatten() for row in state.tensors for t in row]))
    assert torch.linalg.norm(grads[0] - grads[1]) / torch.linalg.norm(grads[0]) < 1e-7

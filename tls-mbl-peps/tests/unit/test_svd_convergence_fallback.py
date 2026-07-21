"""Regression test for the LAPACK convergence fallback (found running the L=8
pilot, 2026-07-20): torch's default CPU SVD driver (gesdd) occasionally raises
LinAlgError on ill-conditioned operands; `_svd_robust` falls back to scipy's
gesvd driver. Simulated via monkeypatch since the real failure isn't
deterministically reproducible from a constructed matrix."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tlsmbl.kernels import svd as svd_mod
from tlsmbl.kernels.svd import ExactSVD, _svd_robust, svd_gauge_fixed


def _crandn(rng: np.random.Generator, *shape: int) -> torch.Tensor:
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    return torch.from_numpy(a).to(torch.complex128)


def _raise_once(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patches torch.linalg.svd to raise LinAlgError on its first call only,
    inside svd.py's namespace (where _svd_robust references it)."""
    calls = [0]
    real_svd = torch.linalg.svd

    def flaky(A: torch.Tensor, full_matrices: bool = True) -> object:
        calls[0] += 1
        if calls[0] == 1:
            raise torch.linalg.LinAlgError(
                "linalg.svd: The algorithm failed to converge (simulated)"
            )
        return real_svd(A, full_matrices=full_matrices)

    monkeypatch.setattr(svd_mod.torch.linalg, "svd", flaky)
    return calls


def test_svd_robust_falls_back_on_single_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _raise_once(monkeypatch)
    A = _crandn(np.random.default_rng(1), 12, 9)
    U, S, Vh = _svd_robust(A)
    assert calls[0] == 1  # only the scipy path ran the recovery, no torch retry
    recon = U @ torch.diag(S.to(U.dtype)) @ Vh
    assert torch.linalg.norm(recon - A) / torch.linalg.norm(A) < 1e-10


def test_svd_robust_falls_back_per_element_in_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _raise_once(monkeypatch)
    rng = np.random.default_rng(2)
    A = torch.stack([_crandn(rng, 8, 6) for _ in range(4)])
    U, S, Vh = _svd_robust(A)
    recon = U @ torch.diag_embed(S.to(U.dtype)) @ Vh
    assert torch.linalg.norm((recon - A).reshape(4, -1), dim=-1).max() < 1e-10


def test_svd_gauge_fixed_survives_convergence_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _raise_once(monkeypatch)
    A = _crandn(np.random.default_rng(3), 10, 7)
    U, S, Vh = svd_gauge_fixed(A, eps_F=1e-12)  # hardened (no-grad) path
    recon = U @ torch.diag(S.to(U.dtype)) @ Vh
    assert torch.linalg.norm(recon - A) / torch.linalg.norm(A) < 1e-10


def test_exact_svd_truncate_survives_convergence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raise_once(monkeypatch)
    A = _crandn(np.random.default_rng(4), 20, 14)
    res = ExactSVD(eps_F=1e-12).truncate(A, chi=6)
    assert res.U.shape == (20, 6)
    assert torch.isfinite(res.S).all()

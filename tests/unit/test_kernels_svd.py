"""Exact backend unit tests: INV-9 gauge property, truncation parity with the numpy
prototype kernel semantics, INV-4 gates."""

import numpy as np
import pytest
import torch

from tlsmbl.core.guards import NumericalCorruption
from tlsmbl.kernels.common import check_operand
from tlsmbl.kernels.svd import ExactSVD, svd_gauge_fixed


def _crandn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(torch.complex128)


def test_gauge_fix_largest_entry_real_positive() -> None:
    rng = np.random.default_rng(7)
    U, S, Vh = svd_gauge_fixed(_t(_crandn(rng, 24, 16)))
    idx = U.abs().argmax(dim=0)
    pivots = U[idx, torch.arange(U.shape[1])]
    assert torch.allclose(pivots.imag, torch.zeros_like(pivots.imag), atol=1e-14)
    assert (pivots.real > 0).all()


def test_gauge_fixed_svd_reconstructs() -> None:
    rng = np.random.default_rng(8)
    A = _t(_crandn(rng, 20, 12))
    U, S, Vh = svd_gauge_fixed(A)
    assert torch.linalg.norm(U @ torch.diag(S.to(U.dtype)) @ Vh - A) < 1e-12


def test_truncate_matches_numpy_reference() -> None:
    """Same operand: torch backend reproduces the numpy prototype's kept triplet and
    discarded weight (gauge fix makes U/Vh comparable directly)."""
    rng = np.random.default_rng(9)
    W = _crandn(rng, 30, 30) * np.exp(-0.4 * np.arange(30))[None, :]
    chi = 8
    res = ExactSVD().truncate(_t(W), chi)
    Un, Sn, Vhn = np.linalg.svd(W, full_matrices=False)
    idx = np.abs(Un).argmax(axis=0)
    ph = Un[idx, np.arange(Un.shape[1])]
    ph /= np.abs(ph)
    Un, Vhn = Un * ph.conj()[None, :], Vhn * ph[:, None]
    disc_np = float((Sn[chi:] ** 2).sum() / (Sn**2).sum())
    np.testing.assert_allclose(res.S.numpy(), Sn[:chi], rtol=1e-12)
    np.testing.assert_allclose(res.U.numpy(), Un[:, :chi], atol=1e-10)
    np.testing.assert_allclose(res.Vh.numpy(), Vhn[:chi], atol=1e-10)
    assert abs(res.disc_weight - disc_np) < 1e-14
    assert res.posterior_err is None


def test_truncate_lossless_when_chi_covers_rank() -> None:
    rng = np.random.default_rng(10)
    A = _crandn(rng, 16, 4)
    W = A @ _crandn(rng, 4, 16)  # rank 4
    res = ExactSVD().truncate(_t(W), 8)
    assert res.disc_weight < 1e-28


def test_check_operand_rejects_nan() -> None:
    W = torch.full((4, 4), torch.nan, dtype=torch.complex128)
    with pytest.raises(NumericalCorruption, match="INV-4"):
        check_operand(W)


def test_check_operand_norm_object_hermiticity() -> None:
    rng = np.random.default_rng(11)
    H = _t(_crandn(rng, 6, 6))
    H = H + H.mH  # Hermitian
    out = check_operand(H, is_norm_object=True)
    assert torch.linalg.norm(out - out.mH) < 1e-14
    with pytest.raises(NumericalCorruption, match="INV-4"):
        check_operand(H + 1.0j * torch.eye(6), is_norm_object=True)  # skew part

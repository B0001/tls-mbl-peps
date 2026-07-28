"""T-INV-3 + T-EQ-BACKENDS(torch) (§14.7/§14.8): the two-sided gate accepts
optimal-quality sketches on fast-decaying spectra (ADR-009's defining case),
falls back conservatively on slow decay, and the accepted sketch's kept subspace
matches the exact backend's."""

import math

import numpy as np
import torch

from tlsmbl.kernels.rsvd import SketchedSVD
from tlsmbl.kernels.svd import ExactSVD


def _operand(spectrum: np.ndarray, m: int, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    k = len(spectrum)

    def crandn(*s: int) -> np.ndarray:
        out: np.ndarray = (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        return out

    U, _ = np.linalg.qr(crandn(m, k))
    V, _ = np.linalg.qr(crandn(n, k))
    return torch.from_numpy((U * spectrum) @ V.conj().T).to(torch.complex128)


def test_gate_accepts_fast_decay_and_matches_optimal() -> None:
    """ADR-009's measured case: sigma_k = e^(-0.5k). The v1 fixed-eta gate rejected
    an optimal sketch here; the two-sided gate must accept, and the sketch error
    must equal the optimal rank-chi error to high accuracy."""
    chi, m = 16, 256
    spectrum = np.exp(-0.5 * np.arange(1, m + 1))
    W = _operand(spectrum, m, m, 1)
    backend = SketchedSVD(seed=7)
    res = backend.truncate(W, chi)
    assert backend.fallback_count == 0
    assert res.posterior_err is not None  # sketched path taken
    approx = res.U @ torch.diag(res.S.to(W.dtype)) @ res.Vh
    err_sketch = float(torch.linalg.matrix_norm(W - approx, ord=2))
    err_opt = float(spectrum[chi])  # optimal rank-chi spectral error
    assert abs(err_sketch - err_opt) / err_opt < 1e-9


def test_gate_falls_back_on_slow_decay() -> None:
    """sigma_k ~ 1/k: probe estimator is structurally pessimistic -> conservative
    fallback (correct per ADR-009; this regime triggers chi-escalation anyway)."""
    chi, m = 16, 256
    spectrum = 1.0 / np.arange(1, m + 1)
    W = _operand(spectrum, m, m, 2)
    backend = SketchedSVD(seed=8)
    res = backend.truncate(W, chi)
    assert backend.fallback_count == 1
    assert res.posterior_err is None  # exact result returned
    assert backend.fallback_rate == 1.0


def test_eq_backends_subspace_angle() -> None:
    """T-EQ-BACKENDS on ADR-009's canonical instance (256x256, sigma_k = e^(-0.5k),
    chi=16, k=24 -- the exact shape the prototype measured at 1.6e-6 rad): principal
    angle between exact and sketched kept column spaces within 5e-6 rad."""
    chi, m = 16, 256
    spectrum = np.exp(-0.5 * np.arange(1, m + 1))
    W = _operand(spectrum, m, m, 3)
    r_exact = ExactSVD().truncate(W, chi)
    r_sketch = SketchedSVD(seed=9).truncate(W, chi)
    assert r_sketch.posterior_err is not None
    overlap = r_exact.U.mH @ r_sketch.U
    sv = torch.linalg.svdvals(overlap)
    angle_rad = math.acos(min(float(sv.min()), 1.0))
    assert angle_rad < 5e-6, f"principal angle {angle_rad:.3e} rad"
    np.testing.assert_allclose(r_sketch.S.numpy(), r_exact.S.numpy(), rtol=1e-9)


def test_sketch_reproducible_per_seed() -> None:
    chi, m = 8, 128
    spectrum = np.exp(-0.3 * np.arange(1, m + 1))
    W = _operand(spectrum, m, m, 4)
    a = SketchedSVD(seed=11).truncate(W, chi)
    b = SketchedSVD(seed=11).truncate(W, chi)
    assert torch.equal(a.U, b.U) and torch.equal(a.S, b.S)
    c = SketchedSVD(seed=12).truncate(W, chi)
    assert not torch.equal(a.U, c.U)  # different stream, different sketch


def test_no_headroom_falls_back() -> None:
    """chi = min(m, n): k cannot exceed chi, the gate cannot see sigma_{chi+1},
    and the backend must go exact."""
    W = _operand(np.exp(-0.5 * np.arange(1, 9)), 8, 8, 5)
    backend = SketchedSVD(seed=13)
    res = backend.truncate(W, chi=8)
    assert backend.fallback_count == 1
    assert res.posterior_err is None

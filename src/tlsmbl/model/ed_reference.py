"""Exact-diagonalization oracle (ARCHITECTURE.md §7.3) -- the correctness anchor.

Sparse H in the full 2^N basis via Kronecker assembly of the canonical term list.
Mandatory support L in {3, 4} (2^16 = 65,536). Site 0 is the slowest bit = first
Kronecker factor; Z = diag(1, -1) so bit 1 corresponds to sigma^z = -1 (§6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from tlsmbl.core.types import HamiltonianTerms, Site, site_index

I2 = np.eye(2)
Z = np.diag([1.0, -1.0])
X = np.array([[0.0, 1.0], [1.0, 0.0]])

_MAX_DENSE_N = 12  # 2^12 x 2^12 dense is fine; beyond that Lanczos only


def op_chain(ops: dict[int, np.ndarray], N: int) -> sparse.csr_matrix:
    """Kronecker chain with identity everywhere except the given site indices."""
    M = sparse.identity(1, format="csr")
    for s in range(N):
        M = sparse.kron(M, sparse.csr_matrix(ops.get(s, I2)), format="csr")
    return M


def build_H(terms: HamiltonianTerms) -> sparse.csr_matrix:
    L = terms.L
    N = L * L
    H = sparse.csr_matrix((2**N, 2**N))
    for site, op, c in terms.onsite:
        H = H + c * op_chain({site_index(site, L): Z if op == "z" else X}, N)
    for site_a, site_b, J in terms.pair:
        H = H + J * op_chain({site_index(site_a, L): Z, site_index(site_b, L): Z}, N)
    return H


@dataclass(frozen=True)
class EDResult:
    energies: np.ndarray  # k lowest eigenvalues, ascending
    ground: np.ndarray  # ground-state vector, 2^N


def ed_ground(terms: HamiltonianTerms, k: int = 4) -> EDResult:
    """k lowest eigenpairs via sparse Lanczos (dense cross-check path for small N)."""
    H = build_H(terms)
    N = terms.L**2
    if N <= _MAX_DENSE_N:
        w, v = np.linalg.eigh(H.toarray())
        return EDResult(energies=w[:k].copy(), ground=v[:, 0].copy())
    # Deterministic Lanczos start vector: reproducible fixtures (INV-6 discipline).
    v0 = np.full(2**N, 1.0 / np.sqrt(2**N))
    w, v = eigsh(H, k=k, which="SA", maxiter=20000, v0=v0)
    order = np.argsort(w)
    return EDResult(energies=w[order], ground=v[:, order[0]])


def ed_observables(
    terms: HamiltonianTerms, psi: np.ndarray
) -> tuple[dict[Site, float], dict[Site, float], dict[tuple[Site, Site], float]]:
    """<sigma^z_i>, <sigma^x_i>, and <sigma^z_i sigma^z_j> on the H-support pairs."""
    L = terms.L
    N = L * L
    nrm = float(np.vdot(psi, psi).real)

    def ev(ops: dict[int, np.ndarray]) -> float:
        return float(np.vdot(psi, op_chain(ops, N) @ psi).real) / nrm

    sz = {(x, y): ev({site_index((x, y), L): Z}) for y in range(L) for x in range(L)}
    sx = {(x, y): ev({site_index((x, y), L): X}) for y in range(L) for x in range(L)}
    zz = {
        (i, j): ev({site_index(i, L): Z, site_index(j, L): Z})
        for i, j, _ in terms.pair
    }
    return sz, sx, zz


def free_energy_analytic(eps: np.ndarray, delta: np.ndarray) -> float:
    """J=0 ground energy: -1/2 sum_i sqrt(eps_i^2 + delta_i^2). Test anchor."""
    return float(-0.5 * np.sqrt(eps**2 + delta**2).sum())

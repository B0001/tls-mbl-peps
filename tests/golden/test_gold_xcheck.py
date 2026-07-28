"""T-GOLD-XCHECK (§14.2, blocking Phase 2): the same instances contracted by quimb
(tests-only dependency, §4) agree with our environment contraction to 1e-9 --
isolating contraction bugs from optimization bugs.

The quimb network is built directly from shared bond indices and contracted by
quimb's own path finder: fully independent of our E-1..E-5 conventions."""

import numpy as np
import pytest
import quimb.tensor as qtn
import torch

from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.boundary import build_env
from tlsmbl.peps.energy import Z, energy_differentiable, sandwich
from tlsmbl.peps.state import PEPSState

L = 3
SEED = 20260719
SPEC = TensorSpec()


def _quimb_ket(state: PEPSState) -> qtn.TensorNetwork:
    """One qtn.Tensor per site; bonds h{y}_{x} (right of (x,y)) and v{y}_{x}
    (below (x,y)); dim-1 boundary legs squeezed out."""
    tensors = []
    for y in range(L):
        for x in range(L):
            a = state.tensors[y][x].numpy()
            inds = [f"p{y}{x}"]
            keep = [0]
            for axis, ind in (
                (1, f"h{y}_{x - 1}"),
                (2, f"v{y - 1}_{x}"),
                (3, f"h{y}_{x}"),
                (4, f"v{y}_{x}"),
            ):
                if a.shape[axis] > 1:
                    inds.append(ind)
                    keep.append(axis)
            arr = a.transpose(keep + [ax for ax in range(5) if ax not in keep]).reshape(
                [a.shape[ax] for ax in keep]
            )
            tensors.append(qtn.Tensor(arr, inds=inds))
    return qtn.TensorNetwork(tensors)


def _quimb_expect(state: PEPSState, ops: dict[tuple[int, int], np.ndarray]) -> float:
    ket = _quimb_ket(state)
    bra = ket.H.reindex({f"p{y}{x}": f"q{y}{x}" for y in range(L) for x in range(L)})
    center = [
        qtn.Tensor(
            ops.get((x, y), np.eye(2)).astype(np.complex128),
            inds=[f"q{y}{x}", f"p{y}{x}"],
        )
        for y in range(L)
        for x in range(L)
    ]
    num = (ket | qtn.TensorNetwork(center) | bra).contract()
    ident = [
        qtn.Tensor(np.eye(2, dtype=np.complex128), inds=[f"q{y}{x}", f"p{y}{x}"])
        for y in range(L)
        for x in range(L)
    ]
    den = (ket | qtn.TensorNetwork(ident) | bra).contract()
    return float((num / den).real)


@pytest.mark.parametrize("D", [2, 3])
def test_xcheck_site_observables(D: int) -> None:
    state = PEPSState.random(L, D, SPEC, np.random.SeedSequence(SEED + D))
    env = build_env(state, D * D, ExactSVD(), want_grad=False)
    Znp = np.diag([1.0, -1.0])
    with torch.no_grad():
        for y in range(L):
            norm = sandwich(env.tops[y], state, y, env.bottoms[y + 1])
            for x in range(L):
                ours = float(
                    (sandwich(env.tops[y], state, y, env.bottoms[y + 1], {x: Z}) / norm).real
                )
                theirs = _quimb_expect(state, {(x, y): Znp})
                assert abs(ours - theirs) < 1e-9, f"site ({x},{y}): {ours} vs {theirs}"


@pytest.mark.parametrize("D", [2, 3])
def test_xcheck_total_energy(D: int) -> None:
    params = ModelParams(L=L, g_J=0.3, R_c=3, seed_realization=SEED)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(SEED)))
    terms = build_terms(real)
    state = PEPSState.random(L, D, SPEC, np.random.SeedSequence(SEED + D))
    ours = float(energy_differentiable(state, terms, D * D, ExactSVD()))
    Znp, Xnp = np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])
    theirs = sum(
        c * _quimb_expect(state, {site: Znp if op == "z" else Xnp})
        for site, op, c in terms.onsite
    ) + sum(J * _quimb_expect(state, {i: Znp, j: Znp}) for i, j, J in terms.pair)
    assert abs(ours - theirs) / max(abs(theirs), 1e-12) < 1e-9

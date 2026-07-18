"""Energy assembly and the certified-energy factory (ARCHITECTURE.md §8.4, INV-1).

`energy_certified` is the ONLY public energy API; an `EnergyReport` cannot be
constructed anywhere else (INV gate by construction, §0 rule 2).

Assembly: every term is an E-4 sandwich ratio against the row norm (scales cancel,
so the detached compression normalization is exact). Cross-row pairs use §8.4
operator-dressed environments. The INV-1 up/down certificate evaluates the full
energy twice with mirrored roles: E_down dresses top environments (source spin
absorbed downward), E_up dresses bottoms (absorbed upward), exercising both sweep
directions end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tlsmbl.core.guards import finite
from tlsmbl.core.types import HamiltonianTerms, Site
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.peps.boundary import (
    BoundaryMPS,
    build_bottoms,
    build_env,
    build_tops,
)
from tlsmbl.peps.doublelayer import double_layer
from tlsmbl.peps.state import PEPSState

Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)


class EnvironmentNotConverged(RuntimeError):
    """INV-1 failure; the optimizer catches this and escalates chi."""


@finite
def sandwich(
    T: BoundaryMPS,
    state: PEPSState,
    y: int,
    B: BoundaryMPS,
    ops: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """E-4a/b/c: <T| row_y (with operator insertions {x: O}) |B>, relative scale."""
    L = state.L
    F = torch.ones(1, 1, 1, dtype=state.tensors[0][0].dtype)
    for x in range(L):
        a = double_layer(state.tensors[y][x], (ops or {}).get(x))
        F = torch.einsum("tmb,tvT->mbvT", F, T.tensors[x])  # E-4a
        F = torch.einsum("mbvT,mvnw->bTnw", F, a)  # E-4b
        F = torch.einsum("bTnw,bwB->TnB", F, B.tensors[x])  # E-4c
    return F.reshape(())


def _assemble(
    state: PEPSState,
    terms: HamiltonianTerms,
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
    dress: str,  # "top" | "bottom"
) -> tuple[torch.Tensor, float]:
    """Differentiable energy; returns (E, row_consistency). Cross-row pairs dressed
    per `dress` direction (the two directions form INV-1's up/down certificate)."""
    L = state.L
    tops, _ = build_tops(state, chi, backend, want_grad=want_grad)
    bottoms, _ = build_bottoms(state, chi, backend, want_grad=want_grad)
    norms = [sandwich(tops[y], state, y, bottoms[y + 1]) for y in range(L)]
    with torch.no_grad():
        scaled = [
            float(n.real) * np.exp(tops[y].log_norm + bottoms[y + 1].log_norm)
            for y, n in enumerate(norms)
        ]
        row_consistency = max(abs(s / scaled[0] - 1.0) for s in scaled)

    cross = sorted(
        {i if dress == "top" else j for (i, j, _) in terms.pair if i[1] != j[1]}
    )
    dressed: dict[Site, list[BoundaryMPS]] = {}
    for s in cross:
        if dress == "top":
            dressed[s], _ = build_tops(state, chi, backend, want_grad=want_grad, insert={s: Z})
        else:
            dressed[s], _ = build_bottoms(state, chi, backend, want_grad=want_grad, insert={s: Z})

    E = torch.zeros((), dtype=torch.float64)
    for (x, y), op, c in terms.onsite:
        op_mat = Z if op == "z" else X
        v = sandwich(tops[y], state, y, bottoms[y + 1], {x: op_mat}) / norms[y]
        E = E + c * v.real
    for i, j, J in terms.pair:
        (x1, y1), (x2, y2) = i, j
        if y1 == y2:
            v = sandwich(tops[y1], state, y1, bottoms[y1 + 1], {x1: Z, x2: Z}) / norms[y1]
        elif dress == "top":
            # Dressed and undressed environments carry different detached
            # normalization scales; the ratio needs the constant log-norm offset
            # restored (it cancels algebraically, so gradients stay exact).
            rescale = np.exp(dressed[i][y2].log_norm - tops[y2].log_norm)
            v = rescale * sandwich(dressed[i][y2], state, y2, bottoms[y2 + 1], {x2: Z}) / norms[y2]
        else:
            rescale = np.exp(dressed[j][y1 + 1].log_norm - bottoms[y1 + 1].log_norm)
            v = rescale * sandwich(tops[y1], state, y1, dressed[j][y1 + 1], {x1: Z}) / norms[y1]
        E = E + J * v.real
    return E, row_consistency


def energy_differentiable(
    state: PEPSState, terms: HamiltonianTerms, chi: int, backend: TruncationBackend
) -> torch.Tensor:
    """The AD-graph energy (top-dressed direction). LBFGS closes over this."""
    E, _ = _assemble(state, terms, chi, backend, want_grad=True, dress="top")
    return E


@dataclass(frozen=True)
class EnvCertificate:
    chi: int
    max_disc_weight: float
    updown_gap: float
    row_consistency: float
    fallback_count: int


_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class EnergyReport:
    """Constructible ONLY via `energy_certified` (INV gate by construction)."""

    e_total: float
    e_per_site: float
    env: EnvCertificate
    tail_bound: float
    chi_stability: tuple[float, float] | None
    grad_norm: float
    n_iters: int
    wall_s: float
    certified: bool
    _token: object = None

    def __post_init__(self) -> None:
        if self._token is not _FACTORY_TOKEN:
            raise TypeError(
                "EnergyReport cannot be constructed directly; use "
                "peps.energy.energy_certified (INV-1)."
            )


def energy_certified(
    state: PEPSState,
    terms: HamiltonianTerms,
    chi: int,
    backend: TruncationBackend,
    *,
    eps_env: float,
    eps_env_E: float,
    tail_bound: float = 0.0,
    grad_norm: float = float("nan"),
    n_iters: int = 0,
    wall_s: float = 0.0,
) -> EnergyReport:
    """Runs the INV-1 gates and mints the report, or raises EnvironmentNotConverged."""
    env = build_env(state, chi, backend, want_grad=False)
    max_disc = max(env.disc_weights)
    if max_disc > eps_env:
        raise EnvironmentNotConverged(
            f"INV-1: max discarded weight {max_disc:.3e} > eps_env {eps_env:.1e} at chi={chi}"
        )
    with torch.no_grad():
        E_down, row_c = _assemble(state, terms, chi, backend, want_grad=False, dress="top")
        E_up, _ = _assemble(state, terms, chi, backend, want_grad=False, dress="bottom")
    gap = abs(float(E_down) - float(E_up))
    if gap > eps_env_E:
        raise EnvironmentNotConverged(
            f"INV-1: up/down energy gap {gap:.3e} > eps_env_E {eps_env_E:.1e} at chi={chi}"
        )
    N = state.L**2
    return EnergyReport(
        e_total=float(E_down),
        e_per_site=float(E_down) / N,
        env=EnvCertificate(
            chi=chi,
            max_disc_weight=max_disc,
            updown_gap=gap,
            row_consistency=row_c,
            fallback_count=0,
        ),
        tail_bound=tail_bound,
        chi_stability=None,
        grad_norm=grad_norm,
        n_iters=n_iters,
        wall_s=wall_s,
        certified=True,
        _token=_FACTORY_TOKEN,
    )

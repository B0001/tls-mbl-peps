"""LBFGS gradient-descent driver (ARCHITECTURE.md §10.2).

The optimizer sees `torch.view_as_real` leaves; complex tensors are reconstructed
inside the closure so the whole certified-energy graph stays complex128 (ADR-004).
On `EnvironmentNotConverged` the driver -- outside the closure -- escalates
chi += dchi and restarts warm (INV-1 retry, max `retry_max`).

Line-search trial rejection (found running the L=8 pilot, 2026-07-20): strong_wolfe
occasionally evaluates an oversized trial step whose PEPS tensors overflow the
double-layer contraction, producing non-finite entries that reach the canonicalization
SVD as a corrupted operand -- surfacing as a LAPACK convergence failure (or, with the
kernels/svd.py scipy fallback, a plain "contains inf/nan" error). This is an ordinary,
recoverable line-search exploration artifact, not a certification-time INV-7
violation: the closure catches it and returns a large finite loss with zero gradient
so strong_wolfe backs off to a smaller step, instead of the whole realization
crashing on what the optimizer should have just tried again with a smaller t.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from tlsmbl.core.types import HamiltonianTerms
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.peps.energy import (
    EnvironmentNotConverged,
    energy_certified,
    energy_differentiable,
)
from tlsmbl.peps.state import PEPSState

_REJECTED_STEP_LOSS = 1e30


@dataclass
class OptResult:
    state: PEPSState
    energy: float
    grad_norm: float
    n_iters: int
    chi: int
    converged: bool


def _leaves(state: PEPSState) -> list[torch.Tensor]:
    return [
        torch.view_as_real(state.tensors[y][x].detach().clone()).requires_grad_(True)
        for y in range(state.L)
        for x in range(state.L)
    ]


def _rebuild(state: PEPSState, leaves: list[torch.Tensor]) -> PEPSState:
    L = state.L
    tensors = [
        [torch.view_as_complex(leaves[y * L + x]) for x in range(L)] for y in range(L)
    ]
    return PEPSState(tensors=tensors, L=L, D=state.D, d=state.d)


def optimize_lbfgs(
    state: PEPSState,
    terms: HamiltonianTerms,
    chi: int,
    backend: TruncationBackend,
    *,
    max_outer: int = 400,
    inner_iters: int = 20,
    tol_E: float = 1e-8,
    tol_g_scale: float = 1e-6,
    history_size: int = 20,
    eps_env: float = 1e-8,
    eps_env_E: float = 1e-7,
    dchi: int | None = None,
    retry_max: int = 3,
) -> OptResult:
    N = state.L**2
    g_stop = tol_g_scale * (2 * N * state.D**4 * state.d) ** 0.5
    dchi = dchi if dchi is not None else state.D**2

    for attempt in range(retry_max + 1):
        try:
            # Arm the INV-1 gates once up front at this chi: refuse to optimize in
            # an uncertifiable environment rather than discover it at the end.
            energy_certified(
                state, terms, chi, backend, eps_env=eps_env, eps_env_E=eps_env_E
            )
            break
        except EnvironmentNotConverged:
            if attempt == retry_max:
                raise
            chi += dchi

    leaves = _leaves(state)
    opt = torch.optim.LBFGS(
        leaves,
        history_size=history_size,
        max_iter=inner_iters,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        try:
            E = energy_differentiable(_rebuild(state, leaves), terms, chi, backend)
            E.backward()  # type: ignore[no-untyped-call]
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"rejecting line-search trial step (non-finite state): {exc}",
                stacklevel=2,
            )
            for p in leaves:
                p.grad = torch.zeros_like(p)
            return torch.tensor(_REJECTED_STEP_LOSS, dtype=torch.float64)
        return E

    last = float("inf")
    last_good = float("inf")
    gnorm = float("inf")
    stable = 0
    n_iters = 0
    converged = False
    for _ in range(max_outer):
        loss = opt.step(closure)  # type: ignore[no-untyped-call]
        assert isinstance(loss, torch.Tensor)
        E = float(loss.detach())
        n_iters += inner_iters
        if E >= _REJECTED_STEP_LOSS / 10:
            # The line search exhausted every trial on rejected (non-finite)
            # steps and fell back to returning one of them as "accepted" --
            # possible under a burst of bad trials. Don't let this corrupt
            # convergence bookkeeping or ever become the reported energy.
            warnings.warn(
                "outer LBFGS iteration collapsed onto a rejected trial step; "
                "continuing without updating convergence state",
                stacklevel=2,
            )
            continue
        grads = [p.grad.flatten() for p in leaves if p.grad is not None]
        rel = abs(E - last) / max(abs(E), 1e-12)
        gnorm = float(torch.linalg.norm(torch.cat(grads)))
        stable = stable + 1 if rel < tol_E else 0
        last = last_good = E
        if stable >= 5 and gnorm < g_stop:
            converged = True
            break

    final = _rebuild(state, [p.detach() for p in leaves])
    final = PEPSState(
        tensors=[[t.clone() for t in row] for row in final.tensors],
        L=state.L,
        D=state.D,
        d=state.d,
    )
    return OptResult(
        state=final,
        energy=last_good,
        grad_norm=gnorm,
        n_iters=n_iters,
        chi=chi,
        converged=converged,
    )

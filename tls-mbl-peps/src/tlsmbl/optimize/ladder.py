"""D-ladder (ARCHITECTURE.md §10.3): optimize at increasing bond dimension, warm-
starting each rung by growing the previous state; finalize with the INV-2 audit and
a 1/D linear extrapolation of E(D)."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from tlsmbl.core.types import DisorderRealization, HamiltonianTerms, TensorSpec
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.optimize.init import product_init
from tlsmbl.optimize.lbfgs_driver import OptResult, optimize_lbfgs
from tlsmbl.peps.state import PEPSState


@dataclass
class LadderRung:
    D: int
    chi: int
    energy: float
    grad_norm: float
    n_iters: int
    wall_s: float
    converged: bool


@dataclass
class LadderResult:
    state: PEPSState
    rungs: list[LadderRung]
    extrapolated_E: float | None  # 1/D linear fit, None for a single rung
    fit_residual: float | None


def run_ladder(
    real: DisorderRealization,
    terms: HamiltonianTerms,
    ladder: list[int],
    spec: TensorSpec,
    seed_seq: np.random.SeedSequence,
    backend: TruncationBackend,
    *,
    chi_factor: int = 1,
    **lbfgs_kwargs: object,
) -> LadderResult:
    init_seq, *grow_seqs = seed_seq.spawn(len(ladder))
    state: PEPSState | None = None
    rungs: list[LadderRung] = []
    res: OptResult | None = None
    for rung_idx, D in enumerate(ladder):
        chi = chi_factor * D * D
        if state is None:
            state = product_init(real, D, spec, init_seq)
        else:
            state = state.grow(D, spec, grow_seqs[rung_idx - 1])
        t0 = perf_counter()
        res = optimize_lbfgs(state, terms, chi, backend, **lbfgs_kwargs)  # type: ignore[arg-type]
        state = res.state
        rungs.append(
            LadderRung(
                D=D,
                chi=res.chi,
                energy=res.energy,
                grad_norm=res.grad_norm,
                n_iters=res.n_iters,
                wall_s=perf_counter() - t0,
                converged=res.converged,
            )
        )
    assert state is not None and res is not None
    extrapolated: float | None = None
    residual: float | None = None
    if len(rungs) >= 2:
        xs = np.array([1.0 / r.D for r in rungs])
        ys = np.array([r.energy for r in rungs])
        coeffs, res_arr, *_ = np.polyfit(xs, ys, 1, full=True)
        extrapolated = float(coeffs[1])  # intercept at 1/D -> 0
        residual = float(res_arr[0]) if len(res_arr) else 0.0
    return LadderResult(
        state=state, rungs=rungs, extrapolated_E=extrapolated, fit_residual=residual
    )

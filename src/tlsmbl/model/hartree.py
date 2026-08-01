"""Hartree tail bound and outer self-consistency loop, INV-5 (ARCHITECTURE.md §3, §7.4).

v1 default is the pure-truncation baseline: h_mf = 0, K_max = 1, with the rigorous tail
bound reported regardless. The loop is behind `config.model.hartree.enabled` and is
implemented here (§16 P5's last exit item).

WHAT THE LOOP DOES (§7.4, verbatim):

    h_mf <- 0
    for outer in 1..K_max:
        converge PEPS (inner problem) -> measure m[x,y] = <sigma^z>
        h_new[i] = sum_{j: r_ij > R_c} (g_J c_ij / r^3) m[j]
        h_mf <- (1-alpha) h_mf + alpha h_new
        stop when max|h_mf - h_new| < tau_MF

`hartree_loop` owns only the field update. The inner PEPS solve arrives as a callable, so
this module never imports the peps/ or optimize/ layers -- the dependency would be
backwards (model/ is below them) and would make the tail physics untestable without a
full variational solve.

LAZY TAIL COUPLINGS (NR-5). `model/sampling.py` draws only 1 <= r <= R_c, so the r > R_c
couplings do not exist and must be generated here. Storing them is O(N^2) (L=16: ~32k
pairs), so they are generated on demand from a counter-based Philox stream keyed by the
realization's spawned tail seed, with the PAIR ITSELF as the counter:

    counter = s_a * N + s_b        (s = y*L + x, canonical s_a < s_b, N = L^2)

Because the counter is a pure function of the pair, the same coupling comes back no
matter what order pairs are requested in, how many times, or which outer iteration asks
-- which is what makes the loop reproducible without an O(N^2) table. A sequential
generator would not have this property: it would tie each value to its position in the
request order.

INV-5 UNDER THE LOOP -- the bound does NOT shrink. `tail_bound` is a rigorous bound on
what the truncation at R_c neglects. Running the loop does not delete that error, it
*moves* it: the tail's mean-field (Hartree) part is now treated, and what remains
neglected is the tail's CORRELATIONS, which the same 2*pi*g_J/R_c expression still
bounds (it bounds the full tail coupling, of which the correlated part is a piece).
Reporting a smaller bound because the loop ran would be claiming rigor the loop does not
provide -- the mean-field treatment is an approximation, not an exact resummation. So
`tail_bound` is untouched, and `HartreeResult.bound_covers` records which statement is
being made. Any future tightening needs its own ADR and its own proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable, Iterator

import numpy as np

# What the INV-5 bound is a bound ON, in each mode. Same magnitude either way.
COVERS_FULL_TAIL = "neglected r>R_c tail (h_mf = 0, pure-truncation baseline)"
COVERS_CORRELATIONS = "neglected r>R_c tail CORRELATIONS (mean-field part treated)"


def tail_bound(g_J: float, R_c: int) -> float:
    """Rigorous per-site bound on the neglected r > R_c dipolar tail (lattice form):
    delta_e_tail = g_J * sum_{r > R_c} r^-3 ~= 2*pi*g_J / R_c."""
    return 2.0 * math.pi * g_J / R_c


def tail_certified(g_J: float, R_c: int, e_per_site: float, tau_tail: float) -> bool:
    """INV-5 gate: the bound must be small relative to the reported energy, else the
    artifact is marked UNCERTIFIED."""
    return tail_bound(g_J, R_c) <= tau_tail * abs(e_per_site)


@dataclass(frozen=True)
class HartreeResult:
    """Outcome of the outer loop. `converged=False` is reported, never raised: the caller
    decides whether an unconverged mean field invalidates the artifact.

    !! `h_mf` IS NOT THE FIELD THE FINAL STATE WAS SOLVED IN. !!
    §7.4 damps *after* measuring, so the last thing this loop does is take one more
    damped step past the field it last called `solve` with. `h_mf` is that step-ahead
    field -- useful as the warm start for a continuation, WRONG to publish alongside a
    certified energy, because the state was optimized in the pre-damping field
    (measured difference at L=4: 4.5e-5, far above any tolerance the report quotes).
    A caller that certifies an artifact must persist the field it passed to `solve` --
    equivalently, the field handed to `before_iteration` for the final iteration, which
    is what `ensemble/orchestrate.py` stores. Pinned by
    tests/unit/test_hartree_loop.py::test_returned_field_is_one_damped_step_past_the_solved_one.
    """

    h_mf: np.ndarray  # (L, L) float64, indexed [y, x] -- step-ahead; see above
    n_iters: int
    converged: bool
    max_delta: float  # final max|h_mf - h_new|
    history: tuple[float, ...]  # max_delta per outer iteration
    tail_bound: float  # unchanged by the loop, by construction
    bound_covers: str


def _tail_pairs(L: int, R_c: int) -> Iterator[tuple[int, int, float]]:
    """Canonically ordered site pairs with r > R_c, as (s_a, s_b, r) with s_a < s_b.

    Mirrors `sampling.py::_qualifying_pairs`'s enumeration and canonical ordering; this
    is the complement of its 1 <= r <= R_c window, so every pair is handled by exactly
    one of the two and none is double counted.
    """
    for s_a in range(L * L):
        ya, xa = divmod(s_a, L)  # s = y*L + x (§6)
        for s_b in range(s_a + 1, L * L):
            yb, xb = divmod(s_b, L)
            r = math.hypot(xa - xb, ya - yb)
            if r > R_c:
                yield s_a, s_b, r


def tail_coupling(tail_seed: int, L: int, s_a: int, s_b: int, g_J: float, r: float) -> float:
    """One r > R_c coupling J_ij = g_J * c_ij / r^3, generated on demand.

    `c_ij ~ U(-1, 1)`, matching `sampling.py`'s draw for the retained bonds exactly -- the
    tail must come from the same distribution as the bonds it continues, or the loop
    measures an artifact of the sampler rather than the physics.

    Counter-based (Philox) keyed on the pair, so the value is independent of request
    order and of how many times it is asked for. `s_a < s_b` is required so that a pair
    has one counter, not two.
    """
    if not 0 <= s_a < s_b < L * L:
        raise ValueError(f"pair must be canonically ordered 0 <= s_a < s_b < {L * L}")
    bit_gen = np.random.Philox(key=tail_seed, counter=s_a * (L * L) + s_b)
    c = float(np.random.Generator(bit_gen).uniform(-1.0, 1.0))
    return g_J * c / r**3


def tail_field(m: np.ndarray, *, tail_seed: int, L: int, R_c: int, g_J: float) -> np.ndarray:
    """h_new[i] = sum_{j: r_ij > R_c} J_ij m[j], the §7.4 mean-field update.

    `m` is (L, L) indexed [y, x]. Couplings are generated lazily and never stored: the
    accumulation is O(N^2) work but O(N) memory.
    """
    if m.shape != (L, L):
        raise ValueError(f"m must have shape {(L, L)}, got {m.shape}")
    flat_m = m.reshape(-1)
    h = np.zeros(L * L, dtype=np.float64)
    for s_a, s_b, r in _tail_pairs(L, R_c):
        J = tail_coupling(tail_seed, L, s_a, s_b, g_J, r)
        h[s_a] += J * flat_m[s_b]  # symmetric bond: contributes to both endpoints
        h[s_b] += J * flat_m[s_a]
    return h.reshape(L, L)


def hartree_loop(
    solve: Callable[[np.ndarray], np.ndarray],
    *,
    L: int,
    R_c: int,
    g_J: float,
    tail_seed: int,
    K_max: int = 8,
    alpha: float = 0.5,
    tol: float = 1e-4,
    h_mf0: np.ndarray | None = None,
    start_iter: int = 1,
    history0: Sequence[float] = (),
    before_iteration: Callable[[int, np.ndarray, tuple[float, ...]], None] | None = None,
) -> HartreeResult:
    """§7.4's outer self-consistency loop.

    `solve(h_mf) -> m[y, x]` converges the inner PEPS problem in the given mean field and
    returns the measured <sigma^z>. It is a callable so this module stays below the
    tensor layers.

    `K_max = 1` with the returned field left at zero reproduces the v1 baseline exactly:
    the loop still calls `solve` once (the caller needs that state anyway) but the
    reported `h_mf` is whatever the single damped update produced -- pass `K_max=1` and
    ignore `h_mf`, or simply leave `hartree.enabled` false, to stay on the baseline.

    Damping (`alpha`, default 0.5) is what makes this converge rather than oscillate: the
    map m -> h -> m is sign-flipping for antiferromagnetic tail couplings, so alpha = 1
    can two-cycle indefinitely. Non-convergence within K_max is REPORTED, not raised.

    RESUMPTION. `h_mf0` / `start_iter` / `history0` restart a loop that was interrupted,
    and `before_iteration(n, h_mf)` is called at the top of each iteration *before*
    `solve`. `ensemble/orchestrate.py` uses the hook to checkpoint the field and clear the
    previous iteration's ladder, which is what lets the outer loop survive a kill without
    this module knowing anything about zarr. Keeping resumption here rather than
    reimplementing the recurrence in the orchestrator means §7.4 has exactly one
    implementation.
    """
    if K_max < 1:
        raise ValueError(f"K_max must be >= 1 (got {K_max})")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1] (got {alpha})")
    if start_iter < 1:
        raise ValueError(f"start_iter must be >= 1 (got {start_iter})")
    h_mf = (
        np.zeros((L, L), dtype=np.float64)
        if h_mf0 is None
        else np.array(h_mf0, dtype=np.float64)
    )
    if h_mf.shape != (L, L):
        raise ValueError(f"h_mf0 must have shape {(L, L)}, got {h_mf.shape}")
    history: list[float] = list(history0)
    converged = False
    max_delta = float("inf")
    n = start_iter - 1
    for n in range(start_iter, K_max + 1):
        if before_iteration is not None:
            # (iteration, field about to be solved in, history so far) -- everything a
            # checkpoint needs to resume this exact iteration.
            before_iteration(n, h_mf, tuple(history))
        m = np.asarray(solve(h_mf), dtype=np.float64)
        h_new = tail_field(m, tail_seed=tail_seed, L=L, R_c=R_c, g_J=g_J)
        # §7.4's stopping test compares the CURRENT field to the proposed one, i.e. it
        # measures how far from self-consistency we still are, before damping.
        max_delta = float(np.max(np.abs(h_mf - h_new))) if h_new.size else 0.0
        history.append(max_delta)
        h_mf = (1.0 - alpha) * h_mf + alpha * h_new
        if max_delta < tol:
            converged = True
            break
    return HartreeResult(
        h_mf=h_mf,
        n_iters=n,
        converged=converged,
        max_delta=max_delta,
        history=tuple(history),
        # Unchanged by construction: the loop moves the neglected error, it does not
        # shrink it. See the module docstring.
        tail_bound=tail_bound(g_J, R_c),
        bound_covers=COVERS_CORRELATIONS,
    )

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
import torch.utils.checkpoint as checkpoint

from tlsmbl.core.guards import finite
from tlsmbl.core.types import HamiltonianTerms, Site
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.peps.boundary import (
    BoundaryMPS,
    build_bottoms,
    build_env,
    build_tops,
    extend_bottom_batched,
    extend_top_batched,
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


class RowSplice:
    """§8.4 prefix/suffix transfer caching for one row sandwich <T| row_y |B>.

    Column prefix transfers F->_x and suffixes F<-_x are built once (O(L) E-4
    steps each); any single insertion then costs one spliced contraction and a
    same-row pair costs only the explicit segment between the two columns
    (bounded by R_c). Transfer legs are (top_bond, row_horizontal, bottom_bond)
    at the left boundary of column x."""

    def __init__(self, T: BoundaryMPS, state: PEPSState, y: int, B: BoundaryMPS) -> None:
        self.T, self.state, self.y, self.B = T, state, y, B
        L = state.L
        dtype = state.tensors[0][0].dtype
        one = torch.ones(1, 1, 1, dtype=dtype)
        self.prefix: list[torch.Tensor] = [one]
        for x in range(L):
            self.prefix.append(self._absorb_left(self.prefix[x], x, None))
        self.suffix: list[torch.Tensor] = [one] * (L + 1)
        for x in range(L - 1, -1, -1):
            s = torch.einsum("TnB,tvT->nBtv", self.suffix[x + 1], T.tensors[x])
            a = double_layer(state.tensors[y][x])
            s = torch.einsum("nBtv,mvnw->Btmw", s, a)
            self.suffix[x] = torch.einsum("Btmw,bwB->tmb", s, B.tensors[x])

    def _absorb_left(
        self, F: torch.Tensor, x: int, op: torch.Tensor | None
    ) -> torch.Tensor:
        a = double_layer(self.state.tensors[self.y][x], op)
        F = torch.einsum("tmb,tvT->mbvT", F, self.T.tensors[x])
        F = torch.einsum("mbvT,mvnw->bTnw", F, a)
        return torch.einsum("bTnw,bwB->TnB", F, self.B.tensors[x])

    @property
    def norm(self) -> torch.Tensor:
        return self.prefix[-1].reshape(())

    def insert(self, ops: dict[int, torch.Tensor]) -> torch.Tensor:
        """<T| row with insertions |B>: prefix at the first insertion column,
        explicit E-4 steps across the spanned segment, suffix after the last."""
        xs = sorted(ops)
        x1, x2 = xs[0], xs[-1]
        F = self.prefix[x1]
        for x in range(x1, x2 + 1):
            F = self._absorb_left(F, x, ops.get(x))
        return torch.einsum("tmb,tmb->", F, self.suffix[x2 + 1])


def _assemble(
    state: PEPSState,
    terms: HamiltonianTerms,
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
    dress: str,  # "top" | "bottom"
    factored: bool = False,
) -> tuple[torch.Tensor, float]:
    """Differentiable energy; returns (E, row_consistency). Cross-row pairs dressed
    per `dress` direction (the two directions form INV-1's up/down certificate)."""
    L = state.L
    tops, _ = build_tops(state, chi, backend, want_grad=want_grad, factored=factored)
    bottoms, _ = build_bottoms(
        state, chi, backend, want_grad=want_grad, factored=factored
    )
    rows = [RowSplice(tops[y], state, y, bottoms[y + 1]) for y in range(L)]
    norms = [rows[y].norm for y in range(L)]
    with torch.no_grad():
        scaled = [
            float(n.real) * np.exp(tops[y].log_norm + bottoms[y + 1].log_norm)
            for y, n in enumerate(norms)
        ]
        row_consistency = max(abs(s / scaled[0] - 1.0) for s in scaled)

    # §8.4: one dressed environment per source site serves all its partners,
    # EXTENDED from the cached undressed environment at the source row rather
    # than rebuilt from the lattice edge (<= R_c rows of new compressions each).
    # Sources sharing a row are batched into one LAPACK-batched compression
    # (exact SVD -- reference backend, always certifiable -- regardless of the
    # sketched setting; batching beats per-call sketching at D <= 4 sizes).
    #
    # Built and consumed ONE ROW-GROUP AT A TIME, not precomputed into a dict
    # spanning the whole lattice: at L=16 with O(1e3) cross-row pairs, holding
    # every source's dressed chain alive for the whole assembly is what blew up
    # peak memory past what an 8-worker/32GB cloud VM could handle (found
    # launching the L=16 production run, 2026-07-23 -- measured 3.4 GB for a
    # single forward+backward vs 365 MB with cross-row pairs removed). Grouping
    # by row and consuming each group's contribution immediately lets Python
    # free it before the next group is built.
    cross_pairs: dict[Site, list[tuple[Site, Site, float]]] = {}
    same_row_pairs: list[tuple[Site, Site, float]] = []
    for i, j, J in terms.pair:
        if i[1] == j[1]:
            same_row_pairs.append((i, j, J))
        else:
            src = i if dress == "top" else j
            cross_pairs.setdefault(src, []).append((i, j, J))
    by_row: dict[int, list[Site]] = {}
    for s in cross_pairs:
        by_row.setdefault(s[1], []).append(s)

    E = torch.zeros((), dtype=torch.float64)
    for (x, y), op, c in terms.onsite:
        op_mat = Z if op == "z" else X
        v = rows[y].insert({x: op_mat}) / norms[y]
        E = E + c * v.real
    for i, j, J in same_row_pairs:
        x1, y1 = i
        x2, _ = j
        v = rows[y1].insert({x1: Z, x2: Z}) / norms[y1]
        E = E + J * v.real

    def _row_group_energy(y_src: int, sources: tuple[Site, ...]) -> torch.Tensor:
        """One row-group's ENTIRE contribution: build its batched dressed
        environment AND consume it through every partner's sandwich() call.
        This whole unit -- not just the internal compression step -- is what
        gets checkpointed below: at L=16 the dominant memory cost turned out
        to be the O(1e3) per-pair sandwich() activations downstream of the
        dressed environment, not the environment's own construction (measured
        2026-07-23: 50 cross-row pairs alone added ~37 MB over baseline,
        extrapolating to ~1.7 GB+ for the full 2354-pair L=16 lattice --
        checkpointing only extend_*_batched's internal compress step left
        this essentially untouched, 3398 -> 3067 MB)."""
        xs = [s[0] for s in sources]
        if dress == "top":
            extent = max(max(j[1] for _, j, _ in cross_pairs[s]) for s in sources)
            batch = extend_top_batched(
                state, tops[y_src], y_src, extent, chi, backend, xs, Z,
                want_grad=want_grad, factored=factored,
            )
        else:
            extent = min(min(i[1] for i, _, _ in cross_pairs[s]) + 1 for s in sources)
            batch = extend_bottom_batched(
                state, bottoms[y_src + 1], y_src, extent, chi, backend, xs, Z,
                want_grad=want_grad, factored=factored,
            )
        group_E = torch.zeros((), dtype=torch.float64)
        for b, s in enumerate(sources):
            dressed_s = {lvl: env.element(b) for lvl, env in batch.items()}
            # Group this source's partners by their OWN row: a raw sandwich()
            # per partner does a full O(L) contraction from scratch every
            # time, so a source with several partners on the same row paid
            # for that row's chain repeatedly -- the actual dominant cost at
            # L=16 (measured: 50 cross-row pairs alone added ~37 MB / a
            # proportional time cost over baseline). RowSplice amortizes the
            # O(L) prefix/suffix build once per (source, partner-row) and
            # each additional same-row partner costs O(1).
            by_partner_row: dict[int, list[tuple[Site, Site, float]]] = {}
            for i, j, J in cross_pairs[s]:
                key_row = j[1] if dress == "top" else i[1]
                by_partner_row.setdefault(key_row, []).append((i, j, J))
            for y2, pairs_here in by_partner_row.items():
                if dress == "top":
                    # Dressed and undressed environments carry different
                    # detached normalization scales; the ratio needs the
                    # constant log-norm offset restored (cancels
                    # algebraically, so gradients stay exact).
                    rescale = np.exp(dressed_s[y2].log_norm - tops[y2].log_norm)
                    splice = RowSplice(dressed_s[y2], state, y2, bottoms[y2 + 1])
                    for _, j, J in pairs_here:
                        v = rescale * splice.insert({j[0]: Z}) / norms[y2]
                        group_E = group_E + J * v.real
                else:
                    y1 = y2
                    rescale = np.exp(dressed_s[y1 + 1].log_norm - bottoms[y1 + 1].log_norm)
                    splice = RowSplice(tops[y1], state, y1, dressed_s[y1 + 1])
                    for i, _, J in pairs_here:
                        v = rescale * splice.insert({i[0]: Z}) / norms[y1]
                        group_E = group_E + J * v.real
        return group_E

    for y_src, sources in sorted(by_row.items()):
        sources_t = tuple(sorted(sources))
        if want_grad:
            E = E + checkpoint.checkpoint(
                _row_group_energy, y_src, sources_t, use_reentrant=False
            )
        else:
            E = E + _row_group_energy(y_src, sources_t)
    return E, row_consistency


def energy_differentiable(
    state: PEPSState,
    terms: HamiltonianTerms,
    chi: int,
    backend: TruncationBackend,
    *,
    factored: bool = False,
) -> torch.Tensor:
    """The AD-graph energy (top-dressed direction). LBFGS closes over this."""
    E, _ = _assemble(
        state, terms, chi, backend, want_grad=True, dress="top", factored=factored
    )
    return E


@dataclass(frozen=True)
class EnvCertificate:
    """Every field describes THIS environment except `fallback_count`/`sketch_stats`
    -- read their notes before quoting them."""

    chi: int
    max_disc_weight: float
    updown_gap: float
    row_consistency: float
    # NOT a count for this environment alone: it is the backend's REALIZATION-CUMULATIVE
    # total at the moment the report was minted, so it includes every truncation the
    # D-ladder performed on the way here, not just the ones in the certified contraction.
    # That is the useful scope for the §11 audit (INV-3's disable is per realization, and
    # a rate over a single environment is too small a sample to mean anything), but it
    # would be wrong to read it as "this energy required N fallbacks". Per-compression
    # counts do exist -- `kernels.zipup.CompressStats.fallback_count` is a true delta.
    fallback_count: int
    # INV-3 audit (§11: REPORT.md echoes fallback rates). Same realization-cumulative
    # scope as above. None for the exact backend, which has no sketch to fall back from.
    sketch_stats: dict[str, float | int | bool | None] | None = None


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
    factored: bool = False,
) -> EnergyReport:
    """Runs the INV-1 gates and mints the report, or raises EnvironmentNotConverged."""
    env = build_env(state, chi, backend, want_grad=False, factored=factored)
    max_disc = max(env.disc_weights)
    if max_disc > eps_env:
        raise EnvironmentNotConverged(
            f"INV-1: max discarded weight {max_disc:.3e} > eps_env {eps_env:.1e} at chi={chi}"
        )
    with torch.no_grad():
        E_down, row_c = _assemble(
            state, terms, chi, backend, want_grad=False, dress="top",
            factored=factored,
        )
        E_up, _ = _assemble(
            state, terms, chi, backend, want_grad=False, dress="bottom",
            factored=factored,
        )
    gap = abs(float(E_down) - float(E_up))
    if gap > eps_env_E:
        raise EnvironmentNotConverged(
            f"INV-1: up/down energy gap {gap:.3e} > eps_env_E {eps_env_E:.1e} at chi={chi}"
        )
    N = state.L**2
    # INV-3 audit: read the backend's own cumulative counters rather than reporting a
    # placeholder. Cumulative is the right scope -- the backend instance is built once
    # per realization from its spawned sketch stream (orchestrate.py::_backend), so
    # these counts describe exactly the realization being certified.
    stats_fn = getattr(backend, "stats", None)
    sketch_stats = stats_fn() if callable(stats_fn) else None
    return EnergyReport(
        e_total=float(E_down),
        e_per_site=float(E_down) / N,
        env=EnvCertificate(
            chi=chi,
            max_disc_weight=max_disc,
            updown_gap=gap,
            row_consistency=row_c,
            fallback_count=int(getattr(backend, "fallback_count", 0)),
            sketch_stats=sketch_stats,
        ),
        tail_bound=tail_bound,
        chi_stability=None,
        grad_norm=grad_norm,
        n_iters=n_iters,
        wall_s=wall_s,
        certified=True,
        _token=_FACTORY_TOKEN,
    )

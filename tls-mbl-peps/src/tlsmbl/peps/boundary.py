"""Boundary-MPS environment construction (ARCHITECTURE.md §8.2; einsum E-3).

Top MPS absorbs rows downward, bottom MPS absorbs rows upward (mirror of E-3);
both orientations are built explicitly (INV-1's up/down certificate and §8.4
dressed-environment caching need both).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint as checkpoint

from tlsmbl.core.types import Site
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.kernels.zipup import compress
from tlsmbl.peps.doublelayer import double_layer
from tlsmbl.peps.state import PEPSState


@dataclass
class BoundaryMPS:
    tensors: list[torch.Tensor]  # M[l, v, r]
    log_norm: float  # detached; add when converting sandwiches to absolute scale


def trivial_mps(L: int, dtype: torch.dtype) -> BoundaryMPS:
    return BoundaryMPS([torch.ones(1, 1, 1, dtype=dtype) for _ in range(L)], 0.0)


def absorb_row_top(M: BoundaryMPS, row: list[torch.Tensor]) -> list[torch.Tensor]:
    """E-3: top boundary MPS meets the row below it; new physical leg = row's down."""
    out = []
    for m, a in zip(M.tensors, row):
        t = torch.einsum("lvr,avbw->lawrb", m, a)
        out.append(t.reshape(m.shape[0] * a.shape[0], a.shape[3], m.shape[2] * a.shape[2]))
    return out


def absorb_row_bottom(M: BoundaryMPS, row: list[torch.Tensor]) -> list[torch.Tensor]:
    """Mirror of E-3: bottom MPS (physical leg up) absorbs the row above it."""
    out = []
    for m, a in zip(M.tensors, row):
        t = torch.einsum("lwr,avbw->lavrb", m, a)
        out.append(t.reshape(m.shape[0] * a.shape[0], a.shape[1], m.shape[2] * a.shape[2]))
    return out


@dataclass
class EnvBundle:
    """tops[y] = rows 0..y-1 absorbed (tops[0] trivial); bottoms[y] = rows y..L-1
    absorbed (bottoms[L] trivial). disc_weights: every compression's discarded
    weights, INV-1's input."""

    tops: list[BoundaryMPS]
    bottoms: list[BoundaryMPS]
    chi: int
    disc_weights: list[float]


def _row(
    state: PEPSState, y: int, insert: dict[Site, torch.Tensor] | None
) -> list[torch.Tensor]:
    return [
        double_layer(state.tensors[y][x], (insert or {}).get((x, y)))
        for x in range(state.L)
    ]


def build_tops(
    state: PEPSState,
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
    insert: dict[Site, torch.Tensor] | None = None,
) -> tuple[list[BoundaryMPS], list[float]]:
    L = state.L
    dtype = state.tensors[0][0].dtype
    tops = [trivial_mps(L, dtype)]
    discs: list[float] = []
    for y in range(L):
        fat = absorb_row_top(tops[-1], _row(state, y, insert))
        t, stats = compress(fat, chi, backend, want_grad=want_grad)
        tops.append(BoundaryMPS(t, tops[-1].log_norm + stats.log_norm))
        discs.extend(stats.disc_weights)
    return tops, discs


def build_bottoms(
    state: PEPSState,
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
    insert: dict[Site, torch.Tensor] | None = None,
) -> tuple[list[BoundaryMPS], list[float]]:
    L = state.L
    dtype = state.tensors[0][0].dtype
    reversed_bottoms = [trivial_mps(L, dtype)]  # index r holds rows (L-r)..L-1
    discs: list[float] = []
    for y in range(L - 1, -1, -1):
        fat = absorb_row_bottom(reversed_bottoms[-1], _row(state, y, insert))
        b, stats = compress(fat, chi, backend, want_grad=want_grad)
        reversed_bottoms.append(BoundaryMPS(b, reversed_bottoms[-1].log_norm + stats.log_norm))
        discs.extend(stats.disc_weights)
    return list(reversed(reversed_bottoms)), discs


def build_env(
    state: PEPSState, chi: int, backend: TruncationBackend, *, want_grad: bool
) -> EnvBundle:
    tops, d1 = build_tops(state, chi, backend, want_grad=want_grad)
    bottoms, d2 = build_bottoms(state, chi, backend, want_grad=want_grad)
    return EnvBundle(tops=tops, bottoms=bottoms, chi=chi, disc_weights=d1 + d2)


def extend_top(
    state: PEPSState,
    base: BoundaryMPS,
    y_from: int,
    y_to: int,
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
    insert: dict[Site, torch.Tensor] | None = None,
) -> dict[int, BoundaryMPS]:
    """§8.4 incremental dressing: absorb rows y_from..y_to-1 into `base`
    (= an existing tops[y_from]), returning the environment at each level
    y_from+1..y_to. A dressed environment differs from the undressed one only
    from the operator's row onward, so rows above are reused, never rebuilt --
    per source this is <= R_c rows of work instead of L."""
    out: dict[int, BoundaryMPS] = {}
    cur = base
    for y in range(y_from, y_to):
        fat = absorb_row_top(cur, _row(state, y, insert))
        t, stats = compress(fat, chi, backend, want_grad=want_grad)
        cur = BoundaryMPS(t, cur.log_norm + stats.log_norm)
        out[y + 1] = cur
    return out


def extend_bottom(
    state: PEPSState,
    base: BoundaryMPS,
    y_from: int,
    y_to: int,
    chi: int,
    backend: TruncationBackend,
    *,
    want_grad: bool,
    insert: dict[Site, torch.Tensor] | None = None,
) -> dict[int, BoundaryMPS]:
    """Mirror of extend_top: absorb rows y_from down to y_to into `base`
    (= an existing bottoms[y_from + 1]), returning bottoms-style environments
    at levels y_from..y_to (level y covers rows y..L-1)."""
    out: dict[int, BoundaryMPS] = {}
    cur = base
    for y in range(y_from, y_to - 1, -1):
        fat = absorb_row_bottom(cur, _row(state, y, insert))
        b, stats = compress(fat, chi, backend, want_grad=want_grad)
        cur = BoundaryMPS(b, cur.log_norm + stats.log_norm)
        out[y] = cur
    return out


@dataclass
class BatchedBoundaryMPS:
    """B independent boundary MPSs sharing shapes: tensors (B, l, v, r)."""

    tensors: list[torch.Tensor]
    log_norm: torch.Tensor  # (B,) detached

    def element(self, b: int) -> BoundaryMPS:
        return BoundaryMPS(
            [t[b] for t in self.tensors], float(self.log_norm[b])
        )


def _batched_row_step(
    state: PEPSState,
    y: int,
    y_from: int,
    xs_t: torch.Tensor,
    op: torch.Tensor,
    chi: int,
    backend: TruncationBackend,
    want_grad: bool,
    orientation: str,  # "top" | "bottom"
    B: int,
    cur_tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """One row of §8.4 batched dressing: absorb + compress_batched. This whole
    step is the unit checkpointed by extend_{top,bottom}_batched (ARCHITECTURE.md
    §8.5 memory control, applied to the batched dressed-environment path):
    recomputed in backward instead of keeping its activations live for the
    whole energy assembly. compress_batched always resolves to ExactSVD, so
    recompute is bit-identical with no RNG-determinism concerns.
    """
    from tlsmbl.kernels.zipup import compress_batched

    L = state.L
    fat: list[torch.Tensor] = []
    for x in range(L):
        plain = double_layer(state.tensors[y][x])
        m = cur_tensors[x]
        if orientation == "top":
            t_plain = torch.einsum("Blvr,avbw->Blawrb", m, plain)
        else:
            t_plain = torch.einsum("Blwr,avbw->Blavrb", m, plain)
        if y == y_from and bool((xs_t == x).any()):
            dressed = double_layer(state.tensors[y][x], op)
            if orientation == "top":
                t_dressed = torch.einsum("Blvr,avbw->Blawrb", m, dressed)
            else:
                t_dressed = torch.einsum("Blwr,avbw->Blavrb", m, dressed)
            mask = (xs_t == x).view(B, 1, 1, 1, 1, 1)
            t = torch.where(mask, t_dressed, t_plain)
        else:
            t = t_plain
        mid = plain.shape[3] if orientation == "top" else plain.shape[1]
        fat.append(
            t.reshape(B, m.shape[1] * plain.shape[0], mid, m.shape[3] * plain.shape[2])
        )
    comp, stats = compress_batched(fat, chi, backend, want_grad=want_grad)
    return (*comp, stats.log_norm)


def extend_top_batched(
    state: PEPSState,
    base: BoundaryMPS,
    y_from: int,
    y_to: int,
    chi: int,
    backend: TruncationBackend,
    xs: list[int],
    op: torch.Tensor,
    *,
    want_grad: bool,
) -> dict[int, BatchedBoundaryMPS]:
    """Batched §8.4 dressing for all sources in row y_from at columns `xs`,
    sharing the undressed base: one batch element per source, absorbed and
    compressed together (LAPACK-batched SVDs). Only the source row differs
    between elements -- and only at each element's own column -- so absorption
    uses the shared plain double layer everywhere and swaps in the single
    shared dressed tensor on the matching (element, column) slots."""
    B = len(xs)
    xs_t = torch.tensor(xs)
    cur_tensors = [m.unsqueeze(0).expand(B, *m.shape) for m in base.tensors]
    cur_log = torch.full((B,), base.log_norm, dtype=torch.float64)
    out: dict[int, BatchedBoundaryMPS] = {}
    for y in range(y_from, y_to):
        args = (state, y, y_from, xs_t, op, chi, backend, want_grad, "top", B)
        if want_grad:
            results = checkpoint.checkpoint(
                _batched_row_step, *args, cur_tensors, use_reentrant=False
            )
        else:
            results = _batched_row_step(*args, cur_tensors)
        cur_tensors = list(results[:-1])
        cur_log = cur_log + results[-1]
        out[y + 1] = BatchedBoundaryMPS(tensors=cur_tensors, log_norm=cur_log)
    return out


def extend_bottom_batched(
    state: PEPSState,
    base: BoundaryMPS,
    y_from: int,
    y_to: int,
    chi: int,
    backend: TruncationBackend,
    xs: list[int],
    op: torch.Tensor,
    *,
    want_grad: bool,
) -> dict[int, BatchedBoundaryMPS]:
    """Mirror of extend_top_batched: absorb rows y_from down to y_to (level y
    covers rows y..L-1) for all sources in row y_from at columns `xs`."""
    B = len(xs)
    xs_t = torch.tensor(xs)
    cur_tensors = [m.unsqueeze(0).expand(B, *m.shape) for m in base.tensors]
    cur_log = torch.full((B,), base.log_norm, dtype=torch.float64)
    out: dict[int, BatchedBoundaryMPS] = {}
    for y in range(y_from, y_to - 1, -1):
        args = (state, y, y_from, xs_t, op, chi, backend, want_grad, "bottom", B)
        if want_grad:
            results = checkpoint.checkpoint(
                _batched_row_step, *args, cur_tensors, use_reentrant=False
            )
        else:
            results = _batched_row_step(*args, cur_tensors)
        cur_tensors = list(results[:-1])
        cur_log = cur_log + results[-1]
        out[y] = BatchedBoundaryMPS(tensors=cur_tensors, log_norm=cur_log)
    return out

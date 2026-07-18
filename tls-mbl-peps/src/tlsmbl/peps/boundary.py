"""Boundary-MPS environment construction (ARCHITECTURE.md §8.2; einsum E-3).

Top MPS absorbs rows downward, bottom MPS absorbs rows upward (mirror of E-3);
both orientations are built explicitly (INV-1's up/down certificate and §8.4
dressed-environment caching need both).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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

"""PEPS state (ARCHITECTURE.md §8.1).

Site tensor layout `A[p, l, u, r, d]` (§6): physical, left, up, right, down; boundary
legs have dimension 1, never omitted. Tensors are stored row-major as
`tensors[y][x]` (y = row, x = column) matching every row-sweep loop in the codebase;
`L` is an explicit parameter everywhere (never a module global -- CLAUDE.md gotcha).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tlsmbl.core.types import TensorSpec


def _leg_dims(x: int, y: int, L: int, D: int) -> tuple[int, int, int, int]:
    """(Dl, Du, Dr, Dd) for site (x, y): interior legs D, boundary legs 1."""
    return (
        1 if x == 0 else D,
        1 if y == 0 else D,
        1 if x == L - 1 else D,
        1 if y == L - 1 else D,
    )


def _crandn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    out: np.ndarray = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)
    return out


@dataclass
class PEPSState:
    tensors: list[list[torch.Tensor]]  # [y][x], shape (d, Dl, Du, Dr, Dd)
    L: int
    D: int
    d: int = 2

    def leg_dims(self, x: int, y: int) -> tuple[int, int, int, int]:
        s = self.tensors[y][x].shape
        return (s[1], s[2], s[3], s[4])

    @classmethod
    def random(
        cls, L: int, D: int, spec: TensorSpec, seed_seq: np.random.SeedSequence, d: int = 2
    ) -> PEPSState:
        """i.i.d. CN(0, 1/sqrt(D^3 d)) entries; one spawned stream per site so the
        state is reproducible under INV-6 and matches the prototype convention."""
        spawns = seed_seq.spawn(L * L)
        tensors: list[list[torch.Tensor]] = []
        for y in range(L):
            row = []
            for x in range(L):
                Dl, Du, Dr, Dd = _leg_dims(x, y, L, D)
                rng = np.random.default_rng(spawns[y * L + x])
                a = _crandn(rng, d, Dl, Du, Dr, Dd) / np.sqrt(D**3 * d)
                row.append(torch.from_numpy(a).to(spec.dtype))
            tensors.append(row)
        return cls(tensors=tensors, L=L, D=D, d=d)

    @classmethod
    def from_product(
        cls,
        vectors: list[list[np.ndarray]],  # [y][x] local d-vectors
        D: int,
        spec: TensorSpec,
        seed_seq: np.random.SeedSequence,
        noise: float = 1e-2,
        d: int = 2,
    ) -> PEPSState:
        """D-padded product state: A[p, 0, 0, 0, 0] = v_p plus `noise`-scale CN noise
        everywhere to break gauge degeneracy (§8.1/§10.1)."""
        L = len(vectors)
        spawns = seed_seq.spawn(L * L)
        tensors: list[list[torch.Tensor]] = []
        for y in range(L):
            row = []
            for x in range(L):
                Dl, Du, Dr, Dd = _leg_dims(x, y, L, D)
                rng = np.random.default_rng(spawns[y * L + x])
                a = noise * _crandn(rng, d, Dl, Du, Dr, Dd)
                a[:, 0, 0, 0, 0] += vectors[y][x]
                row.append(torch.from_numpy(a).to(spec.dtype))
            tensors.append(row)
        return cls(tensors=tensors, L=L, D=D, d=d)

    def grow(
        self, D_new: int, spec: TensorSpec, seed_seq: np.random.SeedSequence, noise: float = 1e-3
    ) -> PEPSState:
        """D-ladder operator (§10.3): zero-pad interior legs to D_new, then add
        `noise`-scale CN noise on the new slices only."""
        if D_new < self.D:
            raise ValueError(f"grow: D_new={D_new} < current D={self.D}")
        L = self.L
        spawns = seed_seq.spawn(L * L)
        tensors: list[list[torch.Tensor]] = []
        for y in range(L):
            row = []
            for x in range(L):
                old = self.tensors[y][x]
                Dl, Du, Dr, Dd = _leg_dims(x, y, L, D_new)
                rng = np.random.default_rng(spawns[y * L + x])
                fresh = noise * _crandn(rng, self.d, Dl, Du, Dr, Dd)
                pad = torch.from_numpy(fresh).to(old.dtype)
                s = old.shape
                pad[:, : s[1], : s[2], : s[3], : s[4]] = old
                row.append(pad)
            tensors.append(row)
        return PEPSState(tensors=tensors, L=L, D=D_new, d=self.d)

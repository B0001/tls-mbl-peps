"""Tier-1 observables (ARCHITECTURE.md §12): rigorous functionals of the certified
variational state, measured in the physical frame via the SDRG pushforward.

Pair observables are assembled generically: each physical sigma^z pushes forward to
a weighted sum of one-site operators; a pair expectation is the double sum of
sandwich insertions, with same-site components merged by operator product (the
2x2 matrix product -- possibly non-Hermitian -- is a legal E-2 insertion)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from tlsmbl.core.types import DisorderRealization, Site
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.peps.boundary import BoundaryMPS, build_bottoms, build_tops
from tlsmbl.peps.energy import X, Z, sandwich
from tlsmbl.peps.state import PEPSState
from tlsmbl.sdrg.circuit import SDRGCircuit

_OPS = {"z": Z, "x": X}


@dataclass
class StaticObservables:
    sz: dict[Site, float]  # physical frame
    sx: dict[Site, float]
    q_ea: float
    czz_r: dict[int, float]  # r-binned mean connected <zz>, physical frame
    n_res_r: dict[int, float]  # resonance census per r bin (disorder property)


class _Measurer:
    """Caches undressed environments and dressed tops per (site, op)."""

    def __init__(
        self,
        state: PEPSState,
        chi: int,
        backend: TruncationBackend,
        factored: bool = False,
    ) -> None:
        self.state = state
        self.chi = chi
        self.backend = backend
        self.factored = factored
        self.tops, _ = build_tops(
            state, chi, backend, want_grad=False, factored=factored
        )
        self.bottoms, _ = build_bottoms(
            state, chi, backend, want_grad=False, factored=factored
        )
        self.norms = [
            sandwich(self.tops[y], state, y, self.bottoms[y + 1])
            for y in range(state.L)
        ]
        self._dressed: dict[tuple[Site, str], list[BoundaryMPS]] = {}

    def one(self, site: Site, kind: str) -> complex:
        x, y = site
        v = sandwich(
            self.tops[y], self.state, y, self.bottoms[y + 1], {x: _OPS[kind]}
        ) / self.norms[y]
        return complex(v)

    def _one_mat(self, site: Site, M: torch.Tensor) -> complex:
        x, y = site
        v = sandwich(self.tops[y], self.state, y, self.bottoms[y + 1], {x: M}) / self.norms[y]
        return complex(v)

    def _dressed_tops(self, site: Site, kind: str) -> list[BoundaryMPS]:
        key = (site, kind)
        if key not in self._dressed:
            tops, _ = build_tops(
                self.state, self.chi, self.backend, want_grad=False,
                insert={site: _OPS[kind]}, factored=self.factored,
            )
            self._dressed[key] = tops
        return self._dressed[key]

    def pair(self, a: Site, ka: str, b: Site, kb: str) -> complex:
        """<O_a O_b> with generic one-site ops; same-site merges by matrix product."""
        if a == b:
            return self._one_mat(a, _OPS[ka] @ _OPS[kb])
        (x1, y1), (x2, y2) = a, b
        if y1 == y2:
            v = sandwich(
                self.tops[y1], self.state, y1, self.bottoms[y1 + 1],
                {x1: _OPS[ka], x2: _OPS[kb]},
            ) / self.norms[y1]
            return complex(v)
        if y1 > y2:
            return self.pair(b, kb, a, ka)
        dt = self._dressed_tops(a, ka)
        rescale = math.exp(dt[y2].log_norm - self.tops[y2].log_norm)
        v = rescale * sandwich(
            dt[y2], self.state, y2, self.bottoms[y2 + 1], {x2: _OPS[kb]}
        ) / self.norms[y2]
        return complex(v)


def resonance_census(real: DisorderRealization) -> dict[int, float]:
    """n_res(r): pair count with |E_i - E_j| < |J_ij| binned by [r, r+1) (§12)."""
    E = np.sqrt((real.eps + real.h_mf) ** 2 + real.delta**2)
    counts: dict[int, int] = {}
    for ((x1, y1), (x2, y2)), J in real.J.items():
        r_bin = int(math.hypot(x1 - x2, y1 - y2))
        counts.setdefault(r_bin, 0)
        if abs(float(E[y1, x1] - E[y2, x2])) < abs(J):
            counts[r_bin] += 1
    return {r: float(c) for r, c in sorted(counts.items())}


def measure_static(
    state: PEPSState,
    real: DisorderRealization,
    chi: int,
    backend: TruncationBackend,
    circuit: SDRGCircuit | None,
    *,
    factored: bool = False,
) -> StaticObservables:
    L = state.L
    m = _Measurer(state, chi, backend, factored=factored)

    def push(site: Site) -> list[tuple[Site, str, float]]:
        if circuit is None:
            return [(site, "z", 1.0)]
        return circuit.pushforward_z(site)

    def push_x(site: Site) -> list[tuple[Site, str, float]]:
        if circuit is None:
            return [(site, "x", 1.0)]
        # x-pushforward: rotations mix, cluster absorbs transverse components
        acc: list[tuple[Site, str, float]] = [(site, "x", 1.0)]
        # reuse circuit machinery by symmetry: sigma^x on a rotated site
        # -> sin(th) z~ + cos(th) x~ ; on an absorbed site -> dropped
        from tlsmbl.sdrg.rules import BondCluster, SiteRotation

        for op in reversed(circuit.ops):
            out: list[tuple[Site, str, float]] = []
            for s, kind, w in acc:
                if isinstance(op, SiteRotation) and s == op.site:
                    if kind == "z":
                        out += [(s, "z", w * math.cos(op.theta)), (s, "x", -w * math.sin(op.theta))]
                    else:
                        out += [(s, "z", w * math.sin(op.theta)), (s, "x", w * math.cos(op.theta))]
                elif isinstance(op, BondCluster) and s == op.absorbed:
                    if kind == "z":
                        out.append((op.host, "z", w * op.sign))
                else:
                    out.append((s, kind, w))
            acc = out
        return acc

    sz: dict[Site, float] = {}
    sx: dict[Site, float] = {}
    for y in range(L):
        for x in range(L):
            site = (x, y)
            sz[site] = sum(w * m.one(s, k).real for s, k, w in push(site))
            sx[site] = sum(w * m.one(s, k).real for s, k, w in push_x(site))
    q_ea = float(np.mean([v**2 for v in sz.values()]))

    r_max = min(L / 2, 2 * real.params.R_c)
    czz_sums: dict[int, list[float]] = {}
    for (a, b) in real.J:
        r = math.hypot(a[0] - b[0], a[1] - b[1])
        if r > r_max:
            continue
        pa, pb = push(a), push(b)
        val = sum(
            wa * wb * m.pair(s1, k1, s2, k2).real
            for s1, k1, wa in pa
            for s2, k2, wb in pb
        )
        czz_sums.setdefault(int(r), []).append(val - sz[a] * sz[b])
    czz_r = {r: float(np.mean(v)) for r, v in sorted(czz_sums.items())}

    return StaticObservables(
        sz=sz, sx=sx, q_ea=q_ea, czz_r=czz_r, n_res_r=resonance_census(real)
    )

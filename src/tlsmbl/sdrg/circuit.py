"""SDRG circuit and observable pushforward (ARCHITECTURE.md §9.4).

The circuit records the ordered decimation ops; `pushforward` conjugates a
physical-frame sigma^z observable through the ops in reverse so it can be
measured on the PEPS ground state of the transformed Hamiltonian. Site rotations
are exact; cluster ops are exact-2-local within the doublet (leakage ledgered).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tlsmbl.core.types import Site
from tlsmbl.sdrg.rules import BondCluster, SDRGOp, SiteRotation


@dataclass
class SDRGCircuit:
    ops: list[SDRGOp] = field(default_factory=list)
    dropped_norm: float = 0.0
    projection_error: float = 0.0

    @property
    def depth_estimate(self) -> int:
        return len(self.ops)

    def pushforward_z(self, site: Site) -> list[tuple[Site, str, float]]:
        """Physical sigma^z_site expressed in the transformed frame, as a weighted
        sum of one-site operators [(host_site, 'z'|'x', weight)]."""
        acc: list[tuple[Site, str, float]] = [(site, "z", 1.0)]
        for op in reversed(self.ops):
            out: list[tuple[Site, str, float]] = []
            for s, kind, w in acc:
                if isinstance(op, SiteRotation) and s == op.site:
                    if kind == "z":
                        # sigma^z -> cos(th) sigma~z - sin(th) sigma~x  (exact)
                        out.append((s, "z", w * math.cos(op.theta)))
                        out.append((s, "x", -w * math.sin(op.theta)))
                    else:
                        # sigma^x -> sin(th) sigma~z + cos(th) sigma~x
                        out.append((s, "z", w * math.sin(op.theta)))
                        out.append((s, "x", w * math.cos(op.theta)))
                elif isinstance(op, BondCluster) and s == op.absorbed and kind == "z":
                    # moment map: absorbed spin's z follows the cluster spin (x s)
                    out.append((op.host, "z", w * op.sign))
                elif isinstance(op, BondCluster) and s == op.absorbed:
                    # transverse component on the absorbed spin leaks outside the
                    # doublet; dropped (ledgered at decimation time)
                    continue
                else:
                    out.append((s, kind, w))
            acc = out
        return acc

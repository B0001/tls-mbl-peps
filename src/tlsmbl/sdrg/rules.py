"""SDRG decimation rules (ARCHITECTURE.md §9.1-§9.3; coefficients pinned by the
executed T-SDRG-3SITE, ADR-013 -- `prototypes/sdrg_3site.py` is the arbiter for
any change).

Normalization: H = sum (eps/2) Z + (Delta/2) X + sum J ZZ.

  site (dominant E_i, theta = atan2(Delta_i, eps_i)):
      eps_j -= 2 J_ij cos(theta)                       [keep_first_order=False]
      or J_ij -> J_ij cos(theta) kept as a zz bond     [keep_first_order=True]
      J_jk += -2 J_ji J_ik sin^2(theta) / E_i          (unordered j != k)
      E0   += -E_i/2 - sin^2(theta) sum_j J_ij^2 / E_i
  bond (dominant |J_ij|, s = -sign(J_ij)):
      eps_c = eps_i + s eps_j;   J_ck = J_ik + s J_jk   (exact within doublet)
      Delta_c = -Delta_i Delta_j / (2|J_ij|);  E0 += -|J_ij| - (D_i^2+D_j^2)/(8|J_ij|)

The effective model is mutated in place; every decimation returns the op record
the circuit accumulates (§9.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tlsmbl.core.types import HamiltonianTerms, Site
from tlsmbl.sdrg.ledger import Ledger


@dataclass(frozen=True)
class SiteRotation:
    site: Site
    theta: float
    E: float  # rotated local scale; pinned sigma~z = -1


@dataclass(frozen=True)
class BondCluster:
    host: Site  # cluster spin lives here
    absorbed: Site  # pinned-trivial partner
    sign: int  # s = -sign(J): +1 F, -1 AF
    gap: float  # 2|J|, to the discarded sector


SDRGOp = SiteRotation | BondCluster


def _key(a: Site, b: Site, L: int) -> tuple[Site, Site]:
    ia, ib = a[1] * L + a[0], b[1] * L + b[0]
    return (a, b) if ia < ib else (b, a)


@dataclass
class EffectiveModel:
    """Mutable working state of the decimation loop. `active` sites still carry
    quantum degrees of freedom; decimated sites stay in the lattice (ADR-002)
    but leave this set."""

    L: int
    eps: dict[Site, float]
    dlt: dict[Site, float]
    J: dict[tuple[Site, Site], float]
    E0: float = 0.0
    active: set[Site] = field(default_factory=set)
    pinned_fields: dict[Site, float] = field(default_factory=dict)  # E_i of pinned sites

    @classmethod
    def from_terms(cls, terms: HamiltonianTerms) -> EffectiveModel:
        eps: dict[Site, float] = {}
        dlt: dict[Site, float] = {}
        for site, op, c in terms.onsite:
            if op == "z":
                eps[site] = eps.get(site, 0.0) + 2 * c
            else:
                dlt[site] = dlt.get(site, 0.0) + 2 * c
        sites = set(eps) | set(dlt)
        for s in sites:
            eps.setdefault(s, 0.0)
            dlt.setdefault(s, 0.0)
        J = {_key(i, j, terms.L): Jv for i, j, Jv in terms.pair}
        return cls(L=terms.L, eps=eps, dlt=dlt, J=J, active=set(sites))

    def partners(self, i: Site) -> list[tuple[Site, float]]:
        """All live J-bonds incident on i -- including bonds to PINNED sites: their
        retained sigma~z couplings still transform under later decimations (skipping
        them leaves untransformed first-order terms of size J(1-cos theta))."""
        out = []
        for (a, b), Jv in self.J.items():
            if a == i:
                out.append((b, Jv))
            elif b == i:
                out.append((a, Jv))
        return out

    def site_scale(self, i: Site) -> float:
        return math.hypot(self.eps[i], self.dlt[i])


def site_decimate(
    m: EffectiveModel, i: Site, ledger: Ledger, *, keep_first_order: bool
) -> SiteRotation:
    """§9.2: exact rotation + PT2 through the sigma~x channel."""
    E = m.site_scale(i)
    theta = math.atan2(m.dlt[i], m.eps[i])
    sin2 = math.sin(theta) ** 2
    partners = m.partners(i)

    for j, Jij in partners:
        if keep_first_order:
            m.J[_key(i, j, m.L)] = Jij * math.cos(theta)  # 2-local, PEPS sees it
        else:
            m.eps[j] -= 2 * Jij * math.cos(theta)  # pinned <sigma~z> = -1 shift
            del m.J[_key(i, j, m.L)]
    # NOTE: -E_i/2 is NOT added to E0 here. In the ADR-002 composition the pinned
    # site keeps its rotated local term (E_i/2) sigma~z in H-tilde, and the PEPS
    # ground state realizes the -E_i/2 itself; only the PT2 diagonal is scalar.
    # (The 3-site prototype drops the site entirely, so its E0 carries -E/2; the
    # parity test accounts for the difference explicitly.)
    m.E0 += -sin2 * sum(Jij**2 for _, Jij in partners) / E
    for a in range(len(partners)):
        for b in range(a + 1, len(partners)):
            (j, Jji), (k, Jik) = partners[a], partners[b]
            m.J[_key(j, k, m.L)] = (
                m.J.get(_key(j, k, m.L), 0.0) - 2 * Jji * Jik * sin2 / E
            )
    # Beyond-PT2 estimate: third-order class terms (§9.2 step 4).
    third = sum(
        abs(Jji * Jik) * abs(m.J.get(_key(j, k, m.L), 0.0))
        for aa, (j, Jji) in enumerate(partners)
        for k, Jik in partners[aa + 1 :]
    ) / max(E**2, 1e-300)
    ledger.add_dropped(third, f"site {i} PT3 tail")

    m.active.discard(i)
    m.pinned_fields[i] = E
    m.eps[i] = 0.0
    m.dlt[i] = 0.0
    return SiteRotation(site=i, theta=theta, E=E)


def bond_decimate(m: EffectiveModel, i: Site, j: Site, ledger: Ledger) -> BondCluster:
    """§9.3: project onto the aligned doublet; cluster spin hosted on i."""
    Jij = m.J.pop(_key(i, j, m.L))
    s = -1 if Jij > 0 else +1
    aJ = abs(Jij)

    partners_j = m.partners(j)
    m.eps[i] = m.eps[i] + s * m.eps[j]
    delta_i, delta_j = m.dlt[i], m.dlt[j]
    m.dlt[i] = -delta_i * delta_j / (2 * aJ)
    m.E0 += -aJ - (delta_i**2 + delta_j**2) / (8 * aJ)
    leak_bonds = 0.0
    for k, Jjk in partners_j:
        if k == i:
            continue
        key_ik = _key(i, k, m.L)
        Jik = m.J.get(key_ik, 0.0)
        m.J[key_ik] = Jik + s * Jjk
        leak_bonds += (Jik - s * Jjk) ** 2
        del m.J[_key(j, k, m.L)]
    ledger.add_projection(
        (delta_i**2 + delta_j**2 + leak_bonds) / (2 * aJ) ** 2,
        f"bond ({i},{j}) doublet leakage",
    )
    m.active.discard(j)
    m.eps[j] = 0.0
    m.dlt[j] = 0.0
    return BondCluster(host=i, absorbed=j, sign=s, gap=2 * aJ)

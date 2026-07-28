"""T-SDRG-3SITE (§14.3, blocking Phase 4) via the executed prototype as oracle:
production rules must reproduce the fixed-denominator Schur downfolding to machine
precision (Tier I) and converge with local order >= 2.9 (Tier II) -- same gates,
same draws as prototypes/sdrg_3site.py (ADR-013 arbiter)."""

import numpy as np
import pytest

import sdrg_3site as proto
from tlsmbl.core.types import HamiltonianTerms, Site
from tlsmbl.sdrg.ledger import Ledger
from tlsmbl.sdrg.rules import EffectiveModel, bond_decimate, site_decimate

I2, Zm, Xm = np.eye(2), np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])
S0, S1, S2 = (0, 0), (1, 0), (2, 0)
SITES = [S0, S1, S2]


def _terms(p: dict) -> HamiltonianTerms:
    onsite = []
    for idx, site in enumerate(SITES):
        onsite.append((site, "z", p["eps"][idx] / 2.0))
        onsite.append((site, "x", p["dlt"][idx] / 2.0))
    pair = [(SITES[a], SITES[b], J) for (a, b), J in p["J"].items()]
    return HamiltonianTerms(L=3, onsite=onsite, pair=pair)


def _heff_two_site(m: EffectiveModel, a: Site, b: Site) -> np.ndarray:
    key = (a, b) if a[1] * 3 + a[0] < b[1] * 3 + b[0] else (b, a)
    J = m.J.get(key, 0.0)
    # The Schur oracle integrates the decimated site out entirely, so its pinned
    # energy -E_i/2 (kept as a lattice term in production, ADR-002) is folded into
    # the scalar here for comparison.
    e0 = m.E0 + sum(-E / 2 for E in m.pinned_fields.values())
    return (
        m.eps[a] / 2 * np.kron(Zm, I2)
        + m.dlt[a] / 2 * np.kron(Xm, I2)
        + m.eps[b] / 2 * np.kron(I2, Zm)
        + m.dlt[b] / 2 * np.kron(I2, Xm)
        + J * np.kron(Zm, Zm)
        + e0 * np.eye(4)
    )


def _decimate_site(p: dict) -> np.ndarray:
    m = EffectiveModel.from_terms(_terms(p))
    site_decimate(m, S1, Ledger(), keep_first_order=False)
    return _heff_two_site(m, S0, S2)


def _decimate_bond(p: dict) -> np.ndarray:
    m = EffectiveModel.from_terms(_terms(p))
    bond_decimate(m, S0, S1, Ledger())
    return _heff_two_site(m, S0, S2)


CASES = [
    ("site_field", lambda lam, d: proto.site_case(lam, d), _decimate_site, proto.schur_site),
    ("bond_AF", lambda lam, d: proto.bond_case(lam, +1, d), _decimate_bond, proto.schur_bond),
    ("bond_F", lambda lam, d: proto.bond_case(lam, -1, d), _decimate_bond, proto.schur_bond),
]


@pytest.mark.parametrize("label,mk,ours,schur", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("draw", [0, 1, 2])
def test_tier1_rule_equals_schur(label: str, mk, ours, schur, draw: int) -> None:
    p = mk(0.32, draw)
    Hr, Hs = ours(p), schur(p)
    rel = np.abs(Hr - Hs).max() / max(np.abs(Hs).max(), 1e-30)
    assert rel < 1e-12, f"{label} draw{draw}: identity {rel:.2e}"


@pytest.mark.parametrize("label,mk,ours,schur", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("draw", [0, 1, 2])
def test_tier2_convergence_order(label: str, mk, ours, schur, draw: int) -> None:
    lams = [0.32, 0.16, 0.08, 0.04, 0.02]
    errs = []
    for lam in lams:
        p = mk(lam, draw)
        w_full = np.linalg.eigvalsh(proto.H_full(p))[:4]
        w_eff = np.linalg.eigvalsh(ours(p))
        errs.append(np.abs(np.sort(w_full) - np.sort(w_eff)).max())
    orders = [np.log2(errs[k] / errs[k + 1]) for k in range(len(lams) - 1)]
    assert min(orders[-2:]) >= 2.9, f"{label} draw{draw}: orders {orders}"

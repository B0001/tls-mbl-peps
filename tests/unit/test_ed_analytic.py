"""§16 P1 exit: ED reproduces analytic cases. All on an L=2 lattice with terms placed
by hand, so the answers are closed-form."""

import numpy as np

from tlsmbl.core.types import HamiltonianTerms
from tlsmbl.model.ed_reference import ed_ground, ed_observables


def test_single_site_ground_energy() -> None:
    """One dressed site: E0 = -sqrt(eps^2 + delta^2)/2."""
    eps, delta = 0.7, 0.4
    terms = HamiltonianTerms(
        L=2, onsite=[((0, 0), "z", eps / 2), ((0, 0), "x", delta / 2)], pair=[]
    )
    assert abs(ed_ground(terms).energies[0] + np.hypot(eps, delta) / 2) < 1e-12


def test_classical_ising_pair() -> None:
    """delta = 0: E0 = min over s1, s2 of e1/2 s1 + e2/2 s2 + J s1 s2."""
    e1, e2, J = 0.9, -0.5, 0.3
    terms = HamiltonianTerms(
        L=2,
        onsite=[((0, 0), "z", e1 / 2), ((1, 0), "z", e2 / 2)],
        pair=[((0, 0), (1, 0), J)],
    )
    exact = min(
        e1 / 2 * s1 + e2 / 2 * s2 + J * s1 * s2 for s1 in (1, -1) for s2 in (1, -1)
    )
    assert abs(ed_ground(terms).energies[0] - exact) < 1e-12


def test_resonant_pair_two_site() -> None:
    """eps = 0, delta1 = delta2 = d, coupling J: E0 = -sqrt(J^2 + d^2)."""
    d, J = 0.6, 0.25
    terms = HamiltonianTerms(
        L=2,
        onsite=[((0, 0), "x", d / 2), ((1, 0), "x", d / 2)],
        pair=[((0, 0), (1, 0), J)],
    )
    assert abs(ed_ground(terms).energies[0] + np.hypot(J, d)) < 1e-12


def test_observables_polarized_site() -> None:
    """Strong-field site (eps >> delta): <sz> -> -1, matches exact 1-site formula
    <sz> = -eps/sqrt(eps^2+delta^2), <sx> = -delta/sqrt(eps^2+delta^2)."""
    eps, delta = 1.0, 0.3
    terms = HamiltonianTerms(
        L=2, onsite=[((0, 0), "z", eps / 2), ((0, 0), "x", delta / 2)], pair=[]
    )
    res = ed_ground(terms)
    sz, sx, _ = ed_observables(terms, res.ground)
    E = np.hypot(eps, delta)
    assert abs(sz[(0, 0)] + eps / E) < 1e-10
    assert abs(sx[(0, 0)] + delta / E) < 1e-10


def test_zz_correlator_ising_pair() -> None:
    """Classical AF pair ground state: <z1 z2> = -1."""
    terms = HamiltonianTerms(
        L=2,
        onsite=[((0, 0), "z", 1e-6)],  # tiny tiebreak field
        pair=[((0, 0), (1, 0), 0.5)],
    )
    res = ed_ground(terms)
    _, _, zz = ed_observables(terms, res.ground)
    assert abs(zz[((0, 0), (1, 0))] + 1.0) < 1e-9

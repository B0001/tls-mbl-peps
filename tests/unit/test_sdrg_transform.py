"""SDRG loop, circuit pushforward, and T-INV-8 bypass (§9, §14.8)."""

import math


from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import ModelParams
from tlsmbl.model.ed_reference import build_H, ed_ground
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.sdrg.circuit import SDRGCircuit
from tlsmbl.sdrg.rules import BondCluster, SiteRotation
from tlsmbl.sdrg.transform import sdrg_transform


def _terms(L: int = 3, g_J: float = 0.3, k: int = 0):
    params = ModelParams(L=L, g_J=g_J, R_c=3, seed_realization=k)
    return build_terms(sample_realization(params, realization_streams(20260716, k).disorder))


def test_transform_runs_and_reports_flow() -> None:
    terms = _terms()
    res = sdrg_transform(
        terms, omega_stop=0.3, f_max=0.4, keep_first_order=True, tau_sdrg=0.05
    )
    assert not res.bypassed
    assert res.circuit is not None
    assert len(res.omega_sequence) == len(res.circuit.ops)
    # decimation scales are non-increasing up to PT2-generated couplings
    assert res.omega_sequence == sorted(res.omega_sequence, reverse=True) or len(
        res.omega_sequence
    ) <= 2
    assert res.ledger.total < 0.05 * terms.norm_local


def test_transformed_spectrum_approximates_original() -> None:
    """E0 + ground energy of the transformed H must approximate the original ground
    energy to the ledgered accuracy scale (weak coupling: PT2 is very accurate)."""
    terms = _terms(g_J=1e-3)
    res = sdrg_transform(
        terms, omega_stop=0.3, f_max=0.4, keep_first_order=True, tau_sdrg=0.05
    )
    assert not res.bypassed
    e_orig = ed_ground(terms).energies[0]
    e_tilde = ed_ground(res.terms).energies[0] + res.E0
    assert abs(e_orig - e_tilde) < 1e-4, f"{e_orig} vs {e_tilde}"


def test_inv8_bypass_on_tiny_tau() -> None:
    terms = _terms(g_J=0.3)
    res = sdrg_transform(
        terms, omega_stop=0.05, f_max=1.0, keep_first_order=False, tau_sdrg=1e-18
    )
    assert res.bypassed
    assert res.circuit is None
    assert res.terms is terms  # untouched original
    assert res.E0 == 0.0


def test_pushforward_site_rotation() -> None:
    circuit = SDRGCircuit(ops=[SiteRotation(site=(0, 0), theta=0.7, E=1.0)])
    out = circuit.pushforward_z((0, 0))
    assert sorted(out) == sorted(
        [((0, 0), "z", math.cos(0.7)), ((0, 0), "x", -math.sin(0.7))]
    )
    # other sites untouched
    assert circuit.pushforward_z((1, 0)) == [((1, 0), "z", 1.0)]


def test_pushforward_cluster_moment_map() -> None:
    circuit = SDRGCircuit(
        ops=[BondCluster(host=(0, 0), absorbed=(1, 0), sign=-1, gap=2.0)]
    )
    assert circuit.pushforward_z((1, 0)) == [((0, 0), "z", -1.0)]
    assert circuit.pushforward_z((0, 0)) == [((0, 0), "z", 1.0)]


def test_pushforward_composed_cluster_then_rotation() -> None:
    """Ops applied in order [rotation on host, cluster into host]: pushforward walks
    in reverse -- absorbed z maps to host, then host z rotates."""
    circuit = SDRGCircuit(
        ops=[
            SiteRotation(site=(0, 0), theta=0.5, E=1.0),
            BondCluster(host=(0, 0), absorbed=(1, 0), sign=+1, gap=2.0),
        ]
    )
    out = circuit.pushforward_z((1, 0))
    assert sorted(out) == sorted(
        [((0, 0), "z", math.cos(0.5)), ((0, 0), "x", -math.sin(0.5))]
    )


def test_transform_preserves_hermiticity_and_pinned_sites() -> None:
    terms = _terms(g_J=0.3)
    res = sdrg_transform(
        terms, omega_stop=0.3, f_max=0.4, keep_first_order=True, tau_sdrg=0.05
    )
    assert not res.bypassed
    H = build_H(res.terms)
    assert abs(H - H.getH()).max() == 0.0
    # every site still appears in the lattice term list (ADR-002: pinned, not removed)
    sites_with_terms = {s for s, _, _ in res.terms.onsite}
    assert len(sites_with_terms) == 9

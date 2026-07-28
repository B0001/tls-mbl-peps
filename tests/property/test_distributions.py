"""T-PROP-DIST (ARCHITECTURE.md §14.5): distributional and structural properties of
the disorder sampler."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import ModelParams
from tlsmbl.model.sampling import sample_realization


def _params(L: int = 4, g_J: float = 1e-3, R_c: int = 3, seed: int = 0, **kw) -> ModelParams:
    return ModelParams(L=L, g_J=g_J, R_c=R_c, seed_realization=seed, **kw)


def _sample(L: int = 4, master: int = 20260715, k: int = 0, **kw):
    streams = realization_streams(master, k)
    return sample_realization(_params(L=L, seed=k, **kw), streams.disorder)


def test_ln_delta_uniform_ks() -> None:
    """P(Delta) ~ 1/Delta means ln Delta ~ U[ln delta_min, 0]."""
    delta_min = 1e-3
    draws = np.concatenate(
        [_sample(L=8, k=k).delta.ravel() for k in range(20)]
    )  # 1280 draws
    u = np.log(draws) / np.log(delta_min)  # maps to U[0, 1]
    assert stats.kstest(u, "uniform").pvalue > 1e-3


def test_eps_bounded_and_centered() -> None:
    eps = np.concatenate([_sample(L=8, k=k).eps.ravel() for k in range(20)])
    assert np.all(np.abs(eps) <= 1.0)
    assert abs(eps.mean()) < 0.05


@given(st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20, deadline=None)
def test_sampler_bitwise_deterministic(k: int) -> None:
    a, b = _sample(k=k), _sample(k=k)
    np.testing.assert_array_equal(a.eps, b.eps)
    np.testing.assert_array_equal(a.delta, b.delta)
    assert a.J == b.J
    assert a.rng_fingerprint == b.rng_fingerprint


def test_pair_list_cutoff_and_canonical_order() -> None:
    L, R_c = 5, 3
    real = _sample(L=L, R_c=R_c)
    sites = [(x, y) for y in range(L) for x in range(L)]
    order = {s: i for i, s in enumerate(sites)}
    expected = {
        (a, b)
        for ia, a in enumerate(sites)
        for b in sites[ia + 1 :]
        if 1.0 <= math.hypot(a[0] - b[0], a[1] - b[1]) <= R_c
    }
    assert set(real.J) == expected  # exactly the r in [1, R_c] pairs, no self-pairs
    for a, b in real.J:
        assert order[a] < order[b]  # canonical (row-major a < b) keys only


def test_coupling_magnitude_bound() -> None:
    """|J_ij| <= g_J / r^3 <= g_J since r >= 1."""
    g_J = 0.3
    real = _sample(g_J=g_J)
    for (a, b), J in real.J.items():
        r = math.hypot(a[0] - b[0], a[1] - b[1])
        assert abs(J) <= g_J / r**3 + 1e-15


def test_polaron_kappa_only_suppresses() -> None:
    base = _sample()
    supp = _sample(polaron_kappa=2.0)
    assert np.all(supp.delta <= base.delta + 1e-15)
    np.testing.assert_array_equal(supp.eps, base.eps)  # eps draw unaffected

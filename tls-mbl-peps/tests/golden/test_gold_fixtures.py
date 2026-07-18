"""T-GOLD fixture integrity (§16 P1 exit): the stored ED fixtures regenerate exactly
from the model layer. Guards against silent sampler / assembly / ED drift; these are
the reference values T-GOLD-ED (Phase 2) certifies PEPS energies against.

The full 12-point grid runs in CI; locally the L=3 subset keeps the suite fast unless
TLSMBL_FULL_GOLD=1.
"""

import json
import os

import pytest

from gen_fixtures import FIXTURES, GJS, KS, LS, fixture_name, generate_one

FULL = os.environ.get("TLSMBL_FULL_GOLD") == "1"
GRID = [
    (L, g_J, k)
    for L in (LS if FULL else (3,))
    for g_J in GJS
    for k in (KS if FULL else KS[:1])
]


@pytest.mark.parametrize("L,g_J,k", GRID)
def test_fixture_regenerates(L: int, g_J: float, k: int) -> None:
    stored = json.loads((FIXTURES / fixture_name(L, g_J, k)).read_text())
    fresh = generate_one(L, g_J, k)
    assert fresh["rng_fingerprint"] == stored["rng_fingerprint"]  # bitwise disorder
    assert fresh["n_pair_terms"] == stored["n_pair_terms"]
    for a, b in zip(fresh["energies"], stored["energies"]):
        assert abs(a - b) < 1e-10
    for key in ("sz", "sx", "zz"):
        assert fresh[key].keys() == stored[key].keys()
        for name, v in fresh[key].items():
            assert abs(v - stored[key][name]) < 1e-8


def test_all_twelve_fixtures_present() -> None:
    missing = [
        fixture_name(L, g_J, k)
        for L in LS
        for g_J in GJS
        for k in KS
        if not (FIXTURES / fixture_name(L, g_J, k)).exists()
    ]
    assert missing == []

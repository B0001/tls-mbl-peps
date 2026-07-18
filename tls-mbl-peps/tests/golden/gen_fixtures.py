"""Generate the stored T-GOLD ED fixtures (§16 P1 exit; consumed by T-GOLD-ED in P2).

Seed convention (ADR-014): master_seed 20260716 (the prototype golden master),
realization index k in {0, 1, 2}, disorder stream via core/rng.realization_streams.
Grid: L in {3, 4} x g_J in {1e-3, 0.3}. Same (master, k, L) share the disorder draw;
only the J magnitudes scale with g_J.

Run from the repo root:  uv run python tests/golden/gen_fixtures.py
Never overwrite silently: refuses if fixtures exist, unless --force.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import ModelParams, Site
from tlsmbl.model.ed_reference import ed_ground, ed_observables
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization

MASTER = 20260716
KS = (0, 1, 2)
LS = (3, 4)
GJS = (1e-3, 0.3)
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _site_key(s: Site) -> str:
    return f"{s[0]},{s[1]}"


def fixture_name(L: int, g_J: float, k: int) -> str:
    return f"ed_L{L}_gJ{g_J:g}_k{k}.json"


def generate_one(L: int, g_J: float, k: int) -> dict:
    params = ModelParams(L=L, g_J=g_J, R_c=3, seed_realization=k)
    real = sample_realization(params, realization_streams(MASTER, k).disorder)
    terms = build_terms(real)
    res = ed_ground(terms, k=4)
    sz, sx, zz = ed_observables(terms, res.ground)
    return {
        "master_seed": MASTER,
        "k": k,
        "L": L,
        "g_J": g_J,
        "R_c": 3,
        "rng_fingerprint": real.rng_fingerprint,
        "n_pair_terms": len(terms.pair),
        "energies": [float(e) for e in res.energies],
        "sz": {_site_key(s): v for s, v in sz.items()},
        "sx": {_site_key(s): v for s, v in sx.items()},
        "zz": {f"{_site_key(i)}|{_site_key(j)}": v for (i, j), v in zz.items()},
    }


def main(force: bool = False) -> None:
    FIXTURES.mkdir(exist_ok=True)
    for L in LS:
        for g_J in GJS:
            for k in KS:
                path = FIXTURES / fixture_name(L, g_J, k)
                if path.exists() and not force:
                    print(f"exists, skipping: {path.name} (--force to regenerate)")
                    continue
                fx = generate_one(L, g_J, k)
                path.write_text(json.dumps(fx, indent=1))
                print(f"wrote {path.name}: E0 = {fx['energies'][0]:+.12f}")


if __name__ == "__main__":
    np.seterr(all="raise")
    main(force="--force" in sys.argv)

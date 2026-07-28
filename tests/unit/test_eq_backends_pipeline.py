"""T-EQ-BACKENDS pipeline half (§14.7): full 3x3 contraction with the sketched
backend agrees with the exact backend at certified chi."""

import numpy as np

from tlsmbl.core.types import ModelParams, TensorSpec
from tlsmbl.kernels.rsvd import SketchedSVD
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.sampling import sample_realization
from tlsmbl.peps.energy import energy_differentiable
from tlsmbl.peps.state import PEPSState

SEED = 20260720


def test_pipeline_energy_exact_vs_sketched() -> None:
    params = ModelParams(L=3, g_J=0.3, R_c=3, seed_realization=SEED)
    real = sample_realization(params, np.random.default_rng(np.random.SeedSequence(SEED)))
    terms = build_terms(real)
    state = PEPSState.random(3, 3, TensorSpec(), np.random.SeedSequence(SEED))
    e_exact = float(energy_differentiable(state, terms, 9, ExactSVD()))
    sketched = SketchedSVD(seed=SEED)
    e_sketch = float(energy_differentiable(state, terms, 9, sketched))
    # <= 10 * tau_chi with the default tau_chi = 1e-6 (§14.7)
    assert abs(e_exact - e_sketch) <= 1e-5, f"{e_exact} vs {e_sketch}"
    assert sketched.call_count > 0

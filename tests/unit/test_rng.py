import numpy as np
import pytest

from tlsmbl.core.rng import (
    MissingSeedError,
    realization_seed_sequence,
    realization_streams,
    require_master_seed,
)


def test_require_master_seed_refuses_none() -> None:
    with pytest.raises(MissingSeedError):
        require_master_seed(None)


def test_require_master_seed_passthrough() -> None:
    assert require_master_seed(42) == 42


def test_realization_streams_bit_reproducible() -> None:
    a = realization_streams(20260715, 3)
    b = realization_streams(20260715, 3)
    assert a.torch_init_seed == b.torch_init_seed
    assert a.torch_sketch_seed == b.torch_sketch_seed
    np.testing.assert_array_equal(a.disorder.random(8), b.disorder.random(8))


def test_realization_streams_differ_by_realization_index() -> None:
    a = realization_streams(20260715, 0)
    b = realization_streams(20260715, 1)
    assert a.torch_init_seed != b.torch_init_seed
    assert a.torch_sketch_seed != b.torch_sketch_seed


def test_realization_seed_sequence_differs_by_master_seed() -> None:
    s1 = realization_seed_sequence(1, 0)
    s2 = realization_seed_sequence(2, 0)
    assert s1.entropy != s2.entropy

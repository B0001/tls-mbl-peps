import pytest
import torch

from tlsmbl.core.guards import NumericalCorruption, finite


def test_finite_passes_clean_tensor() -> None:
    @finite
    def f() -> torch.Tensor:
        return torch.tensor([1.0, 2.0])

    assert torch.equal(f(), torch.tensor([1.0, 2.0]))


def test_finite_raises_on_nan_with_provenance() -> None:
    @finite
    def f(site: tuple[int, int] | None = None) -> torch.Tensor:
        return torch.tensor([float("nan"), 1.0])

    with pytest.raises(NumericalCorruption, match=r"site"):
        f(site=(2, 3))


def test_finite_raises_on_inf_inside_tuple_output() -> None:
    @finite
    def f() -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor([1.0]), torch.tensor([float("inf")]))

    with pytest.raises(NumericalCorruption):
        f()

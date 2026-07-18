from tlsmbl.core.units import Quantity, coarse_graining_length, dimensionless_coupling


def test_dimensionless_coupling() -> None:
    assert dimensionless_coupling(Quantity(2.0), Quantity(3.0)) == 6.0


def test_coarse_graining_length_unit_case() -> None:
    assert coarse_graining_length(Quantity(1.0), Quantity(1.0), Quantity(1.0)) == 1.0


def test_coarse_graining_length_scales_as_inverse_sqrt() -> None:
    a = coarse_graining_length(Quantity(4.0), Quantity(1.0), Quantity(1.0))
    assert abs(a - 0.5) < 1e-12

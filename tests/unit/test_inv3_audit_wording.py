"""ADR-017: the §11 INV-3 audit line must name *which* reason a fallback rate is absent.

Before ADR-017 all three causes printed one sentence, which read as reassurance in the
one case (a sketching backend that recorded nothing) that warrants the opposite.
"""

from tlsmbl.ensemble.aggregate import _inv3_rate_phrase


def test_present_rate_is_printed_regardless_of_backend() -> None:
    assert _inv3_rate_phrase(0.25, "sketched") == "25.0%"
    assert _inv3_rate_phrase(0.0, None) == "0.0%"


def test_exact_backend_certifies_itself_as_having_no_sketch_path() -> None:
    phrase = _inv3_rate_phrase(None, "exact")
    assert "exact backend" in phrase
    assert "predates" not in phrase, "an exact run is not an un-audited run"


def test_legacy_artifact_keeps_the_honest_hedge_and_nothing_more() -> None:
    """Artifacts written before the field existed (runs/pilot_L8.zarr) must still be
    readable and must still decline to guess -- but the hedge no longer has to cover
    the exact-backend case it used to."""
    phrase = _inv3_rate_phrase(None, None)
    assert "predates the INV-3 audit" in phrase
    assert "exact backend" not in phrase


def test_sketching_backend_without_stats_is_flagged_not_excused() -> None:
    phrase = _inv3_rate_phrase(None, "sketched")
    assert "sketched" in phrase
    assert "no realization reported sketch statistics" in phrase


def test_unknown_future_backend_is_tolerated_and_named() -> None:
    """ADR-008's Rust kernel must be recordable in an artifact that a reader build has
    never heard of, without the reader rejecting it or mislabeling it exact."""
    phrase = _inv3_rate_phrase(None, "rust")
    assert "rust" in phrase
    assert "exact backend" not in phrase

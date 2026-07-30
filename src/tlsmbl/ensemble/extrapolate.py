"""E(D) bond-dimension extrapolation (ARCHITECTURE.md §10.3, §11; the §16/§18
definition-of-done requires E(D) extrapolation in REPORT.md).

Fits the 1/D form the configs and §10.3 name:

    E(D) = E_inf + c / D                (least squares in x = 1/D)

and reports the *remaining gap* |E(D_max) - E_inf| as the honest statement of the
ansatz-truncation uncertainty: it is the part of the energy the ladder never
measured, not a bound.

Two things this module refuses to paper over.

1. **Rung count.** `configs/pilot_L8.yaml` runs ladder [2, 3], only
   `configs/bench_L16_D4.yaml` runs [2, 3, 4]. Two points determine the line
   exactly, so their "residuals" are structurally zero and carry no information
   about the fit -- there is no per-realization fit uncertainty to quote. Such a
   result is labeled `method="secant_2pt"`, gets `rms_residual=None`, and says so
   in `reason`. Three or more rungs give `method="fit_1_over_D"` with real
   residuals.

2. **Bad input beats a pretty number.** Every verdict carries `ok` plus a
   machine-readable `reason`; nothing raises and nothing emits a number with
   `ok=False`. Refusals: fewer than two rungs; energies not decreasing in D (a
   variational ansatz cannot get worse with more parameters -- a rise means an
   unconverged rung, per CLAUDE.md's "optimizer floor, not ansatz" gotcha); and
   E_inf above the best measured E(D_max), which is variationally impossible and
   always means the fit, not the physics. Rungs whose stored `converged` flag is
   False (or absent) *taint* the result -- reported, not silently dropped, because
   the extrapolation is only as good as its worst rung.

Bootstrap conventions (resample count, percentile CI, `(mean, lo, hi)` triples)
match `aggregate.py` so the REPORT.md numbers are comparable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import zarr

from tlsmbl.io import store

_BOOT = 10_000  # §11: 1e4-resample bootstrap, same as aggregate.py

METHOD_FIT = "fit_1_over_D"  # >= 3 rungs: least squares, residuals informative
METHOD_SECANT = "secant_2pt"  # exactly 2 rungs: exact 2-point solve, no fit error
METHOD_NONE = "none"

# --- per-realization verdicts -------------------------------------------------
OK_FIT = "ok_fit_1_over_D"
OK_SECANT = "ok_secant_2pt_unfitted_no_uncertainty"
REFUSED_TOO_FEW_RUNGS = "refused_fewer_than_2_rungs"
REFUSED_BAD_RUNG_D = "refused_bad_rung_bond_dimension"
REFUSED_NON_FINITE = "refused_non_finite_rung_energy"
REFUSED_NON_MONOTONE = "refused_energy_non_monotone_in_D"
REFUSED_ABOVE_VARIATIONAL = "refused_e_inf_above_e_at_d_max"

# --- ensemble verdicts --------------------------------------------------------
OK_ENSEMBLE = "ok_disorder_mean_bootstrap_ci"
OK_ENSEMBLE_SECANT = "ok_secant_2pt_disorder_ci_only_no_fit_uncertainty"
OK_ENSEMBLE_MIXED = "ok_mixed_rung_counts_across_realizations"
OK_ENSEMBLE_SINGLE = "ok_single_usable_realization_no_ci"
REFUSED_NO_USABLE = "refused_no_usable_realizations"


@dataclass(frozen=True)
class DExtrapolation:
    """One realization's E(D) -> E_inf record.

    `ok=False` means: do not quote `e_inf` (it is None). `tainted=True` means the
    numbers exist but at least one rung did not converge, so the extrapolation
    inherits that rung's optimizer floor.
    """

    ok: bool
    reason: str
    method: str
    rungs: tuple[tuple[int, float], ...]  # (D, E) sorted by ascending D
    e_inf: float | None
    slope: float | None  # the fitted c in E_inf + c/D (>0 for a sane ladder)
    residuals: tuple[float, ...]  # E_i - (e_inf + c/D_i), ordered like `rungs`
    rms_residual: float | None  # None for the 2-rung secant: structurally zero
    d_max: int | None
    e_d_max: float | None  # best (lowest) measured energy
    remaining_gap: float | None  # |E(D_max) - E_inf|: unmeasured ansatz error
    tainted: bool
    tainted_rungs: tuple[int, ...]


@dataclass(frozen=True)
class EnsembleExtrapolation:
    """Disorder-averaged extrapolation over realizations that passed their gates.

    CI triples are `(mean, lo, hi)` 95% percentile bootstrap over realizations --
    disorder scatter, never a per-realization fit uncertainty. The bounds are NaN
    when a single realization is usable (a bootstrap CI from one sample is noise),
    and the triples are None when none is.
    """

    ok: bool
    reason: str
    method: str
    n_total: int
    n_used: int
    n_tainted: int
    refusals: dict[str, int]  # reason -> count, over the excluded realizations
    e_inf: tuple[float, float, float] | None
    slope: tuple[float, float, float] | None
    remaining_gap: tuple[float, float, float] | None
    per_realization: tuple[DExtrapolation, ...]


def _boot_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean -- identical convention to aggregate.py."""
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _refuse(
    reason: str,
    items: tuple[tuple[int, float], ...],
    tainted_rungs: tuple[int, ...],
) -> DExtrapolation:
    """A refusal keeps the raw inputs (for the audit trail) and emits no number."""
    return DExtrapolation(
        ok=False,
        reason=reason,
        method=METHOD_NONE,
        rungs=items,
        e_inf=None,
        slope=None,
        residuals=(),
        rms_residual=None,
        d_max=items[-1][0] if items else None,
        e_d_max=items[-1][1] if items else None,
        remaining_gap=None,
        tainted=bool(tainted_rungs),
        tainted_rungs=tainted_rungs,
    )


def extrapolate_energy(
    rungs: Mapping[int, float],
    *,
    converged: Mapping[int, bool] | None = None,
    tol_rise: float = 1e-10,
) -> DExtrapolation:
    """Extrapolates one realization's ladder energies to D -> infinity.

    `rungs` maps bond dimension D to the total energy at that rung (as stored by
    `store.write_rung`; use `read_rung_energies` to pull both out of a run).
    `converged` is the optional parallel map of per-rung convergence flags -- a
    rung that is False *or missing* taints the result, since an absent flag is
    not evidence of convergence.

    `tol_rise` is the relative slack on the monotonicity gate: rises below
    tol_rise * max(1, |E|) are approximate-contraction float noise, anything
    larger is an unconverged rung and is refused.
    """
    items = tuple(sorted((int(d), float(e)) for d, e in rungs.items()))
    tainted_rungs = (
        () if converged is None else tuple(d for d, _ in items if not converged.get(d, False))
    )

    if len(items) < 2:
        return _refuse(REFUSED_TOO_FEW_RUNGS, items, tainted_rungs)
    if any(d < 1 for d, _ in items):
        return _refuse(REFUSED_BAD_RUNG_D, items, tainted_rungs)
    if not all(np.isfinite(e) for _, e in items):
        return _refuse(REFUSED_NON_FINITE, items, tainted_rungs)

    # Variational monotonicity: E(D) must not increase with D. The ladder warm-
    # starts each rung from the grown previous state, so a rise cannot be an
    # ansatz property -- it is a rung that stopped short.
    for (_, e_lo), (_, e_hi) in zip(items, items[1:]):
        if e_hi - e_lo > tol_rise * max(1.0, abs(e_lo)):
            return _refuse(REFUSED_NON_MONOTONE, items, tainted_rungs)

    x = np.array([1.0 / d for d, _ in items])
    y = np.array([e for _, e in items])
    # Design matrix [1, 1/D]: intercept is E_inf (the 1/D -> 0 limit), slope is c.
    design = np.stack([np.ones_like(x), x], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    e_inf, slope = float(coeffs[0]), float(coeffs[1])
    residuals = tuple(float(v) for v in y - design @ coeffs)
    d_max, e_d_max = items[-1]

    # E_inf above the best measured energy is variationally impossible (D -> inf
    # contains the D_max manifold), so it convicts the fit, not the state.
    if e_inf > e_d_max + tol_rise * max(1.0, abs(e_d_max)):
        return _refuse(REFUSED_ABOVE_VARIATIONAL, items, tainted_rungs)

    two_point = len(items) == 2
    return DExtrapolation(
        ok=True,
        reason=OK_SECANT if two_point else OK_FIT,
        method=METHOD_SECANT if two_point else METHOD_FIT,
        rungs=items,
        e_inf=e_inf,
        slope=slope,
        residuals=residuals,
        # 2 points fix the line exactly: the zeros in `residuals` are structural,
        # so quoting an RMS residual would fake a goodness-of-fit statistic.
        rms_residual=None if two_point else float(np.sqrt(np.mean(np.square(residuals)))),
        d_max=d_max,
        e_d_max=e_d_max,
        remaining_gap=abs(e_d_max - e_inf),
        tainted=bool(tainted_rungs),
        tainted_rungs=tainted_rungs,
    )


def extrapolate_ensemble(
    per_realization: Sequence[Mapping[int, float]],
    *,
    rng: np.random.Generator,
    n_boot: int = _BOOT,
    converged: Sequence[Mapping[int, bool]] | None = None,
    tol_rise: float = 1e-10,
) -> EnsembleExtrapolation:
    """Disorder-averages the per-realization extrapolations with bootstrap CIs.

    `converged`, when given, is positionally parallel to `per_realization`.
    Realizations that fail their gates are excluded and counted in `refusals`;
    tainted ones are *kept* (their numbers are real, just floor-limited) and
    counted in `n_tainted`, mirroring aggregate.py's label-don't-hide policy for
    uncertified realizations.
    """
    if converged is not None and len(converged) != len(per_realization):
        raise ValueError("converged must be positionally parallel to per_realization")
    records = tuple(
        extrapolate_energy(
            r,
            converged=None if converged is None else converged[i],
            tol_rise=tol_rise,
        )
        for i, r in enumerate(per_realization)
    )
    used = [r for r in records if r.ok]
    refusals: dict[str, int] = {}
    for r in records:
        if not r.ok:
            refusals[r.reason] = refusals.get(r.reason, 0) + 1
    n_tainted = sum(r.tainted for r in used)

    methods = {r.method for r in used}
    if not used:
        method = METHOD_NONE
        reason = REFUSED_NO_USABLE
    elif len(methods) > 1:
        method, reason = METHOD_FIT, OK_ENSEMBLE_MIXED
    elif len(used) == 1:
        method, reason = used[0].method, OK_ENSEMBLE_SINGLE
    elif methods == {METHOD_SECANT}:
        method, reason = METHOD_SECANT, OK_ENSEMBLE_SECANT
    else:
        method, reason = METHOD_FIT, OK_ENSEMBLE

    base = EnsembleExtrapolation(
        ok=bool(used),
        reason=reason,
        method=method,
        n_total=len(records),
        n_used=len(used),
        n_tainted=n_tainted,
        refusals=refusals,
        e_inf=None,
        slope=None,
        remaining_gap=None,
        per_realization=records,
    )
    if not used:
        return base
    e_inf = np.array([r.e_inf for r in used], dtype=float)
    slope = np.array([r.slope for r in used], dtype=float)
    gap = np.array([r.remaining_gap for r in used], dtype=float)
    if len(used) == 1:
        # Mean but no interval: one sample resamples to itself, and a zero-width
        # CI reads as precision. NaN bounds cannot be mistaken for one.
        nan = float("nan")
        return dataclasses.replace(
            base,
            e_inf=(float(e_inf[0]), nan, nan),
            slope=(float(slope[0]), nan, nan),
            remaining_gap=(float(gap[0]), nan, nan),
        )
    # Fixed draw order (e_inf, slope, gap) so a fixed-seed rng is reproducible.
    return dataclasses.replace(
        base,
        e_inf=_boot_ci(e_inf, rng, n_boot),
        slope=_boot_ci(slope, rng, n_boot),
        remaining_gap=_boot_ci(gap, rng, n_boot),
    )


def read_rung_energies(g: zarr.Group) -> tuple[dict[int, float], dict[int, bool]]:
    """Reads one realization group's ladder into (energies, converged) by D.

    Read-only: the rung records were written by `store.write_rung` during the
    §10.3 ladder, so extrapolation never needs to re-run the optimizer.
    """
    energies: dict[int, float] = {}
    flags: dict[int, bool] = {}
    for d in sorted(store.rungs_done(g)):
        record = store.read_attr_dict(store.subgroup(g, f"peps/D{d}"), "record")
        energies[d] = float(record["energy"])
        flags[d] = bool(record["converged"])
    return energies, flags

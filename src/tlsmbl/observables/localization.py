"""Localization length xi (ARCHITECTURE.md §12, reported per §18's definition of done):
log-linear fit of the r-binned connected correlator with a bootstrap CI over disorder.

Pure statistics layer: numpy only, no torch, no state -- the input is exactly what
`ensemble/aggregate.py` already collects from the stored `czz_r` observable, one
`{r: Czz(r)}` dict per certified realization (int keys, float values; `observables/
static.py::measure_static` already pair-averages within each realization).

WHAT IS FITTED, AND WHY (this is a decision, recorded here rather than in an ADR
because it is local to this estimator):

We fit ONE curve, `log <|Czz(r)|>_disorder = a - r/xi`, to the disorder mean of the
per-realization MAGNITUDES -- not the log of the signed disorder mean, and not the
mean of per-realization fits.

- Not the signed disorder mean: `Czz(r)` is signed (dipolar J_ij carry both signs, so
  pairs cancel), and the mean of a signed quantity can be driven to any small value by
  cancellation. Taking a log of that manufactures a decay rate out of a cancellation
  residue. The measured pilot (`runs/pilot_L8.zarr/REPORT.md`, L=8, g_J=1e-3) is the
  worked example: -3.5e-07, +1.3e-09, -2.3e-10 for r=1,2,3 -- the sign alternates and
  a naive log-linear fit still returns a confident-looking xi ~ 0.27.
- Not the mean of per-realization fits: a single realization contributes ~3 r bins,
  each already a small average over pairs, so its slope is noise-dominated, log(0) is
  reachable when a bin cancels, and the mean of per-realization slopes is not the
  slope of the ensemble envelope (Jensen). Fitting the ensemble envelope once and
  bootstrapping the realization list is the honest split: the point estimate is an
  ensemble property, the CI carries the disorder uncertainty.
- The sign structure is not discarded: sign alternation of the signed disorder mean
  across the fitted r window is a refusal condition (`sign_inconsistent`) precisely
  because it says the envelope is a cancellation residue.

REFUSAL, NOT INVENTION. `fit_xi` never raises and never emits a bare number: it
returns an `XiFit` whose `ok` flag, machine-readable `reason` code and human-readable
`detail` say why a length could not be resolved. In the regime this solver actually
runs in (deep localized, correlators at the contraction noise floor) refusing is the
expected outcome, and "xi unresolved at L=8, g_J=1e-3" is a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

_BOOT = 10_000  # same convention as ensemble/aggregate.py::_boot_ci

# Default noise floor on |Czz(r)|. Czz is a DIFFERENCE of O(1) expectation values
# (<z z> - <z><z>; q_EA ~ 0.83 in the pilot, so each term is O(1)), hence its absolute
# error is set by the absolute accuracy of a boundary-MPS expectation value, i.e. by
# `env.eps_env` (1e-8 in configs/pilot_L8.yaml and configs/benchmark.yaml) -- NOT by
# the much smaller INV-1 diagnostics (pilot: discarded weight 1.7e-17, up/down gap
# 7.1e-15), which bound the truncation of an already-normalized contraction, not the
# cancellation error of the subtraction. A |Czz| at or below this scale is
# indistinguishable from a contraction artifact, so it carries no decay information.
# Callers with a different `env.eps_env` should pass it explicitly.
DEFAULT_NOISE_FLOOR = 1.0e-8

_MIN_BINS = 3  # 2 points fit any line exactly: R^2 is then vacuous
_MIN_REALIZATIONS = 2  # a 1-realization bootstrap returns a zero-width, false CI
_XI_MIN = 0.5  # below half a lattice spacing there is nothing to call a length
_R2_MIN = 0.9  # a log-linear fit this poor is not an exponential
_BOOT_MIN_FRAC = 0.5  # fraction of resamples that must yield a decaying fit

XiReason = Literal[
    "ok",
    "no_data",  # empty input, or no r bin shared by all realizations
    "too_few_realizations",  # < _MIN_REALIZATIONS: bootstrap would be degenerate
    "too_few_bins",  # < _MIN_BINS usable r bins
    "at_noise_floor",  # bins dropped at the noise floor left too few
    "sign_inconsistent",  # signed disorder mean alternates across the fit window
    "non_monotone",  # magnitude envelope rises with r
    "no_decay",  # fitted slope >= 0 (or a flat envelope)
    "poor_fit",  # R^2 < _R2_MIN: not log-linear
    "xi_unresolved_large",  # xi > max fitted r: no decay resolved at this L
    "xi_below_lattice",  # xi < _XI_MIN: faster than one lattice spacing
    "bootstrap_unstable",  # too few resamples produced a decaying fit
]


@dataclass(frozen=True)
class XiFit:
    """Verdict-carrying result of the xi fit. `ok` false => xi/ci/r2 are None."""

    ok: bool
    reason: XiReason
    detail: str
    xi: float | None
    ci: tuple[float, float] | None  # 2.5/97.5 bootstrap quantiles over realizations
    r2: float | None  # coefficient of determination of the log-linear fit
    residual_rms: float | None  # rms fit residual, in units of log|Czz|
    r_used: tuple[int, ...]  # r bins actually fitted
    r_available: tuple[int, ...]  # r bins shared by all realizations
    n_realizations: int
    noise_floor: float
    n_boot: int
    boot_frac_ok: float  # fraction of resamples that yielded a decaying fit

    def summary_line(self) -> str:
        """One-line rendering for REPORT.md (§18), verdict included either way."""
        if not self.ok or self.xi is None or self.ci is None:
            return f"xi: unresolved ({self.reason}) -- {self.detail}"
        return (
            f"xi: {self.xi:.3f} [{self.ci[0]:.3f}, {self.ci[1]:.3f}] "
            f"(R^2={self.r2:.4f}, r bins {list(self.r_used)}, "
            f"{self.n_realizations} realizations)"
        )


def _fail(
    reason: XiReason,
    detail: str,
    *,
    r_used: tuple[int, ...] = (),
    r_available: tuple[int, ...] = (),
    n_realizations: int = 0,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    n_boot: int = 0,
    r2: float | None = None,
    residual_rms: float | None = None,
    xi: float | None = None,
) -> XiFit:
    """Not-ok result. `xi`/`r2` may still be populated: a rejected fit's numbers are
    kept for the audit trail, and `ok=False` is what callers must branch on."""
    return XiFit(
        ok=False,
        reason=reason,
        detail=detail,
        xi=xi,
        ci=None,
        r2=r2,
        residual_rms=residual_rms,
        r_used=r_used,
        r_available=r_available,
        n_realizations=n_realizations,
        noise_floor=noise_floor,
        n_boot=n_boot,
        boot_frac_ok=0.0,
    )


def _ols_slopes(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Least-squares slopes of y = a + slope*x, one per row of y (y is (n, k)).

    Closed form rather than lstsq: it is exact for a straight line, and the bootstrap
    needs 10^4 fits over the same 3-5 abscissae without per-row Python overhead.
    """
    xc = x - x.mean()
    sxx = float(xc @ xc)
    return np.asarray((y @ xc) / sxx, dtype=np.float64)


def fit_xi(
    czz_per_realization: list[dict[int, float]],
    *,
    rng: np.random.Generator,
    n_boot: int = _BOOT,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    r2_min: float = _R2_MIN,
    xi_min: float = _XI_MIN,
) -> XiFit:
    """Fit log<|Czz(r)|> = a - r/xi over disorder, with a bootstrap CI (§12).

    `czz_per_realization` is one `{r: Czz(r)}` dict per CERTIFIED realization (the
    caller decides certification; this layer does no filtering). Only r bins present
    in every realization are used -- averaging over a varying realization subset would
    make the envelope's bins incommensurate.

    Returns an `XiFit`; never raises on data quality. See the module docstring for the
    fit choice and for why refusal is the expected outcome deep in the localized phase.
    """
    n = len(czz_per_realization)
    if n == 0:
        return _fail("no_data", "no realizations supplied", noise_floor=noise_floor)
    if n < _MIN_REALIZATIONS:
        return _fail(
            "too_few_realizations",
            f"{n} realization(s): a bootstrap CI over disorder needs >= "
            f"{_MIN_REALIZATIONS}",
            n_realizations=n,
            noise_floor=noise_floor,
        )

    shared = set(czz_per_realization[0])
    for d in czz_per_realization[1:]:
        shared &= set(d)
    r_all = tuple(sorted(int(r) for r in shared))
    if not r_all:
        return _fail(
            "no_data",
            "no r bin is present in every realization",
            n_realizations=n,
            noise_floor=noise_floor,
        )

    # (n_realizations, n_bins) signed correlator table; magnitudes are what we fit.
    signed = np.array(
        [[float(d[r]) for r in r_all] for d in czz_per_realization], dtype=np.float64
    )
    mag_mean = np.abs(signed).mean(axis=0)
    signed_mean = signed.mean(axis=0)

    # Noise floor: keep the leading contiguous run of bins above it. The TAIL of an
    # exponential legitimately sinks to the floor and is simply uninformative; a floor
    # hit strictly inside the retained window would instead mean the envelope is not
    # clean, and the prefix rule refuses that by truncating there.
    above = mag_mean > noise_floor
    k = int(np.argmin(above)) if not above.all() else len(r_all)
    r_used = r_all[:k]
    if len(r_used) < _MIN_BINS:
        floored = int((~above).sum())
        if floored:
            return _fail(
                "at_noise_floor",
                f"only {len(r_used)} of {len(r_all)} r bins have <|Czz|> above the "
                f"noise floor {noise_floor:.1e} "
                f"({', '.join(f'r={r}: {v:.3e}' for r, v in zip(r_all, mag_mean, strict=True))})",
                r_used=r_used,
                r_available=r_all,
                n_realizations=n,
                noise_floor=noise_floor,
            )
        return _fail(
            "too_few_bins",
            f"{len(r_all)} r bin(s) available, need >= {_MIN_BINS} for a fit with a "
            "meaningful residual",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
        )

    mag = mag_mean[:k]
    sgn = np.sign(signed_mean[:k])
    if len(set(sgn.tolist())) > 1:
        return _fail(
            "sign_inconsistent",
            "signed disorder-mean Czz(r) changes sign across the fit window "
            f"({', '.join(f'r={r}: {v:+.3e}' for r, v in zip(r_used, signed_mean[:k], strict=True))})"
            ": the magnitude envelope would be a cancellation residue, not a decay",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
        )

    if bool((np.diff(mag) > 0).any()):
        return _fail(
            "non_monotone",
            "<|Czz(r)|> is not non-increasing in r "
            f"({', '.join(f'r={r}: {v:.3e}' for r, v in zip(r_used, mag, strict=True))})"
            ": no decay envelope to fit",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
        )

    x = np.asarray(r_used, dtype=np.float64)
    y = np.log(mag)
    slope = float(_ols_slopes(x, y[None, :])[0])
    # Unweighted in log space: with 3-5 bins the per-bin log uncertainties are
    # comparable, and the disorder uncertainty is supplied by the bootstrap, not by
    # per-bin weights.
    ss_tot = float(((y - y.mean()) ** 2).sum())
    resid = y - (y.mean() + slope * (x - x.mean()))
    ss_res = float((resid**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    rms = float(np.sqrt(ss_res / len(x)))

    if slope >= 0.0:
        return _fail(
            "no_decay",
            f"fitted log-slope {slope:+.3e} >= 0: the envelope does not decay over "
            f"r in {list(r_used)}",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
            r2=r2,
            residual_rms=rms,
        )
    xi = -1.0 / slope

    if r2 < r2_min:
        return _fail(
            "poor_fit",
            f"log-linear R^2 {r2:.4f} < {r2_min:.2f} (rms residual {rms:.3f} in "
            "log|Czz|): the envelope is not exponential over the fitted range",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
            r2=r2,
            residual_rms=rms,
            xi=xi,
        )
    # Resolvability, both ends: xi comparable to the fitted r span means the data are
    # consistent with no decay at this L; xi below half a lattice spacing means the
    # correlator dies within one bin and the "length" is an extrapolation.
    r_span = float(max(r_used))
    if xi > r_span:
        return _fail(
            "xi_unresolved_large",
            f"xi {xi:.3f} exceeds the largest fitted separation r={int(r_span)}: no "
            "decay resolved at this L (need a larger L or R_c)",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
            r2=r2,
            residual_rms=rms,
            xi=xi,
        )
    if xi < xi_min:
        return _fail(
            "xi_below_lattice",
            f"xi {xi:.3f} < {xi_min} lattice spacings: the correlator decays faster "
            "than the bin spacing, so the slope is not a resolvable length",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
            r2=r2,
            residual_rms=rms,
            xi=xi,
        )

    # Bootstrap over REALIZATIONS (rows), refitting the same r window each resample:
    # the window is part of the estimator's definition, so re-selecting it per resample
    # would fold window flicker into the CI. Same _BOOT/np.random.Generator convention
    # as aggregate.py::_boot_ci; index resampling because rows, not scalars, resample.
    mags = np.abs(signed[:, :k])
    idx = rng.choice(n, size=(n_boot, n), replace=True)
    boot_mag = mags[idx].mean(axis=1)  # (n_boot, k)
    ok_rows = (boot_mag > noise_floor).all(axis=1)
    slopes = np.full(n_boot, np.nan, dtype=np.float64)
    if bool(ok_rows.any()):
        slopes[ok_rows] = _ols_slopes(x, np.log(boot_mag[ok_rows]))
    good = np.isfinite(slopes) & (slopes < 0.0)
    frac = float(good.mean())
    if frac < _BOOT_MIN_FRAC:
        return _fail(
            "bootstrap_unstable",
            f"only {frac:.1%} of {n_boot} resamples produced a decaying above-floor "
            f"envelope (need >= {_BOOT_MIN_FRAC:.0%}): the point fit is not robust to "
            "the realization sample",
            r_used=r_used,
            r_available=r_all,
            n_realizations=n,
            noise_floor=noise_floor,
            r2=r2,
            residual_rms=rms,
            xi=xi,
        )
    xi_boot = -1.0 / slopes[good]
    ci = (float(np.quantile(xi_boot, 0.025)), float(np.quantile(xi_boot, 0.975)))

    return XiFit(
        ok=True,
        reason="ok",
        detail=(
            f"log-linear fit over r in {list(r_used)}, {n} realizations, "
            f"noise floor {noise_floor:.1e}"
        ),
        xi=xi,
        ci=ci,
        r2=r2,
        residual_rms=rms,
        r_used=r_used,
        r_available=r_all,
        n_realizations=n,
        noise_floor=noise_floor,
        n_boot=n_boot,
        boot_frac_ok=frac,
    )

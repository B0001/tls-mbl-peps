"""Tier-2 decoherence estimates (ARCHITECTURE.md §12) -- model-dependent.

Every model input is echoed verbatim into the output and the fixed DISCLAIMER
travels with it: the static solver cannot derive the TLS relaxation rates
gamma_i; they come from the declared phenomenological model
gamma_i = gamma0 * (E_i/W)^3 * coth(E_i / 2T). This is the only module (with
io/) allowed to convert back to physical units (§2).

§12 asks for three Tier-2 outputs. All three are here:

1. **Effective fluctuator table** (`fluctuator_table`) -- per site, the physical-frame
   splitting E_i and the transverse weight. Both weight definitions are reported and
   *labeled*: `WEIGHT_BARE` is Delta~_i/E_i from the fields alone, `WEIGHT_STATE` is
   taken from the measured local polarization when the caller supplies the certified
   state's <sigma^x>/<sigma^z>. They coincide for a decoupled TLS in its local ground
   state and diverge exactly insofar as the interacting state is not a product -- which
   is why the label travels with the number instead of being inferred.

2. **Gamma_1** (`gamma_1`) -- unchanged public behaviour, now sharing one E_i definition
   with the table via `splittings` rather than recomputing the formula.

3. **Spectral-diffusion proxy** (`spectral_diffusion`) -- variance of the qubit
   dispersive shift over the thermally-active fluctuator ensemble. §12 requires the
   formula to be documented here, so, explicitly:

       chi_i   = g_i^2 * (E_i - omega_q) / ((E_i - omega_q)^2 + gamma_i^2)
       Var     = sum_i (2 chi_i)^2 * p_i (1 - p_i),   p_i = 1 / (1 + exp(E_i / T))

   chi_i is the *dispersive* (real, dispersive-limit) shift, the counterpart of the
   absorptive Lorentzian in Gamma_1. Its denominator is regularized by the SAME declared
   linewidth gamma_i that Gamma_1 uses -- not by an arbitrary epsilon -- so a fluctuator
   sitting exactly on resonance (E_i = omega_q) gives chi_i = 0 and its whole weight
   appears in Gamma_1 instead, which is the physically correct split rather than a
   divergence papered over. A fluctuator switches between shifts +/-chi_i, so the shift
   jumps by 2*chi_i, and independent telegraph switchers contribute
   (2 chi_i)^2 p_i(1-p_i) each. p_i(1-p_i) = 1/(4 cosh^2(E_i/2T)) is the thermal
   activity factor: it vanishes for E_i >> T (frozen fluctuators contribute nothing),
   so the sum is automatically restricted to the thermally-active ensemble.

RIGOROUS vs DECLARED. E_i and the bare transverse weight are functionals of the sampled
disorder; `WEIGHT_STATE` is a functional of the certified variational state. Everything
downstream of g0/gamma0/T/omega_q -- gamma_i, Gamma_1, chi_i, the variance -- is
model-dependent and carries DISCLAIMER. Tier-1 (`observables/static.py`) is the rigorous
output; nothing here may be read as certified.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from tlsmbl.core.types import DisorderRealization, Site

DISCLAIMER = (
    "Tier-2 output: Gamma_1 depends on the declared phenomenological inputs "
    "(g0, gamma0, T, omega_q) echoed in this record; the static solver cannot "
    "derive TLS relaxation rates. Tier-1 observables are the rigorous outputs."
)


WEIGHT_BARE = "bare_field"  # Delta~_i / E_i, from the sampled fields alone
WEIGHT_STATE = "measured_state"  # from the certified state's local polarization


@dataclass(frozen=True)
class Tier2Inputs:
    omega_q: float  # qubit frequency, units of W
    g0: float  # coupling scale, units of W
    gamma0: float  # relaxation scale, units of W
    T: float  # temperature, units of W
    # The config carries these as unit-tagged STRINGS (config.Tier2Config: "5.0GHz"),
    # which §12 requires echoed verbatim. Converting a physical frequency into units of W
    # needs W itself in that unit, which the solver does not carry (W == 1.0 internally,
    # §2), so this layer refuses to guess: `raw` preserves whatever the config said and
    # travels into every record next to the floats actually used.
    raw: dict[str, str] | None = None

    def __post_init__(self) -> None:
        # A non-positive T makes coth and the thermal occupation meaningless rather than
        # merely inaccurate, and a non-positive gamma0 removes the linewidth that
        # regularizes the dispersive denominator. Refuse at construction (INV-style gate)
        # instead of returning a nan from deep inside a sum.
        if not self.T > 0.0:
            raise ValueError(f"Tier-2 T must be > 0 (got {self.T})")
        if not self.gamma0 > 0.0:
            raise ValueError(f"Tier-2 gamma0 must be > 0 (got {self.gamma0})")
        for name in ("omega_q", "g0"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Tier-2 {name} must be finite")


class Tier2InputsUnavailable(ValueError):
    """Raised when `observables.tier2.enabled` is set but the declared inputs cannot be
    used as given. Never downgraded to a warning: Tier-2 with guessed inputs is worse
    than no Tier-2, because the number looks the same either way."""


def inputs_from_config(
    omega_q: str | None, g0: str | None, gamma0: str | None, T: str | None
) -> Tier2Inputs:
    """Builds `Tier2Inputs` from `config.Tier2Config`'s unit-tagged strings (§13).

    The config deliberately keeps these as strings so §12's "echoed verbatim" is
    literally true. Here they must become floats *in units of W*, and that is where this
    function refuses rather than guesses:

    - A bare numeric string ("0.5", "1e-3") is taken as already dimensionless, i.e. in
      units of W. That is the only reading available -- the solver sets W == 1.0
      internally (§2) and never carries W in SI.
    - A unit-tagged string ("5.0GHz", "50mK") cannot be converted without W in that same
      unit, which no layer of this solver holds. Converting would require inventing a W,
      so this raises `Tier2InputsUnavailable` naming the offending field. The fix is for
      the caller to declare inputs in units of W, or for a future ingestion layer (§2's
      "converted exactly once, at ingestion") to supply W.

    Whatever the strings said is preserved in `raw` and echoed into every record.
    """
    fields = {"omega_q": omega_q, "g0": g0, "gamma0": gamma0, "T": T}
    missing = sorted(k for k, v in fields.items() if v is None)
    if missing:
        raise Tier2InputsUnavailable(
            f"observables.tier2.enabled requires {', '.join(missing)}; Tier-2 estimates "
            "have no defaults because every input is a declared model choice (§12)"
        )
    values: dict[str, float] = {}
    for name, text in fields.items():
        assert text is not None
        try:
            values[name] = float(text)
        except ValueError:
            raise Tier2InputsUnavailable(
                f"observables.tier2.{name} = {text!r} carries a physical unit, but "
                "converting it to units of W needs W in that unit and the solver does "
                "not carry one (W == 1.0 internally, §2). Declare Tier-2 inputs in "
                "units of W, or add an ingestion layer that supplies W."
            ) from None
    return Tier2Inputs(
        omega_q=values["omega_q"],
        g0=values["g0"],
        gamma0=values["gamma0"],
        T=values["T"],
        raw={k: v for k, v in fields.items() if v is not None},
    )


@dataclass(frozen=True)
class Fluctuator:
    """One row of §12's effective fluctuator table."""

    site: Site  # (x, y), matching observables/static.py's keying
    E: float  # physical-frame splitting, units of W
    transverse_weight: float
    weight_kind: Literal["bare_field", "measured_state"]
    gamma: float  # declared relaxation rate (model-dependent)
    activity: float  # p_i(1 - p_i), the thermal activity factor
    detuning: float  # E_i - omega_q


@dataclass(frozen=True)
class Gamma1Estimate:
    gamma_1: float  # units of W
    inputs: Tier2Inputs  # echoed verbatim
    n_fluctuators: int
    disclaimer: str


@dataclass(frozen=True)
class SpectralDiffusion:
    """Variance of the qubit dispersive shift over the thermally-active ensemble."""

    variance: float  # units of W^2
    rms: float  # sqrt(variance), units of W
    n_active: int  # fluctuators with activity above _ACTIVE_EPS
    inputs: Tier2Inputs
    disclaimer: str


_ACTIVE_EPS = 1e-300  # "contributes at all" -- only excludes exact underflow


def splittings(real: DisorderRealization) -> np.ndarray:
    """Physical-frame splittings E_i = sqrt((eps_i + h_mf_i)^2 + Delta~_i^2), shape (L, L),
    indexed [y, x] like every array on DisorderRealization.

    Single definition shared by the whole module -- `gamma_1` used to inline it, and a
    second copy in the fluctuator table would be a formula waiting to drift. Note
    E_i >= |Delta~_i| >= delta_min > 0 structurally, so E_i cannot vanish and the
    Delta~_i/E_i weight is always well defined; the guard below is for corrupt input,
    not for the physical regime.
    """
    E = np.sqrt((real.eps + real.h_mf) ** 2 + real.delta**2)
    if not np.all(np.isfinite(E)) or np.any(E <= 0.0):
        raise ValueError(
            "non-finite or non-positive splitting E_i: eps/delta/h_mf are corrupt "
            "(E_i >= delta_min > 0 must hold by construction)"
        )
    return np.asarray(E, dtype=np.float64)


def _gamma(E: np.ndarray, inputs: Tier2Inputs, W: float) -> np.ndarray:
    """Declared model: gamma_i = gamma0 (E_i/W)^3 coth(E_i / 2T).

    coth via 1/tanh. tanh underflows to 0 only for an argument of exactly 0, which
    E_i > 0 and T > 0 exclude; for a large argument tanh saturates at 1 and coth -> 1,
    so both limits are stable without clamping.
    """
    return np.asarray(
        inputs.gamma0 * (E / W) ** 3 / np.tanh(E / (2.0 * inputs.T)), dtype=np.float64
    )


def _activity(E: np.ndarray, T: float) -> np.ndarray:
    """Thermal activity factor p_i(1 - p_i) for p_i = 1/(1 + exp(E_i/T)).

    Evaluated as u/(1+u)^2 with u = exp(-E_i/T). This is algebraically exact -- p = u/(1+u)
    and 1-p = 1/(1+u) -- and is the only one of the three obvious forms that is stable at
    both ends:

    - `p*(1-p)` with p = 1/(1+exp(E/T)) overflows exp for E/T > 709 and then evaluates
      0*(1-0) through an inf.
    - `1/(4 cosh^2(E/2T))` overflows cosh for E/2T > 710, and *clamping the argument* to
      avoid that is worse than the overflow: it floors the activity at a spurious
      constant (~1e-261) instead of letting it decay, so two very different temperatures
      return identical answers. Found by test_guards_fire_instead_of_producing_inf_or_nan,
      which measured T=1e-8 and T=1e-4 giving the same variance.
    - u/(1+u)^2 underflows u to 0.0 for E/T > ~745 and then returns exactly 0.0, which is
      the physically right answer: a fluctuator with E >> T is frozen and cannot switch.
    """
    u = np.exp(-E / T)
    return np.asarray(u / (1.0 + u) ** 2, dtype=np.float64)


def fluctuator_table(
    real: DisorderRealization,
    inputs: Tier2Inputs,
    *,
    sx: dict[Site, float] | None = None,
    sz: dict[Site, float] | None = None,
) -> tuple[Fluctuator, ...]:
    """§12's effective fluctuator table, one row per site in row-major (s = y*L + x) order.

    Pass `sx`/`sz` (from `observables/static.py::StaticObservables`) to report the
    transverse weight from the CERTIFIED STATE's local polarization,
    |<sigma^x>| / sqrt(<sigma^x>^2 + <sigma^z>^2), labeled `WEIGHT_STATE`. Omit them and
    the bare-field weight Delta~_i/E_i is reported, labeled `WEIGHT_BARE`. Both are
    supported because they answer different questions; neither is silently substituted
    for the other, and no state-dependent number is fabricated when no state is given.
    """
    if (sx is None) != (sz is None):
        raise ValueError("pass both sx and sz, or neither: a weight needs both components")
    E = splittings(real)
    gam = _gamma(E, inputs, real.params.W)
    act = _activity(E, inputs.T)
    L = real.params.L
    rows: list[Fluctuator] = []
    for y in range(L):  # row-major: s = y*L + x, site 0 first (§6)
        for x in range(L):
            site: Site = (x, y)
            if sx is None or sz is None:
                weight, kind = float(real.delta[y, x] / E[y, x]), WEIGHT_BARE
            else:
                px, pz = float(sx[site]), float(sz[site])
                norm = math.hypot(px, pz)
                # A fully depolarized local state has no polarization direction to read
                # a weight off; report 0.0 rather than 0/0.
                weight = abs(px) / norm if norm > 0.0 else 0.0
                kind = WEIGHT_STATE
            rows.append(
                Fluctuator(
                    site=site,
                    E=float(E[y, x]),
                    transverse_weight=weight,
                    weight_kind=kind,  # type: ignore[arg-type]
                    gamma=float(gam[y, x]),
                    activity=float(act[y, x]),
                    detuning=float(E[y, x] - inputs.omega_q),
                )
            )
    return tuple(rows)


def gamma_1(real: DisorderRealization, inputs: Tier2Inputs) -> Gamma1Estimate:
    """Gamma_1(omega_q) = sum_i 2 g_i^2 gamma_i / (gamma_i^2 + (E_i - omega_q)^2),
    g_i = g0 * (Delta~_i / E_i), gamma_i = gamma0 (E_i/W)^3 coth(E_i / 2T)."""
    E = splittings(real)
    gam = _gamma(E, inputs, real.params.W)
    gi = inputs.g0 * (real.delta / E)
    total = float(
        np.sum(2.0 * gi**2 * gam / (gam**2 + (E - inputs.omega_q) ** 2))
    )
    return Gamma1Estimate(
        gamma_1=total,
        inputs=inputs,
        n_fluctuators=E.size,
        disclaimer=DISCLAIMER,
    )


def spectral_diffusion(
    real: DisorderRealization, inputs: Tier2Inputs
) -> SpectralDiffusion:
    """Variance of the qubit dispersive shift; formula in the module docstring.

    The resonant denominator is regularized by the declared linewidth gamma_i, so
    E_i = omega_q gives chi_i = 0 (all that fluctuator's weight is absorptive and shows
    up in Gamma_1) instead of a divergence.
    """
    E = splittings(real)
    gam = _gamma(E, inputs, real.params.W)
    act = _activity(E, inputs.T)
    gi = inputs.g0 * (real.delta / E)
    det = E - inputs.omega_q
    chi = gi**2 * det / (det**2 + gam**2)
    var = float(np.sum((2.0 * chi) ** 2 * act))
    return SpectralDiffusion(
        variance=var,
        rms=math.sqrt(var),
        n_active=int(np.count_nonzero(act > _ACTIVE_EPS)),
        inputs=inputs,
        disclaimer=DISCLAIMER,
    )


@dataclass(frozen=True)
class Tier2Record:
    """One realization's complete Tier-2 output, JSON-able for zarr attrs (§11).

    `tier` is a literal "2" so no consumer can mistake this for a Tier-1 certified
    observable, and DISCLAIMER is present at the top level as well as inside the nested
    estimates. The full fluctuator table is summarized rather than dumped: L=16 is 256
    rows per realization, which does not belong in run attrs. `table` carries the
    extremes that matter for a decoherence readout (the most strongly coupled and the
    most nearly resonant fluctuator) plus the aggregate weights.
    """

    tier: Literal[2]
    gamma_1: float
    spectral_diffusion_variance: float
    spectral_diffusion_rms: float
    n_fluctuators: int
    n_active: int
    weight_kind: str
    mean_transverse_weight: float
    max_transverse_weight: float
    min_abs_detuning: float
    inputs: dict[str, Any]  # floats used, plus `raw` echoed verbatim
    disclaimer: str

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = asdict(self)
        # Round-trip through json here so a caller storing this into zarr attrs cannot
        # be surprised by a numpy scalar that survived the dataclass.
        return dict(json.loads(json.dumps(d)))

    def report_lines(self) -> list[str]:
        """Tier-2 block for REPORT.md (§18), inputs echoed and disclaimed."""
        raw = self.inputs.get("raw") or {}
        echoed = ", ".join(f"{k}={v}" for k, v in sorted(raw.items())) or ", ".join(
            f"{k}={self.inputs[k]:.6g}"
            for k in ("omega_q", "g0", "gamma0", "T")
            if k in self.inputs
        )
        return [
            f"- Gamma_1(omega_q): {self.gamma_1:.6e} (units of W)",
            f"- spectral-diffusion rms shift: {self.spectral_diffusion_rms:.6e} "
            f"(variance {self.spectral_diffusion_variance:.6e})",
            f"- fluctuators: {self.n_fluctuators} ({self.n_active} thermally active); "
            f"transverse weight [{self.weight_kind}] mean "
            f"{self.mean_transverse_weight:.4f}, max {self.max_transverse_weight:.4f}",
            f"- closest resonance |E_i - omega_q|: {self.min_abs_detuning:.6e}",
            f"- declared model inputs (echoed verbatim): {echoed}",
            f"- {self.disclaimer}",
        ]


def tier2_record(
    real: DisorderRealization,
    inputs: Tier2Inputs,
    *,
    sx: dict[Site, float] | None = None,
    sz: dict[Site, float] | None = None,
) -> Tier2Record:
    """Assembles the full Tier-2 record for one realization."""
    table = fluctuator_table(real, inputs, sx=sx, sz=sz)
    g1 = gamma_1(real, inputs)
    sd = spectral_diffusion(real, inputs)
    weights = [f.transverse_weight for f in table]
    return Tier2Record(
        tier=2,
        gamma_1=g1.gamma_1,
        spectral_diffusion_variance=sd.variance,
        spectral_diffusion_rms=sd.rms,
        n_fluctuators=len(table),
        n_active=sd.n_active,
        weight_kind=table[0].weight_kind if table else WEIGHT_BARE,
        mean_transverse_weight=float(np.mean(weights)) if weights else 0.0,
        max_transverse_weight=float(np.max(weights)) if weights else 0.0,
        min_abs_detuning=min(abs(f.detuning) for f in table) if table else float("nan"),
        inputs={
            "omega_q": inputs.omega_q,
            "g0": inputs.g0,
            "gamma0": inputs.gamma0,
            "T": inputs.T,
            "raw": dict(inputs.raw) if inputs.raw else None,
        },
        disclaimer=DISCLAIMER,
    )

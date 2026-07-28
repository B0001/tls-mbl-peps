#!/usr/bin/env python3
"""tlsmbl kernel microbenchmark (deliverable D3, architecture §15/§20).

Prototypes kernels/svd.py (ExactSVD) and kernels/rsvd.py (SketchedSVD) per
architecture §8.3/§8.6, with INV-7 (finite guard), INV-9 (gauge fix) and the
INV-3 posterior gate implemented. Operand: the zip-up matricization E-5,
W in C^{(chi*D^2) x (chi*D^2)} at chi = D^2, i.e. n = D^4.

Modes:
  timing D [D ...]      time both backends on Gaussian operands (spectrum-
                        independent cost); appends JSONL records
  timing-sketched D...  sketched backend only (large D where exact is skipped)
  validate              spectral-family validation of the INV-3 gate + accuracy
                        vs exact at D=4 and D=6
  report                aggregate JSONL -> fits, speedups, final JSON + table

NOTE: numpy (scipy-openblas64) stands in for torch in this container; both
dispatch to the same LAPACK zgesdd / BLAS zgemm kernel class, so measured
D-scaling exponents transfer. Constant factors will differ from torch builds.
"""
import json, math, os, platform, sys, time
import numpy as np
from pathlib import Path as _P
_RESULTS = _P(__file__).resolve().parent / "results"; _RESULTS.mkdir(exist_ok=True)

JSONL = str(_RESULTS / "bench_records.jsonl")
FINAL = str(_RESULTS / "kernel-bench-results.json")
P_OVER, Q_POW, ETA, S_PROBES = 8, 1, 1e-6, 6
REPEATS = {2: 9, 3: 9, 4: 7, 5: 5, 6: 3, 7: 1, 8: 3}
MASTER_SEED = 20260716


def crandn(rng, *shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def svd_gauge_fixed(A):
    """INV-9: largest-modulus entry of each U column rotated real-positive."""
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    idx = np.abs(U).argmax(axis=0)
    ph = U[idx, np.arange(U.shape[1])]
    ph = ph / np.maximum(np.abs(ph), 1e-300)
    return U * ph.conj()[None, :], S, Vh * ph[:, None]


def exact_truncate(W, chi):
    if not np.isfinite(W).all():
        raise ValueError("INV-7: non-finite operand")
    U, S, Vh = svd_gauge_fixed(W)
    tot = float((S ** 2).sum())
    disc = float((S[chi:] ** 2).sum() / tot) if tot > 0 else 0.0
    return U[:, :chi], S[:chi], Vh[:chi], disc


def rsvd_truncate(W, chi, rng, p=P_OVER, q=Q_POW, eta=ETA, s_probes=S_PROBES):
    """HMT randomized truncation with INV-3 posterior gate (architecture §8.6)."""
    if not np.isfinite(W).all():
        raise ValueError("INV-7: non-finite operand")
    m, n = W.shape
    k = min(chi + p, min(m, n))
    G = crandn(rng, n, k)
    Y = W @ G
    for _ in range(q):                      # subspace/power iteration w/ re-orth
        Q0, _ = np.linalg.qr(Y)
        Y = W @ (W.conj().T @ Q0)
    Q, _ = np.linalg.qr(Y)
    B = Q.conj().T @ W
    Ub, S, Vh = svd_gauge_fixed(B)
    U = Q @ Ub
    Om = crandn(rng, n, s_probes)           # posterior estimate, INV-3
    WOm = W @ Om
    R = WOm - Q @ (Q.conj().T @ WOm)
    est = float(10.0 * np.sqrt(2.0 / np.pi) * np.linalg.norm(R, axis=0).max())
    # Gate v2 (ADR-009): two-sided. Sketch quality is judged against the best
    # any rank-chi truncation could do (c_gate * sigma_hat_{chi+1}, estimated
    # from the sketch's own spectrum since k > chi); whether rank chi itself
    # is adequate is INV-1's job (disc weight -> chi escalation).
    c_gate = 10.0
    sig_floor = float(S[chi]) if len(S) > chi else 0.0
    thresh = max(eta * float(S[0]), c_gate * sig_floor)
    passed = bool(est <= thresh)
    return U[:, :chi], S[:chi], Vh[:chi], est, passed, thresh


def _time(fn, warmup, reps):
    if warmup:
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def append(rec):
    with open(JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def mode_timing(Ds, sketched_only=False):
    for D in Ds:
        chi, n = D * D, D ** 4
        rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(D)[-1])
        W = crandn(rng, n, n)               # spectrum-independent timing operand
        reps = REPEATS[D]
        warm = D <= 6
        if not sketched_only:
            ts = _time(lambda: exact_truncate(W, chi), warm, reps)
            append({"mode": "timing", "backend": "exact", "D": D, "n": n, "chi": chi,
                    "times_s": ts, "median_s": float(np.median(ts)), "min_s": float(min(ts))})
            print(f"D={D} n={n} exact   median={np.median(ts):.4g}s (x{reps})", flush=True)
        ts = _time(lambda: rsvd_truncate(W, chi, rng), warm, reps)
        gate = rsvd_truncate(W, chi, rng)[4]
        append({"mode": "timing", "backend": "sketched", "D": D, "n": n, "chi": chi,
                "times_s": ts, "median_s": float(np.median(ts)), "min_s": float(min(ts)),
                "gate_passed_on_gaussian": gate})
        print(f"D={D} n={n} sketched median={np.median(ts):.4g}s (x{reps}) gate_on_gaussian={gate}", flush=True)
        del W


def _spectrum_matrix(rng, n, sig):
    U, _ = np.linalg.qr(crandn(rng, n, n))
    V, _ = np.linalg.qr(crandn(rng, n, n))
    return (U * sig[None, :]) @ V.conj().T


def mode_validate():
    out = []
    for D in (4, 6):
        chi, n = D * D, D ** 4
        rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED + 1).spawn(D)[-1])
        for fam, sig in (("localized_exp", np.exp(-0.5 * np.arange(n))),
                         ("slow_powerlaw", 1.0 / (1.0 + np.arange(n)))):
            W = _spectrum_matrix(rng, n, sig)
            Ue, Se, Vhe, disc = exact_truncate(W, chi)
            Us, Ss, Vhs, est, passed, thresh = rsvd_truncate(W, chi, rng)
            rec = {"mode": "validate", "gate": "v2_ADR-009", "D": D, "n": n,
                   "chi": chi, "family": fam,
                   "posterior_est": est, "gate_threshold": thresh,
                   "gate_passed": passed,
                   "exact_disc_weight": disc,
                   "optimal_rel_err": float(sig[chi] / sig[0])}
            if passed:
                R = W - (Us * Ss[None, :]) @ Vhs
                rec["sketched_rel_err"] = float(np.linalg.svd(R, compute_uv=False)[0] / sig[0])
                pa = np.linalg.svd(Ue.conj().T @ Us, compute_uv=False)
                rec["max_principal_angle_deg"] = float(np.degrees(np.arccos(np.clip(pa.min(), 0, 1))))
                rec["action"] = "sketched result accepted"
            else:
                rec["action"] = "INV-3 fallback to ExactSVD (production behavior)"
            out.append(rec)
            append(rec)
            print(f"D={D} {fam:14s} gate={'PASS' if passed else 'FAIL->fallback'} "
                  f"est={est:.3e} thresh={rec['gate_threshold']:.3e}", flush=True)
    return out


def _fit(recs, backend, Dmin, Dmax):
    pts = sorted((r["D"], r["median_s"]) for r in recs
                 if r["backend"] == backend and Dmin <= r["D"] <= Dmax)
    if len(pts) < 2:
        return None
    x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return {"slope": float(slope), "log_intercept": float(intercept),
            "rms_resid": float(np.sqrt((resid ** 2).mean())),
            "D_range": [pts[0][0], pts[-1][0]], "n_points": len(pts)}


def mode_report():
    recs = [json.loads(l) for l in open(JSONL)]
    timing = [r for r in recs if r["mode"] == "timing"]
    fits = {
        "exact_largeD": _fit(timing, "exact", 4, 8),
        "sketched_largeD": _fit(timing, "sketched", 4, 8),
        "exact_allD": _fit(timing, "exact", 2, 8),
        "sketched_allD": _fit(timing, "sketched", 2, 8),
    }
    gap = (fits["exact_largeD"]["slope"] - fits["sketched_largeD"]["slope"]
           if fits["exact_largeD"] and fits["sketched_largeD"] else None)
    speedups = {}
    for D in sorted({r["D"] for r in timing}):
        te = [r["median_s"] for r in timing if r["D"] == D and r["backend"] == "exact"]
        tsk = [r["median_s"] for r in timing if r["D"] == D and r["backend"] == "sketched"]
        if te and tsk:
            speedups[D] = te[0] / tsk[0]
    proj8 = None
    f = fits["exact_largeD"]
    if f and 8 not in speedups:
        proj8 = math.exp(f["log_intercept"] + f["slope"] * math.log(8))
    final = {
        "meta": {"stack": f"numpy {np.__version__} / scipy-openblas64, "
                          f"python {platform.python_version()}, cores={os.cpu_count()}",
                 "note": "numpy stands in for torch; identical LAPACK/BLAS kernel class, "
                         "exponents transfer, constants differ",
                 "operand": "Gaussian W in C^{D^4 x D^4}, chi=D^2, complex128",
                 "rsvd": {"oversample": P_OVER, "power_iters": Q_POW, "eta": ETA,
                          "probes": S_PROBES},
                 "theory": {"exact_exponent": 12, "sketched_exponent": 10, "gap": 2.0,
                            "ci_gate_T-PERF": "gap >= 1.6"}},
        "timing": timing, "fits": fits, "exponent_gap_largeD": gap,
        "speedup_exact_over_sketched": speedups,
        "projected_exact_D8_s": proj8,
        "validation": [r for r in recs if r["mode"] == "validate"],
    }
    with open(FINAL, "w") as f_:
        json.dump(final, f_, indent=1)
    print(json.dumps({"fits": fits, "gap": gap, "speedups": speedups,
                      "projected_exact_D8_s": proj8}, indent=1))


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "timing":
        mode_timing([int(a) for a in sys.argv[2:]])
    elif m == "timing-sketched":
        mode_timing([int(a) for a in sys.argv[2:]], sketched_only=True)
    elif m == "validate":
        mode_validate()
    elif m == "report":
        mode_report()

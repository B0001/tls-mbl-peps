#!/usr/bin/env python3
"""tlsmbl 4x4 golden battery (T-GOLD-ED at L=4, architecture §14).

Reuses the certified 3x3 modules with L patched to 4 (all machinery late-binds
the module-global L; the only L=3-limited function, peps_statevector, is not
needed here -- sparse-Lanczos ED is the oracle at 2^16).

New at L=4 vs L=3: chi = c*D^2 is genuinely truncating (two-row cut rank D^4),
so this battery validates the certified-approximation stack: disc weights,
INV-2 chi-stability (via the certified numpy engine at chi and 2chi), and the
honest variational statement E >= E0 - env_error.

Modes: ed | consistency | adfd | opt GJ D CHI BUDGET_S MAXITER_CAP
"""
import json, sys, time
import numpy as np
from pathlib import Path as _P
_RESULTS = _P(__file__).resolve().parent / "results"; _RESULTS.mkdir(exist_ok=True)

import golden_3x3 as g3
g3.L = 4
import ad_phase2 as p2
p2.L = 4

import jax
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

L = 4
RES = str(_RESULTS / "phase3-4x4-results.json")


def record(rec):
    try:
        data = json.load(open(RES))
    except FileNotFoundError:
        data = {"records": []}
    data["records"].append(rec)
    json.dump(data, open(RES, "w"), indent=1)


def ed_ground(onsite, pair):
    H = g3.build_H(onsite, pair)
    w, v = eigsh(H, k=1, which="SA")
    resid = float(np.linalg.norm(H @ v[:, 0] - w[0] * v[:, 0]))
    return float(w[0]), resid, H


def mode_ed():
    for gJ in (1e-3, 0.3):
        real, onsite, pair, seed = p2.setup(gJ)
        t0 = time.time()
        E0, resid, H = ed_ground(onsite, pair)
        # J=0 analytic cross-check of the 16-site Kronecker assembly
        E_free, r_free, _ = ed_ground(onsite, [])
        E_ana = -0.5 * float(np.sqrt(real["eps"] ** 2 + real["delta"] ** 2).sum())
        print(f"gJ={gJ}: E0={E0:+.12f}  resid={resid:.2e}  nnz={H.nnz}  "
              f"npair={len(pair)}  J0 vs analytic {abs(E_free-E_ana):.2e}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        record({"mode": "ed", "gJ": gJ, "E0": E0, "residual": resid,
                "nnz": int(H.nnz), "n_pair_terms": len(pair),
                "J0_analytic_err": abs(E_free - E_ana),
                "pass": bool(resid < 1e-6 and abs(E_free - E_ana) < 1e-9)})


def mode_consistency(chi=8):
    real, onsite, pair, seed = p2.setup(0.3)
    D = 2
    A0 = g3.random_peps(D, seed + 7 * D)
    flatten, unflatten, n = p2.make_codec(A0)
    obs = g3.bmps_observables(A0, chi, onsite, pair)
    E_np = obs["E"]
    E_jx = float(p2.make_energy(unflatten, onsite, pair, chi)(jnp.asarray(flatten(A0))))
    rel = abs(E_jx - E_np) / abs(E_np)
    print(f"4x4 consistency D={D} chi={chi}: jax {E_jx:+.12f}  numpy {E_np:+.12f}  "
          f"rel {rel:.3e}  disc={obs['disc']:.2e}  row_cons={obs['row_consistency']:.2e}",
          flush=True)
    record({"mode": "consistency", "D": D, "chi": chi, "rel_err": rel,
            "disc_weight": obs["disc"], "row_consistency": obs["row_consistency"],
            "pass": bool(rel < 1e-11)})


def mode_adfd():
    real, onsite, pair, seed = p2.setup(0.3)
    D, chi, nc = 2, 8, 12
    A0 = g3.random_peps(D, seed + 7 * D)
    flatten, unflatten, n = p2.make_codec(A0)
    x0 = jnp.asarray(flatten(A0))
    energy = p2.make_energy(unflatten, onsite, pair, chi)
    t0 = time.time(); Ej = jax.jit(energy); E = float(Ej(x0))
    t1 = time.time(); g = np.asarray(jax.jit(jax.grad(energy))(x0)); t2 = time.time()
    disc = g3.bmps_observables(A0, chi, onsite, pair)["disc"]
    rng = np.random.default_rng(13)
    idx = rng.choice(n, size=nc, replace=False)
    h, errs = 1e-5, []
    for i in idx:
        xp, xm = np.asarray(x0).copy(), np.asarray(x0).copy()
        xp[i] += h; xm[i] -= h
        fd = (float(Ej(jnp.asarray(xp))) - float(Ej(jnp.asarray(xm)))) / (2 * h)
        errs.append(abs(g[i] - fd) / max(abs(fd), 1e-3 * np.abs(g).max()))
    print(f"4x4 T-AD-FD D={D} chi={chi} disc={disc:.2e}: max rel err = {max(errs):.3e} "
          f"(E compile {t1-t0:.0f}s, grad compile {t2-t1:.0f}s)", flush=True)
    record({"mode": "adfd", "D": D, "chi": chi, "disc_weight": disc,
            "max_rel_err": float(max(errs)), "n_coords": nc,
            "pass": bool(max(errs) < 1e-6)})


def mode_opt(gJ, D, chi, budget_s, cap):
    real, onsite, pair, seed = p2.setup(gJ)
    print("ED...", flush=True)
    E0, resid, _ = ed_ground(onsite, pair)
    A0 = p2.product_init(real, D, seed + 99)
    flatten, unflatten, n = p2.make_codec(A0)
    energy = p2.make_energy(unflatten, onsite, pair, chi)
    t0 = time.time()
    vg = jax.jit(jax.value_and_grad(energy))
    x0 = jnp.asarray(flatten(A0)); _ = vg(x0)
    tc = time.time() - t0
    t1 = time.time()
    for _ in range(3): vg(x0)
    t_iter = (time.time() - t1) / 3
    maxiter = int(min(cap, max(50, budget_s / max(t_iter, 1e-4))))
    print(f"compile {tc:.0f}s, {t_iter*1e3:.0f} ms/iter -> maxiter={maxiter}", flush=True)

    ckpt = {"x": np.asarray(x0)}
    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g)
    def cb(xk):
        ckpt["x"] = np.asarray(xk)
    t2 = time.time()
    res = minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=maxiter, maxcor=20, ftol=1e-15, gtol=1e-12))
    gap = res.fun - E0
    print(f"gJ={gJ} D={D} chi={chi}: E={res.fun:+.12f}  E0={E0:+.12f}  "
          f"gap={gap:.3e}  rel={gap/abs(E0):.3e}  iters={res.nit}  "
          f"({time.time()-t2:.0f}s)", flush=True)

    # INV-2 stability + engine cross-check via the certified numpy engine
    A_fin = unflatten(np.asarray(res.x))
    A_fin = [[np.asarray(t) for t in row] for row in A_fin]
    ob1 = g3.bmps_observables(A_fin, chi, onsite, pair)
    ob2 = g3.bmps_observables(A_fin, 2 * chi, onsite, pair)
    stab = abs(ob1["E"] - ob2["E"])
    eng_x = abs(ob1["E"] - res.fun)
    bound_ok = bool(res.fun >= E0 - max(1e-9, stab))
    print(f"INV-2 |E(chi)-E(2chi)|={stab:.3e}  jax-vs-numpy@chi={eng_x:.3e}  "
          f"disc(chi)={ob1['disc']:.2e} disc(2chi)={ob2['disc']:.2e}  "
          f"E>=E0-env_err: {bound_ok}", flush=True)
    record({"mode": "opt", "gJ": gJ, "D": D, "chi": chi, "E_opt": float(res.fun),
            "E0_ED": E0, "rel_gap": float(gap / abs(E0)), "iters": int(res.nit),
            "t_iter_ms": t_iter * 1e3, "compile_s": tc,
            "inv2_stability": float(stab), "engine_crosscheck": float(eng_x),
            "disc_chi": ob1["disc"], "disc_2chi": ob2["disc"],
            "variational_mod_env": bound_ok})


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "ed":
        mode_ed()
    elif m == "consistency":
        mode_consistency(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif m == "adfd":
        mode_adfd()
    elif m == "opt":
        mode_opt(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                 float(sys.argv[5]), int(sys.argv[6]))

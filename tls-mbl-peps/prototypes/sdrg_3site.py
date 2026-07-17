#!/usr/bin/env python3
"""tlsmbl T-SDRG-3SITE (architecture §14.3, blocking Phase 4).

Two tiers per decimation type:
  Tier I  (sign/coefficient arbiter): the SDRG-rule effective Hamiltonian must
          EQUAL the fixed-denominator Schur block  H_gg - H_ge H_eg / gap
          computed in the exactly-rotated frame, to machine precision.
  Tier II (physics): eigenvalues of the effective 2-body problem must converge
          to the exact low-sector spectrum of the full 8x8 problem with local
          order >= 3 as ALL non-dominant terms are scaled by lambda.

Site decimation (dominant field E_2, center of a 3-chain, incl. a direct J_13
bond to test accumulation) and bond decimation (dominant |J_12|, probe site 3),
the latter for both alignment signs (AF and F).

Coefficients under this codebase's normalization  H = sum (eps/2)Z + (Delta/2)X
+ sum J ZZ  (derived here, asserted by Tier I; they CORRECT the prose of §9):
  site:  eps_j -= 2 J_ij cos(th);  J_jk += -2 J_ij J_ik sin^2(th)/E_i  (j<k);
         E0 += -E_i/2 - sin^2(th) * sum_j J_ij^2 / E_i
  bond:  eps_c = eps_1 + s*eps_2;  J_c3 = J_13 + s*J_23;  s = -sign(J_12)
         Delta_c = -Delta_1*Delta_2 / (2|J_12|)      [sign = tau^x gauge]
         E0 += -|J_12| - (Delta_1^2 + Delta_2^2) / (8|J_12|)
"""
import json
import numpy as np
from pathlib import Path as _P
_RESULTS = _P(__file__).resolve().parent / "results"; _RESULTS.mkdir(exist_ok=True)
import golden_3x3 as g3

I2 = np.eye(2)
Zm = np.diag([1.0, -1.0])
Xm = np.array([[0.0, 1.0], [1.0, 0.0]])
RES = []


def chain(ops):
    return g3.op_chain(ops, 3).toarray()


def H_full(p):
    H = np.zeros((8, 8))
    for s in range(3):
        H += p["eps"][s] / 2 * chain({s: Zm}) + p["dlt"][s] / 2 * chain({s: Xm})
    for (a, b), J in p["J"].items():
        H += J * chain({a: Zm, b: Zm})
    return H


# ---------------- site decimation: dominant field on center site 1 ----------
def site_case(lam, draw):
    rng = np.random.default_rng(draw)
    c = rng.uniform(-0.4, 0.4, size=7)
    p = {"eps": [c[0] * lam, 0.8, c[1] * lam],
         "dlt": [c[2] * lam, 0.6, c[3] * lam],
         "J": {(0, 1): c[4] * lam, (1, 2): c[5] * lam, (0, 2): c[6] * lam}}
    return p


def sdrg_site(p):
    e2, d2 = p["eps"][1], p["dlt"][1]
    E = np.hypot(e2, d2)
    th = np.arctan2(d2, e2)
    J01, J12, J02 = p["J"][(0, 1)], p["J"][(1, 2)], p["J"][(0, 2)]
    eps0 = p["eps"][0] - 2 * J01 * np.cos(th)
    eps2 = p["eps"][2] - 2 * J12 * np.cos(th)
    Jp = J02 - 2 * J01 * J12 * np.sin(th) ** 2 / E
    E0 = -E / 2 - np.sin(th) ** 2 * (J01 ** 2 + J12 ** 2) / E
    Heff = (eps0 / 2 * np.kron(Zm, I2) + p["dlt"][0] / 2 * np.kron(Xm, I2)
            + eps2 / 2 * np.kron(I2, Zm) + p["dlt"][2] / 2 * np.kron(I2, Xm)
            + Jp * np.kron(Zm, Zm) + E0 * np.eye(4))
    return Heff


def schur_site(p):
    E = np.hypot(p["eps"][1], p["dlt"][1])
    th = np.arctan2(p["dlt"][1], p["eps"][1])
    u = np.array([[np.cos(th / 2), -np.sin(th / 2)],
                  [np.sin(th / 2), np.cos(th / 2)]])
    U = np.kron(np.kron(I2, u), I2)
    Ht = U.T @ H_full(p) @ U
    g = [i for i in range(8) if (i >> 1) & 1 == 1]      # sigma~z_1 = -1 sector
    e = [i for i in range(8) if (i >> 1) & 1 == 0]
    Hgg, Hge, Heg = Ht[np.ix_(g, g)], Ht[np.ix_(g, e)], Ht[np.ix_(e, g)]
    return Hgg - Hge @ Heg / E


# ---------------- bond decimation: dominant |J_01|, probe site 2 ------------
def bond_case(lam, sign, draw):
    rng = np.random.default_rng(1000 + draw)
    c = rng.uniform(-0.4, 0.4, size=8)
    p = {"eps": [c[0] * lam, c[1] * lam, c[2] * lam],
         "dlt": [c[3] * lam, c[4] * lam, c[5] * lam],
         "J": {(0, 1): sign * 1.0, (0, 2): c[6] * lam, (1, 2): c[7] * lam}}
    return p


def sdrg_bond(p):
    J = p["J"][(0, 1)]
    s = -np.sign(J)
    aJ = abs(J)
    eps_c = p["eps"][0] + s * p["eps"][1]
    Jc = p["J"][(0, 2)] + s * p["J"][(1, 2)]
    d_c = -p["dlt"][0] * p["dlt"][1] / (2 * aJ)
    E0 = -aJ - (p["dlt"][0] ** 2 + p["dlt"][1] ** 2) / (8 * aJ)
    Heff = (eps_c / 2 * np.kron(Zm, I2) + d_c / 2 * np.kron(Xm, I2)
            + p["eps"][2] / 2 * np.kron(I2, Zm) + p["dlt"][2] / 2 * np.kron(I2, Xm)
            + Jc * np.kron(Zm, Zm) + E0 * np.eye(4))
    return Heff


def schur_bond(p):
    J = p["J"][(0, 1)]
    if J > 0:                                   # AF doublet {|01>,|10>} x probe
        g, e = [2, 3, 4, 5], [0, 1, 6, 7]
    else:                                       # F  doublet {|00>,|11>} x probe
        g, e = [0, 1, 6, 7], [2, 3, 4, 5]
    H = H_full(p)
    Hgg, Hge, Heg = H[np.ix_(g, g)], H[np.ix_(g, e)], H[np.ix_(e, g)]
    return Hgg - Hge @ Heg / (2 * abs(J))


# ---------------- harness ----------------
def run(label, mk_case, sdrg_fn, schur_fn):
    lams = [0.32, 0.16, 0.08, 0.04, 0.02]
    for draw in range(3):
        # Tier I: exact identity at a non-perturbative lambda
        p = mk_case(0.32, draw)
        Hr, Hs = sdrg_fn(p), schur_fn(p)
        idn = np.abs(Hr - Hs).max() / max(np.abs(Hs).max(), 1e-30)
        # Tier II: local convergence order vs full exact
        errs = []
        for lam in lams:
            p = mk_case(lam, draw)
            Hr = sdrg_fn(p)
            w_full = np.linalg.eigvalsh(H_full(p))[:4]
            w_eff = np.linalg.eigvalsh(Hr)
            errs.append(np.abs(np.sort(w_full) - np.sort(w_eff)).max())
        orders = [np.log2(errs[k] / errs[k + 1]) for k in range(len(lams) - 1)]
        rec = {"case": label, "draw": draw, "tier1_identity_rel": float(idn),
               "errs": [float(x) for x in errs],
               "local_orders": [float(o) for o in orders],
               "min_order_small_lam": float(min(orders[-2:])),
               "pass": bool(idn < 1e-12 and min(orders[-2:]) >= 2.9)}
        RES.append(rec)
        print(f"[{'PASS' if rec['pass'] else 'FAIL'}] {label} draw{draw}: "
              f"identity {idn:.2e}  err(0.32)={errs[0]:.2e} -> err(0.02)={errs[-1]:.2e}  "
              f"orders {['%.2f' % o for o in orders]}", flush=True)


if __name__ == "__main__":
    run("site_field", site_case, sdrg_site, schur_site)
    run("bond_AF", lambda l, d: bond_case(l, +1, d), sdrg_bond, schur_bond)
    run("bond_F", lambda l, d: bond_case(l, -1, d), sdrg_bond, schur_bond)
    ok = all(r["pass"] for r in RES)
    json.dump({"all_pass": ok, "records": RES},
              open(str(_RESULTS / "sdrg3site-results.json"), "w"), indent=1)
    print(("ALL PASS" if ok else "FAILURES PRESENT") +
          f" ({sum(r['pass'] for r in RES)}/{len(RES)})")

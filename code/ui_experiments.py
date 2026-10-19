"""Simulation experiments for the universal e-process (evigil_ui) and the failure of independence-based monitors.
Usage (from repository root, PYTHONPATH=code): python code/ui_experiments.py"""
import json, os, sys, time, numpy as np
from scipy.stats import norm
from evigil import load_national, TAU, THRESH
from evigil_ext import projections, log_e_known, log_e_unknown
from evigil_ui import UI, rw_cov

rng = np.random.default_rng(20261019)
res = {}
d = load_national(); leg = d[d.reg == 0].reset_index(drop=True)
s_leg, t_leg = leg.ai_use_se.values, leg.t.values; N = len(s_leg)
LT = np.log(THRESH)          # old monitors: 2/alpha = 40 for each of two one-sided monitors
LU = np.log(20.0)            # universal e-process: single two-sided e-process, 1/alpha = 20

def rw_noise(t, p, R):
    T = t - t[0]; dT = np.diff(T); L = np.zeros((R, len(t)))
    L[:, 1:] = np.cumsum(rng.standard_normal((R, len(t) - 1)) * np.sqrt(p * dT), axis=1); return L

def old_known(Y, s, t):
    pr = projections(s, t, np.zeros(len(s), int)); A = Y @ pr["C"].T; B = pr["B"]; ok = B > 1e-12; Bs = np.where(ok, B, 1.0)
    return (((log_e_known(A, Bs) >= LT) | (log_e_known(-A, Bs) >= LT)) & ok).any(1)

def old_unknown(Y, s, t):
    n_ = len(s); pr = projections(s, t, np.zeros(n_, int)); hit = np.zeros(Y.shape[0], bool)
    for n in range(4, n_ + 1):
        B = pr["B"][n - 1]; nu = pr["nu"][n - 1]
        if B <= 1e-12 or nu < 2: continue
        yy = Y[:, :n]; A = yy @ pr["C"][n - 1][:n]; rss = np.einsum("rj,jk,rk->r", yy, pr["P"][n - 1][:n, :n], yy)
        Bv = np.full(Y.shape[0], B); nv = np.full(Y.shape[0], float(nu))
        hit |= (log_e_unknown(A, Bv, rss, nv) >= LT) | (log_e_unknown(-A, Bv, rss, nv) >= LT)
    return hit

# ---- A. failure of independence-based monitors as the horizon grows (equal spacing, s = 0.15)
fail = {}
for n in (20, 40, 80, 160):
    t = np.arange(1, n + 1, dtype=float); s = np.full(n, 0.15)
    for p in (0.0, 0.01, 0.03, 0.06):
        R = 600 if n <= 80 else 300
        Y = 5 + 0.12 * t + rng.standard_normal((R, n)) * s + rw_noise(t, p, R)
        row = dict(basic=float(old_known(Y, s, t).mean()))
        if n <= 80: row["unknown_scale"] = float(old_unknown(Y[:300], s, t).mean())
        fail[f"n{n}_p{p}"] = row
    print("A", n, flush=True)
res["failure"] = fail

# ---- B. validity of the universal e-process on the BTOS design
ui = UI(s_leg, t_leg)
base = 5 + 0.12 * (t_leg - t_leg[0]); Rb = 1000
def sup_hit(Y): return float((ui.log_e(Y).max(1) >= LU).mean())
nulls = {"iid": lambda: rng.standard_normal((Rb, N)) * s_leg,
         "scale1.5": lambda: rng.standard_normal((Rb, N)) * s_leg * 1.5}
for p in (0.03, 0.06, 0.2): nulls[f"rw_p{p}"] = (lambda p=p: rw_noise(t_leg, p, Rb) + rng.standard_normal((Rb, N)) * s_leg)
def ar1(rho):
    z = rng.standard_normal((Rb, N)); e = z.copy()
    for j in range(1, N): e[:, j] = np.sqrt(1 - rho**2) * z[:, j] + rho * e[:, j - 1]
    return e * s_leg
nulls["ar1_0.3 (outside class)"] = lambda: ar1(0.3)
res["validity"] = {k: sup_hit(base + f()) for k, f in nulls.items()}
res["validity_old_basic_rw0.06"] = float(old_known(base + rw_noise(t_leg, 0.06, Rb) + rng.standard_normal((Rb, N)) * s_leg, s_leg, t_leg).mean())
print("B", res["validity"], flush=True)

# ---- C. power under roughness: slope change and level shift at wave 25; oracle ceiling
k0 = 24; Rc = 500; power = {}
orc_cache = {}
for p in (0.0, 0.03, 0.06):
    for fam, sizes in (("slope", (0.1, 0.2, 0.4)), ("level", (0.5, 1.0, 2.0))):
        for delta in sizes:
            eff = delta * (np.maximum(t_leg - t_leg[k0], 0) if fam == "slope" else (np.arange(N) >= k0).astype(float))
            Y = base + eff + (rw_noise(t_leg, p, Rc) if p > 0 else 0) + rng.standard_normal((Rc, N)) * s_leg
            le = ui.log_e(Y); hit = le >= LU
            first = np.where(hit.any(1), hit.argmax(1), -1); after = first >= 0
            row = dict(power=float(after.mean()), median_delay=float(np.median(first[after] - k0)) if after.any() else None,
                       median_max_logE=float(np.median(le.max(1))))
            # oracle nuisance (true p, kappa^2 = 1), family and tau known, k unknown: ceiling
            key = (p, fam, delta)
            uo = UI(s_leg, t_leg, r_rel=(p / float(np.median(s_leg)**2),), kap2=(1.0,), families=(fam,),
                    **({"tau_slope": (delta,)} if fam == "slope" else {"tau_level": (delta,)}))
            lo = uo.log_e(Y); ho = lo >= LU; fo = np.where(ho.any(1), ho.argmax(1), -1)
            row["oracle_power"] = float((fo >= 0).mean()); row["oracle_median_delay"] = float(np.median(fo[fo >= 0] - k0)) if (fo >= 0).any() else None
            power[f"p{p}_{fam}_{delta}"] = row
    print("C p", p, flush=True)
res["power"] = power
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/ui_experiments.json", "w"), indent=1)
print(json.dumps(res["failure"], indent=0)); print(res["validity"])

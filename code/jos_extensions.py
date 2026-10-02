"""Simulations for the extensions: unknown scale, redesign variants, log scale, mixture over tau.
Usage (from repository root): python code/jos_extensions.py   (about 10 minutes)"""
import json, os, numpy as np
from scipy.stats import norm
from evigil import load_national, TAU, THRESH
from evigil_ext import *

rng = np.random.default_rng(20261003)
d = load_national(); leg = d[d.reg == 0].reset_index(drop=True)
s_leg, t_leg = leg.ai_use_se.values, leg.t.values
N = len(s_leg); res = {}
lt = np.log(THRESH)

def alarms_known_mix(Y, pr, mode="known"):
    A = Y @ pr["C"].T; B = pr["B"]; ok = B > 1e-12; Bs = np.where(ok, B, 1.0)
    f = log_e_known if mode == "known" else (lambda a, b: log_e_mix(a, b))
    return ((f(A, Bs) >= lt) | (f(-A, Bs) >= lt)) & ok

def alarms_unknown(Y, pr, tau=TAU):
    Rn = Y.shape[0]; out = np.zeros((Rn, N), bool)
    for n in range(4, N + 1):
        B = pr["B"][n - 1]; nu = pr["nu"][n - 1]
        if B <= 1e-12 or nu < 2: continue
        yy = Y[:, :n]; A = yy @ pr["C"][n - 1][:n]
        rss = np.einsum("rj,jk,rk->r", yy, pr["P"][n - 1][:n, :n], yy)
        Bv = np.full(Rn, B); nv = np.full(Rn, float(nu))
        out[:, n - 1] = (log_e_unknown(A, Bv, rss, nv, tau) >= lt) | (log_e_unknown(-A, Bv, rss, nv, tau) >= lt)
    return out

pr0 = projections(s_leg, t_leg, np.zeros(N, int))
th0 = 5 + 0.15 * (t_leg - t_leg[0])

# E1: unknown scale
Ru = 2500
for kap in (1.0, 1.5, 2.0):
    Y = th0 + rng.standard_normal((Ru, N)) * s_leg * kap
    res[f"unknown_typeI_kappa{kap}"] = dict(unknown=float(alarms_unknown(Y, pr0).any(1).mean()),
                                            known=float(alarms_known_mix(Y, pr0).any(1).mean()))
k0 = 24
for delta in (0.02, 0.03, 0.05):
    th = 5 + 0.10 * (t_leg - t_leg[0]) + delta * np.maximum(t_leg - t_leg[k0], 0)
    out = {}
    for kap in (1.0, 1.5):
        Y = th + rng.standard_normal((Ru, N)) * s_leg * kap
        a = alarms_unknown(Y, pr0); f = np.where(a.any(1), a.argmax(1), -1)
        after = f >= k0
        out[f"kappa{kap}"] = dict(false_before=float(((f >= 0) & (f < k0)).mean()), by_end=float(after.mean()),
                                  median_delay=float(np.median(f[after] - k0)) if after.any() else None)
    res[f"unknown_power_delta{delta}"] = out
print("E1 done", flush=True)

# E2: redesign variants
def redesign_variants(k, kind, label_shift=0, reps=3000):
    n0 = 20; n = n0 + k
    s = np.concatenate([s_leg[:n0], np.full(k, 0.2)])
    t = np.concatenate([t_leg[:n0], t_leg[n0 - 1] + 2 + np.arange(k)])
    path = 5 + 0.25 * (t - t[0]); reg_true = (np.arange(n) >= n0).astype(int)
    th = path + 7.3 * reg_true if kind == "additive" else np.where(reg_true == 1, path * 1.73, path)
    lab = (np.arange(n) >= n0 - label_shift).astype(int)     # label_shift=+1: label one wave early; -1: one wave late
    Y = th + rng.standard_normal((reps, n)) * s
    out = {}
    for name, kw in dict(base=dict(), guard=dict(guard=1), slopes=dict(regime_slopes=True),
                         slopes_guard=dict(regime_slopes=True, guard=1)).items():
        pr = projections(s, t, lab, **kw)
        A = Y @ pr["C"].T; B = pr["B"]; ok = B > 1e-12; Bs = np.where(ok, B, 1)
        al = ((log_e_known(A, Bs) >= lt) | (log_e_known(-A, Bs) >= lt)) & ok
        out[name] = float(al.any(1).mean()); out[name + "_B"] = float(B[-1])
    return out
for k in (2, 5, 10, 20):
    res[f"variants_additive_k{k}"] = redesign_variants(k, "additive")
    res[f"variants_multiplicative_k{k}"] = redesign_variants(k, "multiplicative")
    res[f"variants_labelearly_k{k}"] = redesign_variants(k, "additive", label_shift=1)
    res[f"variants_labellate_k{k}"] = redesign_variants(k, "additive", label_shift=-1)
print("E2 done", flush=True)

# E3: log scale (exponential growth, relative standard error 3 percent)
def logscale(k, kind, reps=3000, tau_log=0.002):
    n0 = 20; n = n0 + k
    t = np.concatenate([t_leg[:n0], t_leg[n0 - 1] + 2 + np.arange(k)]); reg = (np.arange(n) >= n0).astype(int)
    lin = np.exp(1.0 + 0.04 * (t - t[0]))
    mu = lin * np.where(reg == 1, 1.73, 1.0) if kind == "multiplicative" else lin + 7.3 * reg
    se = 0.03 * mu
    Y = mu + rng.standard_normal((reps, n)) * se
    Yl = np.log(np.maximum(Y, 1e-3)); sl = se / mu
    pr = projections(sl, t, reg)
    A = Yl @ pr["C"].T; B = pr["B"]; ok = B > 1e-12; Bs = np.where(ok, B, 1)
    al = ((log_e_known(A, Bs, tau_log) >= lt) | (log_e_known(-A, Bs, tau_log) >= lt)) & ok
    return float(al.any(1).mean())
for k in (2, 5, 10, 20):
    res[f"logscale_k{k}"] = dict(multiplicative=logscale(k, "multiplicative"), additive=logscale(k, "additive"))
print("E3 done", flush=True)

# E4: mixture over the prior scale
R = 20000
sig = lambda: rng.standard_normal((R, N)) * s_leg
for beta in (0.0, 0.3):
    res[f"mix_typeI_slope{beta}"] = float(alarms_known_mix(5 + beta * (t_leg - t_leg[0]) + sig(), pr0, "mix").any(1).mean())
for delta in (0.01, 0.02, 0.03, 0.05):
    th = 5 + 0.10 * (t_leg - t_leg[0]) + delta * np.maximum(t_leg - t_leg[k0], 0)
    Y = th + sig(); out = {}
    for name, md in (("single", "known"), ("mix", "mix")):
        a = alarms_known_mix(Y, pr0, md); f = np.where(a.any(1), a.argmax(1), -1); after = f >= k0
        out[name] = dict(by_end=float(after.mean()), within20=float((after & (f - k0 <= 20)).mean()),
                         median_delay=float(np.median(f[after] - k0)) if after.any() else None)
    res[f"mix_power_delta{delta}"] = out
print("E4 done", flush=True)
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/extensions.json", "w"), indent=1)
for k, v in res.items(): print(k, v)

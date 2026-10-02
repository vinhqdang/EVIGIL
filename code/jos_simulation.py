"""Simulation study: false-alarm control, detection delay and instrument-change
behaviour of the invariant e-process monitor versus repeated testing.
Usage (from repository root): python code/jos_simulation.py"""
import json, numpy as np
from scipy.stats import norm
from evigil import load_national, TAU, THRESH

rng = np.random.default_rng(20261002)
d = load_national()
leg = d[d.reg == 0].reset_index(drop=True)
s, t = leg.ai_use_se.values, leg.t.values
N = len(s)
R = 20000

def build(s, t, reg=None, pool=False):
    """Row n holds the linear map y -> A_n; B_n is information. n = 3..N waves."""
    reg = np.zeros(len(s), int) if reg is None else reg
    w = 1 / s**2
    C = np.zeros((N, N)); B = np.zeros(N)
    for n in range(3, N + 1):
        ss, tn = s[:n], t[:n]
        wn = 1 / ss**2; tt = tn - tn.mean()
        r = np.zeros(n, int) if pool else reg[:n]
        D = np.column_stack([(r == g).astype(float) for g in np.unique(r)] + [tt])
        x = tt**2 / 2
        sw = np.sqrt(wn)
        b, *_ = np.linalg.lstsq(D * sw[:, None], x * sw, rcond=None)
        xt = x - D @ b
        C[n - 1, :n] = wn * xt
        B[n - 1] = (wn * xt * xt).sum()
    return C, B

def logE(A, B, tau):
    a = B + tau**-2
    z = A / np.sqrt(a)
    return np.log(2 / (tau * np.sqrt(a))) + norm.logcdf(z) + z * z / 2

def alarms(Y, C, B, tau=TAU, thresh=THRESH):
    """Return boolean arrays (R, N) of alarm at look n for e-process / uncorrected / Bonferroni."""
    A = Y @ C.T                       # (R, N)
    ok = B > 0
    lt = np.log(thresh)
    Bs = np.where(ok, B, 1.0)
    eup = logE(A, Bs, tau) >= lt
    edn = logE(-A, Bs, tau) >= lt
    e = (eup | edn) & ok
    z = np.abs(A) / np.sqrt(Bs)
    L = ok.sum()
    unc = (z > norm.ppf(1 - 0.025)) & ok
    bon = (z > norm.ppf(1 - 0.025 / L)) & ok
    k = np.cumsum(ok)                  # look counter
    spend = (z > norm.ppf(1 - 0.025 / (np.maximum(k, 1) * (np.maximum(k, 1) + 1)))) & ok
    return e, unc, bon, spend

def simulate(theta, noise=None, kappa=1.0):
    eps = rng.standard_normal((R, N)) * s * kappa if noise is None else noise
    return theta + eps

res = {}
C, B = build(s, t)

# S1: type-I error under locally linear nulls (several slopes)
for beta in (0.0, 0.15, 0.30):
    theta = 5 + beta * (t - t[0])
    e, unc, bon, spend = alarms(simulate(theta), C, B)
    res[f"typeI_slope{beta}"] = dict(eproc=float(e.any(1).mean()), uncorrected=float(unc.any(1).mean()),
                                     bonferroni=float(bon.any(1).mean()), spending=float(spend.any(1).mean()),
                                     unc_alarm_looks=float(unc.sum(1).mean()))

# S2: robustness to violations (panel-overlap correlation, understated variance)
def corr_noise(rho, lag=6):
    z = rng.standard_normal((R, N)); e = z.copy()
    for j in range(lag, N):
        e[:, j] = np.sqrt(1 - rho**2) * z[:, j] + rho * e[:, j - lag]
    return e * s
theta0 = 5 + 0.15 * (t - t[0])
for rho in (0.1, 0.25, 0.5):
    e, unc, bon, spend = alarms(theta0 + corr_noise(rho), C, B)
    res[f"robust_rho{rho}"] = dict(eproc=float(e.any(1).mean()), uncorrected=float(unc.any(1).mean()), bonferroni=float(bon.any(1).mean()), spending=float(spend.any(1).mean()))
for kap in (1.1, 1.25, 1.5):
    e, unc, bon, spend = alarms(simulate(theta0, kappa=kap), C, B)
    res[f"robust_kappa{kap}"] = dict(eproc=float(e.any(1).mean()), uncorrected=float(unc.any(1).mean()), bonferroni=float(bon.any(1).mean()), spending=float(spend.any(1).mean()))

# S3: detection of a takeoff kink at wave k0 (growth rate rises by delta pp/wave)
k0 = 24
for delta in (0.05, 0.10, 0.20):
    theta = 5 + 0.10 * (t - t[0]) + delta * np.maximum(t - t[k0], 0)
    out = {}
    for name, idx in zip(("eproc", "uncorrected", "bonferroni", "spending"), range(4)):
        a = alarms(simulate(theta), C, B)[idx]
        first = np.where(a.any(1), a.argmax(1), -1)
        before = (first >= 0) & (first < k0)
        after = first >= k0
        out[name] = dict(false_before=float(before.mean()), detect_after=float(after.mean()),
                         median_delay=float(np.median(first[after] - k0)) if after.any() else None)
    res[f"power_delta{delta}"] = out

# S4: instrument redesign: level shift of 7 pp between waves, regime-aware vs pooled
N_all = len(d); sa, ta, reg = d.ai_use_se.values, d.t.values, d.reg.values
Nsave = N
N = N_all
Ca, Ba = build(sa, ta, reg); Cp, Bp = build(sa, ta, reg, pool=True)
theta = 5 + 0.10 * (ta - ta[0]) + 7.3 * reg
Rn = rng.standard_normal((R, N)) * sa
for name, (CC, BB) in dict(regime_aware=(Ca, Ba), pooled=(Cp, Bp)).items():
    e, unc, bon, spend = alarms(theta + Rn, CC, BB)
    res[f"redesign_{name}"] = dict(eproc=float(e.any(1).mean()), uncorrected=float(unc.any(1).mean()))
N = Nsave

import os; os.makedirs("results", exist_ok=True)
json.dump(res, open("results/simulation.json", "w"), indent=1)
for k, v in res.items(): print(k, v)

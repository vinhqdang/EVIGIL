"""Extensions applied to the BTOS series: unknown scale, regime variants, log scale, tau mixture,
panel-lag diagnostics and a local-linear-trend state-space comparison.
Usage (from repository root): python code/jos_application_ext.py"""
import json, numpy as np
from scipy.optimize import minimize
from evigil import load_national, TAU
from evigil_ext import *

d = load_national()
y, s, t, reg = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values
dt = list(d.collection_end); date = lambda i: str(dt[i].date()); nleg = int((reg == 0).sum())
R = {}
def summ(dec): return [dict(date=date(x["wave"]), dir=x["dir"], E=x["E"], start=date(x["start"])) for x in dec]

# --- variants of the monitor on the national series
R["known"] = summ(surveil_ext(y, s, t, reg)[0])
R["unknown_scale"] = summ(surveil_ext(y, s, t, reg, mode="unknown")[0])
dec_u, tr_u = surveil_ext(y, s, t, reg, mode="unknown")
R["unknown_scale_last_peak"] = max(max(r[1], r[2]) for r in tr_u[-1][1]) if len(dec_u) < len(tr_u) else None
R["tau_mixture"] = summ(surveil_ext(y, s, t, reg, mode="mix")[0])
R["regime_slopes"] = summ(surveil_ext(y, s, t, reg, regime_slopes=True)[0])
R["guard1"] = summ(surveil_ext(y, s, t, reg, guard=1)[0])
R["slopes_guard1"] = summ(surveil_ext(y, s, t, reg, regime_slopes=True, guard=1)[0])
# log scale (delta method standard errors), prior scale in log units
yl, sl = np.log(y), s / y
for tl in (0.001, 0.002, 0.004):
    R[f"log_tau{tl}"] = summ(surveil_ext(yl, sl, t, reg, tau=tl)[0])

# --- panel-lag diagnostics: autocorrelation of standardized second differences (legacy)
d2 = y[2:nleg] - 2 * y[1:nleg-1] + y[:nleg-2]
z2 = d2 / np.sqrt(s[2:nleg]**2 + 4 * s[1:nleg-1]**2 + s[:nleg-2]**2)
def acf(x, k): x = x - x.mean(); return float((x[:-k] * x[k:]).sum() / (x * x).sum())
R["acf_z2"] = {str(k): acf(z2, k) for k in range(1, 13)}
R["acf_se_bound"] = float(2 / np.sqrt(len(z2)))
# the same diagnostic from second-difference residuals of a constant-scale model is not needed: report lags 6 and 12 only in text

# --- local-linear-trend state-space model with a level intervention at the regime change
def kalman(y, s, t, reg, q, kappa, smooth=True, innov=None):
    n = len(y); x = np.zeros(3); P = np.diag([1e4, 1e2, 1e4])
    xs, Ps, xp, Pp = [], [], [], []; ll = 0.0
    for j in range(n):
        if j > 0:
            dlt = t[j] - t[j-1]
            F = np.array([[1, dlt, 0], [0, 1, 0], [0, 0, 1.0]]); Q = np.zeros((3, 3)); Q[1, 1] = q * dlt
            x = F @ x; P = F @ P @ F.T + Q
        else: F = np.eye(3)
        xp.append(x.copy()); Pp.append(P.copy())
        Z = np.array([1.0, 0, reg[j]]); v = y[j] - Z @ x; S = Z @ P @ Z + (kappa * s[j])**2
        K = P @ Z / S; x = x + K * v; P = P - np.outer(K, Z @ P)
        ll += -0.5 * (np.log(2 * np.pi * S) + v * v / S)
        if innov is not None: innov.append(v / np.sqrt(S))
        xs.append(x.copy()); Ps.append(P.copy())
    if not smooth: return ll
    xm, Pm = xs[-1].copy(), Ps[-1].copy(); sm = [(xm.copy(), Pm.copy())]
    for j in range(n - 2, -1, -1):
        dlt = t[j+1] - t[j]; F = np.array([[1, dlt, 0], [0, 1, 0], [0, 0, 1.0]])
        C = Ps[j] @ F.T @ np.linalg.inv(Pp[j+1])
        xm = xs[j] + C @ (xm - xp[j+1]); Pm = Ps[j] + C @ (Pm - Pp[j+1]) @ C.T
        sm.append((xm.copy(), Pm.copy()))
    return ll, sm[::-1]
def fit(free_kappa):
    f = (lambda p: -kalman(y, s, t, reg, np.exp(p[0]), np.exp(p[1]) if free_kappa else 1.0, smooth=False))
    x0 = [np.log(0.01)] + ([0.3] if free_kappa else [])
    return minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
ss = {}
for name, fk in (("kappa_fixed", False), ("kappa_free", True)):
    o = fit(fk); q = float(np.exp(o.x[0])); kap = float(np.exp(o.x[1])) if fk else 1.0
    inn = []
    ll, sm = kalman(y, s, t, reg, q, kap, innov=inn)
    inn = np.array(inn[8:nleg])    # drop the diffuse start, legacy waves only
    ll0 = -minimize(lambda p: -kalman(y, s, t, reg, 1e-10, np.exp(p[0]) if fk else 1.0, smooth=False),
                    [0.3] if fk else [0.0], method="Nelder-Mead").fun if fk else kalman(y, s, t, reg, 1e-10, 1.0, smooth=False)
    pts = {}
    for dd in ("2024-03-10", "2024-06-30", "2024-11-17", "2025-04-06", "2025-07-27", "2025-10-05"):
        j = [i for i in range(len(y)) if date(i) == dd][0]
        pts[dd] = (float(sm[j][0][1]), float(1.96 * np.sqrt(sm[j][1][1, 1])))
    ss[name] = dict(q=q, kappa=kap, loglik=float(ll), loglik_q0=float(ll0), lr_stat=float(2 * (ll - ll0)),
                    step=float(sm[0][0][2]), slope=pts,
                    innov_acf={str(k): acf(inn, k) for k in (1, 2, 3, 6, 12)}, innov_sd=float(inn.std()), innov_n=len(inn))
R["state_space"] = ss
json.dump(R, open("results/application_ext.json", "w"), indent=1, default=float)
for k, v in R.items(): print(k, v)

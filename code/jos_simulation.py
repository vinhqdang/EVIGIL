"""Simulation study for the JOS manuscript.
Usage (from repository root): python code/jos_simulation.py   (about 3 minutes)"""
import json, os, numpy as np
from scipy.stats import norm
from evigil import load_national, TAU, THRESH, surveil

rng = np.random.default_rng(20261002)
d = load_national()
leg = d[d.reg == 0].reset_index(drop=True)
s_leg, t_leg = leg.ai_use_se.values, leg.t.values
R = 20000
res = {}

def build(s, t, reg=None, pool=False):
    """Row n-1 holds the linear map y -> A_n; B_n is the information (n = 3..N)."""
    N = len(s)
    reg = np.zeros(N, int) if reg is None else reg
    C = np.zeros((N, N)); B = np.zeros(N)
    for n in range(3, N + 1):
        wn = 1 / s[:n]**2; tt = t[:n] - t[:n].mean()
        r = np.zeros(n, int) if pool else reg[:n]
        D = np.column_stack([(r == g).astype(float) for g in np.unique(r)] + [tt])
        x = tt**2 / 2; sw = np.sqrt(wn)
        b, *_ = np.linalg.lstsq(D * sw[:, None], x * sw, rcond=None)
        xt = x - D @ b
        C[n - 1, :n] = wn * xt
        B[n - 1] = (wn * xt * xt).sum()
    return C, B

def logE(A, B, tau):
    a = B + tau**-2; z = A / np.sqrt(a)
    return np.log(2 / (tau * np.sqrt(a))) + norm.logcdf(z) + z * z / 2

def alarms(Y, C, B, tau=TAU, thresh=THRESH, gamma0=0.0, pocock=None):
    A = Y @ C.T; ok = B > 0; Bs = np.where(ok, B, 1.0); lt = np.log(thresh)
    e = ((logE(A - gamma0 * Bs, Bs, tau) >= lt) | (logE(-A - gamma0 * Bs, Bs, tau) >= lt)) & ok
    z = np.abs(A) / np.sqrt(Bs); L = ok.sum(); k = np.maximum(np.cumsum(ok), 1)
    out = dict(eproc=e,
               uncorrected=(z > norm.ppf(.975)) & ok,
               bonferroni=(z > norm.ppf(1 - .025 / L)) & ok,
               spending=(z > norm.ppf(1 - .025 / (k * (k + 1)))) & ok)
    if pocock is not None: out["exact"] = (z > pocock) & ok
    return out

def first_alarm(a):
    return np.where(a.any(1), a.argmax(1), -1)

N = len(s_leg)
C, B = build(s_leg, t_leg)
sig = lambda: rng.standard_normal((R, N)) * s_leg
# exact (Pocock-type) flat boundary: horizon known, calibrated to size 0.05 by simulation
zmax = (np.abs((sig()) @ C.T) / np.sqrt(np.where(B > 0, B, 1))) * (B > 0)
POC = float(np.quantile(zmax.max(1), 0.95)); res["pocock_c"] = POC

# S1: false declaration under linear nulls
for beta in (0.0, 0.15, 0.30):
    a = alarms(5 + beta * (t_leg - t_leg[0]) + sig(), C, B, pocock=POC)
    res[f"typeI_slope{beta}"] = {k: float(v.any(1).mean()) for k, v in a.items()}
    res[f"typeI_slope{beta}"]["unc_alarm_looks"] = float(a["uncorrected"].sum(1).mean())

# S1b: robustness: panel-overlap correlation at lag 6; published SE too small by factor kappa
def corr_noise(rho, lag=6):
    z = rng.standard_normal((R, N)); e = z.copy()
    for j in range(lag, N): e[:, j] = np.sqrt(1 - rho**2) * z[:, j] + rho * e[:, j - lag]
    return e * s_leg
th0 = 5 + 0.15 * (t_leg - t_leg[0])
for rho in (0.25, 0.5, 0.8):
    a = alarms(th0 + corr_noise(rho), C, B, pocock=POC)
    res[f"robust_rho{rho}"] = {k: float(v.any(1).mean()) for k, v in a.items()}
for kap in (1.1, 1.25, 1.5, 1.75):
    a = alarms(th0 + sig() * kap, C, B, pocock=POC)
    res[f"robust_kappa{kap}"] = {k: float(v.any(1).mean()) for k, v in a.items()}

# S1c: the full restart procedure under a linear null; and tolerance null
for beta in (0.0, 0.3):
    nd = []
    for _ in range(3000):
        y = 5 + beta * (t_leg - t_leg[0]) + rng.standard_normal(N) * s_leg
        nd.append(len(surveil(y, s_leg, t_leg, np.zeros(N, int))[0]))
    nd = np.array(nd)
    res[f"restart_slope{beta}"] = dict(p_any=float((nd >= 1).mean()), mean_decl=float(nd.mean()), p_two=float((nd >= 2).mean()))
G0 = 0.001   # growth-rate change of 0.02 pp per wave accumulating over 20 waves
for gam in (0.0, G0, 3 * G0):
    th = 5 + 0.1 * (t_leg - t_leg[0]) + gam * (t_leg - t_leg[0])**2 / 2
    y = th + sig()
    res[f"tolerance_gamma{gam}"] = dict(gamma0_0=float(alarms(y, C, B)["eproc"].any(1).mean()),
                                        gamma0_tol=float(alarms(y, C, B, gamma0=G0)["eproc"].any(1).mean()))

# S2: takeoff detection (kink at wave 25: slope rises by delta from baseline 0.10)
k0 = 24
for delta in (0.01, 0.02, 0.03, 0.05):
    th = 5 + 0.10 * (t_leg - t_leg[0]) + delta * np.maximum(t_leg - t_leg[k0], 0)
    a = alarms(th + sig(), C, B, pocock=POC)
    out = {}
    for name, v in a.items():
        if name == "uncorrected": continue
        f = first_alarm(v); before = (f >= 0) & (f < k0); after = f >= k0
        out[name] = dict(false_before=float(before.mean()),
                         within5=float((after & (f - k0 <= 5)).mean()), within10=float((after & (f - k0 <= 10)).mean()),
                         within20=float((after & (f - k0 <= 20)).mean()), by_end=float(after.mean()),
                         median_delay=float(np.median(f[after] - k0)) if after.any() else None)
    res[f"power_delta{delta}"] = out

# S3: redesign: epoch with 20 legacy waves then k revised waves; linear path, slope 0.25
def redesign_sim(k, kind, label_shift=0, reps=5000):
    n0 = 20; n = n0 + k
    s = np.concatenate([s_leg[:n0], np.full(k, 0.2)])
    t = np.concatenate([t_leg[:n0], t_leg[n0 - 1] + 2 + np.arange(k)])   # a gap of 3 periods, then biweekly
    path = 5 + 0.25 * (t - t[0])
    reg_true = (np.arange(n) >= n0).astype(int)
    shift = 7.3 if kind == "additive" else None
    th = path + 7.3 * reg_true if kind == "additive" else np.where(reg_true == 1, path * 1.73, path)
    lab = (np.arange(n) >= n0 - label_shift).astype(int) if label_shift else reg_true
    y = th + rng.standard_normal((reps, n)) * s
    out = {}
    for name, (rg, pool) in dict(regime_aware=(lab, False), pooled=(lab, True)).items():
        C_, B_ = build(s, t, rg, pool=pool)
        out[name] = float(alarms(y, C_, B_)["eproc"].any(1).mean())
    return out
for k in (2, 5, 10, 20):
    res[f"redesign_additive_k{k}"] = redesign_sim(k, "additive")
    res[f"redesign_multiplicative_k{k}"] = redesign_sim(k, "multiplicative")
    res[f"redesign_labelone_early_k{k}"] = redesign_sim(k, "additive", label_shift=1)  # label placed one wave early

os.makedirs("results", exist_ok=True)
json.dump(res, open("results/simulation.json", "w"), indent=1)
for k, v in res.items(): print(k, v if not isinstance(v, dict) else {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})

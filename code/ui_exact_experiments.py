"""Scale-exact monitor vs profile monitor; information ceiling for any valid monitor.  Usage: PYTHONPATH=code python code/ui_exact_experiments.py"""
import json, numpy as np
from evigil import load_national
from evigil_ui import UI, rw_cov
from evigil_exact import UIX, kl_ceiling, power_ceiling
rng = np.random.default_rng(20261022); LU = np.log(20.0)
d = load_national(); y_all, s_all, t_all, reg_all = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values
leg = d[d.reg == 0].reset_index(drop=True); s, t = leg.ai_use_se.values, leg.t.values; N = len(s)
base = 5 + 0.12 * (t - t[0]); k0 = 24
def rw_noise(tt, p, n):
    T = tt - tt[0]; L = np.zeros((n, len(tt))); L[:, 1:] = np.cumsum(rng.standard_normal((n, len(tt) - 1)) * np.sqrt(p * np.diff(T)), axis=1); return L
def eff(fam, dl): return dl * (np.maximum(t - t[k0], 0) if fam == "slope" else (np.arange(N) >= k0).astype(float))
def noise(p, R): return rng.standard_normal((R, N)) * s + rw_noise(t, p, R)
R = {}; ux, up = UIX(s, t), UI(s, t)
# validity
val = {}
for p in (0.0, 0.03, 0.06, 0.2):
    Y = base + noise(p, 600); val[str(p)] = float((ux.log_e(Y).max(1) >= LU).mean()); print("val", p, val[str(p)], flush=True)
Y = base + (rng.standard_normal((600, N)) * 1.5 * s); val["se1.5"] = float((ux.log_e(Y).max(1) >= LU).mean())
R["validity_exact"] = val
# power: exact vs profile
pw = {}
for p, fam, dl in ((0.0, "slope", 0.2), (0.03, "slope", 0.2), (0.06, "slope", 0.2), (0.06, "slope", 0.4), (0.06, "level", 1.0), (0.06, "level", 2.0), (0.03, "level", 1.0)):
    Y = base + eff(fam, dl) + noise(p, 300)
    pw[f"{p}|{fam}|{dl}"] = dict(exact=float((ux.log_e(Y).max(1) >= LU).mean()), profile=float((up.log_e(Y).max(1) >= LU).mean())); print("pw", p, fam, dl, pw[f"{p}|{fam}|{dl}"], flush=True)
R["power"] = pw
# BTOS
uxf = UIX(s_all, t_all, reg=reg_all); le = uxf.log_e(y_all)[0]
R["btos_exact"] = dict(max_logE=float(le.max()), date=str(d.collection_end[int(le.argmax())].date()))
print(R["btos_exact"])
u0 = UIX(s_all, t_all); l0 = u0.log_e(y_all)[0]; R["positive_control_exact"] = float(l0.max())
# spike-in MDC (exact) on the full design under the fitted null
uf = UI(s_all, t_all, reg=reg_all); fit = uf.reml(y_all); r_hat, k2 = fit["r"], fit["kappa2"]
X = uf._X(len(y_all)); Sig = k2 * (np.diag(s_all**2) + r_hat * rw_cov(t_all)); Si = np.linalg.inv(Sig)
mean = X @ np.linalg.solve(X.T @ Si @ X, X.T @ Si @ y_all); Lc = np.linalg.cholesky(Sig); Nn = len(y_all); kk = 24
def inj(fam, dl): return dl * (np.maximum(t_all - t_all[kk], 0) if fam == "slope" else (np.arange(Nn) >= kk).astype(float))
curve = {}
for fam, grid in (("slope", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)), ("level", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0))):
    row = {}
    for dl in grid:
        Yb = mean + inj(fam, dl) + (Lc @ rng.standard_normal((Nn, 300))).T
        row[str(dl)] = float((uxf.log_e(Yb).max(1) >= LU).mean()); print("mdc", fam, dl, row[str(dl)], flush=True)
    curve[fam] = row
R["spike_in_exact"] = curve
# ceiling: no valid monitor can exceed this power / growth against the alternative at wave 25 of the fitted design
r_null = uf.r_null; ceil = {}
for fam, grid in (("slope", (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)), ("level", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0))):
    row = {}
    for dl in grid:
        K = kl_ceiling(s_all, t_all, reg_all, fam, kk + 1, dl, r_hat, k2, r_null)
        row[str(dl)] = dict(K=float(K), power_ceiling=float(power_ceiling(K)))
    ceil[fam] = row; print("ceil", fam, row, flush=True)
R["ceiling"] = ceil
json.dump(R, open("results/ui_exact.json", "w"), indent=1)

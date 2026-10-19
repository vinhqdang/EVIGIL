"""Cost decomposition for the scale-exact monitor: known-r likelihood ratio, one component with the supremum, full mixture.
Usage: PYTHONPATH=code python code/ui_exact_experiments3.py"""
import json, numpy as np
from evigil import load_national
from evigil_exact import UIX
rng = np.random.default_rng(20261024); LU = np.log(20.0)
d = load_national(); leg = d[d.reg == 0].reset_index(drop=True); s, t = leg.ai_use_se.values, leg.t.values; N = len(s); med = float(np.median(s)**2)
k0 = 24; base = 5 + 0.12 * (t - t[0])
def rw(p, R):
    T = t - t[0]; L = np.zeros((R, N)); L[:, 1:] = np.cumsum(rng.standard_normal((R, N - 1)) * np.sqrt(p * np.diff(T)), axis=1); return L
def eff(f, dl): return dl * (np.maximum(t - t[k0], 0) if f == "slope" else (np.arange(N) >= k0).astype(float))
full = UIX(s, t); out = []
for p, f, dl in ((0.06, "slope", 0.4), (0.06, "slope", 0.2), (0.06, "level", 2.0), (0.06, "level", 1.0), (0.03, "slope", 0.2), (0.03, "level", 1.0)):
    Y = base + eff(f, dl) + rng.standard_normal((400, N)) * s + rw(p, 400)
    # single component: true family, wave k0+1, tau = delta, roughness p (kappa^2 = 1)
    one = UIX(s, t, r_rel=(p / med,), families=(f,), tau_slope=(dl,), tau_level=(dl,), kmin=3)
    for n, pr in one.pre.items():
        j = k0 + 1 - one.kmin
        for ff in pr["Z"]: pr["Z"][ff] = pr["Z"][ff][[j]]
    i_true = one.ri[float(one.r_prior[0])]
    known = (one.log_e(Y, fixed=i_true).max(1) >= LU).mean(); sup1 = (one.log_e(Y).max(1) >= LU).mean(); allc = (full.log_e(Y).max(1) >= LU).mean()
    out.append(dict(p=p, family=f, delta=dl, known=float(known), one_sup=float(sup1), universal=float(allc))); print(out[-1], flush=True)
json.dump(out, open("results/ui_exact3.json", "w"), indent=1)

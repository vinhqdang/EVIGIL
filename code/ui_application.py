"""Application of the universal e-process to the BTOS series.
Usage (from repository root, PYTHONPATH=code): python code/ui_application.py"""
import json, numpy as np, pandas as pd
from evigil import load_national, surveil
from evigil_ui import UI, rw_cov
LU = np.log(20.0)
rng = np.random.default_rng(20261020)
d = load_national(); y, s, t, reg = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values
dt = list(d.collection_end); date = lambda i: str(dt[i].date()); N = len(y); nleg = int((reg == 0).sum())
R = {}

# ---- national series
ui = UI(s, t, reg=reg)
fit = ui.reml(y); r_, k2 = fit["r"], fit["kappa2"]
rg, ll = fit["profile"]; ok = ll >= ll.max() - 1.92
R["reml"] = dict(r=r_, kappa2=k2, p=r_ * k2, p_ci=(float(rg[ok].min() * k2), float(rg[ok].max() * k2)), r_zero_loglik_drop=float(ll.max() - ll[0]))
le, pp = ui.log_e(y, parts=True)
R["national"] = dict(max_logE=float(le[0].max()), argmax=date(int(le[0].argmax())), by_family={f: float(pp[f][0].max()) for f in pp},
                     final_logE=float(le[0, -1]), path=[(date(j), float(le[0, j])) for j in range(ui.nmin - 1, N)])
# sensitivity to the prior design
sens = {}
for name, kw in {"wide_tau": dict(tau_slope=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6), tau_level=(0.25, 0.5, 1, 2, 4, 8)),
                 "rough_prior_only": dict(r_rel=(2.0, 8.0)), "kappa_fixed": dict(kap2=(1.0,)),
                 "slope_only": dict(families=("slope",)), "level_only": dict(families=("level",)),
                 "legacy_only": None}.items():
    if kw is None:
        u2 = UI(s[:nleg], t[:nleg]); l2 = u2.log_e(y[:nleg])[0]
    else:
        u2 = UI(s, t, reg=reg, **kw); l2 = u2.log_e(y)[0]
    sens[name] = float(l2.max())
R["sensitivity_max_logE"] = sens

# ---- size classes and combination of e-processes (average: valid under arbitrary dependence)
siz = pd.read_csv("data/btos_ai_sizeclass.csv", parse_dates=["collection_end"]); rows = {}; les = []
for k in "ABCDEFG":
    g = siz[siz.size_class == k].sort_values("collection_end").reset_index(drop=True)
    gt = ((g.collection_end - d.collection_end[0]).dt.days / 14.0).values
    yk, sk, rk = g.ai_use_pct.values, g.ai_use_se.values, (g.wording == "revised").astype(int).values
    uk = UI(sk, gt, reg=rk); fk = uk.reml(yk); lk = uk.log_e(yk)[0]; les.append(lk)
    rows[k] = dict(r=fk["r"], kappa2=fk["kappa2"], p=fk["r"] * fk["kappa2"], max_logE=float(lk.max()), argmax=str(g.collection_end[int(lk.argmax())].date()))
R["size_classes"] = rows
L = np.array(les); finite = np.where(np.isfinite(L), L, -np.inf)
comb = np.log(np.mean(np.exp(np.where(np.isfinite(finite), finite, -50)), axis=0))      # average of the 7 e-processes
R["size_combined"] = dict(max_logE=float(comb.max()))

# ---- parametric bootstrap: how often does the basic (independence-based) monitor declare under the fitted null?
Sig = k2 * (np.diag(s**2) + r_ * rw_cov(t)); Lc = np.linalg.cholesky(Sig)
# fitted null has regime offset: use the fitted mean (GLS fit of X c)
X = ui._X(N); Si = np.linalg.inv(Sig); c = np.linalg.solve(X.T @ Si @ X, X.T @ Si @ y); mean = X @ c
nd = []
for _ in range(400):
    yy = mean + Lc @ rng.standard_normal(N)
    nd.append(len(surveil(yy, s, t, reg)[0]))
nd = np.array(nd)
R["bootstrap_basic_monitor"] = dict(observed_basic_declarations=2, mean_declarations=float(nd.mean()), p_at_least_1=float((nd >= 1).mean()), p_at_least_2=float((nd >= 2).mean()))

# ---- minimum detectable slope change / level shift for the BTOS design under the fitted null (oracle formula)
def b_of(Sigma, X, z):
    Si = np.linalg.inv(Sigma); P = Si - Si @ X @ np.linalg.solve(X.T @ Si @ X, X.T @ Si); return float(z @ P @ z)
mdsc = {}
for lab, (rr, kk) in {"fitted_roughness": (r_, k2), "no_roughness": (0.0, 1.0)}.items():
    Sg = kk * (np.diag(s**2) + rr * rw_cov(t)); row = {}
    for k in (10, 20, 30, 40):
        zs = np.maximum(t - t[k - 1], 0.0); zl = (np.arange(N) >= k).astype(float)
        row[str(k)] = dict(slope_b=b_of(Sg, X, zs), level_b=b_of(Sg, X, zl))
    mdsc[lab] = row
R["mdsc_b"] = mdsc
json.dump(R, open("results/ui_application.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k != "national"}, indent=1, default=float)); print({k: v for k, v in R["national"].items() if k != "path"})

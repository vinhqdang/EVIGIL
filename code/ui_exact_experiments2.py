"""Scale-exact monitor on BTOS (families, variants, size classes, planned use) and growth-rate check against the ceiling.
Usage: PYTHONPATH=code python code/ui_exact_experiments2.py"""
import json, numpy as np, pandas as pd
from evigil import load_national
from evigil_ui import UI, rw_cov
from evigil_exact import UIX, kl_ceiling
import evigil
rng = np.random.default_rng(20261023); LU = np.log(20.0)
d = load_national(); y, s_, t_, reg = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values
R = {}
u = UIX(s_, t_, reg=reg); le, pp = u.log_e(y, parts=True)
R["national"] = dict(max=float(le[0].max()), date=str(d.collection_end[int(le[0].argmax())].date()), final=float(le[0, -1]),
                     slope=float(pp["slope"][0].max()), level=float(pp["level"][0].max()),
                     level_date=str(d.collection_end[int(pp["level"][0].argmax())].date()))
print(R["national"], flush=True)
var = {}
for name, kw in (("wider_tau", dict(tau_slope=(0.2, 0.4, 0.8, 1.6), tau_level=(1, 2, 4, 8))), ("rough_prior_only", dict(r_rel=(2.0, 8.0, 32.0))),
                 ("slope_only", dict(families=("slope",))), ("level_only", dict(families=("level",))), ("kmin5", dict(kmin=5))):
    ux = UIX(s_, t_, reg=reg, **kw); var[name] = float(ux.log_e(y)[0].max()); print(name, var[name], flush=True)
leg = d[d.reg == 0].reset_index(drop=True); ul = UIX(leg.ai_use_se.values, leg.t.values); var["legacy_only"] = float(ul.log_e(leg.ai_use_pct.values)[0].max())
R["variants"] = var
# positive control: wording change as ordinary data
R["positive_control"] = float(UIX(s_, t_).log_e(y)[0].max())
# size classes, planned use via the loaders in ui_application
siz = pd.read_csv("data/btos_ai_sizeclass.csv", parse_dates=["collection_end"]); sc = {}
for k in "ABCDEFG":
    g = siz[siz.size_class == k].sort_values("collection_end").reset_index(drop=True)
    gt = ((g.collection_end - d.collection_end[0]).dt.days / 14.0).values
    sc[k] = float(UIX(g.ai_use_se.values, gt, reg=(g.wording == "revised").astype(int).values).log_e(g.ai_use_pct.values)[0].max())
R["size_classes"] = sc; print(sc, flush=True)
R["planned_use"] = float(UIX(d.ai_plan_se.values, t_, reg=reg).log_e(d.ai_plan_pct.values)[0].max()); print("planned", R["planned_use"], flush=True)
# growth rate under the alternative vs the ceiling (fitted design, change at wave 25)
uf = UI(s_, t_, reg=reg); fit = uf.reml(y); r_hat, k2 = fit["r"], fit["kappa2"]
X = uf._X(len(y)); Sig = k2 * (np.diag(s_**2) + r_hat * rw_cov(t_)); Si = np.linalg.inv(Sig)
mean = X @ np.linalg.solve(X.T @ Si @ X, X.T @ Si @ y); Lc = np.linalg.cholesky(Sig); Nn = len(y); kk = 24
growth = {}
for fam, grid in (("slope", (0.2, 0.3, 0.4, 0.5, 0.6)), ("level", (1.0, 1.5, 2.0, 2.5, 3.0))):
    for dl in grid:
        inj = dl * (np.maximum(t_ - t_[kk], 0) if fam == "slope" else (np.arange(Nn) >= kk).astype(float))
        Yb = mean + inj + (Lc @ rng.standard_normal((Nn, 400))).T
        lg = u.log_e(Yb)[:, -1]
        K = kl_ceiling(s_, t_, reg, fam, kk + 1, dl, r_hat, k2, uf.r_null)
        growth[f"{fam}|{dl}"] = dict(mean_logE=float(lg.mean()), ceiling=float(K)); print(fam, dl, growth[f"{fam}|{dl}"], flush=True)
R["growth"] = growth
json.dump(R, open("results/ui_exact2.json", "w"), indent=1)

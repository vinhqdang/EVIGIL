"""Second set of experiments: oracle decomposition, calibrated GLR comparator, enlarged null class,
spike-in detection curves and minimum detectable changes on the BTOS design, positive control, model comparison,
planned-use series, and the roughness inflation of independence-based statistics.
Usage (repository root): PYTHONPATH=code python code/ui_experiments2.py"""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from evigil import load_national
from evigil_ui import UI, rw_cov
rng = np.random.default_rng(20261021)
LU = np.log(20.0)
d = load_national(); y_all, s_all, t_all, reg_all = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values
leg = d[d.reg == 0].reset_index(drop=True); s, t = leg.ai_use_se.values, leg.t.values; N = len(s); med = float(np.median(s)**2)
R = {}

def rw_noise(tt, p, n):
    T = tt - tt[0]; dT = np.diff(T); L = np.zeros((n, len(tt)))
    L[:, 1:] = np.cumsum(rng.standard_normal((n, len(tt) - 1)) * np.sqrt(p * dT), axis=1); return L
base = 5 + 0.12 * (t - t[0]); k0 = 24
def effect(fam, delta): return delta * (np.maximum(t - t[k0], 0) if fam == "slope" else (np.arange(N) >= k0).astype(float))
def power_first(le, thr=LU):
    h = le >= thr; first = np.where(h.any(1), h.argmax(1), -1); return float((first >= 0).mean())

ui = UI(s, t)
# ---- 1. oracle decomposition: known-theta LR, single component with sup denominator, full UI
dec = {}; Rc = 400
for p, fam, delta in ((0.06, "slope", 0.4), (0.06, "slope", 0.2), (0.06, "level", 2.0), (0.06, "level", 1.0), (0.03, "slope", 0.2), (0.03, "level", 1.0)):
    Y = base + effect(fam, delta) + rw_noise(t, p, Rc) + rng.standard_normal((Rc, N)) * s
    kw = dict(r_rel=(p / med,), kap2=(1.0,), families=(fam,), **({"tau_slope": (delta,)} if fam == "slope" else {"tau_level": (delta,)}))
    uo = UI(s, t, **kw); idx = uo.ri[float(p)] if float(p) in uo.ri else int(np.argmin(np.abs(uo.rs - p)))
    known = power_first(uo.log_e(Y, fixed=(idx, 1.0))); single = power_first(uo.log_e(Y)); full = power_first(ui.log_e(Y))
    dec[f"p{p}_{fam}_{delta}"] = dict(known_theta_LR=known, single_component_sup=single, full_universal=full)
    print("1", p, fam, delta, dec[f"p{p}_{fam}_{delta}"], flush=True)
R["oracle_decomposition"] = dec

# ---- 2. calibrated GLR (ML) comparator; threshold = least favourable over roughness values of the 95% null quantile of max_n GLR
cal = {}
for rr in (0.0, 0.5, 2.0, 8.0, 32.0):
    Y = base + rw_noise(t, rr * med, 300) + rng.standard_normal((300, N)) * s
    cal[rr] = float(np.quantile(ui.log_glr(Y).max(1), 0.95))
thr = max(cal.values()); R["glr_calibration"] = dict(by_r_rel=cal, threshold=thr)
print("2 cal", cal, flush=True)
glr_pow = {}
for p, fam, delta in ((0.0, "slope", 0.2), (0.06, "slope", 0.2), (0.06, "slope", 0.4), (0.03, "slope", 0.2), (0.0, "level", 1.0), (0.06, "level", 1.0), (0.06, "level", 2.0), (0.03, "level", 1.0)):
    Y = base + effect(fam, delta) + (rw_noise(t, p, 400) if p > 0 else 0) + rng.standard_normal((400, N)) * s
    glr_pow[f"p{p}_{fam}_{delta}"] = dict(glr_calibrated=float((ui.log_glr(Y).max(1) >= thr).mean()), universal=power_first(ui.log_e(Y)))
    print("2", p, fam, delta, glr_pow[f"p{p}_{fam}_{delta}"], flush=True)
R["glr_power"] = glr_pow

# ---- 3. enlarged null class: AR(1) latent level
def ar_cov(v, phi): return lambda tt: v * phi ** np.abs(np.subtract.outer(tt, tt))
extra = {f"ar_v{vr}_phi{ph}": ar_cov(vr * med, ph) for vr in (0.5, 2, 8) for ph in (0.5, 0.9, 0.98)}
ui_x = UI(s, t, extra=extra, extra_in_prior=True)
def ar_level(v, phi, n):
    L = np.zeros((n, N)); L[:, 0] = rng.standard_normal(n) * np.sqrt(v)
    for j in range(1, N):
        a = phi ** (t[j] - t[j - 1]); L[:, j] = a * L[:, j - 1] + np.sqrt(v * (1 - a * a)) * rng.standard_normal(n)
    return L
cls = {}
for v, phi in ((0.1, 0.9), (0.3, 0.98)):
    Y = base + ar_level(v, phi, 400) + rng.standard_normal((400, N)) * s
    cls[f"ar_level_v{v}_phi{phi}"] = dict(rw_class=float((ui.log_e(Y).max(1) >= LU).mean()), enlarged_class=float((ui_x.log_e(Y).max(1) >= LU).mean()))
Y = base + effect("slope", 0.4) + rw_noise(t, 0.06, 400) + rng.standard_normal((400, N)) * s
cls["power_slope0.4_rw0.06"] = dict(rw_class=power_first(ui.log_e(Y)), enlarged_class=power_first(ui_x.log_e(Y)))
R["enlarged_class"] = cls; print("3", cls, flush=True)

# ---- 4. spike-in on the BTOS design under the fitted null: detection curves and minimum detectable changes
uf = UI(s_all, t_all, reg=reg_all); fit = uf.reml(y_all); r_hat, k2 = fit["r"], fit["kappa2"]
X = uf._X(len(y_all)); Sig = k2 * (np.diag(s_all**2) + r_hat * rw_cov(t_all)); Si = np.linalg.inv(Sig)
mean = X @ np.linalg.solve(X.T @ Si @ X, X.T @ Si @ y_all); Lc = np.linalg.cholesky(Sig)
Nn = len(y_all); kk = 24
def inj(fam, delta): return delta * (np.maximum(t_all - t_all[kk], 0) if fam == "slope" else (np.arange(Nn) >= kk).astype(float))
curve = {}
for fam, grid in (("slope", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)), ("level", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0))):
    row = {}
    for dl in grid:
        Y = mean + inj(fam, dl) + (Lc @ rng.standard_normal((Nn, 300))).T
        row[str(dl)] = power_first(uf.log_e(Y)[:, :]); print("4", fam, dl, row[str(dl)], flush=True)
    curve[fam] = row
R["spike_in"] = curve

# ---- 5. positive control: the wording change treated as ordinary data (no regime column)
u0 = UI(s_all, t_all); l0, p0 = u0.log_e(y_all, parts=True)
R["positive_control_wording"] = dict(final_logE=float(l0[0, -1]), max_logE=float(l0[0].max()), level_family_max=float(p0["level"][0].max()),
                                      date_of_max=str(d.collection_end[int(l0[0].argmax())].date()))
print("5", R["positive_control_wording"], flush=True)

# ---- 6. model comparison (maximum likelihood, AIC) for observed and planned use
def irw_cov(tt):
    T = tt - tt[0]; m = np.minimum.outer(T, T); df = np.abs(np.subtract.outer(T, T)); return m**3 / 3 + df * m**2 / 2
def fit_model(y, s_, t_, reg_, quad, comps):
    n = len(y); Xl = [(reg_ == g).astype(float) for g in np.unique(reg_)] + [t_ - t_.mean()]
    if quad: Xl.append((t_ - t_.mean())**2)
    X = np.column_stack(Xl); D = np.diag(s_**2); mats = [c(t_) for c in comps]
    def nll(th):
        Sg = np.exp(th[0]) * D + sum(np.exp(th[i + 1]) * m for i, m in enumerate(mats))
        try: L = np.linalg.cholesky(Sg)
        except np.linalg.LinAlgError: return 1e12
        Li = np.linalg.solve(L, X); z = np.linalg.solve(L, y); beta = np.linalg.lstsq(Li, z, rcond=None)[0]
        res = z - Li @ beta; return 0.5 * (res @ res) + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)
    best = None
    for st in ([0.0] + [np.log(0.05)] * len(mats), [0.0] + [np.log(0.5)] * len(mats), [np.log(2.0)] + [np.log(0.01)] * len(mats)):
        o = minimize(nll, st, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-5, fatol=1e-8))
        if best is None or o.fun < best.fun: best = o
    k = X.shape[1] + 1 + len(mats); return dict(loglik=float(-best.fun), k=k, aic=float(2 * best.fun + 2 * k), params=[float(np.exp(v)) for v in best.x])
models = {"line + independent noise": (False, []), "quadratic + independent noise": (True, []),
          "line + random-walk level": (False, [rw_cov]), "quadratic + random-walk level": (True, [rw_cov]),
          "line + random slope (integrated random walk)": (False, [irw_cov]), "line + random-walk level + random slope": (False, [rw_cov, irw_cov]),
          "line + AR(1) level (phi=0.9)": (False, [lambda tt: 0.9 ** np.abs(np.subtract.outer(tt, tt))])}
mc = {}
for series, (yy, ss) in {"current use": (y_all, s_all), "planned use": (d.ai_plan_pct.values, d.ai_plan_se.values)}.items():
    mc[series] = {name: fit_model(yy, ss, t_all, reg_all, q, c) for name, (q, c) in models.items()}
    print("6", series, {k: round(v["aic"], 1) for k, v in mc[series].items()}, flush=True)
R["model_comparison"] = mc

# ---- 7. planned-use series: roughness and universal e-process
up = UI(d.ai_plan_se.values, t_all, reg=reg_all); yp = d.ai_plan_pct.values; fp = up.reml(yp); lp, pp_ = up.log_e(yp, parts=True)
rg, ll = fp["profile"]
R["planned_use"] = dict(r=fp["r"], kappa2=fp["kappa2"], p=fp["r"] * fp["kappa2"], lr_stat=float(2 * (ll.max() - ll[0])), max_logE=float(lp[0].max()),
                        by_family={f: float(pp_[f][0].max()) for f in pp_})
print("7", R["planned_use"], flush=True)

# ---- 8. roughness inflation of independence-based statistics on the BTOS design
Xf = X; D = np.diag(s_all**2); Di = np.linalg.inv(D); P0 = Di - Di @ Xf @ np.linalg.solve(Xf.T @ Di @ Xf, Xf.T @ Di); Wm = rw_cov(t_all)
infl = {}
for lab, z in (("curvature", (t_all - t_all.mean())**2), ("slope change at wave 20", np.maximum(t_all - t_all[19], 0)), ("level shift at wave 20", (np.arange(Nn) >= 19).astype(float)),
               ("slope change at wave 35", np.maximum(t_all - t_all[34], 0)), ("level shift at wave 35", (np.arange(Nn) >= 34).astype(float))):
    q = float(z @ P0 @ z); lam = float(r_hat * (z @ P0 @ Wm @ P0 @ z) / q)          # Var(Z) = kappa^2 (1 + lambda)
    var = k2 * (1 + lam); infl[lab] = dict(lambda_=lam, var_Z=var, fixed_n_test_false_positive=float(2 * (1 - norm.cdf(1.96 / np.sqrt(var)))))
R["roughness_inflation"] = infl; print("8", infl, flush=True)
json.dump(R, open("results/ui_experiments2.json", "w"), indent=1, default=float)
print("done")

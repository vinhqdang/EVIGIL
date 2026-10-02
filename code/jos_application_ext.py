"""Primary analysis and sensitivity analyses on the BTOS series, and the state-space comparison.
Usage (from repository root): python code/jos_application_ext.py"""
import json, numpy as np
from scipy.optimize import minimize
from evigil import load_national, TAU, surveil
from evigil_ext import *

d = load_national()
y, s, t, reg = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values
dt = list(d.collection_end); date = lambda i: str(dt[i].date()); nleg = int((reg == 0).sum())
R = {}
def summ(dec):
    out = []
    for x in dec:
        r = dict(date=date(x["wave"]), dir=x["dir"], E=x["E"], start=date(x["start"]), n=x["wave"] - x["start"] + 1)
        if "A" in x:
            r["gamma"] = x["A"] / x["B"]
            r["gamma_hw"] = 1.96 * (np.sqrt(x["rss"] / x["nu"]) if "rss" in x else 1.0) / np.sqrt(x["B"])
            if "rss" in x: r["kappa_hat"] = float(np.sqrt(x["rss"] / x["nu"]))
        out.append(r)
    return out
def peak_last(dec, tr):
    if len(dec) >= len(tr): return None
    p = tr[-1][1]; m = max(p, key=lambda r: max(r[1], r[2]))
    return dict(start=date(tr[-1][0]), n_waves=len(p) + 2, max_decel=max(r[2] for r in p), max_accel=max(r[1] for r in p),
                date_max_decel=date(max(p, key=lambda r: r[2])[0]), p_value=min(1.0, 2 / max(max(r[1], r[2]) for r in p)))

# ---- primary monitor: unknown scale, mixture over the prior scale
dec, tr = surveil_ext(y, s, t, reg, mode="unknown_mix")
R["primary"] = summ(dec); R["primary_last"] = peak_last(dec, tr)
R["primary_epochs"] = [dict(start=date(st), n_waves=len(p) + 2) for st, p in tr]
# ---- known scale, single tau (the basic monitor) for comparison
dk, tk = surveil_ext(y, s, t, reg, mode="known"); R["basic"] = summ(dk); R["basic_last"] = peak_last(dk, tk)
# ---- sensitivity of the primary monitor
R["primary_one_wave_dropped"] = summ(surveil_ext(y, s, t, np.where(np.arange(len(y)) == 12, 2, reg), mode="unknown_mix")[0])
R["primary_level_shift_mar2024"] = summ(surveil_ext(y, s, t, np.where(np.arange(len(y)) >= 12, np.where(reg == 1, 2, 1), 0), mode="unknown_mix")[0])
dl_, tl_ = surveil_ext(y, s, t, np.where(np.arange(len(y)) >= 12, np.where(reg == 1, 2, 1), 0), mode="unknown_mix")
R["primary_level_shift_last"] = peak_last(dl_, tl_)
R["primary_start"] = {}
for k in (0, 3, 6):
    dd_, _ = surveil_ext(y[k:], s[k:], t[k:], reg[k:], mode="unknown_mix")
    for x in dd_: x["wave"] += k; x["start"] += k
    R["primary_start"][str(k)] = summ(dd_)
R["unknown_single_tau"] = {}
for tau in (0.005, 0.01, 0.02, 0.04):
    dd_, tt_ = surveil_ext(y, s, t, reg, mode="unknown", tau=tau)
    R["unknown_single_tau"][str(tau)] = dict(decl=summ(dd_), last=peak_last(dd_, tt_))
R["gamma0_known"] = {str(g): summ(surveil_ext(y, s, t, reg, mode="known", gamma0=g)[0]) for g in (0.0, 0.001, 0.002, 0.003, 0.004, 0.005)}
R["regime_slopes_guard"] = summ(surveil_ext(y, s, t, reg, mode="unknown_mix", regime_slopes=True, guard=1)[0])
yl, sl = np.log(y), s / y
R["log_scale_unknown_mix"] = {}
dd_, _ = surveil_ext(yl, sl, t, reg, mode="unknown", tau=0.002); R["log_scale_unknown_tau0.002"] = summ(dd_)

# ---- local-linear-trend comparison, by dense Gaussian conditioning
n = len(y); dlt = np.diff(t)
def mats():
    """x = [L1, b1, c, eta_2..eta_n, eps_2..eps_n]; returns TL, Tb (n x dim)."""
    m = 3 + 2 * (n - 1); TL = np.zeros((n, m)); Tb = np.zeros((n, m))
    TL[0, 0] = 1; Tb[0, 1] = 1
    for j in range(1, n):
        Tb[j] = Tb[j - 1]; Tb[j, 3 + j - 1] = 1                              # b_j = b_{j-1} + eta_j
        TL[j] = TL[j - 1] + dlt[j - 1] * Tb[j - 1]; TL[j, 3 + (n - 1) + j - 1] = 1  # L_j = L_{j-1} + d b_{j-1} + eps_j
    return TL, Tb
TL, Tb = mats()
H = TL.copy(); H[:, 2] = reg                                                   # step enters through x[2]
def prior(q, p):
    v = np.concatenate([[1e3, 1e3, 1e3], q * dlt, p * dlt]); return v
def nll(theta, which, kfix):
    q = np.exp(theta[0]) if which in ("irw", "llt") else 0.0
    p = np.exp(theta[1]) if which == "llt" else (np.exp(theta[0]) if which == "ll" else 0.0)
    kap = np.exp(theta[-1]) if kfix is None else kfix
    S = (H * prior(q, p)) @ H.T + np.diag((kap * s)**2)
    try: L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e9
    z = np.linalg.solve(L, y); return 0.5 * (z @ z) + np.log(np.diag(L)).sum()
def fit(which, kfix):
    k0_ = {"irw": 1, "llt": 2, "ll": 1}[which]
    x0 = [np.log(0.01)] * k0_ + ([] if kfix is not None else [0.3])
    o = minimize(nll, x0, args=(which, kfix), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-8, maxiter=3000))
    return o
dates_ = ("2024-03-10", "2024-06-30", "2024-11-17", "2025-04-06", "2025-07-27", "2025-10-05")
idx = {dd: [i for i in range(n) if date(i) == dd][0] for dd in dates_}
def acf(x, k): x = x - x.mean(); return float((x[:-k] * x[k:]).sum() / (x * x).sum())
SS = {}
for which, lab in (("irw", "smooth_trend"), ("llt", "local_linear_trend"), ("ll", "local_level_constant_slope")):
    for kname, kfix in (("kappa_free", None), ("kappa_1", 1.0)):
        o = fit(which, kfix); th = o.x
        q = float(np.exp(th[0])) if which in ("irw", "llt") else 0.0
        p = float(np.exp(th[1])) if which == "llt" else (float(np.exp(th[0])) if which == "ll" else 0.0)
        kap = float(np.exp(th[-1])) if kfix is None else 1.0
        Sx = prior(q, p); Sy = (H * Sx) @ H.T + np.diag((kap * s)**2)
        G = (H * Sx).T @ np.linalg.inv(Sy)                                     # Sigma_x H' Sy^-1
        mean = G @ y; cov = np.diag(Sx) - G @ (H * Sx)
        sl_m = Tb @ mean; sl_c = Tb @ cov @ Tb.T
        L = np.linalg.cholesky(Sy); inn = np.linalg.solve(L, y)[8:nleg]
        diff = sl_m[idx["2025-07-27"]] - sl_m[idx["2025-04-06"]]
        dsd = float(np.sqrt(sl_c[idx["2025-07-27"], idx["2025-07-27"]] + sl_c[idx["2025-04-06"], idx["2025-04-06"]]
                            - 2 * sl_c[idx["2025-07-27"], idx["2025-04-06"]]))
        SS[f"{lab}_{kname}"] = dict(loglik=float(-o.fun), q=q, p=p, kappa=kap,
            slope={dd: (float(sl_m[i]), float(1.96 * np.sqrt(sl_c[i, i]))) for dd, i in idx.items()},
            diff_jul_apr=(float(diff), float(1.96 * dsd)),
            innov_acf={str(k): acf(inn, k) for k in (1, 2, 3, 6, 12)}, innov_sd=float(inn.std()), innov_n=len(inn))
R["state_space"] = SS
json.dump(R, open("results/application_ext.json", "w"), indent=1, default=float)
def short(l): return [(x["date"], x["dir"], f"{x['E']:.3g}") for x in l]
for k in ("primary", "basic", "primary_one_wave_dropped", "primary_level_shift_mar2024", "regime_slopes_guard", "log_scale_unknown_tau0.002"): print(k, short(R[k]))
print("primary_last", R["primary_last"]); print("level_shift_last", R["primary_level_shift_last"])
for k, v in R["primary_start"].items(): print("start", k, short(v))
for k, v in R["unknown_single_tau"].items(): print("tau", k, short(v["decl"]), v["last"] and round(v["last"]["max_decel"], 1))
for k, v in R["gamma0_known"].items(): print("gamma0", k, short(v))
for k, v in SS.items(): print(k, round(v["loglik"], 2), "kappa", round(v["kappa"], 2), "q", f"{v['q']:.2g}", "p", f"{v['p']:.2g}", {a: tuple(round(c, 2) for c in b) for a, b in v["slope"].items()}, "diff", tuple(round(c, 3) for c in v["diff_jul_apr"]))

# ---- Figure 1: the series with the declarations of the primary monitor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, pandas as pd
plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
BREAK = pd.Timestamp("2025-11-17")
fig, ax = plt.subplots(figsize=(7.2, 3.5))
l = d[d.reg == 0]; r_ = d[d.reg == 1]
ax.errorbar(l.collection_end, l.ai_use_pct, yerr=1.96 * l.ai_use_se, fmt="-o", ms=2.5, lw=1, elinewidth=.5, color="k", label="Legacy wording")
ax.errorbar(r_.collection_end, r_.ai_use_pct, yerr=1.96 * r_.ai_use_se, fmt="s", ms=4, mfc="white", color="k", elinewidth=.5, label="Revised wording")
for k_, x in enumerate(dec):
    ax.axvline(dt[x["wave"]], color="0.35", ls="--", lw=1)
    ax.text(dt[x["wave"]], 19.3 - 0.0 * k_, " " + ("Deceleration" if x["dir"] == "decel" else "Acceleration") + "\n " + dt[x["wave"]].strftime("%-d %b %Y"),
            fontsize=7, va="top", ha="left" if k_ % 2 == 0 else "right")
ax.axvline(BREAK, color="k", ls=":", lw=1.3); ax.text(BREAK, 12.5, "Wording\nchange ", fontsize=7, ha="right")
ax.set_ylim(2, 20); ax.set_ylabel("Share of businesses using AI (%)"); ax.legend(fontsize=7.5, loc="center left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig("manuscripts/jos/fig_series.pdf"); plt.close(fig)

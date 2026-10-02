"""Empirical analysis for the JOS manuscript (BTOS AI-use series).
Usage (from repository root): python code/jos_analysis.py
Writes numbers to results/analysis.json and figures to manuscripts/jos/."""
import json, os, numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from evigil import *

os.makedirs("results", exist_ok=True)
OUT = "manuscripts/jos"
d = load_national()
y, s, t, reg, dt = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values, list(d.collection_end)
date = lambda i: str(dt[i].date())
R = {}
nleg = int((reg == 0).sum())
R["n_waves"], R["n_legacy"] = len(d), nleg

# ---- diagnostics of the published standard errors (legacy regime)
z1 = np.diff(y[:nleg]) / np.hypot(s[1:nleg], s[:nleg-1])
R["acf1_increments"] = float(np.corrcoef(z1[:-1], z1[1:])[0, 1])
d2 = y[2:nleg] - 2 * y[1:nleg-1] + y[:nleg-2]
z2 = d2 / np.sqrt(s[2:nleg]**2 + 4 * s[1:nleg-1]**2 + s[:nleg-2]**2)
R["sd_z2"] = float(z2.std()); R["kappa_mad"] = float(1.4826 * np.median(np.abs(z2 - np.median(z2))))
R["kappa_hat"] = R["kappa_mad"]

def run(tau=TAU, pool=False, thresh=THRESH, kappa=1.0, gamma0=0.0, regv=None, ss=None):
    ss = s if ss is None else ss
    rv = reg if regv is None else regv
    dec, tr = surveil(y, ss * kappa, t, rv, tau=tau, pool_regimes=pool, thresh=thresh, gamma0=gamma0)
    return dec, tr, [dict(date=date(x["wave"]), dir=x["dir"], E=x["E"], start=date(x["start"]),
                          n=x["wave"] - x["start"] + 1, gamma=x["A"] / x["B"], gamma_hw=1.96 / x["B"]**.5) for x in dec]

dec, tr, R["declarations"] = run()
decp, trp, R["declarations_pooled"] = run(pool=True)
R["tau_sensitivity"] = {str(k): run(tau=k)[2] for k in (0.0025, 0.005, 0.01, 0.02, 0.04)}
R["alpha10"] = run(thresh=20.0)[2]
R["kappa_sensitivity"], R["kappa_last_peak"] = {}, {}
for k in (1.0, 1.25, 1.5, round(R["kappa_hat"], 2)):
    dk_, tk_, rk_ = run(kappa=k)
    R["kappa_sensitivity"][str(k)] = rk_
    R["kappa_last_peak"][str(k)] = max(r[2] for r in tk_[-1][1]) if len(dk_) < len(tk_) else None
R["gamma0_sensitivity"] = {str(g): run(gamma0=g)[2] for g in (0.0, 0.0005, 0.001, 0.002)}
# smoothed / constant standard errors
R["se_constant"] = run(ss=np.full_like(s, np.median(s)))[2]

# last epoch: peak evidence and anytime-valid p-value
last = tr[-1][1]
pk = max(last, key=lambda r: r[2])
R["last_epoch"] = dict(start=date(tr[-1][0]), n_waves=len(last) + 2, max_decel=pk[2], date_max_decel=date(pk[0]),
                       p_decel=min(1.0, 2 / pk[2]), max_accel=max(r[1] for r in last))
ep = []
for (st, p), dd in zip(tr, dec + [None] * len(tr)):
    if dd is not None: p = [r for r in p if r[0] <= dd["wave"]]
    ep.append(dict(start=date(st), n_waves=len(p) + 2, declared=dd is not None))
R["epochs"] = ep

# ---- the March 2024 wave
i = 12
R["march2024"] = dict(date=date(i), prev=float(y[i-1]), val=float(y[i]), z=float((y[i]-y[i-1]) / np.hypot(s[i], s[i-1])),
                      within_rank=int((np.abs(z1) >= abs(z1[i-1])).sum()))
one = (np.arange(len(y)) == i).astype(int) * 2 + reg          # wave i alone as its own regime (absorbed)
one = np.where(np.arange(len(y)) == i, 2, reg)
from_i = np.where(np.arange(len(y)) >= i, np.where(reg == 1, 2, 1), 0)    # persistent level shift from wave i
R["march2024_sens"] = dict(drop_wave=run(regv=one)[2], level_shift=run(regv=from_i)[2])

# ---- redesign
jump = y[nleg] - y[nleg-1]
R["redesign"] = dict(jump_pp=float(jump), jump_z=float(jump / np.hypot(s[nleg], s[nleg-1])),
                     within_median=float(np.median(np.abs(z1))), within_max=float(np.abs(z1).max()))

# ---- comparators (curvature z statistic, same restart rule), regime-aware and pooled
def curvature_z(n, start, pool):
    A, B = score(y[start:n], s[start:n], t[start:n], np.zeros(n - start, int) if pool else reg[start:n])
    return A / np.sqrt(B) if B > 0 else 0.0
def repeated(kind, pool):
    start, out = 0, []
    while start + 3 <= len(y):
        ns = list(range(start + 3, len(y) + 1)); hit = None
        for k, n in enumerate(ns, 1):
            zz = abs(curvature_z(n, start, pool))
            c = norm.ppf(.975) if kind == "uncorrected" else norm.ppf(1 - .025 / len(ns)) if kind == "bonferroni" \
                else norm.ppf(1 - .025 / (k * (k + 1)))
            if zz > c: hit = n - 1; break
        if hit is None: break
        out.append(hit); start = hit + 1
    return out
R["comparators"] = {f"{k}_{m}": [date(i_) for i_ in repeated(k, m == "pooled")]
                    for k in ("uncorrected", "bonferroni", "spending") for m in ("aware", "pooled")}
ma = pd.Series(y).rolling(3).mean() - pd.Series(y).rolling(8).mean(); sg = np.sign(ma.values)
R["comparators"]["ma_crossings"] = [date(i_) for i_ in range(1, len(y)) if not np.isnan(sg[i_]) and not np.isnan(sg[i_-1])
                                    and sg[i_] != sg[i_-1] and sg[i_] != 0]
R["max_abs_curv_z_at_redesign"] = float(abs(curvature_z(nleg + 1, 0, True)))

# ---- strata
siz = pd.read_csv("data/btos_ai_sizeclass.csv", parse_dates=["collection_end"])
lab = {"A": "1--4", "B": "5--9", "C": "10--19", "D": "20--49", "E": "50--99", "F": "100--249", "G": "250+"}
rows, zk_all, p95s = [], {}, {}
for k in "ABCDEFG":
    g = siz[siz.size_class == k].sort_values("collection_end").reset_index(drop=True)
    gt = ((g.collection_end - d.collection_end[0]).dt.days / 14.0).values
    yk, sk, rk = g.ai_use_pct.values, g.ai_use_se.values, (g.wording == "revised").astype(int).values
    nl = int((rk == 0).sum())
    dk, _ = surveil(yk, sk, gt, rk); dp, _ = surveil(yk, sk, gt, rk, pool_regimes=True)
    zk = np.diff(yk[:nl]) / np.hypot(sk[1:nl], sk[:nl-1]); zk_all[k] = zk; p95s[k] = float(np.percentile(np.abs(zk), 95))
    jk = yk[nl] - yk[nl-1]
    rows.append(dict(size=lab[k], first=yk[0], last_legacy=yk[nl-1], first_revised=yk[nl], med_se=float(np.median(sk[:nl])),
                     jump_pp=float(jk), jump_z=float(jk / np.hypot(sk[nl], sk[nl-1])), ratio=float(yk[nl] / yk[nl-1]),
                     p95=p95s[k],
                     decl_aware=[(str(g.collection_end[x["wave"]].date()), x["dir"]) for x in dk],
                     decl_pooled_last=(str(g.collection_end[dp[-1]["wave"]].date()), dp[-1]["dir"]) if dp else None))
R["strata"] = rows
# cross-strata co-movement at every legacy wave: number of classes beyond their own p95 with the same sign
co = []
for j in range(nleg - 1):
    zs = np.array([zk_all[k][j] for k in "ABCDEFG"]); ex = np.array([abs(zk_all[k][j]) > p95s[k] for k in "ABCDEFG"])
    for sgn in (1, -1):
        n_ = int((ex & (np.sign(zs) == sgn)).sum())
        if n_ >= 3: co.append(dict(date=date(j + 1), n_classes=n_, sign=sgn, national_z=float(z1[j])))
R["comovement"] = co
R["comovement_max_legacy"] = max([c["n_classes"] for c in co], default=0)
json.dump(R, open("results/analysis.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k != "strata"}, indent=1, default=float))

# ---------------------------- figures (grey-scale safe) ----------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
BREAK = pd.Timestamp("2025-11-17")
fig, ax = plt.subplots(figsize=(7.0, 3.0)); x = np.arange(7)
ax.bar(x, [r["jump_z"] for r in rows], .55, color="0.55", edgecolor="k", label="Jump at wording change")
ax.plot(x, [r["p95"] for r in rows], "kD", ms=5, mfc="white", label="95th percentile, within regime")
ax.set_xticks(x); ax.set_xticklabels([r["size"].replace("--", "-") for r in rows])
ax.set_xlabel("Employment size class"); ax.set_ylabel("Standardized |z|"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_strata.pdf"); plt.close(fig)

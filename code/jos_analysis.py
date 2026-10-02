"""Empirical analysis for the JOS manuscript (BTOS AI-use series).
Usage (from repository root): python code/jos_analysis.py
Writes numbers to results/analysis.json and figures to manuscripts/jos/."""
import json, os, numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from evigil import *
from evigil import path as epath

os.makedirs("results", exist_ok=True)
OUT = "manuscripts/jos"
d = load_national()
y, s, t, reg, dt = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values, list(d.collection_end)
date = lambda i: str(dt[i].date())
R = {}
nleg = int((reg == 0).sum())
R["n_waves"], R["n_legacy"] = len(d), nleg

# lag-1 autocorrelation of standardised increments (legacy)
z = np.diff(y[:nleg]) / np.sqrt(s[1:nleg]**2 + s[:nleg-1]**2)
R["acf1_increments"] = float(np.corrcoef(z[:-1], z[1:])[0, 1])

def run(tau=TAU, pool=False, thresh=THRESH):
    dec, tr = surveil(y, s, t, reg, tau=tau, pool_regimes=pool, thresh=thresh)
    return dec, tr, [dict(date=date(x["wave"]), dir=x["dir"], E=x["E"], start=date(x["start"]),
                          n=x["wave"] - x["start"] + 1) for x in dec]

dec, tr, R["declarations"] = run()
decp, trp, R["declarations_pooled"] = run(pool=True)
R["tau_sensitivity"] = {str(k): run(tau=k)[2] for k in (0.0025, 0.005, 0.01, 0.02, 0.04)}
R["alpha10"] = run(thresh=20.0)[2]
# graded evidence within the last (regime-aware) epoch
lastpath = tr[-1][1]
R["last_epoch"] = dict(start=date(tr[-1][0]),
    max_decel=max(r[2] for r in lastpath), max_accel=max(r[1] for r in lastpath),
    date_max_decel=date(max(lastpath, key=lambda r: r[2])[0]))
# epochs: peak and amber/strong-amber crossing dates per epoch
ep = []
for (st, p), dd in zip(tr, dec + [None] * len(tr)):
    if dd is not None:
        p = [r for r in p if r[0] <= dd["wave"]]
    e = dict(start=date(st), n_waves=len(p) + 2, declared=dd is not None)
    for key, j in (("accel", 1), ("decel", 2)):
        v = [r[j] for r in p]; k = int(np.argmax(v))
        e[key] = dict(max=v[k], date=date(p[k][0]),
                      amber=next((date(r[0]) for r in p if r[j] >= 2), None),
                      strong=next((date(r[0]) for r in p if r[j] >= 10), None))
    ep.append(e)
R["epochs"] = ep

# redesign increment
jump = y[nleg] - y[nleg-1]
R["redesign"] = dict(jump_pp=float(jump), jump_z=float(jump / np.hypot(s[nleg], s[nleg-1])),
                     within_median=float(np.median(np.abs(z))), within_max=float(np.abs(z).max()))

# ---- comparators on the naively spliced series, with the same restart rule
def repeated(kind):
    """Repeated curvature z-test, restarting after each alarm; returns alarm dates."""
    start, out = 0, []
    while start + 3 <= len(y):
        ns = list(range(start + 3, len(y) + 1)); hit = None
        for k, n in enumerate(ns, 1):
            zz = abs(curvature_z(n, start))
            if kind == "uncorrected": c = norm.ppf(.975)
            elif kind == "bonferroni": c = norm.ppf(1 - .025 / len(ns))
            else: c = norm.ppf(1 - .025 / (k * (k + 1)))
            if zz > c: hit = n - 1; break
        if hit is None: break
        out.append(hit); start = hit + 1
    return out
def curvature_z(n, start=0):
    A, B = score(y[start:n], s[start:n], t[start:n], np.zeros(n - start, int))
    return A / np.sqrt(B)
ma = pd.Series(y).rolling(3).mean() - pd.Series(y).rolling(8).mean()
sg = np.sign(ma.values)
macr = [i for i in range(1, len(y)) if not np.isnan(sg[i]) and not np.isnan(sg[i-1]) and sg[i] != sg[i-1] and sg[i] != 0]
R["comparators"] = {k: [date(i) for i in repeated(k)] for k in ("uncorrected", "bonferroni", "spending")}
R["comparators"]["ma_crossings"] = [date(i) for i in macr]
zr = [abs(curvature_z(n)) for n in range(3, len(y) + 1)]
R["max_abs_curv_z_at_redesign"] = float(max(zr[nleg - 3], zr[nleg - 2]))

# ---- strata
siz = pd.read_csv("data/btos_ai_sizeclass.csv", parse_dates=["collection_end"])
lab = {"A": "1--4", "B": "5--9", "C": "10--19", "D": "20--49", "E": "50--99", "F": "100--249", "G": "250+"}
rows = []
for k in "ABCDEFG":
    g = siz[siz.size_class == k].sort_values("collection_end").reset_index(drop=True)
    gt = ((g.collection_end - d.collection_end[0]).dt.days / 14.0).values
    yk, sk, rk = g.ai_use_pct.values, g.ai_use_se.values, (g.wording == "revised").astype(int).values
    nl = int((rk == 0).sum())
    dk, _ = surveil(yk, sk, gt, rk)
    zk = np.diff(yk[:nl]) / np.hypot(sk[1:nl], sk[:nl-1])
    jk = yk[nl] - yk[nl-1]
    dp, _ = surveil(yk, sk, gt, rk, pool_regimes=True)
    rows.append(dict(size=lab[k], first=yk[0], last_legacy=yk[nl-1], first_revised=yk[nl], med_se=float(np.median(sk[:nl])),
                     jump_pp=float(jk), jump_z=float(jk / np.hypot(sk[nl], sk[nl-1])),
                     p95=float(np.percentile(np.abs(zk), 95)),
                     decl_aware=[(str(g.collection_end[x["wave"]].date()), x["dir"]) for x in dk],
                     decl_pooled_last=(str(g.collection_end[dp[-1]["wave"]].date()), dp[-1]["dir"]) if dp else None))
R["strata"] = rows
json.dump(R, open("results/analysis.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k != "strata"}, indent=1, default=float))
for r in rows: print(r)

# ---------------------------- figures (grey-scale safe) ----------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
BREAK = pd.Timestamp("2025-11-17")
fig, ax = plt.subplots(figsize=(7.2, 3.5))
l = d[d.reg == 0]; r_ = d[d.reg == 1]
ax.errorbar(l.collection_end, l.ai_use_pct, yerr=1.96 * l.ai_use_se, fmt="-o", ms=2.5, lw=1, elinewidth=.5, color="k",
            label="Legacy wording")
ax.errorbar(r_.collection_end, r_.ai_use_pct, yerr=1.96 * r_.ai_use_se, fmt="s", ms=4, mfc="white", color="k", elinewidth=.5,
            label="Revised wording")
for x in dec:
    ax.axvline(dt[x["wave"]], color="0.35", ls="--" if x["dir"] == "accel" else "-.", lw=1)
ax.axvline(BREAK, color="k", ls=":", lw=1.3)
ax.set_ylabel("Share of businesses using AI (%)"); ax.legend(fontsize=7.5, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_series.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, ttl, trj in ((axs[0], "(a) Regime-aware monitor", tr), (axs[1], "(b) Pooled (redesign ignored)", trp)):
    for j, (st, p) in enumerate(trj):
        if j < len(trj) - 1:
            p = [r for r in p if r[0] <= (dec if trj is tr else decp)[j]["wave"]]
        x = [dt[r[0]] for r in p]
        ax.semilogy(x, np.maximum([r[1] for r in p], 1e-3), color="k", lw=1.2, label="Acceleration" if j == 0 else None)
        ax.semilogy(x, np.maximum([r[2] for r in p], 1e-3), color="0.45", lw=1.2, ls="--", label="Deceleration" if j == 0 else None)
    ax.axhline(40, color="k", ls=":", lw=.9); ax.axvline(BREAK, color="k", ls="-.", lw=.8)
    ax.set_title(ttl, fontsize=9); ax.set_ylim(1e-3, 1e8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
axs[0].set_ylabel("E-value (log scale)"); axs[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_evalues.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 3.0)); x = np.arange(7)
ax.bar(x, [r["jump_z"] for r in rows], .55, color="0.55", edgecolor="k", label="Jump at wording change")
ax.plot(x, [r["p95"] for r in rows], "kD", ms=5, mfc="white", label="95th percentile, within regime")
ax.set_xticks(x); ax.set_xticklabels([r["size"].replace("--", "-") for r in rows])
ax.set_xlabel("Employment size class"); ax.set_ylabel("Standardised |z|"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_strata.pdf"); plt.close(fig)

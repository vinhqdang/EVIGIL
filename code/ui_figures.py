"""Figures for the robust-monitoring manuscript. Usage: PYTHONPATH=code python code/ui_figures.py"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from evigil import load_national
from evigil_ui import UI, rw_cov
plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
OUT = "manuscripts/robust/"
E = json.load(open("results/ui_experiments.json")); A = json.load(open("results/ui_application.json"))

# Figure 1: failure of independence-based monitors as the horizon grows
fig, ax = plt.subplots(figsize=(5.2, 3.2)); ns = [20, 40, 80, 160]
for p, ls, mk in ((0.0, "-", "o"), (0.01, "--", "s"), (0.03, "-.", "^"), (0.06, ":", "D")):
    ax.plot(ns, [E["failure"][f"n{n}_p{p}"]["basic"] for n in ns], ls=ls, marker=mk, color="k", mfc="white", label=f"roughness $p={p}$")
ax.axhline(0.05, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
ax.set_xlabel("Number of waves $n$"); ax.set_ylabel("Probability of a false declaration"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "fig_failure.pdf"); plt.close(fig)

# Figure 2: power of the universal e-process
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, fam, sizes, lab in ((axs[0], "slope", (0.1, 0.2, 0.4), "Slope change (points per wave)"), (axs[1], "level", (0.5, 1.0, 2.0), "Level shift (points)")):
    for p, ls, mk in ((0.0, "-", "o"), (0.03, "--", "s"), (0.06, ":", "D")):
        ax.plot(sizes, [E["power"][f"p{p}_{fam}_{d}"]["power"] for d in sizes], ls=ls, marker=mk, color="k", mfc="white", label=f"$p={p}$")
        ax.plot(sizes, [E["power"][f"p{p}_{fam}_{d}"]["oracle_power"] for d in sizes], ls=ls, color="0.6", lw=0.8)
    ax.set_xlabel(lab); ax.set_ylim(-0.02, 1.02); ax.set_xticks(sizes)
axs[0].set_ylabel("Probability of a declaration"); axs[0].legend(fontsize=7.5, title="roughness", title_fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "fig_power.pdf"); plt.close(fig)

# Figure 3: BTOS series with the fitted rough-level model, and the universal e-process
d = load_national(); y, s, t, reg = d.ai_use_pct.values, d.ai_use_se.values, d.t.values, d.reg.values; N = len(y)
ui = UI(s, t, reg=reg); fit = ui.reml(y); r_, k2 = fit["r"], fit["kappa2"]
X = ui._X(N); Sig = k2 * (np.diag(s**2) + r_ * rw_cov(t)); Si = np.linalg.inv(Sig)
c = np.linalg.solve(X.T @ Si @ X, X.T @ Si @ y); trend = X @ c
level = trend + (k2 * r_ * rw_cov(t)) @ Si @ (y - trend)             # BLUP of trend + rough level
fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax = axs[0]; l = d[d.reg == 0]; r2 = d[d.reg == 1]
ax.errorbar(l.collection_end, l.ai_use_pct, yerr=1.96 * l.ai_use_se, fmt="o", ms=2.5, elinewidth=.5, color="k", label="Legacy wording")
ax.errorbar(r2.collection_end, r2.ai_use_pct, yerr=1.96 * r2.ai_use_se, fmt="s", ms=4, mfc="white", elinewidth=.5, color="k", label="Revised wording")
dts = d.collection_end
ax.plot(dts[:54], trend[:54], color="0.4", ls="--", lw=1, label="Linear trend (fitted)"); ax.plot(dts[:54], level[:54], color="0.0", lw=1, label="Fitted level with roughness")
ax.plot(dts[54:], trend[54:], color="0.4", ls="--", lw=1); ax.plot(dts[54:], level[54:], color="0.0", lw=1)
ax.set_ylabel("Share of businesses using AI (%)"); ax.legend(fontsize=7, loc="upper left")
le, pp = ui.log_e(y, parts=True); ax = axs[1]
ax.plot(dts[3:], le[0, 3:], color="k", lw=1.1, label="All alternatives"); ax.plot(dts[3:], pp["slope"][0, 3:], color="0.5", ls="--", lw=1, label="Slope change")
ax.plot(dts[3:], pp["level"][0, 3:], color="0.5", ls=":", lw=1.2, label="Level shift"); ax.axhline(np.log(20), color="k", lw=0.8, ls="-.")
ax.text(dts[4], np.log(20) + 0.2, "threshold $1/\\alpha = 20$", fontsize=7); ax.set_ylabel("log E-value"); ax.set_ylim(-5, 4); ax.legend(fontsize=7, loc="lower left", ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig(OUT + "fig_btos.pdf"); plt.close(fig)
print("figures written")

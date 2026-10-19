"""Tables and extra figures for the robust-monitoring manuscript. PYTHONPATH=code python code/ui_tables_figs.py"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
OUT = "manuscripts/robust/"
E = json.load(open("results/ui_experiments.json")); R2 = json.load(open("results/ui_experiments2.json")); A = json.load(open("results/ui_application.json"))
f2 = lambda v: f"{v:.2f}"
def w(name, s): open(OUT + name, "w").write(s)

# oracle decomposition
rows = []
lab = {"slope": "slope change", "level": "level shift"}
for key, v in R2["oracle_decomposition"].items():
    p, fam, dl = key.split("_"); p = p[1:]
    rows.append(f"{p} & {lab[fam]} {dl} & {f2(v['known_theta_LR'])} & {f2(v['single_component_sup'])} & {f2(v['full_universal'])} \\\\")
w("tab_oracle.tex", "\\begin{tabular}{llccc}\n\\toprule\n$p$ & Alternative & Known-$\\theta$ LR & One component, sup & Universal \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# calibrated GLR
rows = []
for key, v in R2["glr_power"].items():
    p, fam, dl = key.split("_"); p = p[1:]
    rows.append(f"{p} & {lab[fam]} {dl} & {f2(v['glr_calibrated'])} & {f2(v['universal'])} \\\\")
w("tab_glr.tex", "\\begin{tabular}{llcc}\n\\toprule\n$p$ & Alternative & Calibrated GLR & Universal e-process \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# model comparison
mc = R2["model_comparison"]; names = list(mc["current use"].keys()); rows = []
for n in names:
    rows.append(f"{n} & {mc['current use'][n]['aic']:.1f} & {mc['planned use'][n]['aic']:.1f} \\\\")
w("tab_models.tex", "\\begin{tabular}{lcc}\n\\toprule\nModel (regime levels in all) & AIC, current use & AIC, planned use \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# roughness inflation
rows = []
for k, v in R2["roughness_inflation"].items():
    rows.append(f"{k} & {v['lambda_']:.0f} & {v['var_Z']:.0f} & {100*v['fixed_n_test_false_positive']:.0f} \\\\")
w("tab_inflation.tex", "\\begin{tabular}{lccc}\n\\toprule\nContrast & Inflation $\\lambda$ & Var of $Z$ & False positives of a nominal 5\\% test (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# spike-in figure with 50 and 80 percent minimum detectable changes
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9)); mdc = {}
for ax, fam, lab_ in ((axs[0], "slope", "Slope change (points per wave)"), (axs[1], "level", "Level shift (points)")):
    xs = np.array([float(k) for k in R2["spike_in"][fam]]); ys = np.array(list(R2["spike_in"][fam].values()))
    ax.plot(xs, ys, "ko-", ms=3.5, mfc="white"); ax.axhline(0.5, color="0.6", lw=0.7, ls=":"); ax.axhline(0.8, color="0.6", lw=0.7, ls="--")
    mdc[fam] = [float(np.interp(q, ys, xs)) for q in (0.5, 0.8)]
    ax.set_xlabel(lab_); ax.set_ylim(-0.02, 1.02)
axs[0].set_ylabel("Probability of a declaration")
fig.tight_layout(); fig.savefig(OUT + "fig_mdc.pdf"); plt.close(fig)
json.dump(mdc, open("results/ui_mdc.json", "w")); print("MDC (50%, 80%):", mdc)

# Figure 1: analytic variance inflation and simulated false-declaration probabilities
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0)); ns = np.array([10, 20, 40, 80, 160, 320])
for p, ls in ((0.01, "--"), (0.0225, "-."), (0.06, ":"), (0.2, "-")):
    axs[0].loglog(ns, 1 + (p / 0.0225) * ns**2 / 42, "k", ls=ls, label=f"$p/\\sigma^2={p/0.0225:.1f}$")
axs[0].axhline(1, color="0.6", lw=0.8); axs[0].set_xlabel("Number of waves $n$"); axs[0].set_ylabel("$\\mathrm{Var}(Z_n)$"); axs[0].legend(fontsize=7, loc="upper left")
ns2 = [20, 40, 80, 160]
for p, ls, mk in ((0.0, "-", "o"), (0.01, "--", "s"), (0.03, "-.", "^"), (0.06, ":", "D")):
    axs[1].plot(ns2, [E["failure"][f"n{n}_p{p}"]["basic"] for n in ns2], ls=ls, marker=mk, color="k", mfc="white", ms=4, label=f"$p={p}$")
axs[1].axhline(0.05, color="0.6", lw=0.8); axs[1].set_xscale("log"); axs[1].set_xticks(ns2); axs[1].set_xticklabels(ns2)
axs[1].set_xlabel("Number of waves $n$"); axs[1].set_ylabel("False-declaration probability"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_failure.pdf"); plt.close(fig)
print("ok")

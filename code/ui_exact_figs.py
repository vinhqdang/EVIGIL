"""Figures and tables for the scale-exact monitor and the ceiling.  Usage: PYTHONPATH=code python code/ui_exact_figs.py"""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "manuscripts/robust/"
E1 = json.load(open("results/ui_exact.json")); E2 = json.load(open("results/ui_exact2.json")); E3 = json.load(open("results/ui_exact3.json"))
P = json.load(open("results/ui_experiments2.json"))["spike_in"]
def interp(xs, ys, q): return float(np.interp(q, ys, xs))
res = {}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, fam, lab in zip(axs, ("slope", "level"), ("Slope change (points per wave)", "Level shift (points)")):
    ex = E1["spike_in_exact"][fam]; pr = P[fam]; ce = E1["ceiling"][fam]
    cur = {"exact": ex, "profile": pr}
    for nm, c, st in (("exact", ex, "-"), ("profile", pr, "--")):
        x = np.array([float(k) for k in c]); y = np.array(list(c.values()))
        ax.plot(x, y, st, color="black", marker="o", ms=3, label={"exact": "scale-exact e-process", "profile": "profile-scale e-process"}[nm])
    x = np.array([float(k) for k in ce]); y = np.array([v["power_ceiling"] for v in ce.values()])
    ax.plot(x, y, ":", color="grey", marker="s", ms=3, label="ceiling for any valid monitor")
    ax.axhline(0.5, color="lightgrey", lw=0.6); ax.axhline(0.8, color="lightgrey", lw=0.6)
    ax.set_xlabel(lab); ax.set_ylabel("Probability of declaring"); ax.set_ylim(-0.02, 1.02)
    res[fam] = {nm: [interp(np.array([float(k) for k in c]), np.array(list(c.values())), q) for q in (0.5, 0.8)] for nm, c in cur.items()}
    res[fam]["ceiling"] = [interp(x, y, q) for q in (0.5, 0.8)]
axs[0].legend(fontsize=6.5, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT + "fig_mdc.pdf"); plt.close(fig)
json.dump(res, open("results/ui_mdc_exact.json", "w"), indent=1); print(json.dumps(res, indent=1))
# decomposition table
rows = []
nm = {("slope"): "slope change", ("level"): "level shift"}
for r in E3:
    rows.append(f"{r['p']} & {nm[r['family']]} {r['delta']} & {r['known']:.2f} & {r['one_sup']:.2f} & {r['universal']:.2f} \\\\")
open(OUT + "tab_oracle.tex", "w").write("\\begin{tabular}{llccc}\n\\toprule\n$p$ & Alternative & Known-$r$ LR & One component, sup & Universal \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# growth table
g = E2["growth"]; rows = []
for k, v in g.items():
    f, dl = k.split("|"); rows.append(f"{nm[f]} {dl} & {v['mean_logE']:.1f} & {v['ceiling']:.1f} & {v['ceiling'] - v['mean_logE']:.1f} \\\\")
open(OUT + "tab_growth.tex", "w").write("\\begin{tabular}{lccc}\n\\toprule\nAlternative & Mean $\\log E_{56}$ & Ceiling $K$ & Gap \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# power table exact vs profile
rows = []
for k, v in E1["power"].items():
    p, f, dl = k.split("|"); rows.append(f"{p} & {nm[f]} {dl} & {v['profile']:.2f} & {v['exact']:.2f} \\\\")
open(OUT + "tab_exactpower.tex", "w").write("\\begin{tabular}{llcc}\n\\toprule\n$p$ & Alternative & Profile scale & Exact scale \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

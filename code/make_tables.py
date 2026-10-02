"""Build LaTeX tables for the JOS manuscript from results/*.json."""
import json
S = json.load(open("results/simulation.json")); A = json.load(open("results/analysis.json"))
OUT = "manuscripts/jos/"
f3 = lambda v: f"{v:.3f}"
def w(name, s): open(OUT + name, "w").write(s)

rows = []
sc = [("Linear null, slope 0.00", "typeI_slope0.0"), ("Linear null, slope 0.15", "typeI_slope0.15"),
      ("Linear null, slope 0.30", "typeI_slope0.3"),
      ("Overlap correlation 0.25", "robust_rho0.25"), ("Overlap correlation 0.50", "robust_rho0.5"),
      ("Standard errors understated by 10\\%", "robust_kappa1.1"),
      ("Standard errors understated by 25\\%", "robust_kappa1.25"),
      ("Standard errors understated by 50\\%", "robust_kappa1.5")]
for lab, k in sc:
    v = S[k]
    rows.append(f"{lab} & {f3(v['eproc'])} & {f3(v['bonferroni'])} & {f3(v['spending'])} & {f3(v['uncorrected'])} \\\\")
w("tab_typeI.tex", "\\begin{tabular}{lcccc}\n\\toprule\nScenario & E-process & Bonferroni & Spending & Uncorrected \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for dl in ("0.05", "0.1", "0.2"):
    v = S[f"power_delta{dl}"]
    c = lambda m: f"{v[m]['false_before']:.3f} / {v[m]['detect_after']:.3f} / {v[m]['median_delay']:.0f}"
    rows.append(f"{float(dl):.2f} & {c('eproc')} & {c('bonferroni')} & {c('spending')} & {c('uncorrected')} \\\\")
w("tab_power.tex", "\\begin{tabular}{lcccc}\n\\toprule\nSlope increase & E-process & Bonferroni & Spending & Uncorrected \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

import datetime, math
def sci(x):
    if x < 1000: return f"{x:.1f}"
    e = int(math.floor(math.log10(x))); return f"${x/10**e:.1f}\\times 10^{{{e}}}$"
fmt = lambda d: datetime.date.fromisoformat(d).strftime("%-d %b %Y")
rows = []
for e, dd in zip(A["epochs"], A["declarations"] + [None]):
    dirn = {"decel": "deceleration", "accel": "acceleration"}
    if dd:
        k = dd["dir"]; x = e[k]
        rows.append(f"{fmt(e['start'])} & {e['n_waves']} & {dirn[k].capitalize()} & {fmt(dd['date'])} & {fmt(x['amber'])} & {fmt(x['strong'])} & ${dd['E']:.1e}$ \\\\".replace("e+0", "\\times 10^{").replace("e+", "\\times 10^{") if False else
                    f"{fmt(e['start'])} & {e['n_waves']} & {dirn[k].capitalize()} & {fmt(dd['date'])} & {fmt(x['amber'])} & {fmt(x['strong'])} & {sci(x['max'])} \\\\")
    else:
        x = e["decel"]
        rows.append(f"{fmt(e['start'])} & {e['n_waves']} & None declared & --- & {fmt(x['amber'])} & {fmt(x['strong'])} & {x['max']:.1f} \\\\")
w("tab_decl.tex", "\\begin{tabular}{lrlcccr}\n\\toprule\nEpoch start & Waves & Declaration & Date & E $\\ge 2$ & E $\\ge 10$ & Peak E \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for r in A["strata"]:
    dl = "; ".join(f"{fmt(a)} ({'dec.' if b=='decel' else 'acc.'})" for a, b in r["decl_aware"]) or "none"
    pooled = f"{fmt(r['decl_pooled_last'][0])}" if r["decl_pooled_last"] and r["decl_pooled_last"][0] == "2025-11-30" else "no"
    rows.append(f"{r['size']} & {r['last_legacy']:.1f} & {r['first_revised']:.1f} & {r['jump_z']:.1f} & {r['p95']:.2f} & {pooled} \\\\")
w("tab_strata.tex", "\\begin{tabular}{lccccc}\n\\toprule\nSize class & Last legacy & First revised & Jump $|z|$ & 95th pct. within & Pooled declares at redesign \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print(open(OUT+"tab_decl.tex").read()); print(open(OUT+"tab_strata.tex").read())

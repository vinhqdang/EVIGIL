"""Build LaTeX tables for the JOS manuscript from results/*.json."""
import json, math, datetime
S = json.load(open("results/simulation.json")); A = json.load(open("results/analysis.json"))
OUT = "manuscripts/jos/"
f3 = lambda v: f"{v:.3f}"
fmt = lambda d: datetime.date.fromisoformat(d).strftime("%-d %b %Y")
def w(name, s): open(OUT + name, "w").write(s)
def sci(x):
    if x < 1000: return f"{x:.1f}"
    e = int(math.floor(math.log10(x))); return f"${x/10**e:.1f}\\times 10^{{{e}}}$"

rows = []
sc = [("Linear path, slope 0.00", "typeI_slope0.0"), ("Linear path, slope 0.15", "typeI_slope0.15"),
      ("Linear path, slope 0.30", "typeI_slope0.3"),
      ("Lag-6 correlation 0.25", "robust_rho0.25"), ("Lag-6 correlation 0.50", "robust_rho0.5"), ("Lag-6 correlation 0.80", "robust_rho0.8"),
      ("True SE 1.10 times published", "robust_kappa1.1"), ("True SE 1.25 times published", "robust_kappa1.25"),
      ("True SE 1.50 times published", "robust_kappa1.5"), ("True SE 1.75 times published", "robust_kappa1.75")]
for lab, k in sc:
    v = S[k]
    rows.append(f"{lab} & {f3(v['eproc'])} & {f3(v['bonferroni'])} & {f3(v['spending'])} & {f3(v['exact'])} & {f3(v['uncorrected'])} \\\\")
rows.insert(3, "\\midrule")
rows.insert(7, "\\midrule")
w("tab_typeI.tex", "\\begin{tabular}{lccccc}\n\\toprule\nScenario & E-process & Bonferroni & Spending & Exact boundary & Uncorrected \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for dl in ("0.01", "0.02", "0.03", "0.05"):
    v = S[f"power_delta{dl}"]
    c = lambda m: f"{v[m]['within20']:.2f} / {v[m]['by_end']:.2f} / {v[m]['median_delay']:.0f}"
    rows.append(f"{float(dl):.2f} & {c('eproc')} & {c('bonferroni')} & {c('spending')} & {c('exact')} \\\\")
w("tab_power.tex", "\\begin{tabular}{lcccc}\n\\toprule\nSlope increase & E-process & Bonferroni & Spending & Exact boundary \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for k in (2, 5, 10, 20):
    a = S[f"redesign_additive_k{k}"]; m = S[f"redesign_multiplicative_k{k}"]; l = S[f"redesign_labelone_late_k{k}"]
    rows.append(f"{k} & {f3(a['regime_aware'])} & {f3(m['regime_aware'])} & {f3(l['regime_aware'])} & {f3(a['pooled'])} \\\\")
w("tab_redesign.tex", "\\begin{tabular}{lcccc}\n\\toprule\nWaves after redesign & Additive shift & Multiplicative effect & Label one wave late & Pooled analysis \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
dirn = {"decel": "Deceleration", "accel": "Acceleration"}
for e, dd in zip(A["epochs"], A["declarations"] + [None]):
    if dd:
        rows.append(f"{fmt(e['start'])} & {e['n_waves']} & {dirn[dd['dir']]} & {fmt(dd['date'])} & ${dd['gamma']:.4f} \\pm {dd['gamma_hw']:.4f}$ & {sci(dd['E'])} \\\\")
    else:
        L = A["last_epoch"]
        rows.append(f"{fmt(e['start'])} & {e['n_waves']} & None & --- & --- & {L['max_decel']:.1f} \\\\")
w("tab_decl.tex", "\\begin{tabular}{lrllcr}\n\\toprule\nEpoch start & Waves & Declaration & Date & Curvature (pp per wave$^2$) & Peak E \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for k, lst in A["kappa_sensitivity"].items():
    c = [f"{fmt(x['date'])} ({sci(x['E'])})" for x in lst]
    lp = A["kappa_last_peak"][k]
    rows.append(f"{float(k):.2f} & {c[0] if c else '---'} & {c[1] if len(c) > 1 else '---'} & {lp:.1f} \\\\")
w("tab_kappa.tex", "\\begin{tabular}{lccc}\n\\toprule\nSE inflation & First declaration (E) & Second declaration (E) & Peak E, third epoch \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for r in A["strata"]:
    pooled = fmt(r["decl_pooled_last"][0]) if r["decl_pooled_last"] and r["decl_pooled_last"][0] == "2025-11-30" else "no"
    rows.append(f"{r['size']} & {r['last_legacy']:.1f} & {r['first_revised']:.1f} & {r['ratio']:.2f} & {r['jump_z']:.1f} & {r['p95']:.2f} & {pooled} \\\\")
w("tab_strata.tex", "\\begin{tabular}{lcccccc}\n\\toprule\nSize class & Last legacy & First revised & Ratio & Jump $|z|$ & 95th pct. within & Pooled declares \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

C = A["comparators"]
def lst(k): return ", ".join(fmt(x) for x in C[k])
print(open(OUT + "tab_decl.tex").read()); print(open(OUT + "tab_kappa.tex").read())
for k in C: print(k, lst(k))

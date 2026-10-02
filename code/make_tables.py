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

E = json.load(open("results/extensions.json")); X = json.load(open("results/application_ext.json"))
rows = []
for k in (2, 5, 10, 20):
    v = lambda sc: E[f"variants_{sc}_k{k}"]["base"]
    rows.append(f"{k} & {f3(v('additive'))} & {f3(v('multiplicative'))} & {f3(v('labelearly'))} & {f3(v('labellate'))} & {f3(S[f'redesign_additive_k{k}']['pooled'])} \\\\")
w("tab_redesign.tex", "\\begin{tabular}{lccccc}\n\\toprule\nWaves after redesign & Additive shift & Multiplicative effect & Label one wave early & Label one wave late & Pooled analysis \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
b0 = E["variants_additive_k20"]["base_B"]
for name, lab in (("base", "Regime levels (baseline)"), ("guard", "Guard waves"), ("slopes", "Regime slopes"), ("slopes_guard", "Regime slopes and guard waves")):
    c = [f3(E[f"variants_{sc}_k20"][name]) for sc in ("additive", "multiplicative", "labelearly", "labellate")]
    rows.append(f"{lab} & {' & '.join(c)} & {E['variants_additive_k20'][name + '_B'] / b0:.2f} \\\\")
w("tab_variants.tex", "\\begin{tabular}{lccccc}\n\\toprule\nMonitor & Additive shift & Multiplicative effect & Label early & Label late & Information \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for kap in ("1.0", "1.5", "2.0"):
    v = E[f"unknown_typeI_kappa{kap}"]; rows.append(f"False declaration, true SE {float(kap):.1f} times published & {f3(v['unknown'])} & {f3(v['known'])} \\\\")
rows.append("\\midrule")
for dl in ("0.02", "0.03", "0.05"):
    u = E[f"unknown_power_delta{dl}"]["kappa1.0"]; kn = S[f"power_delta{dl}"]["eproc"]
    rows.append(f"Detection by end, slope increase {dl}, true SE as published & {f3(u['by_end'])} & {f3(kn['by_end'])} \\\\")
for dl in ("0.03", "0.05"):
    u = E[f"unknown_power_delta{dl}"]["kappa1.5"]
    rows.append(f"Detection by end, slope increase {dl}, true SE 1.5 times published & {f3(u['by_end'])} & --- \\\\")
w("tab_unknown.tex", "\\begin{tabular}{lcc}\n\\toprule\nScenario & Unknown scale & Known scale \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

import datetime
fm = lambda d: datetime.date.fromisoformat(d).strftime("%-d %b %Y")
def dl(lst): return "; ".join(f"{fm(x['date'])} ({'dec.' if x['dir']=='decel' else 'acc.'})" for x in lst) or "none"
rows = []
for lab, key in (("Known scale, single $\\tau$", "known"), ("Unknown scale", "unknown_scale"), ("Mixture over $\\tau$", "tau_mixture"),
                 ("Regime slopes and guard waves", "slopes_guard1"), ("Log scale, $\\tau = 0.002$", "log_tau0.002")):
    rows.append(f"{lab} & {dl(X[key])} \\\\")
w("tab_extapp.tex", "\\begin{tabular}{lp{9.5cm}}\n\\toprule\nMonitor & Declarations \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
sl = X["state_space"]["kappa_free"]["slope"]
for dd, (m, h) in sl.items():
    rows.append(f"{fm(dd)} & ${m:.2f} \\pm {h:.2f}$ & ${X['state_space']['kappa_fixed']['slope'][dd][0]:.2f} \\pm {X['state_space']['kappa_fixed']['slope'][dd][1]:.2f}$ \\\\")
w("tab_ssm.tex", "\\begin{tabular}{lcc}\n\\toprule\nDate & Slope, scale estimated & Slope, published SE \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

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

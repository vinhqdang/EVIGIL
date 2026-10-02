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
    rows.append(f"{k} & {f3(v('additive'))} & {f3(v('multiplicative'))} & {f3(v('labelearly'))} & {f3(v('labellate'))} & {f3(E[f'variants_additive_k{k}']['pooled'])} \\\\")
w("tab_redesign.tex", "\\begin{tabular}{lccccc}\n\\toprule\nWaves after redesign & Additive shift & Multiplicative effect & Label one wave early & Label one wave late & Pooled analysis \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
b0 = E["variants_additive_k20"]["base_B"]
for name, lab in (("base", "Regime levels (baseline)"), ("guard", "Guard waves"), ("slopes", "Regime slopes"), ("slopes_guard", "Regime slopes and guard waves")):
    c = [f3(E[f"variants_{sc}_k20"][name]) for sc in ("additive", "multiplicative", "labelearly", "labellate")]
    rows.append(f"{lab} & {' & '.join(c)} & {E['variants_additive_k20'][name + '_B'] / b0:.2f} \\\\")
w("tab_variants.tex", "\\begin{tabular}{lccccc}\n\\toprule\nMonitor & Additive shift & Multiplicative effect & Label early & Label late & Information \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for kap in ("1.0", "1.5", "2.0"):
    v = E[f"unknown_typeI_kappa{kap}"]
    mx = E.get(f"unknownmix_typeI_kappa{kap}")
    rows.append(f"True SE {float(kap):.1f} times published & {f3(v['unknown'])} & {f3(mx) if mx is not None else '---'} & {f3(v['known'])} \\\\")
rows.append("\\midrule")
for lab, key in (("Lag-1 error correlation $-0.3$", "ar1_rho-0.3"), ("Lag-1 error correlation 0.1", "ar1_rho0.1"), ("Lag-1 error correlation 0.3", "ar1_rho0.3"),
                 ("Lag-1 error correlation 0.5", "ar1_rho0.5"), ("True SD constant at 0.15 (published weights vary)", "misweight_const_sd0.15"),
                 ("Wave-specific scale errors, lognormal SD 0.3", "misweight_lognormal0.3"), ("Wave-specific scale errors, lognormal SD 0.5", "misweight_lognormal0.5")):
    v = E[key]; rows.append(f"{lab} & {f3(v['unknown'])} & --- & {f3(v['known'])} \\\\")
rows.append("\\midrule")
for dl in ("0.02", "0.03", "0.05"):
    u = E[f"unknown_power_delta{dl}"]["kappa1.0"]; kn = S[f"power_delta{dl}"]["eproc"]; um = E.get(f"unknownmix_power_delta{dl}")
    rows.append(f"Detection by end, slope increase {dl} & {f3(u['by_end'])} & {f3(um['by_end']) if um else '---'} & {f3(kn['by_end'])} \\\\")
for dl in ("0.02", "0.03", "0.05"):
    u = E[f"unknown_power_delta{dl}"]["kappa1.5"]
    rows.append(f"Detection by end, slope increase {dl}, true SE 1.5 times published & {f3(u['by_end'])} & --- & --- \\\\")
w("tab_unknown.tex", "\\begin{tabular}{lccc}\n\\toprule\nScenario & Unknown scale & Unknown scale, mixture & Known scale \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

import datetime
fm = lambda d: datetime.date.fromisoformat(d).strftime("%-d %b %Y")
def dl(lst): return "; ".join(f"{fm(x['date'])} ({'dec.' if x['dir']=='decel' else 'acc.'})" for x in lst) or "none"
P = X["primary"]; PL = X["primary_last"]
rows = []
for i, e in enumerate(X["primary_epochs"]):
    if i < len(P):
        x = P[i]
        rows.append(f"{fm(e['start'])} & {x['n']} & {'Deceleration' if x['dir']=='decel' else 'Acceleration'} & {fm(x['date'])} & ${x['gamma']:.4f} \\pm {x['gamma_hw']:.4f}$ & {sci(x['E'])} \\\\")
    else:
        rows.append(f"{fm(e['start'])} & {e['n_waves']} & None & --- & --- & {PL['max_decel']:.1f} \\\\")
w("tab_decl.tex", "\\begin{tabular}{lrllcr}\n\\toprule\nEpoch start & Waves & Declaration & Date & Curvature (pp per wave$^2$) & E \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
def row(lab, lst, last=None): rows.append(f"{lab} & {dl(lst)}" + (f" (last epoch peak {last:.1f})" if last else "") + " \\\\")
row("Primary monitor", X["primary"], PL["max_decel"])
row("Primary, wave of 10 March 2024 absorbed by its own level", X["primary_one_wave_dropped"])
row("Primary, level shift from 10 March 2024", X["primary_level_shift_mar2024"], max(X["primary_level_shift_last"]["max_decel"], X["primary_level_shift_last"]["max_accel"]))
for k in ("3", "6"):
    row(f"Primary, monitoring starts {k} waves later", X["primary_start"][k])
for tau in ("0.005", "0.01", "0.02", "0.04"):
    v = X["unknown_single_tau"][tau]; row(f"Unknown scale, single $\\tau = {tau}$", v["decl"], v["last"]["max_decel"] if v["last"] else None)
row("Primary, regime slopes and guard waves", X["regime_slopes_guard"])
row("Unknown scale on the log scale, $\\tau = 0.002$", X["log_scale_unknown_tau0.002"])
row("Basic monitor (known scale, $\\tau = 0.01$)", X["basic"], X["basic_last"]["max_decel"])
for g in ("0.003", "0.004", "0.005"):
    row(f"Basic monitor, tolerance $\\gamma_0 = {g}$", X["gamma0_known"][g])
row("Basic monitor, SE multiplied by 1.55", A["kappa_sensitivity"]["1.55"])
w("tab_sens.tex", "\\begin{tabular}{lp{8.6cm}}\n\\toprule\nVariant & Declarations \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
SSM = X["state_space"]
for key, lab in (("smooth_trend_kappa_free", "Smooth trend (slope disturbance only)"), ("local_linear_trend_kappa_free", "Local linear trend (level and slope disturbances)"),
                 ("local_level_constant_slope_kappa_free", "Random-walk level, constant slope")):
    v = SSM[key]; a_ = v["slope"]["2025-04-06"]; j_ = v["slope"]["2025-07-27"]; df = v["diff_jul_apr"]
    rows.append(f"{lab} & {v['loglik']:.1f} & {v['kappa']:.2f} & {v['q']:.1e} & {v['p']:.3f} & ${a_[0]:.2f} \\pm {a_[1]:.2f}$ & ${j_[0]:.2f} \\pm {j_[1]:.2f}$ & ${df[0]:.2f} \\pm {df[1]:.2f}$ \\\\")
w("tab_ssm.tex", "\\begin{tabular}{lccccccc}\n\\toprule\nModel & Log-lik. & Scale & Slope var. & Level var. & Slope Apr 2025 & Slope Jul 2025 & Jul minus Apr \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for k in (2, 5, 10, 20):
    b0 = E[f"variants_additive_k{k}"]["base_B"]
    rows.append(f"{k} & {E[f'variants_additive_k{k}']['guard_B']/b0:.2f} & {E[f'variants_additive_k{k}']['slopes_B']/b0:.2f} & {E[f'variants_additive_k{k}']['slopes_guard_B']/b0:.2f} \\\\")
w("tab_info.tex", "\\begin{tabular}{lccc}\n\\toprule\nWaves after redesign & Guard waves & Regime slopes & Both \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for r in A["strata"]:
    pooled = fmt(r["decl_pooled_last"][0]) if r["decl_pooled_last"] and r["decl_pooled_last"][0] == "2025-11-30" else "no"
    rows.append(f"{r['size']} & {r['last_legacy']:.1f} & {r['first_revised']:.1f} & {r['ratio']:.2f} & {r['jump_z']:.1f} & {r['p95']:.2f} & {pooled} \\\\")
w("tab_strata.tex", "\\begin{tabular}{lcccccc}\n\\toprule\nSize class & Last legacy (\\%) & First revised (\\%) & Ratio & Jump $|z|$ & 95th pct. within & Pooled declares at wording change \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

C = A["comparators"]
def lst(k): return ", ".join(fmt(x) for x in C[k])
print(open(OUT + "tab_decl.tex").read())
for k in C: print(k, lst(k))

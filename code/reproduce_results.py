"""
Replication script for:
"Anytime-Valid Monitoring of Technology Adoption Dynamics: Evidence from
Biweekly Surveillance of Enterprise Artificial Intelligence Diffusion in the
United States, 2023-2025" (Journal of Engineering and Technology Management,
double-blind submission).

Inputs (supplementary data, same directory):
  btos_ai_national.csv   - spliced national AI series with design-based SEs
  btos_ai_sizeclass.csv  - employment-size-class series with design-based SEs
Both files are exact extracts of the U.S. Census Bureau BTOS public downloads
(National.xlsx and "Employment Size Class.xlsx"), vintages of 8 Dec 2025
(legacy-wording rows) and 15 Dec 2025 (revised-wording rows); see Section 4.

Outputs: every number reported in Sections 5.1-5.6 and Tables 2-3, and
Figures 2-5, written to ./figures/.

Usage:  python reproduce_results.py
Requires: numpy, pandas, matplotlib (Python >= 3.9).
"""
import numpy as np
import pandas as pd
from statistics import NormalDist
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ND = NormalDist()
ALPHA = 0.05
THRESH = 2.0 / ALPHA        # 40: declaration threshold per one-sided monitor
TAU = 1.0                   # prior scale on standardized drift
W_ANCHOR = 8                # waves in the anchoring window
Path("figures").mkdir(exist_ok=True)

# ----------------------------- data -----------------------------------------
nat = pd.read_csv("btos_ai_national.csv", parse_dates=["collection_end"])
siz = pd.read_csv("btos_ai_sizeclass.csv", parse_dates=["collection_end"])

leg = nat[nat.wording == "legacy"].sort_values("collection_end").reset_index(drop=True)
rev = nat[nat.wording == "revised"].sort_values("collection_end").reset_index(drop=True)
y, s, d = leg.ai_use_pct.to_numpy(), leg.ai_use_se.to_numpy(), list(leg.collection_end)
yn, sn, dn = rev.ai_use_pct.to_numpy(), rev.ai_use_se.to_numpy(), list(rev.collection_end)

print(f"[data] legacy waves: {len(y)} ({d[0].date()} -> {d[-1].date()}, "
      f"{y[0]}% -> {y[-1]}%); revised waves: {len(yn)} "
      f"({[float(v) for v in yn]}); national SE median "
      f"{np.median(s):.2f} pp (range {s.min():.2f}-{s.max():.2f})")

# ------------------------ e-process machinery --------------------------------
def e_one_sided(S, n, tau=TAU):
    """Robbins one-sided half-normal mixture e-process, Eq. (3) of the paper."""
    a = n + 1.0 / tau**2
    return (2.0 / (tau * np.sqrt(a))) * ND.cdf(S / np.sqrt(a)) * np.exp(S**2 / (2 * a))

def monitor(y, s, dates, beta, beta_se, t0, tau=TAU, kappa=1.0):
    """Run the two one-sided monitors from increment index t0 with baseline
    slope beta (+/- beta_se). Returns per-wave records."""
    Sup = Sdn = 0.0
    n = 0
    out = []
    for t in range(t0, len(y)):
        sig = kappa * np.sqrt(s[t]**2 + s[t-1]**2 + beta_se**2)
        z = ((y[t] - y[t-1]) - beta) / sig
        n += 1
        Sup += z
        Sdn -= z
        out.append(dict(t=t, date=dates[t],
                        e_up=e_one_sided(Sup, n, tau),
                        e_dn=e_one_sided(Sdn, n, tau)))
    return out

def wls_slope(y, s, idx):
    """Weighted-least-squares local-linear slope and its SE on waves idx."""
    yy, ss = y[idx], s[idx]
    x = np.arange(len(idx), dtype=float)
    w = 1.0 / np.maximum(ss, 1e-6)**2
    W = np.diag(w)
    X = np.column_stack([np.ones(len(idx)), x])
    cov = np.linalg.pinv(X.T @ W @ X)
    b = cov @ (X.T @ W @ yy)
    return float(b[1]), float(np.sqrt(cov[1, 1]))

def first(recs, key, th):
    return next((r for r in recs if r[key] >= th), None)

# --------------------------- Episode 1: takeoff ------------------------------
tr1 = monitor(y, s, d, beta=0.0, beta_se=0.0, t0=1)
det = first(tr1, "e_up", THRESH)
amb = first(tr1, "e_up", 2.0)
strg = first(tr1, "e_up", 10.0)
i10 = tr1.index(strg)
idet = tr1.index(det)
itr = tr1.index(min(tr1[i10:idet], key=lambda r: r["e_up"]))
_ = None
peak = max(tr1[i10:itr], key=lambda r: r["e_up"])
trough = min(tr1[i10:idet], key=lambda r: r["e_up"])
print(f"[ep1] amber {amb['date'].date()} (E={amb['e_up']:.2f}); "
      f"strong amber {strg['date'].date()} (E={strg['e_up']:.1f}); "
      f"early peak {peak['date'].date()} (E={peak['e_up']:.1f}); "
      f"trough {trough['date'].date()} (E={trough['e_up']:.2f}); "
      f"DECLARED {det['date'].date()} (E={det['e_up']:.1f}, "
      f"{tr1.index(det)+1} increments)")

# --------------------- Episode 2: anchored momentum --------------------------
td = det["t"]
beta2, beta2_se = wls_slope(y, s, list(range(td - W_ANCHOR + 1, td + 1)))
print(f"[anchor] slope {beta2:.3f} +/- {beta2_se:.3f} pp/wave on "
      f"{d[td-W_ANCHOR+1].date()} -> {d[td].date()}")
tr2 = monitor(y, s, d, beta=beta2, beta_se=beta2_se, t0=td + 1)
amb2 = first(tr2, "e_dn", 2.0)
pk2 = max(tr2, key=lambda r: r["e_dn"])
print(f"[ep2] deceleration amber from {amb2['date'].date()}; "
      f"peak E={pk2['e_dn']:.1f} on {pk2['date'].date()}; "
      f"final E_dn={tr2[-1]['e_dn']:.1f} on {tr2[-1]['date'].date()}; "
      f"max E_up={max(r['e_up'] for r in tr2):.2f}; "
      f"declaration at alpha=0.05: {first(tr2,'e_dn',THRESH) is not None}; "
      f"at alpha=0.10 (thr 20): "
      f"{(lambda r: r['date'].date() if r else None)(first(tr2,'e_dn',20.0))}")

# --------------------- Episode 3: wording-change break -----------------------
jump = yn[0] - y[-1]
jz = jump / np.sqrt(sn[0]**2 + s[-1]**2)
wz = np.abs(np.diff(y)) / np.sqrt(s[1:]**2 + s[:-1]**2)
print(f"[break] +{jump:.1f} pp = {jz:.1f} SE; within-regime |z| "
      f"median {np.median(wz):.2f}, max {wz.max():.1f} "
      f"(transition increment excluded by design)")

# ------------------------------ robustness -----------------------------------
for tau in (0.5, 2.0):
    r = first(monitor(y, s, d, 0.0, 0.0, 1, tau=tau), "e_up", THRESH)
    print(f"[robust] tau={tau}: declaration "
          f"{r['date'].date() if r else None} (E={r['e_up']:.1f})" if r else
          f"[robust] tau={tau}: no declaration")
rk = monitor(y, s, d, 0.0, 0.0, 1, kappa=1.5)
print(f"[robust] kappa=1.5: max E_up={max(r['e_up'] for r in rk):.1f}; "
      f"declaration: {first(rk,'e_up',THRESH) is not None}")

# ------------------------------ comparators ----------------------------------
ys = np.concatenate([y, yn]); ss = np.concatenate([s, sn]); ds = d + dn
brk = len(y)

def rep_ztest(y, s, k=4, bonf=False, alpha=0.05):
    alarms, looks, zmax_brk = [], 0, 0.0
    for t in range(2 * k, len(y)):
        looks += 1
        a, b = y[t-k+1:t+1], y[t-2*k+1:t-k+1]
        sa, sb = s[t-k+1:t+1], s[t-2*k+1:t-k+1]
        z = (a.mean() - b.mean()) / np.sqrt((sa**2).sum()/k**2 + (sb**2).sum()/k**2)
        if brk <= t <= brk + 1:
            zmax_brk = max(zmax_brk, abs(z))
        crit = ND.inv_cdf(1 - (alpha/2)/looks) if bonf else 1.959964
        if abs(z) > crit:
            alarms.append(t)
    return alarms, looks, zmax_brk

def episodes(al):
    return 0 if not al else 1 + sum(al[i] != al[i-1] + 1 for i in range(1, len(al)))

def ma_cross(y, short=3, long=8):
    yy = pd.Series(y)
    sg = np.sign((yy.rolling(short).mean() - yy.rolling(long).mean()).to_numpy())
    return [t for t in range(1, len(y))
            if not np.isnan(sg[t]) and not np.isnan(sg[t-1])
            and sg[t] != sg[t-1] and sg[t] != 0]

al_t, looks, zbrk = rep_ztest(ys, ss)
al_b, _, _ = rep_ztest(ys, ss, bonf=True)
al_m = ma_cross(ys)
at_brk = lambda al: any(brk <= t <= brk + 2 for t in al)
print(f"[comp] looks={looks}; z-test: {len(al_t)} alarm waves / "
      f"{episodes(al_t)} episodes, first {ds[al_t[0]].date()}, "
      f"at break {at_brk(al_t)} (|z|={zbrk:.1f}); "
      f"Bonferroni: {len(al_b)} waves / {episodes(al_b)} episodes, "
      f"first {ds[al_b[0]].date()}, at break {at_brk(al_b)}; "
      f"MA(3,8): crossings {[ds[t].date().isoformat() for t in al_m]}, "
      f"at break {at_brk(al_m)}")

# ----------------------- strata: Table 3 and diagnostic ----------------------
labels = {"A": "1-4", "B": "5-9", "C": "10-19", "D": "20-49",
          "E": "50-99", "F": "100-249", "G": "250+"}
rows = []
for k in "ABCDEFG":
    lk = siz[(siz.size_class == k) & (siz.wording == "legacy")].sort_values("collection_end")
    rk_ = siz[(siz.size_class == k) & (siz.wording == "revised")].sort_values("collection_end")
    yk, sk, dk = lk.ai_use_pct.to_numpy(), lk.ai_use_se.to_numpy(), list(lk.collection_end)
    trk = monitor(yk, sk, dk, 0.0, 0.0, 1)
    jmp = rk_.ai_use_pct.iloc[0] - yk[-1]
    jzk = jmp / np.sqrt(rk_.ai_use_se.iloc[0]**2 + sk[-1]**2)
    wzk = np.abs(np.diff(yk)) / np.sqrt(sk[1:]**2 + sk[:-1]**2)
    rows.append([labels[k], yk[0], yk[-1], rk_.ai_use_pct.iloc[0],
                 round(float(np.median(sk)), 2),
                 round(max(r["e_up"] for r in trk), 1),
                 round(float(jmp), 1), round(float(jzk), 1),
                 round(float(np.percentile(wzk, 95)), 2)])
tab3 = pd.DataFrame(rows, columns=["size_class", "sep2023", "oct2025",
                                   "nov2025_revised", "median_se", "max_E_up",
                                   "jump_pp", "jump_z", "p95_within_z"])
tab3.to_csv("figures/table3_strata.csv", index=False)
print("[table 3]\n", tab3.to_string(index=False))

# -------------------------------- figures ------------------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
BREAK = datetime(2025, 11, 17)

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.errorbar(d, y, yerr=1.96*s, fmt="-o", ms=2.2, lw=1.0, elinewidth=0.5,
            color="#1f4e79", label="Current use, legacy wording")
ax.errorbar(dn, yn, yerr=1.96*sn, fmt="-s", ms=3.2, lw=1.0, elinewidth=0.5,
            color="#b02418", label="Current use, revised wording")
pl = nat[nat.wording == "legacy"]; pr = nat[nat.wording == "revised"]
ax.plot(pl.collection_end, pl.ai_plan_pct, "--", lw=0.8, color="#7fa6c9",
        label="Planned use (6 months), legacy")
ax.plot(pr.collection_end, pr.ai_plan_pct, "--", lw=0.8, color="#d98880",
        label="Planned use, revised")
ax.axvline(det["date"], color="#2e7d32", ls="--", lw=1.0)
ax.axvspan(amb2["date"], tr2[-1]["date"], color="#f4a623", alpha=0.18)
ax.axvline(BREAK, color="#b02418", ls=":", lw=1.2)
ax.set_ylabel("Share of businesses using AI (%)"); ax.set_ylim(2, 21)
ax.legend(fontsize=7, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig("figures/fig2_series.png", dpi=300)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.semilogy([r["date"] for r in tr1[:idet+1]], [r["e_up"] for r in tr1[:idet+1]],
            color="#1f4e79", lw=1.3, label="Epoch 1: upward (takeoff)")
ax.semilogy([r["date"] for r in tr1[:idet+1]], [r["e_dn"] for r in tr1[:idet+1]],
            color="#1f4e79", lw=0.8, ls=":", alpha=0.6, label="Epoch 1: downward")
ax.semilogy([r["date"] for r in tr2], [r["e_dn"] for r in tr2],
            color="#b02418", lw=1.3, label="Epoch 2: downward (deceleration)")
ax.semilogy([r["date"] for r in tr2], [r["e_up"] for r in tr2],
            color="#b02418", lw=0.8, ls=":", alpha=0.6, label="Epoch 2: upward")
for th, c in [(40, "k"), (10, "0.4"), (2, "0.6")]:
    ax.axhline(th, color=c, ls="--", lw=0.8)
ax.set_ylabel("e-value (log scale)"); ax.set_ylim(0.01, 2000)
ax.legend(fontsize=7, loc="lower right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig("figures/fig3_evalues.png", dpi=300)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(ds, ys, color="0.6", lw=1.0)
ax.scatter([ds[t] for t in al_t], [ys[t] for t in al_t], marker="x", s=26,
           color="#1f4e79", label=f"z-test alarms ({len(al_t)})")
ax.scatter([ds[t] for t in al_b], [ys[t]+0.55 for t in al_b], marker="v", s=16,
           color="#6a51a3", label=f"Bonferroni alarms ({len(al_b)})")
ax.scatter([ds[t] for t in al_m], [ys[t]-0.7 for t in al_m], marker="s", s=24,
           facecolors="none", edgecolors="#b02418",
           label=f"MA crossings ({len(al_m)})")
ax.axvline(det["date"], color="#2e7d32", ls="--", lw=1.0)
ax.axvline(BREAK, color="k", ls=":", lw=1.0)
ax.set_ylabel("Share using AI (%)"); ax.legend(fontsize=7, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig("figures/fig4_alarms.png", dpi=300)

fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = np.arange(7)
ax.bar(x, tab3.jump_z, 0.55, color="#b02418", alpha=0.85,
       label="Jump at wording change (|z| units)")
ax.plot(x, tab3.p95_within_z, "D", color="#1f4e79", ms=5,
        label="95th pct. within-regime |z|")
ax.set_xticks(x); ax.set_xticklabels(tab3.size_class)
ax.set_xlabel("Employment size class (employees)")
ax.set_ylabel("Standardized magnitude |z|")
ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fig5_strata.png", dpi=300)

print("[done] figures in ./figures/")

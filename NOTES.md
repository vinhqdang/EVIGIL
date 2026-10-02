# Project notes

Paper: anytime-valid (e-process) monitoring of technology diffusion using U.S. Census BTOS biweekly AI-adoption data, plus a 1990–2026 retrospective (internet users, e-commerce share). Status: rejected by JET-M (desk) and TF&SC (post-review); need a new venue — see submission_tracking/SUBMISSION_LOG.md.

## Layout
- manuscripts/{jetm,tfsc,frl}/  — three LaTeX (elsarticle) versions with cover letter, highlights, title page. jetm and tfsc are the full 48-page paper; frl is the 2,233-word letter (finance framing, no 1990–2026 section).
- code/reproduce_results.py — BTOS analysis (reads data/btos_ai_*.csv; regenerates Tables 2–3, Figs 2–5). code/reproduce_retro.py — Table 4 (long-run series). Both verified to reproduce every number in the paper.
- data/ — 4 CSVs: BTOS national (56 waves), BTOS by size class, FRED ITNETUSERP2USA, FRED ECOMPCTSA.
- figures/ — figure1 (framework), 2–5 (BTOS results), 6 (1990–2026 three-panel), 7 (technology-wave timeline).
- transcript/ — full earlier conversation (code for fig1/fig6/fig7 generation, raw-data extraction, search results).
- Build: `pdflatex main.tex` twice inside a manuscript folder (needs texlive-publishers for elsarticle.cls; figures must sit next to main.tex).

## Method (one paragraph)
Standardised increments z_t=(Δ_t−β)/σ_t with σ_t² = SE_t²+SE_{t−1}² (+ se(β)²). Robbins half-normal-mixture one-sided e-process E_n = 2/(τ√a)·Φ(S/√a)·exp(S²/2a), a=n+τ⁻², τ=1; two monitors, declare at E≥40 (α=0.05); epochs re-anchored by WLS slope on the 8 waves ending at declaration; pre-announced instrument changes excluded (do-operator); cross-strata diagnostic for unannounced breaks. Long-run series use a plug-in MAD variance (approximate validity).

## Key verified results
Takeoff declared 20 Apr 2025, E=42.2 (41 increments); early peak 17.1 (28 Jan 2024), trough 0.40 (7 Apr 2024). Anchor slope 0.299±0.027 pp/wave. Deceleration monitor peak E=24.3 on 7 Sep 2025, no declaration. Nov 2025 wording change: +7.3 pp = 25.7 SE; naive z-test |z|=32.9 alarms; z-test 37/48 alarms, Bonferroni 31/48, MA(3,8) 4 crossings. Strata jump z 6.1–13.8 vs within-regime p95 1.74–2.85. Robustness: τ=0.5→same date (E=58.8), τ=2→4 May 2025; κ=1.5 → no declaration. Retro: internet takeoff 1995, saturation 2008; e-commerce 2004Q1, 2018Q1, 2020Q2, 2023Q1.

## Guardrails
- Never fabricate data; BTOS AI series starts Sep 2023 only; no penetration series exist for data mining/cloud/big data/deep learning (Fig 7 is landmark-dated by design).
- All references carry DOI or verified URL (Rogers 2003 = ISBN only).
- Archived BTOS data end 14 Dec 2025: re-download current Census BTOS files and rerun reproduce_results.py to extend the AI panel before any new submission.
- Raw Census files were obtained from GitHub EIG-Research/ai-btos (census.gov blocked in the sandbox); two vintages (20251208 legacy rows, 20251215 revised rows) are spliced. Raw extraction scripts were not preserved — only the CSVs; see transcript if re-extraction is needed.
- Double-blind: manuscripts say "Author information removed"; author details only on title page.

## Open items
1. Add TF&SC reviewer reports to submission_tracking/.
2. Pick new venue (technical journals first), check its guide for authors, then retarget framing (derivations up front, drop diffusion-theory/tech-waves material if venue is statistical).
3. Extend AI panel past Dec 2025 with fresh BTOS download.

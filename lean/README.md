# Lean 4 check of the proof of Proposition 1

Lean 4 (v4.35.0-rc3) with Mathlib. Build with `lake build` in this directory (Mathlib's cache is
downloaded by `lake update`). All files compile without `sorry`, and every theorem depends only on
the standard axioms `propext`, `Classical.choice`, `Quot.sound`.

| File | Theorem | What it checks |
|---|---|---|
| `Projection.lean` | `log_lr_eq`, `sq_norm_resid_sub`, `inner_resid_resid` | In a real inner product space, with `R` and `m` the residuals of the data and of the quadratic regressor after orthogonal projection onto the nuisance space, `-(‖R - γ m‖² - ‖R‖²)/2 = γ A - γ² B/2` with `A = ⟪m, y⟫`, `B = ‖m‖²`. |
| `Information.lean` | `infoB_mono` | The information `B_n` (weighted residual sum of squares of `q` on a design with fixed columns and `n` rows) is nondecreasing. |
| `Gaussian.lean` | `gaussian_increment_condMgf` | An increment independent of the past with law `N(m, v)` has conditional mgf `exp(m u + v u²/2)`. |
| `Supermartingale.lean` | `expProc_supermartingale` | If the increments of `A` have conditional mgf `exp(u γ ΔB + u² ΔB/2)` and `B` is nondecreasing, `exp(u A_n - u² B_n/2)` is a supermartingale for `u ≥ 0`, `γ ≤ 0`. |
| `Mixture.lean` | `mixture_supermartingale` | A mixture (Fubini) of supermartingales over a probability measure is a supermartingale. |
| `Ville.lean` | `ville` | Ville's inequality for nonnegative supermartingales, from optional stopping: `ε P(∃ n, f n ≥ ε) ≤ E f 0`. |
| `ClosedForm.lean` | `mixture_closed_form` | `∫_{u>0} e^{uA - u²B/2} · 2/(τ√(2π)) e^{-u²/(2τ²)} du = 2/(τ√a) · Φ(A/√a) · e^{A²/(2a)}`, `a = B + τ⁻²`. |
| `Main.lean` | `prop1_time_uniform` | Combines the above: under the hypotheses, `ε P(∃ n, E_n ≥ ε) ≤ 1` for the mixture `E_n`, i.e. `P(∃ n, E_n ≥ 2/α) ≤ α/2`. |

## What is not formalized

* That the score `A_n` of the regression residuals has conditionally Gaussian increments with
  mean `γ ΔB` and variance `ΔB` given the residual sigma-field (the innovation representation).
  It enters `prop1_time_uniform` as the hypothesis `hcond`; `gaussian_increment_condMgf` covers the
  case of independent increments only.
* The identification of the density ratio of the Gaussian residual vector with
  `exp(-(‖R - γ m‖² - ‖R‖²)/2)`, and the nesting of the residual sigma-fields.
* The link between `infoB` and the projector form of `B_n`.
* Adaptedness and joint integrability of the mixture (hypotheses `hadp`, `hjoint`).
* Proposition 2 (unknown scale), the polar-coordinate density, and Corollary 1 (restarts).

import Mathlib

/-!
# Likelihood-ratio identity for the curvature statistic (Appendix A of the manuscript)

In a real inner product space `V` (the `W`-weighted space of the manuscript), let `K` be the
column space of the nuisance design `D_n`, `P` the orthogonal projection onto `K`,
`R = y - P y` the residual of the data and `m = q - P q` the residual of the quadratic regressor.
Then with `A = ⟪m, y⟫` and `B = ‖m‖²`:

* `⟪m, R⟫ = A`,
* `‖R - γ m‖² = ‖R‖² - 2 γ A + γ² B`,

which is the identity behind `L_n(γ) = exp (γ A_n - γ² B_n / 2)`.
-/

open scoped RealInnerProductSpace

namespace Evigil

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (K : Submodule ℝ V) [K.HasOrthogonalProjection]

/-- The residual of `v` after projecting out the nuisance space `K`. -/
noncomputable def resid (v : V) : V := v - K.starProjection v

/-- The residual of the quadratic regressor is orthogonal to the nuisance space. -/
theorem resid_orthogonal (q : V) {x : V} (hx : x ∈ K) : ⟪resid K q, x⟫ = 0 := by
  have h := K.sub_starProjection_mem_orthogonal q
  rw [Submodule.mem_orthogonal'] at h
  simpa [resid, real_inner_comm] using h x hx

/-- `⟪m, R⟫ = ⟪m, y⟫`: the score `A_n` may be computed from the raw data. -/
theorem inner_resid_resid (y q : V) : ⟪resid K q, resid K y⟫ = ⟪resid K q, y⟫ := by
  have h : ⟪resid K q, K.starProjection y⟫ = 0 :=
    resid_orthogonal K q (K.starProjection_apply_mem y)
  unfold resid at *
  rw [inner_sub_right, h, sub_zero]

/-- Expansion of the residual sum of squares at curvature `γ`:
`‖R - γ m‖² = ‖R‖² - 2 γ A + γ² B`. -/
theorem sq_norm_resid_sub (y q : V) (γ : ℝ) :
    ‖resid K y - γ • resid K q‖ ^ 2
      = ‖resid K y‖ ^ 2 - 2 * γ * ⟪resid K q, y⟫ + γ ^ 2 * ‖resid K q‖ ^ 2 := by
  rw [norm_sub_sq_real, inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs,
    real_inner_comm, inner_resid_resid]
  ring

/-- The log likelihood ratio of the residual vector under curvature `γ` against curvature `0`,
for unit variance Gaussian noise, equals `γ A - γ² B / 2`. -/
theorem log_lr_eq (y q : V) (γ : ℝ) :
    -(‖resid K y - γ • resid K q‖ ^ 2 - ‖resid K y‖ ^ 2) / 2
      = γ * ⟪resid K q, y⟫ - γ ^ 2 * ‖resid K q‖ ^ 2 / 2 := by
  rw [sq_norm_resid_sub]; ring

end Evigil

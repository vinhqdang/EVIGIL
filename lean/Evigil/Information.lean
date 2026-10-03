import Mathlib

/-!
# Monotonicity of the information `B_n`

`B_n` is the weighted residual sum of squares of the quadratic regressor `q` after regression on
the nuisance design (regime levels and slope).  A regime that enters later has a design column
that is zero on the earlier rows, so the columns can be taken to be fixed (`Fin p`) and only the
number of rows grows.  Then `B_n` is nondecreasing in `n`, which is needed for `ΔB ≥ 0` in the
supermartingale step.
-/

open Finset

namespace Evigil

variable {p : ℕ}

/-- Weighted residual sum of squares of `q` on `n` rows of the design `d`. -/
noncomputable def infoB (w q : ℕ → ℝ) (d : ℕ → Fin p → ℝ) (n : ℕ) : ℝ :=
  ⨅ c : Fin p → ℝ, ∑ j ∈ range n, w j * (q j - ∑ i, d j i * c i) ^ 2

theorem infoB_nonneg {w q : ℕ → ℝ} (hw : ∀ j, 0 ≤ w j) (d : ℕ → Fin p → ℝ) (n : ℕ) :
    0 ≤ infoB w q d n :=
  Real.iInf_nonneg fun c => Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (sq_nonneg _)

theorem infoB_mono {w q : ℕ → ℝ} (hw : ∀ j, 0 ≤ w j) (d : ℕ → Fin p → ℝ) :
    Monotone (infoB w q d) := by
  intro n m hnm
  unfold infoB
  refine le_ciInf fun c => ?_
  refine (ciInf_le ⟨0, ?_⟩ c).trans ?_
  · rintro _ ⟨c', rfl⟩
    exact Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (sq_nonneg _)
  · exact Finset.sum_le_sum_of_subset_of_nonneg (range_mono hnm)
      fun j _ _ => mul_nonneg (hw j) (sq_nonneg _)

end Evigil

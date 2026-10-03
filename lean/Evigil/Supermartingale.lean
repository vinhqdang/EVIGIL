import Mathlib
import Evigil.Ville

/-!
# The exponential process `exp (u A_n - u² B_n / 2)` as a supermartingale

This is the one-step argument of Appendix A (composite one-sided null).  Assume the score process
`A` has conditionally Gaussian increments with mean `γ ΔB` and variance `ΔB`, in the form of the
conditional moment generating function, where `B` is a deterministic nondecreasing information
process.  Then, for `u ≥ 0` and `γ ≤ 0`, `M n = exp (u A n - u² B n / 2)` is a nonnegative
supermartingale (a martingale if `γ = 0`).  Combined with `Evigil.ville` this gives the
time-uniform bound of Proposition 1 for a fixed mixing value `u`.
-/

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal NNReal

namespace Evigil

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]
  {𝒢 : Filtration ℕ m0}

/-- The exponential process. -/
noncomputable def expProc (u : ℝ) (A : ℕ → Ω → ℝ) (B : ℕ → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  exp (u * A n ω - u ^ 2 * B n / 2)

theorem expProc_nonneg (u : ℝ) (A : ℕ → Ω → ℝ) (B : ℕ → ℝ) : 0 ≤ expProc u A B :=
  fun n ω => (exp_pos _).le

/-- The one-step algebra: `M (n+1) = M n · exp(u ΔA) · exp(-u² ΔB / 2)`. -/
theorem expProc_succ (u : ℝ) (A : ℕ → Ω → ℝ) (B : ℕ → ℝ) (n : ℕ) (ω : Ω) :
    expProc u A B (n + 1) ω
      = expProc u A B n ω * exp (u * (A (n + 1) ω - A n ω)) * exp (-(u ^ 2 * (B (n + 1) - B n) / 2)) := by
  unfold expProc
  rw [← exp_add, ← exp_add]; congr 1; ring

/-- **Supermartingale step.**  If `A` is adapted, `exp (u A n)` is integrable, `B` is nondecreasing,
and the increments of `A` have conditional mgf `exp (u γ ΔB + u² ΔB / 2)` given the past, then for
`u ≥ 0`, `γ ≤ 0` the process `exp (u A n - u² B n / 2)` is a supermartingale. -/
theorem expProc_supermartingale {u γ : ℝ} (hu : 0 ≤ u) (hγ : γ ≤ 0)
    {A : ℕ → Ω → ℝ} {B : ℕ → ℝ} (hB : Monotone B) (hA : StronglyAdapted 𝒢 A)
    (hint : ∀ n, Integrable (fun ω => exp (u * A n ω)) μ)
    (hincr : ∀ n, Integrable (fun ω => exp (u * (A (n + 1) ω - A n ω))) μ)
    (hcond : ∀ n, μ[fun ω => exp (u * (A (n + 1) ω - A n ω)) | 𝒢 n]
      =ᵐ[μ] fun _ => exp (u * γ * (B (n + 1) - B n) + u ^ 2 * (B (n + 1) - B n) / 2)) :
    Supermartingale (expProc u A B) 𝒢 μ := by
  have hM_meas : ∀ n, StronglyMeasurable[𝒢 n] (expProc u A B n) := fun n => by
    unfold expProc
    exact Real.continuous_exp.comp_stronglyMeasurable
      (((hA n).const_mul u).sub stronglyMeasurable_const)
  have hM_int : ∀ n, Integrable (expProc u A B n) μ := fun n => by
    have : expProc u A B n = fun ω => exp (-(u ^ 2 * B n / 2)) * exp (u * A n ω) := by
      ext ω; unfold expProc; rw [← exp_add]; congr 1; ring
    rw [this]; exact (hint n).const_mul _
  refine supermartingale_nat (fun n => hM_meas n |>.mono (𝒢.le n |> fun _ => le_rfl)) hM_int ?_
  intro n
  set c : ℝ := exp (-(u ^ 2 * (B (n + 1) - B n) / 2)) with hc
  set g : Ω → ℝ := fun ω => exp (u * (A (n + 1) ω - A n ω)) with hg
  set F : Ω → ℝ := fun ω => expProc u A B n ω * c with hF
  have hFmeas : StronglyMeasurable[𝒢 n] F := (hM_meas n).mul stronglyMeasurable_const
  have hfact : expProc u A B (n + 1) = F * g := by
    ext ω; simp only [Pi.mul_apply, hF, hg, hc]; rw [expProc_succ]; ring
  have hprod : Integrable (F * g) μ := hfact ▸ hM_int (n + 1)
  have hpull := condExp_mul_of_stronglyMeasurable_left hFmeas hprod (hincr n)
  rw [hfact]
  filter_upwards [hpull, hcond n] with ω h1 h2
  rw [h1, Pi.mul_apply, h2]
  have hMn : 0 ≤ expProc u A B n ω := (exp_pos _).le
  have hdB : 0 ≤ B (n + 1) - B n := sub_nonneg.mpr (hB (Nat.le_succ n))
  have key : c * exp (u * γ * (B (n + 1) - B n) + u ^ 2 * (B (n + 1) - B n) / 2)
      = exp (u * γ * (B (n + 1) - B n)) := by
    rw [hc, ← exp_add]; congr 1; ring
  have hle : exp (u * γ * (B (n + 1) - B n)) ≤ 1 := by
    apply exp_le_one_iff.mpr
    have : u * γ ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hu hγ
    exact mul_nonpos_of_nonpos_of_nonneg this hdB
  calc F ω * exp (u * γ * (B (n + 1) - B n) + u ^ 2 * (B (n + 1) - B n) / 2)
      = expProc u A B n ω * (c * exp (u * γ * (B (n + 1) - B n) + u ^ 2 * (B (n + 1) - B n) / 2)) := by
        simp only [hF]; ring
    _ = expProc u A B n ω * exp (u * γ * (B (n + 1) - B n)) := by rw [key]
    _ ≤ expProc u A B n ω := by nlinarith

end Evigil

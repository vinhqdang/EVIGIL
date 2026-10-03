import Mathlib

/-!
# Ville's inequality for nonnegative supermartingales

The time-uniform guarantee of Proposition 1: if `M` is a nonnegative supermartingale with
`E M₀ ≤ 1`, then `P (∃ n, M n ≥ 1/α) ≤ α`.  Mathlib has Doob's maximal inequality for
submartingales; here the supermartingale version is derived from optional stopping.
-/

open MeasureTheory Filter Finset
open scoped ENNReal NNReal

namespace Evigil

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
  {𝒢 : Filtration ℕ m0} {f : ℕ → Ω → ℝ}

/-- Finite-horizon Ville inequality: `ε μ{max_{k ≤ n} f k ≥ ε} ≤ E f 0`. -/
theorem ville_finite (hsup : Supermartingale f 𝒢 μ) (hnn : 0 ≤ f) {ε : ℝ≥0} (n : ℕ) :
    ε * μ {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_add_one fun k => f k ω}
      ≤ ENNReal.ofReal (∫ ω, f 0 ω ∂μ) := by
  have hsub : Submartingale (-f) 𝒢 μ := hsup.neg
  have hadp : StronglyAdapted 𝒢 f := hsup.stronglyAdapted
  have hstop : IsStoppingTime 𝒢 (fun ω => ((hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ) : WithTop ℕ)) :=
    hadp.adapted.isStoppingTime_hittingBtwn measurableSet_Ici
  have hbdd : ∀ ω, ((hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ) : WithTop ℕ) ≤ n :=
    fun ω => by exact_mod_cast (hittingBtwn_le ω : hittingBtwn f {y : ℝ | (ε : ℝ) ≤ y} 0 n ω ≤ n)
  have hint : Integrable (stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ))) μ := by
    have := hsub.integrable_stoppedValue hstop hbdd
    have h2 : stoppedValue (-f) (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ))
        = -stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) := by
      ext ω; simp [stoppedValue]
    rw [h2] at this
    exact integrable_neg_iff.mp this
  -- optional stopping: E[f_τ] ≤ E[f_0]
  have hmono : ∫ ω, stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) ω ∂μ
      ≤ ∫ ω, f 0 ω ∂μ := by
    have h := hsub.expected_stoppedValue_mono (isStoppingTime_const 𝒢 0) hstop
      (fun ω => by simp) hbdd
    have e1 : stoppedValue (-f) (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ))
        = -stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) := by
      ext ω; simp [stoppedValue]
    have e0 : stoppedValue (-f) (fun _ => ((0 : ℕ) : WithTop ℕ)) = -f 0 := by
      ext ω; simp [stoppedValue]
    have e0' : ∀ x, stoppedValue (-f) (fun _ => ((0 : ℕ) : WithTop ℕ)) x = -f 0 x := fun x => by
      simp [stoppedValue]
    have e1' : ∀ x, stoppedValue (-f) (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) x
        = -stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) x := fun x => by
      simp [stoppedValue]
    simp only [e1', integral_neg] at h
    simp [stoppedValue, integral_neg] at h
    simpa [stoppedValue] using h
  set S : Set Ω := {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_add_one fun k => f k ω} with hS
  have hSmeas : MeasurableSet S :=
    measurableSet_le measurable_const
      (measurable_range_sup'' fun n _ => (hsup.stronglyMeasurable n).measurable.le (𝒢.le n))
  have hge : ∀ ω ∈ S, (ε : ℝ) ≤
      stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) ω := by
    intro ω hx
    simp_rw [hS, Set.mem_setOf_eq, le_sup'_iff, mem_range, Nat.lt_succ_iff] at hx
    refine stoppedValue_hittingBtwn_mem ?_
    simp only [Set.mem_Icc, zero_le, true_and, Set.mem_ofPred_eq]
    exact let ⟨j, hj₁, hj₂⟩ := hx; ⟨j, hj₁, hj₂⟩
  have h1 := setIntegral_ge_of_const_le_real hSmeas (measure_ne_top _ _) hge hint.integrableOn
  have h2 : ∫ ω in S, stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) ω ∂μ
      ≤ ∫ ω, stoppedValue f (fun ω ↦ (hittingBtwn f {y : ℝ | ε ≤ y} 0 n ω : ℕ)) ω ∂μ :=
    setIntegral_le_integral hint (Filter.Eventually.of_forall fun ω => by
      simp only [Pi.zero_apply, stoppedValue]; exact hnn _ ω)
  have h0 : 0 ≤ ∫ ω, f 0 ω ∂μ := integral_nonneg (hnn 0)
  rw [show (ε : ℝ≥0∞) * μ S = ε • μ S by simp [ENNReal.smul_def], ENNReal.le_ofReal_iff_toReal_le, ENNReal.toReal_smul]
  · exact h1.trans (h2.trans hmono)
  · exact ENNReal.mul_ne_top (by simp) (measure_ne_top _ _)
  · exact h0

/-- **Ville's inequality**: for a nonnegative supermartingale `f`,
`ε μ{∃ n, f n ≥ ε} ≤ E f 0`. -/
theorem ville (hsup : Supermartingale f 𝒢 μ) (hnn : 0 ≤ f) {ε : ℝ≥0} :
    ε * μ {ω | ∃ n, (ε : ℝ) ≤ f n ω} ≤ ENNReal.ofReal (∫ ω, f 0 ω ∂μ) := by
  have hmono : Monotone fun n : ℕ =>
      {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_add_one fun k => f k ω} := by
    intro a b hab ω hω
    simp only [Set.mem_setOf_eq, le_sup'_iff, mem_range] at hω ⊢
    obtain ⟨j, hj, hε⟩ := hω
    exact ⟨j, by omega, hε⟩
  have hunion : {ω | ∃ n, (ε : ℝ) ≤ f n ω} = ⋃ n : ℕ,
      {ω | (ε : ℝ) ≤ (range (n + 1)).sup' nonempty_range_add_one fun k => f k ω} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, le_sup'_iff, mem_range]
    constructor
    · rintro ⟨n, hn⟩; exact ⟨n, n, by omega, hn⟩
    · rintro ⟨n, j, _, hj⟩; exact ⟨j, hj⟩
  rw [hunion, hmono.measure_iUnion, ENNReal.mul_iSup]
  exact iSup_le fun n => ville_finite hsup hnn n


end Evigil

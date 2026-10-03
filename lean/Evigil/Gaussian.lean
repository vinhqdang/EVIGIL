import Mathlib

/-!
# Conditional moment generating function of a Gaussian increment

If the increment `X = A (n+1) - A n` of the score process is independent of the past `𝒢 n` and
has law `N (γ ΔB, ΔB)`, its conditional mgf given the past is the constant
`exp (u γ ΔB + u² ΔB / 2)`, which is the hypothesis `hcond` of
`Evigil.expProc_supermartingale`.
-/

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal NNReal

namespace Evigil

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem gaussian_increment_condMgf {X : Ω → ℝ} {m : ℝ} {v : ℝ≥0} (hXm : Measurable X)
    (hX : HasLaw X (gaussianReal m v) μ) {𝒢₀ : MeasurableSpace Ω} (h𝒢 : 𝒢₀ ≤ m0)
    (hind : Indep (MeasurableSpace.comap X inferInstance) 𝒢₀ μ) (u : ℝ) :
    μ[fun ω => exp (u * X ω) | 𝒢₀] =ᵐ[μ] fun _ => exp (m * u + v * u ^ 2 / 2) := by
  have hmeas : Measurable[MeasurableSpace.comap X inferInstance] X := comap_measurable X
  have hf : StronglyMeasurable[MeasurableSpace.comap X inferInstance] (fun ω => exp (u * X ω)) :=
    (Real.continuous_exp.comp_stronglyMeasurable ((hmeas.stronglyMeasurable).const_mul u))
  have h := condExp_indep_eq (μ := μ) hXm.comap_le (h𝒢) hf hind
  refine h.trans (Filter.Eventually.of_forall fun ω => ?_)
  have := mgf_gaussianReal hX u
  simp only [mgf] at this
  exact this

end Evigil

import Mathlib

/-!
# Mixtures of supermartingales are supermartingales

The half-normal mixture over the curvature `δ` in the manuscript is an integral of the
exponential supermartingales over a probability measure.  This file shows (by Fubini) that such
a mixture is again a supermartingale, provided that the mixture is adapted and the family is
jointly integrable.
-/

open MeasureTheory
open scoped ENNReal

namespace Evigil

variable {Ω U : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]
  [MeasurableSpace U] {𝒢 : Filtration ℕ m0}

theorem mixture_supermartingale (ν : Measure U) [IsProbabilityMeasure ν]
    (f : ℕ → Ω → U → ℝ)
    (hsup : ∀ᵐ u ∂ν, Supermartingale (fun n ω => f n ω u) 𝒢 μ)
    (hadp : StronglyAdapted 𝒢 (fun n ω => ∫ u, f n ω u ∂ν))
    (hint : ∀ n, Integrable (Function.uncurry (f n)) (μ.prod ν)) :
    Supermartingale (fun n ω => ∫ u, f n ω u ∂ν) 𝒢 μ := by
  refine supermartingale_of_setIntegral_succ_le hadp (fun n => (hint n).integral_prod_left) ?_
  intro n s hs
  have hsm : MeasurableSet s := 𝒢.le n s hs
  have hint_s : ∀ k, Integrable (Function.uncurry (f k)) ((μ.restrict s).prod ν) :=
    fun k => (hint k).mono_measure (Measure.prod_mono Measure.restrict_le_self le_rfl)
  have swap : ∀ k, ∫ ω in s, ∫ u, f k ω u ∂ν ∂μ = ∫ u, ∫ ω in s, f k ω u ∂μ ∂ν :=
    fun k => integral_integral_swap (hint_s k)
  rw [swap, swap]
  refine integral_mono_ae (hint_s (n + 1)).integral_prod_right (hint_s n).integral_prod_right ?_
  filter_upwards [hsup] with u hu
  exact hu.setIntegral_le (Nat.le_succ n) hs

end Evigil

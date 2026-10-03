import Mathlib
import Evigil.Ville
import Evigil.Supermartingale
import Evigil.Mixture
import Evigil.Gaussian

/-!
# Proposition 1 (time-uniform guarantee), conditional on the Gaussian innovation structure

Let `A` be the score process and `B` the (deterministic, nondecreasing) information process with
`A 0 = 0`, `B 0 = 0`.  For a mixing law `ν` on `u ≥ 0` (the half-normal prior in the manuscript)
let `E n = ∫ exp (u A n - u² B n / 2) dν(u)`.  If for `ν`-almost every `u` the increments of `A`
have conditional mgf `exp (u γ ΔB + u² ΔB / 2)` given the past with `γ ≤ 0`, then

`ε · P (∃ n, E n ≥ ε) ≤ 1`  for every `ε ≥ 0`,

that is, `P (∃ n, E n ≥ 2/α) ≤ α/2`, which is Proposition 1.
-/

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal NNReal

namespace Evigil

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]
  {𝒢 : Filtration ℕ m0}

theorem prop1_time_uniform {γ : ℝ} (hγ : γ ≤ 0)
    {A : ℕ → Ω → ℝ} {B : ℕ → ℝ} (hB : Monotone B) (hA0 : A 0 = 0) (hB0 : B 0 = 0)
    (hA : StronglyAdapted 𝒢 A)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hν : ν (Set.Iio 0) = 0)
    (hint : ∀ᵐ u ∂ν, ∀ n, Integrable (fun ω => exp (u * A n ω)) μ)
    (hincr : ∀ᵐ u ∂ν, ∀ n, Integrable (fun ω => exp (u * (A (n + 1) ω - A n ω))) μ)
    (hcond : ∀ᵐ u ∂ν, ∀ n, μ[fun ω => exp (u * (A (n + 1) ω - A n ω)) | 𝒢 n]
      =ᵐ[μ] fun _ => exp (u * γ * (B (n + 1) - B n) + u ^ 2 * (B (n + 1) - B n) / 2))
    (hadp : StronglyAdapted 𝒢 (fun n ω => ∫ u, expProc u A B n ω ∂ν))
    (hjoint : ∀ n, Integrable (Function.uncurry fun ω u => expProc u A B n ω) (μ.prod ν))
    {ε : ℝ≥0} :
    ε * μ {ω | ∃ n, (ε : ℝ) ≤ ∫ u, expProc u A B n ω ∂ν} ≤ 1 := by
  have hu : ∀ᵐ u ∂ν, 0 ≤ u := by
    rw [ae_iff]; simpa [not_le, Set.Iio] using hν
  have hsup : ∀ᵐ u ∂ν, Supermartingale (fun n ω => expProc u A B n ω) 𝒢 μ := by
    filter_upwards [hu, hint, hincr, hcond] with u hu0 h1 h2 h3
    exact expProc_supermartingale hu0 hγ hB hA h1 h2 h3
  have hmix := mixture_supermartingale (μ := μ) ν (fun n ω u => expProc u A B n ω) hsup hadp hjoint
  have hnn : 0 ≤ fun n ω => ∫ u, expProc u A B n ω ∂ν :=
    fun n ω => integral_nonneg fun u => expProc_nonneg u A B n ω
  have h := ville hmix hnn (ε := ε)
  have h0 : (∫ ω, (∫ u, expProc u A B 0 ω ∂ν) ∂μ) = 1 := by
    simp [expProc, hA0, hB0]
  rw [h0] at h
  simpa using h

end Evigil

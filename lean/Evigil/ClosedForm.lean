import Mathlib

/-!
# Closed form of the half-normal mixture

For `a = B + τ⁻² > 0`,
`∫_{u>0} exp (u A - u² B / 2) · 2/(τ√(2π)) exp (-u²/(2τ²)) du
   = 2/(τ √a) · exp (A²/(2a)) · P (N (A/a, 1/a) > 0)`,
and `P (N (A/a, 1/a) > 0) = Φ (A/√a)`, which gives the formula for `E_n^+` in the manuscript.
-/

open MeasureTheory ProbabilityTheory Real Set
open scoped ENNReal NNReal

namespace Evigil

/-- Completing the square. -/
theorem complete_square (A a u : ℝ) (ha : a ≠ 0) :
    u * A - u ^ 2 * a / 2 = A ^ 2 / (2 * a) - a * (u - A / a) ^ 2 / 2 := by
  field_simp
  ring

theorem mixture_integral_eq (A B τ : ℝ) (hB : 0 ≤ B) (hτ : 0 < τ) {v : ℝ≥0}
    (hv : (v : ℝ) = 1 / (B + τ⁻¹ ^ 2)) :
    ∫ u in Ioi (0 : ℝ), exp (u * A - u ^ 2 * B / 2) *
        (2 / (τ * √(2 * π)) * exp (-u ^ 2 / (2 * τ ^ 2)))
      = 2 / (τ * √(B + τ⁻¹ ^ 2)) * exp (A ^ 2 / (2 * (B + τ⁻¹ ^ 2)))
          * (gaussianReal (A / (B + τ⁻¹ ^ 2)) v).real (Ioi 0) := by
  set a : ℝ := B + τ⁻¹ ^ 2 with ha
  have hapos : 0 < a := by positivity
  have hv0 : v ≠ 0 := by
    intro h; rw [h] at hv; simp at hv; exact hapos.ne' hv.symm
  rw [show (gaussianReal (A / a) v).real (Ioi 0) = ∫ x in Ioi 0, gaussianPDFReal (A / a) v x from ?_]
  · rw [← integral_const_mul]
    refine setIntegral_congr_fun measurableSet_Ioi fun u _ => ?_
    have hsq : u ^ 2 * a = u ^ 2 * B + u ^ 2 / τ ^ 2 := by
      rw [ha]; field_simp
    have e1 : exp (u * A - u ^ 2 * B / 2) * exp (-u ^ 2 / (2 * τ ^ 2))
        = exp (A ^ 2 / (2 * a)) * exp (-(u - A / a) ^ 2 / (2 * (v : ℝ))) := by
      rw [← exp_add, ← exp_add]; congr 1
      have := complete_square A a u hapos.ne'
      rw [hv]
      have h2 : -u ^ 2 / (2 * τ ^ 2) = -(u ^ 2 / τ ^ 2) / 2 := by ring
      rw [h2]
      have h3 : -(u - A / a) ^ 2 / (2 * (1 / a)) = -(a * (u - A / a) ^ 2) / 2 := by field_simp
      rw [h3]
      nlinarith [this, hsq]
    have hsqrt : √(2 * π * (v : ℝ)) = √(2 * π) / √a := by
      rw [hv, show 2 * π * (1 / a) = 2 * π / a by ring, Real.sqrt_div (by positivity)]
    rw [gaussianPDFReal, hsqrt]
    have hs2 : 0 < √(2 * π) := by positivity
    have hsa : 0 < √a := Real.sqrt_pos.mpr hapos
    calc exp (u * A - u ^ 2 * B / 2) * (2 / (τ * √(2 * π)) * exp (-u ^ 2 / (2 * τ ^ 2)))
        = 2 / (τ * √(2 * π)) * (exp (u * A - u ^ 2 * B / 2) * exp (-u ^ 2 / (2 * τ ^ 2))) := by ring
      _ = 2 / (τ * √(2 * π)) * (exp (A ^ 2 / (2 * a)) * exp (-(u - A / a) ^ 2 / (2 * (v : ℝ)))) := by rw [e1]
      _ = _ := by field_simp
  · rw [Measure.real, gaussianReal_apply_eq_integral _ hv0,
      ENNReal.toReal_ofReal (setIntegral_nonneg measurableSet_Ioi fun x _ => gaussianPDFReal_nonneg _ _ _)]

/-- `P (N (m, v) > 0) = Φ (m / √v)`. -/
theorem gaussian_pos_prob (m : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    (gaussianReal m v).real (Ioi 0) = cdf (gaussianReal 0 1) (m / √v) := by
  set c : ℝ := √v with hc
  have hcpos : 0 < c := Real.sqrt_pos.mpr (by exact_mod_cast pos_iff_ne_zero.mpr hv)
  have hstd : (gaussianReal m v) = ((gaussianReal 0 1).map (fun x => c * x)).map (fun x => m + x) := by
    rw [gaussianReal_map_const_mul, gaussianReal_map_const_add]
    congr 1
    · simp
    · congr 1
      ext; simp [hc, Real.sq_sqrt]
  have h1 : (gaussianReal m v).real (Ioi 0) = (gaussianReal 0 1).real (Ioi (-(m / c))) := by
    rw [hstd, Measure.map_map (by fun_prop) (by fun_prop), Measure.real,
      Measure.map_apply (by fun_prop) measurableSet_Ioi, Measure.real]
    congr 2
    ext x
    simp only [Function.comp, mem_preimage, mem_Ioi]
    constructor
    · intro h; rw [neg_div', div_lt_iff₀ hcpos]; linarith
    · intro h; rw [neg_div', div_lt_iff₀ hcpos] at h; linarith
  have h2 : (gaussianReal 0 1).real (Ioi (-(m / c))) = (gaussianReal 0 1).real (Iio (m / c)) := by
    have hneg := gaussianReal_map_neg (μ := (0 : ℝ)) (v := (1 : ℝ≥0))
    simp only [neg_zero] at hneg
    conv_lhs => rw [← hneg]
    rw [Measure.real, Measure.map_apply (by fun_prop) measurableSet_Ioi, Measure.real]
    congr 2
    ext x
    simp [neg_lt]
  haveI : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal one_ne_zero
  rw [h1, h2, cdf_eq_real, Measure.real, Measure.real]
  congr 1
  exact measure_congr Iio_ae_eq_Iic

/-- **Closed form of the half-normal mixture** (equation for `E_n^+` in the manuscript):
with `a = B + τ⁻²`,
`∫_{u>0} exp (u A - u² B / 2) · 2/(τ√(2π)) exp (-u²/(2τ²)) du
  = 2/(τ √a) · Φ (A/√a) · exp (A²/(2a))`. -/
theorem mixture_closed_form (A B τ : ℝ) (hB : 0 ≤ B) (hτ : 0 < τ) :
    ∫ u in Ioi (0 : ℝ), exp (u * A - u ^ 2 * B / 2) *
        (2 / (τ * √(2 * π)) * exp (-u ^ 2 / (2 * τ ^ 2)))
      = 2 / (τ * √(B + τ⁻¹ ^ 2)) * cdf (gaussianReal 0 1) (A / √(B + τ⁻¹ ^ 2))
          * exp (A ^ 2 / (2 * (B + τ⁻¹ ^ 2))) := by
  set a : ℝ := B + τ⁻¹ ^ 2 with ha
  have hapos : 0 < a := by positivity
  have hv : ((Real.toNNReal (1 / a) : ℝ≥0) : ℝ) = 1 / a := Real.coe_toNNReal _ (by positivity)
  have hv0 : Real.toNNReal (1 / a) ≠ 0 := by
    intro h; rw [h] at hv; simp at hv; exact hapos.ne' hv.symm
  rw [mixture_integral_eq A B τ hB hτ hv, gaussian_pos_prob _ hv0, hv]
  have : A / a / √(1 / a) = A / √a := by
    have hsa : 0 < √a := Real.sqrt_pos.mpr hapos
    rw [one_div, Real.sqrt_inv]
    field_simp
    rw [Real.sq_sqrt hapos.le]
  rw [this]; ring

end Evigil

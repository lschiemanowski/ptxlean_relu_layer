import ReluLayer.PTX.FloatArithmetic
import FloatLib.Floats.Formats.BinaryInterchange.Model.RealSemantics
import FloatLib.Floats.Formats.BinaryInterchange.Model.Fields.Proof

namespace ReluLayer.PTX.FP32

open FloatLib.Floats.Formats.BinaryInterchange

/-- Finite encodings exclude both NaNs and infinities; signed zeros and subnormals are included. -/
def isFinite (word : Bits) : Bool := Model.isFinite (decode word)

/-- Exact real value on finite words. FloatLib's total convention returns zero on exceptions. -/
noncomputable def toReal (word : Bits) : ℝ := Model.toReal (decode word)

private theorem fraction_decode (word : Bits) :
    Model.fracField (decode word) = fraction word := by
  change (word &&& BitVec.ofNat 32 (2 ^ fractionBits - 1)).toNat = _
  rw [BitVec.toNat_and, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt (by decide : 2 ^ fractionBits - 1 < 2 ^ 32)]
  exact Nat.and_two_pow_sub_one_eq_mod _ _

private theorem exponent_decode (word : Bits) :
    Model.expField (decode word) = exponent word := by
  change ((word >>> fractionBits) &&& BitVec.ofNat 32 (2 ^ exponentBits - 1)).toNat = _
  rw [BitVec.toNat_and, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt (by decide : 2 ^ exponentBits - 1 < 2 ^ 32),
    Nat.and_two_pow_sub_one_eq_mod, BitVec.toNat_ushiftRight]
  simp [exponent, Nat.shiftRight_eq_div_pow]

private theorem negative_decode (word : Bits) :
    Model.signBit (decode word) = negative word := by
  cases hn : negative word
  · have hp := of_decide_eq_false hn
    apply (Model.signBit_eq_false_iff_toNatBits_lt_signMaskNat (decode word)).2
    simpa [negative, fractionBits, exponentBits, decode, Model.toNatBits,
      Model.ofBits, FloatFormat.signMaskNat, FloatFormat.signBitIndex,
      FloatFormat.binary32, FloatFormat.bitWidth] using Nat.lt_of_not_ge hp
  · have hp := of_decide_eq_true hn
    apply (Model.signBit_eq_true_iff_signMaskNat_le (decode word)).2
    simpa [negative, fractionBits, exponentBits, decode, Model.toNatBits,
      Model.ofBits, FloatFormat.signMaskNat, FloatFormat.signBitIndex,
      FloatFormat.binary32, FloatFormat.bitWidth] using hp

private theorem isNaN_decode (word : Bits) : Model.isNaN (decode word) = isNaN word := by
  change (Model.expField (decode word) == 2 ^ exponentBits - 1 &&
    Model.fracField (decode word) != 0) = _
  rw [exponent_decode, fraction_decode]
  apply Bool.eq_iff_iff.mpr
  simp only [isNaN, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, decide_eq_true_eq]

@[simp] theorem isFinite_positiveZero : isFinite positiveZero = true := by decide

@[simp] theorem toReal_positiveZero : toReal positiveZero = 0 :=
  Model.toReal_zero FloatFormat.binary32 false

/-- The kernel's ReLU output stays finite whenever its input is finite. -/
theorem isFinite_maxPositiveZero (word : Bits) (hfinite : isFinite word = true) :
    isFinite (maxPositiveZero word) = true := by
  unfold maxPositiveZero
  split
  · exact isFinite_positiveZero
  · exact hfinite

/-- Decoding finite kernel words turns the selected maximum into real ReLU exactly. -/
theorem toReal_maxPositiveZero (word : Bits) (hfinite : isFinite word = true) :
    toReal (maxPositiveZero word) = max (toReal word) 0 := by
  have hnan : isNaN word = false := by
    rw [← isNaN_decode]
    exact Model.isNaN_eq_false_of_isFinite_eq_true (decode word) hfinite
  obtain ⟨d, hd⟩ := Model.exists_toDyadic?_of_isFinite hfinite
  have hsign : d.negative = negative word :=
    (Model.sign_eq_signBit_of_toDyadic?_some hd).trans (negative_decode word)
  have hvalue : toReal word = d.toReal := by
    unfold toReal
    rw [Model.toReal_eq, hd]
  have hmag : 0 ≤ (d.significand : ℝ) * (2 : ℝ) ^ d.exponent := by positivity
  cases hn : negative word
  · have hnonneg : 0 ≤ toReal word := by
      rw [hvalue, FloatLib.Numerics.Dyadic.toReal,
        FloatLib.Numerics.Dyadic.cast_signedSignificand, hsign, hn]
      simpa using hmag
    simp [maxPositiveZero, hnan, hn, max_eq_left hnonneg]
  · have hnonpos : toReal word ≤ 0 := by
      rw [hvalue, FloatLib.Numerics.Dyadic.toReal,
        FloatLib.Numerics.Dyadic.cast_signedSignificand, hsign, hn]
      simp only [↓reduceIte, neg_one_mul]
      nlinarith
    simp [maxPositiveZero, hnan, hn, max_eq_right hnonpos]

/-- ReLU cannot increase an absolute discrepancy between real preactivations. -/
theorem relu_abs_sub_le (a b : ℝ) : |max a 0 - max b 0| ≤ |a - b| := by
  rcases le_total a 0 with ha | ha <;> rcases le_total b 0 with hb | hb
  · simp [max_eq_right ha, max_eq_right hb]
  · rw [max_eq_right ha, max_eq_left hb, abs_of_nonpos (by linarith),
      abs_of_nonpos (by linarith)]
    linarith
  · rw [max_eq_left ha, max_eq_right hb, abs_of_nonneg (by linarith),
      abs_of_nonneg (by linarith)]
    linarith
  · simp [max_eq_left ha, max_eq_left hb]

/-- Any preactivation error budget passes through the finite bit-pattern ReLU unchanged. -/
theorem maxPositiveZero_abs_error_le (word : Bits) (hfinite : isFinite word = true) (z : ℝ) :
    |toReal (maxPositiveZero word) - max z 0| ≤ |toReal word - z| := by
  rw [toReal_maxPositiveZero word hfinite]
  exact relu_abs_sub_le (toReal word) z

end ReluLayer.PTX.FP32

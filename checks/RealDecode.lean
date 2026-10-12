import ReluLayer.PTX.RealDecode

open ReluLayer.PTX
open FloatLib.Floats.Formats.BinaryInterchange

-- Actual real decoding, not just the executable classification/maximum.
example : FP32.toReal 0 = 0 := FP32.toReal_positiveZero
example : FP32.toReal 0x80000000 = 0 := Model.toReal_zero FloatFormat.binary32 true
example : FP32.toReal 0x3fc00000 = (3 : ℝ) / 2 := by
  have hd : Model.toDyadic? (FP32.decode 0x3fc00000) =
      some ⟨false, 12582912, -23⟩ := by decide
  unfold FP32.toReal
  rw [Model.toReal_eq, hd]
  norm_num [FloatLib.Numerics.Dyadic.toReal]
example : FP32.toReal (FP32.maxPositiveZero 0xbfc00000) = 0 := by
  have hword : FP32.maxPositiveZero 0xbfc00000 = 0 := by decide
  rw [hword]; exact FP32.toReal_positiveZero

-- The bridge remains symbolic for all finite words and arbitrary comparison values.
example (word : FP32.Bits) (hfinite : FP32.isFinite word = true) :
    FP32.toReal (FP32.maxPositiveZero word) = max (FP32.toReal word) 0 :=
  FP32.toReal_maxPositiveZero word hfinite
example (word : FP32.Bits) (hfinite : FP32.isFinite word = true) (z budget : ℝ)
    (hbudget : |FP32.toReal word - z| ≤ budget) :
    |FP32.toReal (FP32.maxPositiveZero word) - max z 0| ≤ budget :=
  (FP32.maxPositiveZero_abs_error_le word hfinite z).trans hbudget
example : FP32.isFinite 0x7f800000 = false := by decide
example : FP32.isFinite 0xff800000 = false := by decide
example : FP32.isFinite 0x7fc01234 = false := by decide
example : FP32.isFinite 0x7f801234 = false := by decide
example : FP32.isFinite 1 = true := by decide
example : FP32.isFinite 0x80000001 = true := by decide
example : FP32.isFinite 0x7f7fffff = true := by decide
example : FP32.maxPositiveZero 0x80000000 = 0 := by decide

#print axioms FP32.isFinite_maxPositiveZero
#print axioms FP32.toReal_positiveZero
#print axioms FP32.toReal_maxPositiveZero
#print axioms FP32.relu_abs_sub_le
#print axioms FP32.maxPositiveZero_abs_error_le

-- Independent exact rational oracles for the shared finite-value decoder.
def main : IO Unit := do
  let fixtures : List (FP32.Bits × Rat × FP32.Bits) :=
    [(0, 0, 0), (0x80000000, 0, 0),
     (0x3fc00000, 3 / 2, 0x3fc00000), (0xbfc00000, -3 / 2, 0),
     (1, 1 / (2 ^ 149), 1), (0x80000001, -1 / (2 ^ 149), 0),
     (0x00800000, 1 / (2 ^ 126), 0x00800000),
     (0x7f7fffff, (2 ^ 24 - 1) * (2 ^ 104), 0x7f7fffff),
     (0xff7fffff, -(2 ^ 24 - 1) * (2 ^ 104), 0)]
  for (word, exactValue, expectedWord) in fixtures do
    unless FP32.isFinite word && Model.toRat? (FP32.decode word) == some exactValue do
      throw (IO.userError s!"Finite exact-value decoding failed for {word.toNat}.")
    let output := FP32.maxPositiveZero word
    unless output == expectedWord && FP32.isFinite output &&
        Model.toRat? (FP32.decode output) == some (max exactValue 0) do
      throw (IO.userError s!"ReLU encoding/value/finite result failed for {word.toNat}.")
  for word in [0x7f800000, 0xff800000, 0x7fc01234, 0xff801234] do
    unless !FP32.isFinite word && (Model.toRat? (FP32.decode word)).isNone do
      throw (IO.userError "Exceptional encoding was admitted as a finite real input.")
  IO.println "Finite real/ReLU bridge and error-transfer audits, signed zeros, exact normal/subnormal/extreme values and exceptional-input exclusion pass."

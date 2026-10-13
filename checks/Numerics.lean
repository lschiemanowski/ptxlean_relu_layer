import ReluLayer.PTX.Layer
import ReluLayer.PTX.KernelExecution
import ReluLayer.PTX.FloatArithmetic
import ReluLayer.PTX.RoundingError

open ReluLayer.PTX
open FloatLib.Floats.Formats.BinaryInterchange

namespace ArithmeticChecks

-- Exact bit preservation includes NaN payloads and both signed zeros.
example (word : FP32.Bits) : FP32.encode (FP32.decode word) = word := FP32.encode_decode word
example (a b : FP32.Bits) :
    FP32.floatLibArithmetic.addRN a b = FP32.encode (FloatLib.Floats.Formats.BinaryInterchange.Model.Spec.add (FP32.decode a) (FP32.decode b)) :=
  FP32.addRN_eq_reference a b
example (a b c : FP32.Bits) :
    FP32.floatLibArithmetic.fmaRN a b c = FP32.encode (FloatLib.Floats.Formats.BinaryInterchange.Model.Spec.fma (FP32.decode a) (FP32.decode b) (FP32.decode c)) :=
  FP32.fmaRN_eq_reference a b c

#print axioms FP32.encode_decode
#print axioms FP32.addRN_eq_reference
#print axioms FP32.fmaRN_eq_reference

private def assertWord (name : String) (actual expected : FP32.Bits) : IO Unit :=
  unless actual == expected do
    throw (IO.userError s!"{name}: got {actual.toNat}, expected {expected.toNat}")

def check : IO Unit := do
  let op := FP32.floatLibArithmetic
  assertWord "even halfway addition" (op.addRN 0x3f800000 0x33800000) 0x3f800000
  assertWord "odd halfway addition" (op.addRN 0x3f800001 0x33800000) 0x3f800002
  assertWord "subnormal addition" (op.addRN 1 1) 2
  assertWord "negative zeros" (op.addRN 0x80000000 0x80000000) 0x80000000
  assertWord "opposite zeros" (op.addRN 0 0x80000000) 0
  assertWord "overflow" (op.addRN 0x7f7fffff 0x7f7fffff) 0x7f800000
  assertWord "fused cancellation" (op.fmaRN 0x3f800001 0x3f7ffffe 0xbf800000) 0xa8800000
  assertWord "normal-to-subnormal FMA" (op.fmaRN 0x00800000 0x3f000000 0) 0x00400000
  assertWord "underflow halfway to zero" (op.fmaRN 1 0x3f000000 0) 0
  assertWord "underflow halfway to even" (op.fmaRN 3 0x3f000000 0) 2
  unless FP32.isNaN (op.addRN 0x7f800000 0xff800000) &&
      FP32.isNaN (op.fmaRN 0 0x7f800000 0) &&
      FP32.isNaN (op.fmaRN 0x7fc01234 0x3f800000 0) do
    throw (IO.userError "Exceptional arithmetic did not return NaN.")
  for bits in [0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0xff801234] do
    assertWord "word roundtrip" (FP32.encode (FP32.decode bits)) bits
  -- Lean 4.35 native execution is a comparison oracle, not a GPU refinement proof.
  for (a, b, c) in [(0x3f800001, 0x3f7ffffe, 0xbf800000),
      (0x00800000, 0x3f000000, 0), (1, 0x3f000000, 0), (3, 0x3f000000, 0)] do
    let fa := Float32.ofBits (UInt32.ofBitVec a)
    let fb := Float32.ofBits (UInt32.ofBitVec b)
    let fc := Float32.ofBits (UInt32.ofBitVec c)
    assertWord "Lean 4.35 FMA comparison" (op.fmaRN a b c) (fa.fma fb fc).toBits.toBitVec
  -- Concrete arithmetic in the full 1024-column, 31-instruction kernel.
  let weightBytes := layerLaunch.rows * layerLayout.rowStride
  let xBase := weightBytes
  let bBase := xBase + layerLayout.rowStride
  let yBase := bBase + layerLaunch.rows * fp32Bytes
  let arg : Argument → Word64 := fun a => BitVec.ofNat 64 (1000 + (match a with
    | .weights => 0 | .input => xBase | .bias => bBase | .output => yBase))
  let env : Environment :=
    { params := fun symbol => if symbol = "W_ptr" then some (arg .weights)
        else if symbol = "x_ptr" then some (arg .input)
        else if symbol = "b_ptr" then some (arg .bias)
        else if symbol = "y_ptr" then some (arg .output) else none
      toGlobal := fun a => some (BitVec.ofNat 64 (a.toNat - 1000)) }
  let initial : State Register :=
    { regs := fun _ => 99, regs64 := fun _ => 999, pc := 0, predicates := fun _ => true
      globalMemory := fun a =>
        if a.toNat < weightBytes then some 0x3f800000 -- W = 1
        else if a.toNat < bBase then some 0x3f000000 -- x = 1/2
        else if a.toNat < yBase then some 0x3e800000 -- b = 1/4
        else some 0x42c60000 }
  for row in [0, layerLaunch.rows - 1] do
    match run (kernelProgram layerLaunch layerLayout) ⟨⟨BitVec.ofNat 32 (row % layerLaunch.threads), BitVec.ofNat 32 (row / layerLaunch.threads)⟩, env, op⟩ (kernelSteps layerLaunch layerLayout) initial with
    | none => throw (IO.userError "Concrete binary32 kernel execution failed.")
    | some final =>
      let output := BitVec.ofNat 64 (yBase + row * fp32Bytes)
      assertWord "kernel result 512.25" (final.regs .result) 0x44001000
      unless final.returned && final.pc == 31 && final.globalMemory output == some 0x44001000 &&
          final.globalMemory (output + 4) == some 0x42c60000 &&
          final.globalMemory (BitVec.ofNat 64 bBase) == some 0x3e800000 do
        throw (IO.userError "Concrete binary32 kernel output/return/frame failed.")
  IO.println "FloatLib refinement audits, encoding, rounding/FMA/subnormal/exception cases, Lean 4.35 comparison and concrete full-width kernel execution pass."

end ArithmeticChecks

namespace DecodeChecks

-- Bit-pattern oracles cover both signs, infinities, NaNs, and subnormals.
example : FP32.maxPositiveZero 0x80000000 = 0 := by decide
example : FP32.maxPositiveZero 0xbf800000 = 0 := by decide
example : FP32.maxPositiveZero 0xff800000 = 0 := by decide
example : FP32.maxPositiveZero 0x7fc00001 = 0 := by decide
example : FP32.maxPositiveZero 0x7f800001 = 0 := by decide
example : FP32.maxPositiveZero 0xffc00001 = 0 := by decide
example : FP32.maxPositiveZero 0x7f800000 = 0x7f800000 := by decide
example : FP32.maxPositiveZero 1 = 1 := by decide
example : FP32.maxPositiveZero 0x80000001 = 0 := by decide
example : FP32.maxPositiveZero 0 = 0 := by decide
example : FP32.maxPositiveZero 0x3f800000 = 0x3f800000 := by decide

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

#print axioms FP32.isFinite_maxPositiveZero
#print axioms FP32.toReal_positiveZero
#print axioms FP32.toReal_maxPositiveZero
#print axioms FP32.relu_abs_sub_le
#print axioms FP32.maxPositiveZero_abs_error_le

-- Independent exact rational oracles for the shared finite-value decoder.
def check : IO Unit := do
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

end DecodeChecks

namespace ErrorChecks

private def smallLayout : Layout := ⟨2, by decide, by decide⟩

private def tieData : TensorData smallLayout 1 :=
  { weights := fun _ _ => 0x3f800000
    input := fun j => if j.val = 0 then 0x3f800000 else 0x33800000
    bias := fun _ => 0x33800000 }

-- FiniteRowTrace checks the complete trace, rather than only the inputs.
private theorem tieFinite : FiniteRowTrace tieData 0 := by
  constructor <;> decide

example : |FP32.toReal (rowResult FP32.floatLibArithmetic tieData 0) -
    exactRowResult tieData 0| ≤ traceErrorBudget tieData 0 :=
  (rowResult_abs_error_le tieData 0 tieFinite).2

-- All inputs are finite here, but the first accumulator overflows.
private def overflowData : TensorData smallLayout 1 :=
  { weights := fun _ _ => 0x7f7fffff
    input := fun _ => 0x40000000
    bias := fun _ => 0 }

example : (∀ j, FP32.isFinite (overflowData.weights 0 j) = true) ∧
    (∀ j, FP32.isFinite (overflowData.input j) = true) ∧
    FP32.isFinite (overflowData.bias 0) = true := by decide
example : ¬ FiniteRowTrace overflowData 0 := by
  intro hf
  have h := hf.accumulators ⟨1, by decide⟩
  have hn : FP32.isFinite (prefixAccumulator FP32.floatLibArithmetic overflowData 0 1) = false := by
    decide
  rw [hn] at h
  contradiction

#print axioms preactivation_abs_error_le
#print axioms rowResult_abs_error_le
#print axioms kernel_abs_error_le

private def asRat (word : FP32.Bits) : IO Rat :=
  match Model.toRat? (FP32.decode word) with
  | some value => pure value
  | none => throw (IO.userError "Exceptional word in finite rational oracle.")

def check : IO Unit := do
  -- Independent exact-real formulas and hand-computed half-ULP sums.
  -- Cases: accumulated ties, cancellation at the ReLU boundary, gradual underflow.
  let fixtures : List (List FP32.Bits × FP32.Bits × Rat × Rat) :=
    [([0x3f800000, 0x33800000], 0x33800000, 1 + 1 / (2 ^ 23), 3 / (2 ^ 24)),
     ([0x3f800000, 0x33800000], 0xbf800000, 1 / (2 ^ 24),
       2 / (2 ^ 24) + 1 / (2 ^ 150))]
  for (inputs, bias, exactOutput, budget) in fixtures do
    let data : TensorData smallLayout 1 :=
      { weights := fun _ _ => 0x3f800000
        input := fun j => inputs[j.val]!
        bias := fun _ => bias }
    let mut exactDot : Rat := 0
    for x in inputs do exactDot := exactDot + (← asRat x)
    let target := max (exactDot + (← asRat bias)) 0
    let actual ← asRat (rowResult FP32.floatLibArithmetic data 0)
    unless target == exactOutput && actual == (if bias == 0xbf800000 then 0 else 1) &&
        actual != target && abs (actual - target) ≤ budget do
      throw (IO.userError "Accumulated rounding/cancellation oracle failed.")
  let tinyData : TensorData smallLayout 1 :=
    { weights := fun _ j => if j.val = 0 then 1 else 0
      input := fun _ => 0x3f000000
      bias := fun _ => 0 }
  let tiny ← asRat (rowResult FP32.floatLibArithmetic tinyData 0)
  unless tiny == 0 && abs (tiny - 1 / (2 ^ 150)) ≤ 3 / (2 ^ 150) do
    throw (IO.userError "Gradual-underflow error oracle failed.")
  unless !FP32.isFinite (prefixAccumulator FP32.floatLibArithmetic overflowData 0 1) do
    throw (IO.userError "Finite-input overflow was admitted.")
  IO.println "Generic prefix/row error statements, finite trace and overflow exclusion, theorem audits, accumulated ties, ReLU cancellation and subnormal rounding oracles pass."

end ErrorChecks

def main : IO Unit := do
  ArithmeticChecks.check
  DecodeChecks.check
  ErrorChecks.check

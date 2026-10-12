import ReluLayer.PTX.RoundingError

open ReluLayer.PTX
open FloatLib.Floats.Formats.BinaryInterchange

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

def main : IO Unit := do
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

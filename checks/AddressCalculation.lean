import ReluLayer.PTX.Layer
import ReluLayer.PTX.Optional.AddressBounds

open ReluLayer.PTX

-- Arbitrary word arithmetic for integer-only instruction fixtures.
private def testArithmetic : FP32.Arithmetic :=
  ⟨fun w x a => w * x + a, fun a b => a + b⟩

-- Wide multiplication keeps the high half of the unsigned product.
example : (wideProduct 4294967295 4294967295).toNat = 18446744065119617025 := by decide

private def wideInitial : State Nat :=
  ⟨fun _ => 42, 9, fun r => if r = 0 then 18446744073709551615 else 17, fun _ => none, fun _ => false, false⟩
private def aliasSum : State Nat :=
  (Instruction.add64 0 (.reg 0) (.imm 2)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ wideInitial |>.getD wideInitial

example : (aliasSum.regs64 0).toNat = 1 := by decide
example : (aliasSum.regs64 1).toNat = 17 := by decide
example : (aliasSum.regs 0).toNat = 42 := by decide
example : aliasSum.pc = 10 := by decide
example : ((Instruction.mulWide 0 (.reg 0) (.imm 4294967295)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ wideInitial).map (fun s => s.regs64 0) =
    some (wideProduct 42 4294967295) := by decide

-- A small alternative layout catches any hidden dependence on width 1024.
private def tinyLayout : Layout := ⟨3, by decide, by decide⟩
private def tinyInitial : State Register :=
  ⟨fun _ => 1, 0, fun r => match r with
    | .weightBase => 100
    | .biasBase => 200
    | .outputBase => 300
    | _ => 999, fun _ => none, fun _ => false, false⟩

example : ((run (addressCalculation 3) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5 tinyInitial).map
    (fun final => ((final.regs64 .weightRead).toNat,
      (final.regs64 .biasRead).toNat, (final.regs64 .outputWrite).toNat))) =
    some (112, 204, 304) := by decide

example : ∃ final, run (addressCalculation 3) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5 tinyInitial = some final ∧
    AddressResult tinyLayout 2 tinyInitial final :=
  addressCalculation_correct tinyLayout 2 ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ tinyInitial
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

-- The last element fits with the buffer end one byte below 2^64.
private def boundaryLayout : Layout := ⟨1, by decide, by decide⟩
private def boundaryInitial : State Register :=
  ⟨fun _ => 1, 0, fun _ => 18446744073709551607, fun _ => none, fun _ => false, false⟩
example : ∃ final, run (addressCalculation 1) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5 boundaryInitial = some final ∧
    AddressResult boundaryLayout 2 boundaryInitial final :=
  addressCalculation_correct boundaryLayout 2 ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ boundaryInitial
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

-- Without buffer-end bounds, the instruction semantics still permits wraparound.
private def wrappingInitial : State Register :=
  ⟨fun _ => 1, 0, fun _ => 18446744073709551615, fun _ => none, fun _ => false, false⟩
example : ((run (addressCalculation 1) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5 wrappingInitial).map
    (fun final => (final.regs64 .weightRead).toNat)) = some 3 := by decide

#print axioms wideProduct_toNat
#print axioms rowSelection_correct
#print axioms addressCalculation_exec
#print axioms addressCalculation_correct
#print axioms layer_addressCalculation_correct

private def instructionText : Instruction Register → String := Instruction.ptxText

def main (args : List String) : IO Unit := do
  let source ← IO.FS.readFile (args.headD "kernels/relu_layer.ptx")
  let normalized := "\n" ++ String.intercalate "\n"
    ((source.splitOn "\n").map (fun line => line.trimAscii.toString))
  let expected := "\n" ++ String.intercalate "\n"
    ((addressCalculation layerLayout.inputWidth).map instructionText) ++ "\n"
  unless (normalized.splitOn expected).length == 2 do
    throw (IO.userError s!"Expected exactly one address fragment:\n{expected}")
  IO.println "Five address instructions match the formal layout and register operands."

import ReluLayer.PTX.Layer
import ReluLayer.PTX.RowSelection

open ReluLayer.PTX

-- Arbitrary word arithmetic for integer-only instruction fixtures.
private def testArithmetic : FP32.Arithmetic :=
  ⟨fun w x a => w * x + a, fun a b => a + b⟩

-- Aliasing uses the old source value; multiplication and addition both wrap.
private def aliasState : State Nat := ⟨fun r => if r = 0 then 4294967295 else 17, 7, fun _ => 0, fun _ => none, fun _ => false, false⟩
private def aliasResult : State Nat :=
  (Instruction.madLo 0 (.reg 0) (.imm 2) (.imm 3)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ aliasState |>.getD aliasState

example : (aliasResult.regs 0).toNat = 1 := by decide
example : (aliasResult.regs 1).toNat = 17 := by decide
example : aliasResult.pc = 8 := by decide
example : ((Instruction.mov 0 (.reg 0)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ aliasState).map
    (fun s => s.regs 0) = some (aliasState.regs 0) := by decide

-- Fetching outside the fragment fails; exhaustion of fuel alone succeeds.
example : step (rowSelection 5) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ ⟨fun _ => 0, 3, fun _ => 0, fun _ => none, fun _ => false, false⟩ = none := by decide
example : (run (rowSelection 5) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 0 ⟨fun _ => 0, 3, fun _ => 0, fun _ => none, fun _ => false, false⟩).isSome = true := by decide

-- Zero fuel alone does not imply that an active thread has returned.
example : ¬ Returns [] ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ aliasState aliasState := by
  intro h
  exact Bool.noConfusion h.2

-- The state contract specializes to a different launch with arbitrary scratch.
example (t : Thread ⟨7, 5, by decide, by decide⟩) (initial : State Register)
    (hpc : initial.pc = 0) (hactive : initial.returned = false) (env : Environment) :
    ∃ final, run (rowSelection 5) ⟨t.identity, env, testArithmetic⟩ 3 initial = some final ∧
      RowResult ⟨7, 5, by decide, by decide⟩ t initial final :=
  rowSelection_correct _ (by decide) t initial hpc hactive ⟨t.identity, env, testArithmetic⟩ rfl

#print axioms Thread.row_lt
#print axioms rowSelection_correct
#print axioms layerLaunch_rows

private def instructionText : Instruction Register → String := Instruction.ptxText

private def requireOnce (source expected : String) : IO Unit := do
  unless (source.splitOn expected).length == 2 do
    throw (IO.userError s!"Expected exactly one source passage:\n{expected}")

/-- A source comparison for this fragment, not a general or verified PTX parser. -/
def main (args : List String) : IO Unit := do
  let path := args.headD "kernels/relu_layer.ptx"
  let source ← IO.FS.readFile path
  let normalized := "\n" ++ String.intercalate "\n"
    ((source.splitOn "\n").map (fun line => line.trimAscii.toString))
  let expected := String.intercalate "\n"
    ((rowSelection layerLaunch.threads).map instructionText)
  requireOnce normalized ("\n" ++ expected ++ "\n")
  requireOnce normalized s!".reqntid {layerLaunch.threads}, 1, 1\n"
  requireOnce normalized
    s!"// Launch: grid=({layerLaunch.blocks},1,1), block=({layerLaunch.threads},1,1); arguments: W, x, b, y.\n"
  IO.println "Row-selection instructions, required block size, and documented grid match the formal configuration."

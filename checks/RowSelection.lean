import ReluLayer.PTX.RowSelection

open ReluLayer.PTX

-- Aliasing uses the old source value; multiplication and addition both wrap.
private def aliasState : State Nat := ⟨fun r => if r = 0 then 4294967295 else 17, 7⟩
private def aliasResult : State Nat :=
  (Instruction.madLo 0 (.reg 0) (.imm 2) (.imm 3)).execute ⟨0, 0⟩ aliasState

example : (aliasResult.regs 0).toNat = 1 := by decide
example : (aliasResult.regs 1).toNat = 17 := by decide
example : aliasResult.pc = 8 := by decide
example : ((Instruction.mov 0 (.reg 0)).execute ⟨0, 0⟩ aliasState).regs 0 =
    aliasState.regs 0 := by decide

-- Fetching outside the fragment fails; exhaustion of fuel alone succeeds.
example : step (rowSelection 5) ⟨0, 0⟩ ⟨fun _ => 0, 3⟩ = none := by decide
example : (run (rowSelection 5) ⟨0, 0⟩ 0 ⟨fun _ => 0, 3⟩).isSome = true := by decide

-- The arithmetic theorem specializes to launch counts other than the layer's.
example (t : Thread ⟨7, 5, by decide, by decide⟩) (regs : RowReg → Word32) :
    ∃ s, run (rowSelection 5) t.identity 3 ⟨regs, 0⟩ = some s ∧
      s.pc = 3 ∧ (s.regs .row).toNat = t.row ∧ (s.regs .row).toNat < 35 :=
  rowSelection_correct _ (by decide) t regs

#print axioms Thread.row_lt
#print axioms rowSelection_correct
#print axioms layerLaunch_rows
#print axioms layer_rowSelection_correct

private def regName : RowReg → String
  | .thread => "%thread"
  | .block => "%block"
  | .row => "%row"

private def operandText : Operand RowReg → String
  | .reg r => regName r
  | .imm v => toString v.toNat
  | .tidX => "%tid.x"
  | .ctaidX => "%ctaid.x"

private def instructionText : Instruction RowReg → String
  | .mov dst src => s!"mov.u32 {regName dst}, {operandText src};"
  | .madLo dst a b c =>
      s!"mad.lo.u32 {regName dst}, {operandText a}, {operandText b}, {operandText c};"

private def requireOnce (source expected : String) : IO Unit := do
  unless (source.splitOn expected).length == 2 do
    throw (IO.userError s!"Expected exactly one source passage:\n{expected}")

/-- A source comparison for this fragment, not a general or verified PTX parser. -/
def main (args : List String) : IO Unit := do
  let path := args.headD "kernels/relu_layer.ptx"
  let source ← IO.FS.readFile path
  let normalized := String.intercalate "\n"
    ((source.splitOn "\n").map (fun line => line.trimAscii.toString))
  let expected := String.intercalate "\n"
    ((rowSelection layerLaunch.threads).map instructionText)
  requireOnce normalized expected
  requireOnce normalized s!".reqntid {layerLaunch.threads}, 1, 1\n"
  requireOnce normalized
    s!"// Launch: grid=({layerLaunch.blocks},1,1), block=({layerLaunch.threads},1,1); arguments: W, x, b, y.\n"
  IO.println "Row-selection instructions, required block size, and documented grid match the formal configuration."

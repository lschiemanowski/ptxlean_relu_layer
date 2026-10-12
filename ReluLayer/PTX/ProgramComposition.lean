import ReluLayer.PTX.Machine

namespace ReluLayer.PTX

/-- Embedding a fragment changes only its PC, never its data state. -/
def State.shiftPC (offset : Nat) (s : State Reg) : State Reg :=
  { s with pc := offset + s.pc }

/-- Absolute branch targets move with the fragment. -/
def Instruction.relocate (offset : Nat) : Instruction Reg → Instruction Reg
  | .branchIf predicate target => .branchIf predicate (offset + target)
  | inst => inst

private theorem execute_relocate [DecidableEq Reg] (offset : Nat) (inst : Instruction Reg)
    (id : Identity) (s : State Reg) (env : Environment) (ops : Option FP32.Arithmetic) :
    (inst.relocate offset).execute id (s.shiftPC offset) env ops =
      (inst.execute id s env ops).map (State.shiftPC offset) := by
  cases inst <;>
    simp [Instruction.relocate, Instruction.execute, State.shiftPC, State.write,
      State.write64, Operand.read, Operand64.read, Option.map_map, Function.comp_def,
      Nat.add_assoc]
  all_goals first | rfl | split <;> simp_all

private theorem step_relocate [DecidableEq Reg]
    (prelude program : List (Instruction Reg)) (id : Identity) (s : State Reg)
    (env : Environment) (ops : Option FP32.Arithmetic) :
    step (prelude ++ program.map (Instruction.relocate prelude.length)) id
      (s.shiftPC prelude.length) env ops =
      (step program id s env ops).map (State.shiftPC prelude.length) := by
  have hoff : ¬ prelude.length + s.pc < prelude.length := by omega
  simp only [step, State.shiftPC, hoff, List.getElem?_append, Nat.add_sub_cancel_left,
    List.getElem?_map]
  cases ha : s.returned <;> simp only [Bool.false_eq_true, ↓reduceIte, Option.map_none]
  cases h : program[s.pc]? with
  | none => simp
  | some inst => simp only [Option.map_some, Option.bind_some]; simpa [State.shiftPC, ha] using execute_relocate prelude.length inst id s env ops

/-- A successful fragment run embeds after any prelude, with PC and branches relocated. -/
theorem run_relocate [DecidableEq Reg]
    (prelude program : List (Instruction Reg)) (id : Identity) (fuel : Nat)
    (s final : State Reg) (env : Environment) (ops : Option FP32.Arithmetic)
    (h : run program id fuel s env ops = some final) :
    run (prelude ++ program.map (Instruction.relocate prelude.length)) id fuel
      (s.shiftPC prelude.length) env ops = some (final.shiftPC prelude.length) := by
  induction fuel generalizing s with
  | zero => simpa [run] using congrArg (Option.map (State.shiftPC prelude.length)) h
  | succ fuel ih =>
    obtain ⟨next, hs, ht⟩ := Option.bind_eq_some_iff.mp h
    rw [run, step_relocate, hs, Option.map_some, Option.bind_some]
    exact ih next ht

end ReluLayer.PTX

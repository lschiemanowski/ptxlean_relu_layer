import ReluLayer.PTX.Machine

namespace ReluLayer.PTX

variable {Reg : Type}

/-- Embedding a fragment changes only its PC, never its data state. -/
def State.shiftPC (offset : Nat) (s : State Reg) : State Reg :=
  { s with pc := offset + s.pc }

/-- Absolute branch targets move with the fragment. -/
def Instruction.relocate (offset : Nat) : Instruction Reg → Instruction Reg
  | .branchIf predicate target => .branchIf predicate (offset + target)
  | inst => inst

private theorem execute_relocate [DecidableEq Reg] (offset : Nat) (inst : Instruction Reg)
    (ctx : ExecutionContext) (s : State Reg) :
    (inst.relocate offset).execute ctx (s.shiftPC offset) =
      (inst.execute ctx s).map (State.shiftPC offset) := by
  cases inst <;>
    simp [Instruction.relocate, Instruction.execute, State.shiftPC, State.writeNext,
      State.write64Next, Operand.read, Operand64.read, Option.map_map, Function.comp_def,
      Nat.add_assoc]
  all_goals first | rfl | split <;> simp_all

private theorem step_relocate [DecidableEq Reg]
    (prelude program : List (Instruction Reg)) (ctx : ExecutionContext) (s : State Reg) :
    step (prelude ++ program.map (Instruction.relocate prelude.length)) ctx (s.shiftPC
        prelude.length) =
      (step program ctx s).map (State.shiftPC prelude.length) := by
  have hoff : ¬ prelude.length + s.pc < prelude.length := by omega
  simp only [step, State.shiftPC, hoff, List.getElem?_append, Nat.add_sub_cancel_left,
    List.getElem?_map]
  cases ha : s.returned <;> simp only [Bool.false_eq_true, ↓reduceIte, Option.map_none]
  cases h : program[s.pc]? with
  | none => simp
  | some inst => simp only [Option.map_some, Option.bind_some]; simpa [State.shiftPC, ha] using
      execute_relocate prelude.length inst ctx s

/-- A successful fragment run embeds after any prelude, with PC and branches relocated. -/
theorem run_relocate [DecidableEq Reg]
    (prelude program : List (Instruction Reg)) (ctx : ExecutionContext) (fuel : Nat)
    (s final : State Reg)
    (h : run program ctx fuel s = some final) :
    run (prelude ++ program.map (Instruction.relocate prelude.length)) ctx fuel (s.shiftPC
        prelude.length) = some (final.shiftPC prelude.length) := by
  induction fuel generalizing s with
  | zero => simpa [run] using congrArg (Option.map (State.shiftPC prelude.length)) h
  | succ fuel ih =>
    obtain ⟨next, hs, ht⟩ := Option.bind_eq_some_iff.mp h
    rw [run, step_relocate, hs, Option.map_some, Option.bind_some]
    exact ih next ht

end ReluLayer.PTX

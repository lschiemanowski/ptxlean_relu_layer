import ReluLayer.PTX.Registers

namespace ReluLayer.PTX

inductive Argument where
  | weights | input | bias | output
  deriving DecidableEq

def Argument.symbol : Argument → String
  | .weights => "W_ptr"
  | .input => "x_ptr"
  | .bias => "b_ptr"
  | .output => "y_ptr"

def Argument.register : Argument → Register
  | .weights => .weightBase
  | .input => .inputRead
  | .bias => .biasBase
  | .output => .outputBase

/-- Exactly lines 22--29 of the selected kernel. -/
def baseLoading : List (Instruction Register) :=
  [ .ldParam64 .weightBase "W_ptr",
    .ldParam64 .inputRead "x_ptr",
    .ldParam64 .biasBase "b_ptr",
    .ldParam64 .outputBase "y_ptr",
    .cvtaGlobal64 .weightBase (.reg .weightBase),
    .cvtaGlobal64 .inputRead (.reg .inputRead),
    .cvtaGlobal64 .biasBase (.reg .biasBase),
    .cvtaGlobal64 .outputBase (.reg .outputBase) ]

private def finalState (env : Environment) (args : Argument → Word64)
    (initial : State Register) : State Register :=
  let s := initial.write64 .weightBase (args .weights)
  let s := s.write64 .inputRead (args .input)
  let s := s.write64 .biasBase (args .bias)
  let s := s.write64 .outputBase (args .output)
  let s := s.write64 .weightBase (env.space.globalAddress (s.regs64 .weightBase))
  let s := s.write64 .inputRead (env.space.globalAddress (s.regs64 .inputRead))
  let s := s.write64 .biasBase (env.space.globalAddress (s.regs64 .biasBase))
  s.write64 .outputBase (env.space.globalAddress (s.regs64 .outputBase))

/-- The block's exit contract, including register and memory frame conditions. -/
structure BaseResult (env : Environment) (args : Argument → Word64)
    (initial final : State Register) : Prop where
  pc : final.pc = 8
  bases : ∀ arg : Argument, final.regs64 arg.register = env.space.globalAddress (args arg)
  bounds : ∀ arg : Argument, (final.regs64 arg.register).toNat < env.space.globalWindow.size
  regs32 : final.regs = initial.regs
  other64 : ∀ r, r ≠ .weightBase → r ≠ .inputRead → r ≠ .biasBase → r ≠ .outputBase →
    final.regs64 r = initial.regs64 r
  memory : final.globalMemory = initial.globalMemory
  predicates : final.predicates = initial.predicates
  active : final.returned = initial.returned

/-- Four initialized parameter slots and four admitted pointers suffice for all eight steps. -/
theorem baseLoading_correct (env : Environment) (args : Argument → Word64) (id : Identity)
    (initial : State Register) (hpc : initial.pc = 0) (hactive : initial.returned = false)
    (hparams : ∀ arg : Argument, env.params arg.symbol = some (args arg))
    (hspace : ∀ arg : Argument, env.space.admits (args arg) = true)
    (ops : Option FP32.Arithmetic := none) :
    ∃ final, run baseLoading id 8 initial env ops = some final ∧
      BaseResult env args initial final := by
  have hpw := hparams .weights
  have hpx := hparams .input
  have hpb := hparams .bias
  have hpy := hparams .output
  simp only [Argument.symbol] at hpw hpx hpb hpy
  have hw := env.space.toGlobal_of_admits (args .weights) (hspace .weights)
  have hx := env.space.toGlobal_of_admits (args .input) (hspace .input)
  have hb := env.space.toGlobal_of_admits (args .bias) (hspace .bias)
  have hy := env.space.toGlobal_of_admits (args .output) (hspace .output)
  refine ⟨finalState env args initial, ?_, ?_⟩
  · simp [run, step, baseLoading, Instruction.execute, Operand64.read,
      State.write64, finalState, hpc, hactive, hpw, hpx, hpb, hpy, hw, hx, hb, hy]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [finalState, State.write64, hpc]
    · intro arg
      cases arg <;> simp [finalState, State.write64, Argument.register]
    · intro arg
      have heq : (finalState env args initial).regs64 arg.register =
          env.space.globalAddress (args arg) := by
        cases arg <;> simp [finalState, State.write64, Argument.register]
      rw [heq]
      exact (env.space.admitted_bounds (args arg) (hspace arg)).2
    · rfl
    · intro r hw hx hb hy
      simp [finalState, State.write64, hw, hx, hb, hy]
    · rfl
    · rfl
    · rfl

end ReluLayer.PTX

import ReluLayer.PTX.Registers

namespace ReluLayer.PTX

inductive Argument where
  | weights | input | bias | output
  deriving DecidableEq

def Argument.symbol : Argument → String
  | .weights => "W_ptr" | .input => "x_ptr" | .bias => "b_ptr" | .output => "y_ptr"

def Argument.register : Argument → Register
  | .weights => .weightBase | .input => .inputRead | .bias => .biasBase | .output => .outputBase

abbrev Buffers := Argument → Word64

/-- Exactly the four parameter loads and four global conversions of the selected kernel. -/
def baseLoading : List (Instruction Register) :=
  [ .ldParam64 .weightBase "W_ptr", .ldParam64 .inputRead "x_ptr",
    .ldParam64 .biasBase "b_ptr", .ldParam64 .outputBase "y_ptr",
    .cvtaGlobal64 .weightBase (.reg .weightBase),
    .cvtaGlobal64 .inputRead (.reg .inputRead),
    .cvtaGlobal64 .biasBase (.reg .biasBase),
    .cvtaGlobal64 .outputBase (.reg .outputBase) ]

private def finalState (args : Argument → Word64) (buffers : Buffers)
    (initial : State Register) : State Register :=
  let s := initial.write64Next .weightBase (args .weights)
  let s := s.write64Next .inputRead (args .input)
  let s := s.write64Next .biasBase (args .bias)
  let s := s.write64Next .outputBase (args .output)
  let s := s.write64Next .weightBase (buffers .weights)
  let s := s.write64Next .inputRead (buffers .input)
  let s := s.write64Next .biasBase (buffers .bias)
  s.write64Next .outputBase (buffers .output)

/-- Converted bases and the state preserved by the loading block. -/
structure BaseResult (buffers : Buffers) (initial final : State Register) : Prop where
  pc : final.pc = baseLoading.length
  bases : ∀ arg, final.regs64 arg.register = buffers arg
  regs32 : final.regs = initial.regs
  other64 : ∀ r, r ≠ .weightBase → r ≠ .inputRead → r ≠ .biasBase → r ≠ .outputBase →
    final.regs64 r = initial.regs64 r
  memory : final.globalMemory = initial.globalMemory
  predicates : final.predicates = initial.predicates
  active : final.returned = initial.returned

/-- The four supplied generic arguments must convert to the four supplied global buffers. -/
theorem baseLoading_correct (ctx : ExecutionContext) (args : Argument → Word64) (buffers : Buffers)
    (initial : State Register) (hpc : initial.pc = 0) (hactive : initial.returned = false)
    (hparams : ∀ arg, ctx.env.params arg.symbol = some (args arg))
    (hconvert : ∀ arg, ctx.env.toGlobal (args arg) = some (buffers arg)) :
    ∃ final, run baseLoading ctx baseLoading.length initial = some final ∧
      BaseResult buffers initial final := by
  have hpw := hparams .weights
  have hpx := hparams .input
  have hpb := hparams .bias
  have hpy := hparams .output
  simp only [Argument.symbol] at hpw hpx hpb hpy
  have hw := hconvert .weights
  have hx := hconvert .input
  have hb := hconvert .bias
  have hy := hconvert .output
  refine ⟨finalState args buffers initial, ?_, ?_⟩
  · simp [run, step, baseLoading, Instruction.execute, Operand64.read,
      State.write64Next, finalState, hpc, hactive, hpw, hpx, hpb, hpy, hw, hx, hb, hy]
  · constructor
    · simp [finalState, State.write64Next, baseLoading, hpc]
    · intro arg; cases arg <;> simp [finalState, State.write64Next, Argument.register]
    · rfl
    · intro r hw hx hb hy; simp [finalState, State.write64Next, hw, hx, hb, hy]
    · rfl
    · rfl
    · rfl

end ReluLayer.PTX

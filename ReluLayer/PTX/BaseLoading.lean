import ReluLayer.PTX.Registers
import ReluLayer.PTX.ProgramComposition

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
  pc : final.pc = initial.pc + baseLoading.length
  bases : ∀ arg, final.regs64 arg.register = buffers arg
  regs32 : final.regs = initial.regs
  other64 : ∀ r, r ≠ .weightBase → r ≠ .inputRead → r ≠ .biasBase → r ≠ .outputBase →
    final.regs64 r = initial.regs64 r
  memory : final.globalMemory = initial.globalMemory
  predicates : final.predicates = initial.predicates
  active : final.returned = initial.returned

/-- The four supplied generic arguments must convert to the four supplied global buffers. -/
theorem baseLoading_correct (program : List (Instruction Register)) (ctx : ExecutionContext)
    (args : Argument → Word64) (buffers : Buffers) (initial : State Register)
    (hactive : initial.returned = false)
    (hparams : ∀ arg, ctx.env.params arg.symbol = some (args arg))
    (hconvert : ∀ arg, ctx.env.toGlobal (args arg) = some (buffers arg))
    (hcode : CodeAt program initial.pc baseLoading) :
    ∃ final, run program ctx baseLoading.length initial = some final ∧
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
  have h₀ := hcode 0 (by simp [baseLoading])
  have h₁ := hcode 1 (by simp [baseLoading])
  have h₂ := hcode 2 (by simp [baseLoading])
  have h₃ := hcode 3 (by simp [baseLoading])
  have h₄ := hcode 4 (by simp [baseLoading])
  have h₅ := hcode 5 (by simp [baseLoading])
  have h₆ := hcode 6 (by simp [baseLoading])
  have h₇ := hcode 7 (by simp [baseLoading])
  simp [baseLoading] at h₀ h₁ h₂ h₃ h₄ h₅ h₆ h₇
  refine ⟨finalState args buffers initial, ?_, ?_⟩
  · simp [run, step, baseLoading, h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇,
      Instruction.execute, Operand64.read, State.write64Next, finalState,
      hactive, hpw, hpx, hpb, hpy, hw, hx, hb, hy, Nat.add_assoc]
  · constructor
    · simp [finalState, State.write64Next, baseLoading, Nat.add_assoc]
    · intro arg; cases arg <;> simp [finalState, State.write64Next, Argument.register]
    · rfl
    · intro r hw hx hb hy; simp [finalState, State.write64Next, hw, hx, hb, hy]
    · rfl
    · rfl
    · rfl

end ReluLayer.PTX

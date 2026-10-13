import ReluLayer.PTX.RowSelection

namespace ReluLayer.PTX

/-- FP32 storage size in bytes. -/
def fp32Bytes : Nat := 4

/-- The layout supplies the input width; the row stride is derived from it. -/
structure Layout where
  inputWidth : Nat
  inputWidth_pos : 0 < inputWidth
  stride_fits : inputWidth * fp32Bytes < 2 ^ 32

def Layout.rowStride (layout : Layout) : Nat := layout.inputWidth * fp32Bytes

/-- Exactly the two offset multiplies and three pointer additions in the kernel. -/
def addressCalculation (inputWidth : Nat) : List (Instruction Register) :=
  [ .mulWide .rowOffset (.reg .row) (.imm (BitVec.ofNat 32 (inputWidth * fp32Bytes))),
    .mulWide .elementOffset (.reg .row) (.imm (BitVec.ofNat 32 fp32Bytes)),
    .add64 .weightRead (.reg .weightBase) (.reg .rowOffset),
    .add64 .biasRead (.reg .biasBase) (.reg .elementOffset),
    .add64 .outputWrite (.reg .outputBase) (.reg .elementOffset) ]

private def finalState (width : Nat) (initial : State Register) : State Register :=
  let s := initial.write64Next .rowOffset
    (wideProduct (initial.regs .row) (BitVec.ofNat 32 (width * fp32Bytes)))
  let s := s.write64Next .elementOffset
    (wideProduct (s.regs .row) (BitVec.ofNat 32 fp32Bytes))
  let s := s.write64Next .weightRead (s.regs64 .weightBase + s.regs64 .rowOffset)
  let s := s.write64Next .biasRead (s.regs64 .biasBase + s.regs64 .elementOffset)
  s.write64Next .outputWrite (s.regs64 .outputBase + s.regs64 .elementOffset)

/-- Word-level execution facts hold even when pointer addition wraps. -/
structure AddressWords (width : Nat) (initial final : State Register) : Prop where
  pc : final.pc = initial.pc + (addressCalculation width).length
  regs32 : final.regs = initial.regs
  memory : final.globalMemory = initial.globalMemory
  predicates : final.predicates = initial.predicates
  active : final.returned = initial.returned
  inputRead : final.regs64 .inputRead = initial.regs64 .inputRead
  weightBase : final.regs64 .weightBase = initial.regs64 .weightBase
  biasBase : final.regs64 .biasBase = initial.regs64 .biasBase
  outputBase : final.regs64 .outputBase = initial.regs64 .outputBase
  rowOffset : final.regs64 .rowOffset =
    wideProduct (initial.regs .row) (BitVec.ofNat 32 (width * fp32Bytes))
  elementOffset : final.regs64 .elementOffset =
    wideProduct (initial.regs .row) (BitVec.ofNat 32 fp32Bytes)
  weights : final.regs64 .weightRead = initial.regs64 .weightBase +
    wideProduct (initial.regs .row) (BitVec.ofNat 32 (width * fp32Bytes))
  bias : final.regs64 .biasRead = initial.regs64 .biasBase +
    wideProduct (initial.regs .row) (BitVec.ofNat 32 fp32Bytes)
  output : final.regs64 .outputWrite = initial.regs64 .outputBase +
    wideProduct (initial.regs .row) (BitVec.ofNat 32 fp32Bytes)

theorem addressCalculation_exec (program : List (Instruction Register))
    (width : Nat) (ctx : ExecutionContext) (initial : State Register)
    (hactive : initial.returned = false)
    (hcode : CodeAt program initial.pc (addressCalculation width)) :
    ∃ final, run program ctx (addressCalculation width).length initial = some final ∧
      AddressWords width initial final := by
  have h₀ := hcode 0 (by simp [addressCalculation])
  have h₁ := hcode 1 (by simp [addressCalculation])
  have h₂ := hcode 2 (by simp [addressCalculation])
  have h₃ := hcode 3 (by simp [addressCalculation])
  have h₄ := hcode 4 (by simp [addressCalculation])
  simp [addressCalculation] at h₀ h₁ h₂ h₃ h₄
  refine ⟨finalState width initial, ?_, ?_⟩
  · simp [run, step, addressCalculation, h₀, h₁, h₂, h₃, h₄, Instruction.execute,
      Operand.read, Operand64.read, State.write64Next, finalState, hactive, Nat.add_assoc]
  · constructor <;> simp [finalState, State.write64Next, addressCalculation, Nat.add_assoc]

end ReluLayer.PTX

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

/-- Exact modular offsets and pointers, preserving every other state field. -/
def addressesComputed (width : Nat) (initial : State Register) : State Register :=
  let rowOffset := wideProduct (initial.regs .row) (BitVec.ofNat 32 (width * fp32Bytes))
  let elementOffset := wideProduct (initial.regs .row) (BitVec.ofNat 32 fp32Bytes)
  { initial with
    regs64 := fun r => match r with
      | .rowOffset => rowOffset
      | .elementOffset => elementOffset
      | .weightRead => initial.regs64 .weightBase + rowOffset
      | .biasRead => initial.regs64 .biasBase + elementOffset
      | .outputWrite => initial.regs64 .outputBase + elementOffset
      | _ => initial.regs64 r
    pc := initial.pc + (addressCalculation width).length }

theorem addressCalculation_exec (program : List (Instruction Register))
    (width : Nat) (ctx : ExecutionContext) (initial : State Register)
    (hactive : initial.returned = false)
    (hcode : CodeAt program initial.pc (addressCalculation width)) :
    run program ctx (addressCalculation width).length initial =
      some (addressesComputed width initial) := by
  have h₀ := hcode 0 (by simp [addressCalculation])
  have h₁ := hcode 1 (by simp [addressCalculation])
  have h₂ := hcode 2 (by simp [addressCalculation])
  have h₃ := hcode 3 (by simp [addressCalculation])
  have h₄ := hcode 4 (by simp [addressCalculation])
  simp [addressCalculation] at h₀ h₁ h₂ h₃ h₄
  simp [run, step, addressCalculation, h₀, h₁, h₂, h₃, h₄, Instruction.execute,
    Operand.read, Operand64.read, State.write64Next, addressesComputed, hactive, Nat.add_assoc]
  funext r
  cases r <;> rfl

end ReluLayer.PTX

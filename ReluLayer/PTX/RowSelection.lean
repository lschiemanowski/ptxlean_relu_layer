import ReluLayer.PTX.Registers
import ReluLayer.PTX.ProgramComposition
import Mathlib.Data.Nat.Basic

namespace ReluLayer.PTX

/-- One-dimensional launch dimensions, independent of this layer's shapes. -/
structure Launch where
  blocks : Nat
  threads : Nat
  blocks_pos : 0 < blocks
  threads_pos : 0 < threads

def Launch.rows (cfg : Launch) : Nat := cfg.blocks * cfg.threads

variable {cfg : Launch}

structure Thread (cfg : Launch) where
  block : Fin cfg.blocks
  tid : Fin cfg.threads

def Thread.identity (t : Thread cfg) : Identity :=
  ⟨BitVec.ofNat 32 t.tid.val, BitVec.ofNat 32 t.block.val⟩

def Thread.row (t : Thread cfg) : Nat := cfg.threads * t.block.val + t.tid.val

theorem Thread.row_lt (t : Thread cfg) : t.row < cfg.rows := by
  have h₁ := Nat.mul_le_mul_left cfg.threads (Nat.succ_le_of_lt t.block.isLt)
  have h₂ := t.tid.isLt
  simp only [Thread.row, Launch.rows, Nat.mul_succ] at *
  rw [Nat.mul_comm cfg.blocks cfg.threads]
  omega

/-- The owned row as a valid tensor index; the range proof stays out of public statements. -/
def Thread.rowIndex (t : Thread cfg) : Fin cfg.rows := ⟨t.row, t.row_lt⟩

@[simp] theorem Thread.rowIndex_val (t : Thread cfg) : t.rowIndex.val = t.row := rfl

/-- Two identity loads followed by the configured multiply-add. -/
def rowSelection (threads : Nat) : List (Instruction Register) :=
  [ .mov .thread .tidX,
    .mov .block .ctaidX,
    .madLo .row (.reg .block) (.imm (BitVec.ofNat 32 threads)) (.reg .thread) ]

/-- Exact state after the two identity loads and modular row multiply-add. -/
def rowSelected (t : Thread cfg) (initial : State Register) : State
    Register :=
  let s := initial.writeNext .thread t.identity.tidX
  let s := s.writeNext .block t.identity.ctaidX
  s.writeNext .row (s.regs .block * BitVec.ofNat 32 cfg.threads + s.regs .thread)

/-- A fitting launch makes the computed word equal to the mathematical owned row. -/
theorem rowSelected_row (t : Thread cfg) (initial : State Register)
    (hfit : cfg.rows ≤ 2 ^ 32) : ((rowSelected t initial).regs .row).toNat = t.row := by
  simp only [rowSelected, State.writeNext, Thread.identity,
    reduceCtorEq, ↓reduceIte, BitVec.ofNat_mul_ofNat,
    BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
  rw [Nat.mul_comm t.block.val cfg.threads]
  exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le t.row_lt hfit)

/-- Execute the three placed instructions and return their exact state, with arbitrary scratch. -/
theorem rowSelection_correct (program : List (Instruction Register))
    (cfg : Launch) (t : Thread cfg) (initial : State Register)
    (hactive : initial.returned = false) (ctx : ExecutionContext)
    (hid : ctx.id = t.identity) (hcode : CodeAt program initial.pc (rowSelection cfg.threads)) :
    run program ctx (rowSelection cfg.threads).length initial =
      some (rowSelected t initial) := by
  have h₀ := hcode 0 (by simp [rowSelection])
  have h₁ := hcode 1 (by simp [rowSelection])
  have h₂ := hcode 2 (by simp [rowSelection])
  simp [rowSelection] at h₀ h₁ h₂
  simp [run, step, rowSelection, h₀, h₁, h₂, Instruction.execute, Operand.read,
    State.writeNext, rowSelected, hactive, hid, Nat.add_assoc]

end ReluLayer.PTX

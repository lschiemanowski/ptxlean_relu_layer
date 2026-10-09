import ReluLayer.PTX.Machine
import Mathlib.Data.Nat.Basic

namespace ReluLayer.PTX

/-- One-dimensional launch dimensions, independent of this layer's shapes. -/
structure Launch where
  blocks : Nat
  threads : Nat
  blocks_pos : 0 < blocks
  threads_pos : 0 < threads

def Launch.rows (cfg : Launch) : Nat := cfg.blocks * cfg.threads

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

/-- Ordinary register names for the selected fragment. -/
inductive RowReg where
  | thread | block | row
  deriving DecidableEq

/-- The stride is a program parameter, not part of multiply-add semantics. -/
def rowSelection (threads : Nat) : List (Instruction RowReg) :=
  [ .mov .thread .tidX,
    .mov .block .ctaidX,
    .madLo .row (.reg .block) (.imm (BitVec.ofNat 32 threads)) (.reg .thread) ]

private def finalState (cfg : Launch) (t : Thread cfg) (regs : RowReg → Word32) :
    State RowReg :=
  let s : State RowReg := ⟨regs, 0⟩
  let s := s.write .thread t.identity.tidX
  let s := s.write .block t.identity.ctaidX
  s.write .row (s.regs .block * BitVec.ofNat 32 cfg.threads + s.regs .thread)

private theorem run_eq (cfg : Launch) (t : Thread cfg) (regs : RowReg → Word32) :
    run (rowSelection cfg.threads) t.identity 3 ⟨regs, 0⟩ =
      some (finalState cfg t regs) := by
  simp [run, step, rowSelection, Instruction.execute, Operand.read, State.write, finalState]

/-- Executing the fragment succeeds, computes the row without wrap, and stays in range. -/
theorem rowSelection_correct (cfg : Launch) (hfit : cfg.rows ≤ 2 ^ 32)
    (t : Thread cfg) (regs : RowReg → Word32) :
    ∃ s, run (rowSelection cfg.threads) t.identity 3 ⟨regs, 0⟩ = some s ∧
      s.pc = 3 ∧ (s.regs .row).toNat = t.row ∧ (s.regs .row).toNat < cfg.rows := by
  have hrow : ((finalState cfg t regs).regs .row).toNat = t.row := by
    simp only [finalState, State.write, Thread.identity,
      reduceCtorEq, ↓reduceIte, BitVec.ofNat_mul_ofNat,
      BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
    rw [Nat.mul_comm t.block.val cfg.threads]
    exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le t.row_lt hfit)
  exact ⟨finalState cfg t regs, run_eq cfg t regs, rfl, hrow, by rw [hrow]; exact t.row_lt⟩

/-- The sole choice of launch counts for the current concrete kernel. -/
def layerLaunch : Launch := ⟨12, 256, by decide, by decide⟩

theorem layerLaunch_rows : layerLaunch.rows = 3072 := rfl

theorem layer_rowSelection_correct (t : Thread layerLaunch) (regs : RowReg → Word32) :
    ∃ s, run (rowSelection layerLaunch.threads) t.identity 3 ⟨regs, 0⟩ = some s ∧
      s.pc = 3 ∧ (s.regs .row).toNat = t.row ∧ (s.regs .row).toNat < 3072 :=
  rowSelection_correct layerLaunch (by decide) t regs

end ReluLayer.PTX

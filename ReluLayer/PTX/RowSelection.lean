import ReluLayer.PTX.Registers
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

private def finalState (cfg : Launch) (t : Thread cfg) (initial : State Register) : State
    Register :=
  let s := initial.writeNext .thread t.identity.tidX
  let s := s.writeNext .block t.identity.ctaidX
  s.writeNext .row (s.regs .block * BitVec.ofNat 32 cfg.threads + s.regs .thread)

/-- The selected row and the state preserved by the three-instruction block. -/
structure RowResult (cfg : Launch) (t : Thread cfg) (initial final : State Register) : Prop where
  pc : final.pc = (rowSelection cfg.threads).length
  row : (final.regs .row).toNat = t.row
  regs64 : final.regs64 = initial.regs64
  memory : final.globalMemory = initial.globalMemory
  predicates : final.predicates = initial.predicates
  active : final.returned = initial.returned
  other32 : ∀ r, r ≠ .thread → r ≠ .block → r ≠ .row → final.regs r = initial.regs r

/-- Arbitrary scratch values are allowed; a fitting launch computes the owned row without wrap. -/
theorem rowSelection_correct (cfg : Launch) (hfit : cfg.rows ≤ 2 ^ 32)
    (t : Thread cfg) (initial : State Register) (hpc : initial.pc = 0)
    (hactive : initial.returned = false) (ctx : ExecutionContext)
    (hid : ctx.id = t.identity) :
    ∃ final, run (rowSelection cfg.threads) ctx (rowSelection cfg.threads).length
        initial = some final ∧ RowResult cfg t initial final := by
  refine ⟨finalState cfg t initial, ?_, ?_⟩
  · simp [run, step, rowSelection, Instruction.execute, Operand.read, State.writeNext,
      finalState, hpc, hactive, hid]
  · constructor
    · simp [finalState, State.writeNext, rowSelection, hpc]
    · simp only [finalState, State.writeNext, Thread.identity,
        reduceCtorEq, ↓reduceIte, BitVec.ofNat_mul_ofNat,
        BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
      rw [Nat.mul_comm t.block.val cfg.threads]
      exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le t.row_lt hfit)
    · rfl
    · rfl
    · rfl
    · rfl
    · intro r ht hb hr
      simp [finalState, State.writeNext, ht, hb, hr]

end ReluLayer.PTX

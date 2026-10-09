import Mathlib.Data.Nat.Basic

namespace ReluLayer.PTX

/-- The instruction type fixes this width; launch dimensions do not. -/
abbrev Word32 := BitVec 32

/-- Immutable special-register values supplied to one thread. -/
structure Identity where
  tidX : Word32
  ctaidX : Word32

/-- Private mutable state; the PC indexes the modeled instruction fragment. -/
structure State (Reg : Type) where
  regs : Reg → Word32
  pc : Nat

inductive Operand (Reg : Type) where
  | reg (name : Reg)
  | imm (value : Word32)
  | tidX
  | ctaidX

def Operand.read (id : Identity) (s : State Reg) : Operand Reg → Word32
  | .reg r => s.regs r
  | .imm v => v
  | .tidX => id.tidX
  | .ctaidX => id.ctaidX

/-- Only the two unpredicated instruction forms needed for row selection. -/
inductive Instruction (Reg : Type) where
  | mov (dst : Reg) (src : Operand Reg)
  | madLo (dst : Reg) (a b c : Operand Reg)

def State.write [DecidableEq Reg] (s : State Reg) (dst : Reg) (v : Word32) : State Reg :=
  { regs := fun r => if r = dst then v else s.regs r, pc := s.pc + 1 }

/-- All operands are read before the destination is updated. BitVec arithmetic wraps. -/
def Instruction.execute [DecidableEq Reg] (id : Identity) (s : State Reg) :
    Instruction Reg → State Reg
  | .mov dst src => s.write dst (src.read id s)
  | .madLo dst a b c => s.write dst (a.read id s * b.read id s + c.read id s)

/-- Fetch at the current PC; falling outside the fragment is not an instruction. -/
def step [DecidableEq Reg] (program : List (Instruction Reg)) (id : Identity)
    (s : State Reg) : Option (State Reg) :=
  (program[s.pc]?).map (fun inst => inst.execute id s)

/-- Execute exactly `fuel` instructions, reporting a failed fetch as `none`. -/
def run [DecidableEq Reg] (program : List (Instruction Reg)) (id : Identity) :
    Nat → State Reg → Option (State Reg)
  | 0, s => some s
  | fuel + 1, s => (step program id s).bind (run program id fuel)

end ReluLayer.PTX

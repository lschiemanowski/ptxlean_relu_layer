import ReluLayer.PTX.FP32

namespace ReluLayer.PTX

/-- The full unsigned product of two 32-bit words fits in 64 bits. -/
def wideProduct (a b : Word32) : Word64 :=
  BitVec.ofNat 64 (a.toNat * b.toNat)

@[simp] theorem wideProduct_toNat (a b : Word32) :
    (wideProduct a b).toNat = a.toNat * b.toNat := by
  exact Nat.mod_eq_of_lt (BitVec.toNat_mul_toNat_lt (x := a) (y := b))

/-- Immutable special-register values supplied to one thread. -/
structure Identity where
  tidX : Word32
  ctaidX : Word32

/-- One thread's private registers/PC, together with the shared global-memory snapshot. -/
structure State (Reg : Type) where
  regs : Reg → Word32
  pc : Nat
  regs64 : Reg → Word64
  globalMemory : GlobalMemory
  predicates : Reg → Bool := fun _ => false
  returned : Bool := false

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

inductive Operand64 (Reg : Type) where
  | reg (name : Reg)
  | imm (value : Word64)

def Operand64.read (s : State Reg) : Operand64 Reg → Word64
  | .reg r => s.regs64 r
  | .imm v => v

/-- Selected instruction forms needed by the scalar dense-ReLU kernel. -/
inductive Instruction (Reg : Type) where
  | mov (dst : Reg) (src : Operand Reg)
  | madLo (dst : Reg) (a b c : Operand Reg)
  | mulWide (dst : Reg) (a b : Operand Reg)
  | add64 (dst : Reg) (a b : Operand64 Reg)
  | ldParam64 (dst : Reg) (symbol : String)
  | cvtaGlobal64 (dst : Reg) (src : Operand64 Reg)
  | movF32 (dst : Reg) (value : Word32)
  | loadF32 (dst : Reg) (address : Operand64 Reg)
  | fmaRN (dst : Reg) (a b c : Operand Reg)
  | add32 (dst : Reg) (a b : Operand Reg)
  | setpLt (dst : Reg) (a b : Operand Reg)
  | branchIf (predicate : Reg) (target : Nat)
  | addF32RN (dst : Reg) (a b : Operand Reg)
  | maxPositiveZero (dst : Reg) (src : Operand Reg)
  | storeF32 (address : Operand64 Reg) (src : Operand Reg)
  | ret

def State.write [DecidableEq Reg] (s : State Reg) (dst : Reg) (v : Word32) : State Reg :=
  { s with regs := fun r => if r = dst then v else s.regs r, pc := s.pc + 1 }

def State.write64 [DecidableEq Reg] (s : State Reg) (dst : Reg) (v : Word64) : State Reg :=
  { s with regs64 := fun r => if r = dst then v else s.regs64 r, pc := s.pc + 1 }

/-- All operands are read before the destination is updated. BitVec arithmetic wraps. -/
def Instruction.execute [DecidableEq Reg] (id : Identity) (s : State Reg)
    (inst : Instruction Reg) (env : Environment := Environment.empty)
    (ops : Option FP32.Arithmetic := none) : Option (State Reg) :=
  match inst with
  | .mov dst src => some (s.write dst (src.read id s))
  | .madLo dst a b c => some (s.write dst (a.read id s * b.read id s + c.read id s))
  | .mulWide dst a b => some (s.write64 dst (wideProduct (a.read id s) (b.read id s)))
  | .add64 dst a b => some (s.write64 dst (a.read s + b.read s))
  | .ldParam64 dst symbol => (env.params symbol).map (s.write64 dst)
  | .cvtaGlobal64 dst src => (env.space.toGlobal (src.read s)).map (s.write64 dst)
  | .movF32 dst value => some (s.write dst value)
  | .loadF32 dst address => (s.globalMemory (address.read s)).map (s.write dst)
  | .fmaRN dst a b c => ops.map (fun op =>
      s.write dst (op.fmaRN (a.read id s) (b.read id s) (c.read id s)))
  | .add32 dst a b => some (s.write dst (a.read id s + b.read id s))
  | .setpLt dst a b => some { s with
      predicates := fun r => if r = dst then decide ((a.read id s).toNat < (b.read id s).toNat)
        else s.predicates r
      pc := s.pc + 1 }
  | .branchIf predicate target => some { s with
      pc := if s.predicates predicate then target else s.pc + 1 }
  | .addF32RN dst a b => ops.map (fun op =>
      s.write dst (op.addRN (a.read id s) (b.read id s)))
  | .maxPositiveZero dst src => some (s.write dst (FP32.maxPositiveZero (src.read id s)))
  | .storeF32 address src => some { s with
      globalMemory := fun a => if a = address.read s then some (src.read id s) else s.globalMemory a
      pc := s.pc + 1 }
  | .ret => some { s with returned := true, pc := s.pc + 1 }

/-- Active threads fetch at the current PC; returned threads have no next step. -/
def step [DecidableEq Reg] (program : List (Instruction Reg)) (id : Identity)
    (s : State Reg) (env : Environment := Environment.empty)
    (ops : Option FP32.Arithmetic := none) : Option (State Reg) :=
  if s.returned then none else
    (program[s.pc]?).bind (fun inst => inst.execute id s env ops)

/-- Execute exactly `fuel` instructions; failed fetches and excluded instruction uses yield `none`. -/
def run [DecidableEq Reg] (program : List (Instruction Reg)) (id : Identity)
    (fuel : Nat) (s : State Reg) (env : Environment := Environment.empty)
    (ops : Option FP32.Arithmetic := none) : Option (State Reg) :=
  match fuel with
  | 0 => some s
  | fuel + 1 => (step program id s env ops).bind (fun next => run program id fuel next env ops)

theorem run_add [DecidableEq Reg] (program : List (Instruction Reg)) (id : Identity)
    (a b : Nat) (s : State Reg) (env : Environment) (ops : Option FP32.Arithmetic) :
    run program id (a + b) s env ops =
      (run program id a s env ops).bind (fun next => run program id b next env ops) := by
  induction a generalizing s with
  | zero => simp [run]
  | succ a ih => simp [Nat.succ_add, run, ih, Option.bind_assoc]

private theorem step_append_of_step [DecidableEq Reg]
    (program suffix : List (Instruction Reg)) (id : Identity) (s next : State Reg)
    (env : Environment) (ops : Option FP32.Arithmetic)
    (h : step program id s env ops = some next) :
    step (program ++ suffix) id s env ops = some next := by
  have hpc : s.pc < program.length := by
    by_contra hn
    have hf := List.getElem?_eq_none (l := program) (by omega : program.length ≤ s.pc)
    simp [step, hf] at h
  simpa only [step, List.getElem?_append_left hpc] using h

/-- Appending instructions cannot change an already successful prefix execution. -/
theorem run_append_of_run [DecidableEq Reg]
    (program suffix : List (Instruction Reg)) (id : Identity) (fuel : Nat)
    (s final : State Reg) (env : Environment) (ops : Option FP32.Arithmetic)
    (h : run program id fuel s env ops = some final) :
    run (program ++ suffix) id fuel s env ops = some final := by
  induction fuel generalizing s with
  | zero => simpa [run] using h
  | succ fuel ih =>
    obtain ⟨next, hs, ht⟩ := Option.bind_eq_some_iff.mp h
    have he := step_append_of_step program suffix id s next env ops hs
    simpa only [run, he, Option.bind_some] using ih next ht

end ReluLayer.PTX

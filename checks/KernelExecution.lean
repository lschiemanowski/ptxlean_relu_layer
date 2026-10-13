import ReluLayer.PTX.Layer
import ReluLayer.PTX.KernelExecution

open ReluLayer.PTX

-- Integer-only arithmetic isolates instruction behavior from numerical semantics.
private def testArithmetic : FP32.Arithmetic :=
  ⟨fun w x a => w * x + a, fun a b => a + b⟩

namespace RowChecks

-- Aliasing uses the old source value; multiplication and addition both wrap.
private def aliasState : State Nat := ⟨fun r => if r = 0 then 4294967295 else 17, 7, fun _ => 0, fun _ => none, fun _ => false, false⟩
private def aliasResult : State Nat :=
  (Instruction.madLo 0 (.reg 0) (.imm 2) (.imm 3)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ aliasState |>.getD aliasState

example : (aliasResult.regs 0).toNat = 1 := by decide
example : (aliasResult.regs 1).toNat = 17 := by decide
example : aliasResult.pc = 8 := by decide
example : ((Instruction.mov 0 (.reg 0)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ aliasState).map
    (fun s => s.regs 0) = some (aliasState.regs 0) := by decide

-- Fetching outside the fragment fails; exhaustion of fuel alone succeeds.
example : step (rowSelection 5) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ ⟨fun _ => 0, 3, fun _ => 0, fun _ => none, fun _ => false, false⟩ = none := by decide
example : (run (rowSelection 5) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 0 ⟨fun _ => 0, 3, fun _ => 0, fun _ => none, fun _ => false, false⟩).isSome = true := by decide

-- Zero fuel alone does not imply that an active thread has returned.
example : ¬ Returns [] ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ aliasState aliasState := by
  intro h
  exact Bool.noConfusion h.2

-- The state contract specializes to a different launch with arbitrary scratch.
example (t : Thread ⟨7, 5, by decide, by decide⟩) (initial : State Register)
    (hpc : initial.pc = 0) (hactive : initial.returned = false) (env : Environment) :
    ∃ final, run (rowSelection 5) ⟨t.identity, env, testArithmetic⟩ 3 initial = some final ∧
      RowResult ⟨7, 5, by decide, by decide⟩ t initial final :=
  rowSelection_correct (rowSelection 5) _ (by decide) t initial hactive
    ⟨t.identity, env, testArithmetic⟩ rfl (by
      rw [hpc]
      simpa using codeAt_append [] (rowSelection 5) [])

-- The same contract works inside arbitrary enclosing code at a nonzero PC.
example (pre post : List (Instruction Register)) (t : Thread ⟨7, 5, by decide, by decide⟩)
    (initial : State Register) (hpc : initial.pc = pre.length)
    (hactive : initial.returned = false) (env : Environment) :
    ∃ final, run (pre ++ rowSelection 5 ++ post) ⟨t.identity, env, testArithmetic⟩ 3 initial =
        some final ∧ RowResult ⟨7, 5, by decide, by decide⟩ t initial final :=
  rowSelection_correct _ _ (by decide) t initial hactive ⟨t.identity, env, testArithmetic⟩ rfl
    (by rw [hpc]; exact codeAt_append pre (rowSelection 5) post)

-- Neither a return in the prefix nor a row overwrite in the suffix is executed.
private def offsetThread : Thread ⟨7, 5, by decide, by decide⟩ := ⟨2, 3⟩
private def offsetInitial : State Register :=
  ⟨fun _ => 37, 1, fun _ => 42, fun _ => some 9, fun _ => true, false⟩
private def enclosingProgram : List (Instruction Register) :=
  [.ret] ++ rowSelection 5 ++ [.mov .row (.imm 99)]

example : (run enclosingProgram ⟨offsetThread.identity, Environment.empty, testArithmetic⟩ 3
    offsetInitial).map (fun s => (s.pc, (s.regs .row).toNat)) = some (4, 13) := by decide

#print axioms Thread.row_lt
#print axioms rowSelection_correct
#print axioms codeAt_append
#print axioms layerLaunch_rows

end RowChecks

namespace BaseChecks

private def args : Argument → Word64
  | .weights => 1024
  | .input => 1040
  | .bias => 1064
  | .output => 1080

private def env : Environment :=
  { params := fun symbol =>
      if symbol = "W_ptr" then some (args .weights)
      else if symbol = "x_ptr" then some (args .input)
      else if symbol = "b_ptr" then some (args .bias)
      else if symbol = "y_ptr" then some (args .output)
      else none
    toGlobal := fun a => if 1000 ≤ a.toNat && a.toNat < 1200 &&
      !(1100 ≤ a.toNat && a.toNat < 1116) then some (BitVec.ofNat 64 (a.toNat - 1000)) else none }

private def buffers : Buffers := fun arg => BitVec.ofNat 64 ((args arg).toNat - 1000)

private def initial : State Register :=
  ⟨fun _ => 77, 0, fun _ => 999, fun address => if address = 24 then some 42 else none, fun _ => false, false⟩

-- The supplied conversion performs nonidentity address translation.
example : ((run baseLoading ⟨⟨0, 0⟩, env, testArithmetic⟩ 8 initial).map (fun s =>
    ((s.regs64 .weightBase).toNat, (s.regs64 .inputRead).toNat,
      (s.regs64 .biasBase).toNat, (s.regs64 .outputBase).toNat))) =
    some (24, 40, 64, 80) := by decide

example : ∃ final, run baseLoading ⟨⟨0, 0⟩, env, testArithmetic⟩ 8 initial = some final ∧
    BaseResult buffers initial final :=
  baseLoading_correct baseLoading ⟨⟨0, 0⟩, env, testArithmetic⟩ args buffers initial (by decide)
    (fun arg => by cases arg <;> decide) (fun arg => by cases arg <;> decide)
    (by simpa [initial] using codeAt_append [] baseLoading [])

-- The supplied converter has successful and excluded inputs; parameter slots may be absent.
example : env.toGlobal 1000 = some 0 := by decide
example : env.toGlobal 1199 = some 199 := by decide
example : env.toGlobal 1200 = none := by decide
example : env.toGlobal 999 = none := by decide
example : env.toGlobal 1100 = none := by decide
example : env.toGlobal 1115 = none := by decide
example : env.toGlobal 1116 = some 116 := by decide
example : (Instruction.ldParam64 Register.weightBase "missing").execute ⟨⟨0, 0⟩, env, testArithmetic⟩ initial = none := by decide
example : run baseLoading ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 8 initial = none := by decide

-- Conversion reads the old source word when destination and source coincide.
private def aliasInitial : State Register := { initial with regs64 := fun _ => 1024 }
example : ((Instruction.cvtaGlobal64 Register.weightBase (.reg .weightBase)).execute ⟨⟨0, 0⟩, env, testArithmetic⟩ aliasInitial).map (fun s => (s.regs64 .weightBase).toNat) = some 24 := by decide
example : ((run baseLoading ⟨⟨0, 0⟩, env, testArithmetic⟩ 8 initial).map (fun s =>
    ((s.regs .row).toNat, (s.regs64 .weightRead).toNat, s.globalMemory 24, s.pc))) =
    some (77, 999, some 42, 8) := by decide

-- Base loading has the same contract after any skipped prefix and before any suffix.
example (pre post : List (Instruction Register)) :
    ∃ final, run (pre ++ baseLoading ++ post) ⟨⟨0, 0⟩, env, testArithmetic⟩ baseLoading.length
        { initial with pc := pre.length } = some final ∧
      BaseResult buffers { initial with pc := pre.length } final :=
  baseLoading_correct _ ⟨⟨0, 0⟩, env, testArithmetic⟩ args buffers
    { initial with pc := pre.length } (by rfl)
    (fun arg => by cases arg <;> decide) (fun arg => by cases arg <;> decide)
    (codeAt_append pre baseLoading post)

-- A prefix return is skipped, and the suffix cannot overwrite a converted base.
private def offsetProgram : List (Instruction Register) :=
  [.ret] ++ baseLoading ++ [.add64 .weightBase (.imm 0) (.imm 0)]
example : (run offsetProgram ⟨⟨0, 0⟩, env, testArithmetic⟩ baseLoading.length
    { initial with pc := 1 }).map (fun s => (s.pc, (s.regs64 .weightBase).toNat)) =
    some (9, 24) := by decide

#print axioms baseLoading_correct

end BaseChecks

namespace AddressChecks

-- Wide multiplication keeps the high half of the unsigned product.
example : (wideProduct 4294967295 4294967295).toNat = 18446744065119617025 := by decide

private def wideInitial : State Nat :=
  ⟨fun _ => 42, 9, fun r => if r = 0 then 18446744073709551615 else 17, fun _ => none, fun _ => false, false⟩
private def aliasSum : State Nat :=
  (Instruction.add64 0 (.reg 0) (.imm 2)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ wideInitial |>.getD wideInitial

example : (aliasSum.regs64 0).toNat = 1 := by decide
example : (aliasSum.regs64 1).toNat = 17 := by decide
example : (aliasSum.regs 0).toNat = 42 := by decide
example : aliasSum.pc = 10 := by decide
example : ((Instruction.mulWide 0 (.reg 0) (.imm 4294967295)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ wideInitial).map (fun s => s.regs64 0) =
    some (wideProduct 42 4294967295) := by decide

-- A small alternative layout catches any hidden dependence on width 1024.
private def tinyInitial : State Register :=
  ⟨fun _ => 1, 0, fun r => match r with
    | .weightBase => 100
    | .biasBase => 200
    | .outputBase => 300
    | _ => 999, fun _ => none, fun _ => false, false⟩

example : ((run (addressCalculation 3) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5 tinyInitial).map
    (fun final => ((final.regs64 .weightRead).toNat,
      (final.regs64 .biasRead).toNat, (final.regs64 .outputWrite).toNat))) =
    some (112, 204, 304) := by decide

-- Address calculation just below the 64-bit wraparound boundary.
private def boundaryInitial : State Register :=
  ⟨fun _ => 1, 0, fun _ => 18446744073709551607, fun _ => none, fun _ => false, false⟩
example : (run (addressCalculation 1) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5
    boundaryInitial).map (fun final => ((final.regs64 .weightRead).toNat,
      (final.regs64 .biasRead).toNat, (final.regs64 .outputWrite).toNat)) =
    some (18446744073709551611, 18446744073709551611, 18446744073709551611) := by decide

-- Without buffer-end bounds, the instruction semantics still permits wraparound.
private def wrappingInitial : State Register :=
  ⟨fun _ => 1, 0, fun _ => 18446744073709551615, fun _ => none, fun _ => false, false⟩
example : ((run (addressCalculation 1) ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 5 wrappingInitial).map
    (fun final => (final.regs64 .weightRead).toNat)) = some 3 := by decide

-- The modular address contract also applies inside arbitrary enclosing code.
example (pre post : List (Instruction Register)) (width : Nat) (ctx : ExecutionContext)
    (initial : State Register) (hpc : initial.pc = pre.length)
    (hactive : initial.returned = false) :
    ∃ final, run (pre ++ addressCalculation width ++ post) ctx
        (addressCalculation width).length initial = some final ∧ AddressWords width initial final :=
  addressCalculation_exec _ width ctx initial hactive (by
    rw [hpc]
    exact codeAt_append pre (addressCalculation width) post)

-- Prefix instructions are skipped, and the suffix cannot overwrite the computed pointer.
private def offsetProgram : List (Instruction Register) :=
  [.ret, .mov .row (.imm 7)] ++ addressCalculation 3 ++
    [.add64 .weightRead (.imm 0) (.imm 0)]

example : (run offsetProgram ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩
    (addressCalculation 3).length { tinyInitial with pc := 2 }).map
    (fun final => (final.pc, (final.regs64 .weightRead).toNat)) = some (7, 112) := by decide

#print axioms wideProduct_toNat
#print axioms rowSelection_correct
#print axioms addressCalculation_exec

end AddressChecks

namespace ComputationChecks

private def layout : Layout := ⟨2, by decide, by decide⟩
private def buffers : Buffers
  | .weights => 0
  | .input => 16
  | .bias => 24
  | .output => 32
private def data : TensorData layout 2 :=
  ⟨fun i j => BitVec.ofNat 32 (i.val * 2 + j.val + 1),
    fun j => BitVec.ofNat 32 (j.val + 5), fun i => BitVec.ofNat 32 (i.val + 7)⟩
private def memory : GlobalMemory := fun a =>
  if a = 0 then some 1 else if a = 4 then some 2
  else if a = 8 then some 3 else if a = 12 then some 4
  else if a = 16 then some 5 else if a = 20 then some 6
  else if a = 24 then some 7 else if a = 28 then some 8
  else if a = 32 then some 0x3f800000 else none
private def initial : State Register :=
  { regs := fun r => if r = .row then 1 else 99
    pc := 0
    regs64 := fun r => match r with
      | .weightRead => 8 | .inputRead => 16 | .biasRead => 28 | .outputWrite => 36 | _ => 999
    globalMemory := memory
    predicates := fun _ => true }
private def ops : FP32.Arithmetic := ⟨fun w x acc => acc * 10 + w + x, fun a b => a + b⟩

private theorem entry : BodyEntry layout 2 buffers data 1 initial := by
  constructor
  · constructor <;> decide
  all_goals decide

-- Input contracts, permitted read-only aliases and the memory frame.
private def contractMemory : GlobalMemory := fun a =>
  if a = 0 then some 1 else if a = 4 then some 2
  else if a = 8 then some 3 else if a = 12 then some 4
  else if a = 16 then some 5 else if a = 20 then some 6
  else if a = 24 then some 7 else if a = 28 then some 8
  else if a = 36 then some 0x3f800000 else none

private theorem inputsStored : InputsStored layout 2 buffers data contractMemory := by
  constructor <;> decide

-- No output initialization is required. Missing/wrong input words are rejected.
example : contractMemory 32 = none := by decide
example : ¬ InputsStored layout 2 buffers data (fun _ => none) := by
  intro h
  have hh := h.weights (0 : Fin 2) ⟨0, by decide⟩
  contradiction
example : ¬ InputsStored layout 2 buffers data (fun a => if a = 0 then some 9 else contractMemory a) := by
  intro h
  have hh := h.weights (0 : Fin 2) ⟨0, by decide⟩
  exact (by decide : (some (9 : Word32)) ≠ some 1) hh

-- Read-only inputs can share addresses when their specified values agree.
private def sharedBuffers : Buffers := fun arg => if arg = .input then 0 else buffers arg
private def sharedData : TensorData layout 2 :=
  { data with input := fun j => BitVec.ofNat 32 (j.val + 1) }
example : InputsStored layout 2 sharedBuffers sharedData contractMemory := by constructor <;> decide

private def contractInitial : State Register :=
  { regs := fun r => if r = .row then 0 else 77
    pc := 0
    regs64 := fun r => match r with
      | .weightRead => 0
      | .inputRead => 16
      | .biasRead => 24
      | .outputWrite => 32
      | _ => 999
    globalMemory := contractMemory }
example : BodyEntry layout 2 buffers data 0 contractInitial :=
  ⟨inputsStored, rfl, by decide, by decide, by decide, by decide⟩

-- Artificial operations expose order/arity; they are not rounded FP32 arithmetic.
example : prefixAccumulator ops data 0 0 = 0 := by decide
example : prefixAccumulator ops data 0 1 = 6 := by decide
example : prefixAccumulator ops data 0 2 = 68 := by decide
example : prefixAccumulator ops data 0 3 = 68 := by decide
example : rowResult ops data 0 = 75 := by decide
example : rowResult ops data 1 = 98 := by decide

private def written : GlobalMemory := fun a => if a = 32 then some 75 else contractMemory a
example : OutputStored ops data buffers 0 contractMemory written := by
  constructor
  · decide
  · intro address h
    change address ≠ (32 : Word64) at h
    exact ite_eq_right h
example : written 0 = contractMemory 0 ∧ written 20 = contractMemory 20 ∧ written 36 = some 0x3f800000 := by decide

#print axioms prefixAccumulator_zero

-- Loop execution and its invariant, including a nonzero entry.
-- Arbitrary scratch is overwritten; the first branch is taken, the last is not.
example : ((run (dotLoop 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 10 initial).map
    (fun s => (s.pc, (s.regs .column).toNat, (s.regs .accumulator).toNat,
      (s.regs64 .weightRead).toNat, (s.regs64 .inputRead).toNat, s.predicates .more))) =
    some (2, 1, 8, 12, 20, true) := by decide
example : ((run (dotLoop 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 18 initial).map
    (fun s => (s.pc, (s.regs .column).toNat, (s.regs .accumulator).toNat,
      (s.regs64 .weightRead).toNat, (s.regs64 .inputRead).toNat, s.predicates .more))) =
    some (10, 2, 90, 16, 24, false) := by decide
example : ((run (dotLoop 1 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 10 initial).map
    (fun s => (s.pc, (s.regs .accumulator).toNat, s.predicates .more))) =
    some (10, 8, false) := by decide
example : ((run (dotLoop 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 18 initial).map
    (fun s => ((s.regs .row).toNat, (s.regs64 .biasRead).toNat,
      (s.regs64 .outputWrite).toNat, s.globalMemory 8, s.predicates .row))) =
    some (1, 28, 36, some 3, true) := by decide

-- Unsigned comparison, finite-width addition, unavailable loads/operations, and FMA aliasing.
example : ((Instruction.setpLt Register.more (.imm 0x80000000) (.imm 1)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ initial).map (fun s => s.predicates .more) = some false := by decide
example : ((Instruction.add32 Register.column (.imm 0xffffffff) (.imm 1)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ initial).map (fun s => s.regs .column) = some 0 := by decide
example : (Instruction.loadF32 Register.weightValue (.imm 100)).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ initial = none := by decide
example : ((Instruction.fmaRN Register.accumulator (.imm 1) (.imm 2) (.reg .accumulator)).execute ⟨⟨0, 0⟩, Environment.empty, ops⟩ initial).map (fun s => s.regs .accumulator) = some 993 := by decide
example : ((Instruction.branchIf Register.more 7).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ initial).map (fun s => s.pc) = some 7 := by decide
example : ((Instruction.branchIf Register.more 7).execute ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ { initial with pc := 4, predicates := fun _ => false }).map (fun s => s.pc) = some 5 := by decide

-- The full-width invariant includes exit PC, counter, accumulator and memory facts.
example : ∃ final, run (dotLoop 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 18 initial = some final ∧
    LoopInvariant ops data buffers 1 initial 2 final :=
  dotLoop_correct (dotLoop 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ data buffers 1 initial entry
    (by simpa [initial, layout] using codeAt_append [] (dotLoop 2 0) [])

-- A taken branch must return to the body after a skipped prefix, not to PC 2.
private def offsetLoop : List (Instruction Register) :=
  [.ret] ++ dotLoop 2 1 ++ [.mov .row (.imm 99)]
example : (run offsetLoop ⟨⟨0, 0⟩, Environment.empty, ops⟩ 10
    { initial with pc := 1 }).map (fun s => (s.pc, (s.regs .accumulator).toNat)) =
    some (3, 8) := by decide
example : (run offsetLoop ⟨⟨0, 0⟩, Environment.empty, ops⟩ 18
    { initial with pc := 1 }).map (fun s => (s.pc, (s.regs .accumulator).toNat)) =
    some (11, 90) := by decide

#print axioms run_add
#print axioms prefixAccumulator_succ
#print axioms dotLoop_iteration
#print axioms dotLoop_prefix
#print axioms dotLoop_correct

-- Bias, maximum, store and return; the output is initially absent.
private def suffixState : State Register :=
  { regs := fun r => match r with
      | .row => 1 | .column => 2 | .accumulator => 90
      | .biasValue => 0x7fc00000 | .result => 0xdeadbeef | _ => 99
    pc := 0
    regs64 := fun r => if r = .biasRead then 28 else if r = .outputWrite then 36 else 999
    globalMemory := fun a => if a = 28 then some 8 else if a = 32 then some 0x3f800000 else none
    predicates := fun _ => true }
private theorem suffixStateEntry : SuffixEntry ops data buffers 1 suffixState := by constructor <;> decide

private def executedSuffix : Option (State Register) := run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 suffixState
example : executedSuffix.map (fun s => (s.pc, s.returned)) = some (5, true) := by decide
example : executedSuffix.map (fun s => ((s.regs .biasValue).toNat, (s.regs .accumulator).toNat,
    (s.regs .result).toNat)) = some (8, 98, 98) := by decide
example : executedSuffix.bind (fun s => s.globalMemory 36) = some 98 := by decide
example : suffixState.globalMemory 36 = none := by decide
example : executedSuffix.map (fun s => ((s.regs .row).toNat, (s.regs .column).toNat)) = some (1, 2) := by decide
example : executedSuffix.map (fun s => ((s.regs64 .biasRead).toNat, (s.regs64 .outputWrite).toNat)) =
    some (28, 36) := by decide
example : executedSuffix.bind (fun s => s.globalMemory 28) = some 8 := by decide
example : executedSuffix.bind (fun s => s.globalMemory 32) = some 0x3f800000 := by decide
example : executedSuffix.map (fun s => s.predicates .more) = some true := by decide
example : ∃ final, run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 suffixState = some final ∧
    SuffixResult ops data buffers 1 suffixState final :=
  reluSuffix_correct reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ data buffers 1 suffixState suffixStateEntry
    (by simpa [suffixState] using codeAt_append [] reluSuffix [])

-- Feed chosen preactivation patterns through the actual max/store/return sequence.
private def preactivationOps (v : Word32) : FP32.Arithmetic := ⟨fun _ _ a => a, fun _ _ => v⟩
private def stored (v : Word32) : Option Word32 :=
  (run reluSuffix ⟨⟨0, 0⟩, Environment.empty, (preactivationOps v)⟩ 5 suffixState).bind
    (fun s => s.globalMemory 36)
example : stored 0xbf800000 = some 0 := by decide
example : stored 0x80000000 = some 0 := by decide
example : stored 0x7fc00001 = some 0 := by decide
example : stored 0x7f800001 = some 0 := by decide
example : stored 0x7f800000 = some 0x7f800000 := by decide
example : stored 1 = some 1 := by decide

-- Addition reads the old destination; a missing bias excludes execution.
example : ((Instruction.addF32RN Register.accumulator (.reg .accumulator) (.imm 8)).execute ⟨⟨0, 0⟩, Environment.empty, ops⟩ suffixState).map (fun s => s.regs .accumulator) = some 98 := by decide
example : run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 { suffixState with globalMemory := fun _ => none } = none := by decide
example : run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 { suffixState with returned := true } = none := by decide

-- ret stops execution even if a valid store instruction follows it.
private def afterRet : List (Instruction Register) := [.ret, .storeF32 (.imm 36) (.imm 123)]
example : ((run afterRet ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 1 suffixState).map (fun s => (s.returned, s.globalMemory 36))) =
    some (true, none) := by decide
example : run afterRet ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 2 suffixState = none := by decide

#print axioms reluSuffix_correct
#print axioms step_after_return

-- Loop/suffix composition, stale row scratch, and arbitrary enclosing code.
private def executed : Option (State Register) :=
  run (bodyProgram 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ (bodySteps 2) initial

-- The appended program crosses the loop/suffix boundary with no PC reset.
example : ((run (bodyProgram 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 18 initial).map
    (fun s => (s.pc, s.returned, (s.regs .accumulator).toNat))) = some (10, false, 90) := by decide
example : executed.map (fun s => (s.pc, s.returned)) = some (15, true) := by decide
example : executed.map (fun s => ((s.regs .column).toNat, (s.regs .result).toNat)) = some (2, 98) := by decide
example : executed.bind (fun s => s.globalMemory 36) = some 98 := by decide
example : executed.bind (fun s => s.globalMemory 28) = some 8 := by decide
example : executed.bind (fun s => s.globalMemory 32) = some 0x3f800000 := by decide
example : executed.map (fun s => ((s.regs .row).toNat, (s.regs64 .biasRead).toNat,
    (s.regs64 .outputWrite).toNat)) = some (1, 28, 36) := by decide
example : ∃ final, run (bodyProgram 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ (bodySteps 2) initial = some final ∧
    BodyResult ops data buffers 1 initial final :=
  body_correct (bodyProgram 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ data buffers 1 initial entry
    (by simpa [initial, layout] using codeAt_append [] (bodyProgram 2 0) [])

-- The row register may be stale: only the already-selected pointers are used.
private def staleRowInitial : State Register :=
  { initial with regs := fun r => if r = .row then 99 else initial.regs r }
example : BodyEntry layout 2 buffers data 1 staleRowInitial :=
  ⟨entry.inputs, entry.active, entry.weightRead, entry.inputRead, entry.biasRead, entry.outputWrite⟩
example : (run (bodyProgram 2 0) ⟨⟨0, 0⟩, Environment.empty, ops⟩ (bodySteps 2)
    staleRowInitial).map (fun s => ((s.regs .row).toNat, s.globalMemory 36)) =
    some (99, some 98) := by decide

-- The generalized suffix theorem also applies after a non-loop prelude.
private def prelude : List (Instruction Register) := [.mov .row (.imm 1), .mov .column (.imm 2)]
private def suffixInitial : State Register :=
  { initial with pc := prelude.length, regs := fun r => if r = .accumulator then 90 else initial.regs r }
private theorem suffixEntry : SuffixEntry ops data buffers 1 suffixInitial := by
  constructor <;> decide
example : ∃ final, run (prelude ++ reluSuffix) ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 suffixInitial = some final ∧
    SuffixResult ops data buffers 1 suffixInitial final :=
  reluSuffix_correct (prelude ++ reluSuffix) ⟨⟨0, 0⟩, Environment.empty, ops⟩
    data buffers 1 suffixInitial suffixEntry (by
      simpa [suffixInitial] using codeAt_append prelude reluSuffix [])

-- Complete execution at a nonzero entry uses the absolute loop target and returns.
private def offsetProgram : List (Instruction Register) :=
  [.ret, .mov .row (.imm 99)] ++ bodyProgram 2 2 ++ [.mov .row (.imm 99)]
example : (run offsetProgram ⟨⟨0, 0⟩, Environment.empty, ops⟩ (bodySteps 2)
    { initial with pc := 2 }).map (fun s => (s.pc, s.returned, (s.regs .result).toNat)) =
    some (17, true, 98) := by decide

-- Arbitrary prefixes and suffixes have the same body contract at their actual PCs.
example (pre post : List (Instruction Register)) :
    ∃ final, run (pre ++ bodyProgram 2 pre.length ++ post)
        ⟨⟨0, 0⟩, Environment.empty, ops⟩ (bodySteps 2) { initial with pc := pre.length } =
        some final ∧ BodyResult ops data buffers 1 { initial with pc := pre.length } final :=
  body_correct _ ⟨⟨0, 0⟩, Environment.empty, ops⟩ data buffers 1
    { initial with pc := pre.length }
    ⟨entry.inputs, entry.active, entry.weightRead, entry.inputRead,
      entry.biasRead, entry.outputWrite⟩
    (by simpa [layout] using codeAt_append pre (bodyProgram 2 pre.length) post)

#print axioms CodeAt.left
#print axioms reluSuffix_correct
#print axioms body_correct

private def checkFullLoop : IO Unit := do
  -- Full-size execution uses artificial word arithmetic, not a hardware floating-point claim.
  let sumOps : FP32.Arithmetic := ⟨fun w x acc => acc + w * x, fun a b => a + b⟩
  let fullInitial : State Register := { initial with
    regs64 := fun r => if r = .weightRead then 0 else if r = .inputRead then 8192 else 999
    globalMemory := fun a => if a.toNat < 4096 then some 2
      else if 8192 ≤ a.toNat && a.toNat < 12288 then some 3 else none }
  let result := run (dotLoop layerLayout.inputWidth 0) ⟨⟨0, 0⟩, Environment.empty, sumOps⟩
    (dotLoopStart + dotIterationSteps * layerLayout.inputWidth) fullInitial
  match result with
  | none => throw (IO.userError "Full-width loop execution failed.")
  | some final =>
    unless final.pc == dotLoopExit && final.regs .column == 1024 &&
        final.regs .accumulator == 6144 && final.regs64 .weightRead == 4096 &&
        final.regs64 .inputRead == 12288 && !final.predicates .more do
      throw (IO.userError "Full-width loop trace did not match the independent oracle.")

private def checkFullBody : IO Unit := do
  -- Artificial word arithmetic supplies independent full-size execution oracles.
  let sumOps : FP32.Arithmetic := ⟨fun w x acc => acc + w * x, fun a b => a + b⟩
  let fullInitial : State Register := { initial with
    regs64 := fun r => match r with
      | .weightRead => 0 | .inputRead => 8192 | .biasRead => 16380 | .outputWrite => 16384 | _ => 999
    globalMemory := fun a => if a.toNat < 4096 then some 2
      else if 8192 ≤ a.toNat && a.toNat < 12288 then some 3
      else if a = 16380 then some 1 else if a = 16388 then some 0x3f800000 else none }
  match run (bodyProgram layerLayout.inputWidth 0) ⟨⟨0, 0⟩, Environment.empty, sumOps⟩
      (bodySteps layerLayout.inputWidth) fullInitial with
  | none => throw (IO.userError "Full-width body execution failed.")
  | some final =>
    unless final.pc == (bodyProgram layerLayout.inputWidth 0).length && final.returned &&
        final.regs .column == 1024 && final.regs .result == 6145 &&
        final.globalMemory 16384 == some 6145 && final.globalMemory 16380 == some 1 &&
        final.globalMemory 16388 == some 0x3f800000 do
      throw (IO.userError "Full-width body output/frame did not match independent oracles.")

def check : IO Unit := do
  checkFullLoop
  checkFullBody

end ComputationChecks

namespace WholeKernelChecks

private def cfg : Launch := ⟨2, 2, by decide, by decide⟩
private def layout : Layout := ⟨2, by decide, by decide⟩
private def args : Argument → Word64
  | .weights => 1000 | .input => 1032 | .bias => 1040 | .output => 1064
private def env : Environment :=
  { params := fun symbol => if symbol = "W_ptr" then some 1000
      else if symbol = "x_ptr" then some 1032 else if symbol = "b_ptr" then some 1040
      else if symbol = "y_ptr" then some 1064 else none
    toGlobal := fun a => if 1000 ≤ a.toNat && a.toNat < 2000 then
      some (BitVec.ofNat 64 (a.toNat - 1000)) else none }
private def buffers : Buffers := fun arg => BitVec.ofNat 64 ((args arg).toNat - 1000)

private def data : TensorData layout cfg.rows :=
  ⟨fun i j => BitVec.ofNat 32 (i.val * 2 + j.val + 1),
    fun j => BitVec.ofNat 32 (j.val + 5), fun i => BitVec.ofNat 32 (i.val + 7)⟩
private def initial : State Register :=
  { regs := fun _ => 99
    pc := 0
    regs64 := fun _ => 999
    predicates := fun _ => true
    globalMemory := fun a =>
      if a.toNat < 32 then some (BitVec.ofNat 32 (a.toNat / 4 + 1))
      else if a = 32 then some 5 else if a = 36 then some 6
      else if 40 ≤ a.toNat && a.toNat < 56 then some (BitVec.ofNat 32 ((a.toNat - 40) / 4 + 7))
      else if a = 64 then some 0x3f800000 else none }
private def ops : FP32.Arithmetic := ⟨fun w x acc => acc * 10 + w + x, fun a b => a + b⟩
private def thread : Thread cfg := ⟨⟨1, by decide⟩, ⟨0, by decide⟩⟩
private def executed (fuel : Nat) :=
  run (kernelProgram cfg layout) ⟨thread.identity, env, ops⟩ fuel initial

private theorem entry : KernelEntry cfg layout env args buffers data initial := by
  constructor
  · rfl
  · rfl
  · intro arg; cases arg <;> decide
  · intro arg; cases arg <;> decide
  · constructor <;> decide

-- Full execution reuses the theorem for arbitrary launch counts and input width.
example : ∃ final, executed (kernelSteps cfg layout) = some final ∧
    KernelResult cfg layout ops data buffers thread initial final :=
  kernel_correct cfg layout (by decide) ⟨thread.identity, env, ops⟩ data args buffers thread rfl initial entry

-- The only store follows all reads: output/bias aliasing is valid for one thread.
private def aliasArgs : Argument → Word64 :=
  fun arg => if arg = .output then args .bias else args arg
private def aliasEnv : Environment :=
  { env with params := fun symbol =>
      if symbol = "y_ptr" then some (args .bias) else env.params symbol }
private def aliasBuffers : Buffers := fun arg => if arg = .output then buffers .bias else buffers arg
private theorem aliasEntry : KernelEntry cfg layout aliasEnv aliasArgs aliasBuffers data initial := by
  constructor
  · rfl
  · rfl
  · intro arg; cases arg <;> decide
  · intro arg; cases arg <;> decide
  · constructor <;> decide
example : ∃ final, run (kernelProgram cfg layout) ⟨thread.identity, aliasEnv, ops⟩ (kernelSteps cfg layout) initial = some final ∧
    KernelResult cfg layout ops data aliasBuffers thread initial final :=
  kernel_correct cfg layout (by decide) ⟨thread.identity, aliasEnv, ops⟩ data aliasArgs aliasBuffers thread rfl initial aliasEntry
example : (run (kernelProgram cfg layout) ⟨thread.identity, aliasEnv, ops⟩ (kernelSteps cfg layout) initial).bind (fun s => s.globalMemory 48) = some 121 := by decide
example : (run (kernelProgram cfg layout) ⟨thread.identity, aliasEnv, ops⟩ (kernelSteps cfg layout) initial).bind (fun s => s.globalMemory 40) = some 7 := by decide

-- Actual PCs at all block boundaries, including the taken and untaken loop branch.
example : (executed 8).map (fun s => (s.pc, (s.regs64 .inputRead).toNat)) = some (8, 32) := by decide
example : (executed 11).map (fun s => (s.pc, (s.regs .row).toNat)) = some (11, 2) := by decide
example : (executed 16).map (fun s => (s.pc, (s.regs64 .weightRead).toNat)) = some (16, 16) := by decide
example : (executed 16).map (fun s => ((s.regs64 .biasRead).toNat, (s.regs64 .outputWrite).toNat)) = some (48, 72) := by decide
example : (executed 18).map (fun s => (s.pc, (s.regs .column).toNat)) = some (18, 0) := by decide
example : (executed 26).map (fun s => (s.pc, (s.regs .column).toNat)) = some (18, 1) := by decide
example : (executed 34).map (fun s => (s.pc, (s.regs .accumulator).toNat)) = some (26, 112) := by decide
example : (executed 39).map (fun s => (s.pc, s.returned)) = some (31, true) := by decide
example : (executed 39).map (fun s => ((s.regs .row).toNat, (s.regs .result).toNat)) = some (2, 121) := by decide
example : (executed 39).bind (fun s => s.globalMemory 72) = some 121 := by decide
example : (executed 39).bind (fun s => s.globalMemory 48) = some 9 := by decide
example : (executed 39).bind (fun s => s.globalMemory 64) = some 0x3f800000 := by decide
example : executed 40 = none := by decide
example : (kernelProgram layerLaunch layerLayout).length = 31 := rfl
example : kernelSteps layerLaunch layerLayout = 8215 := rfl
example : (kernelProgram layerLaunch layerLayout)[18]? = some (.loadF32 .weightValue (.reg .weightRead)) := rfl
example : (kernelProgram layerLaunch layerLayout)[25]? = some (.branchIf .more 18) := rfl
example : run (kernelProgram cfg layout) ⟨thread.identity, { env with params := fun _ => none }, ops⟩ 1 initial = none := by decide
example : run (kernelProgram cfg layout) ⟨thread.identity, { env with toGlobal := fun _ => none }, ops⟩ 8 initial = none := by decide

-- Direct branch construction yields exactly the previous relocated instruction list.
example (cfg : Launch) (layout : Layout) :
    kernelProgram cfg layout = kernelSetup cfg layout ++
      (bodyProgram layout.inputWidth 0).map (fun inst => match inst with
        | .branchIf predicate target => .branchIf predicate ((kernelSetup cfg layout).length + target)
        | inst => inst) := by
  simp [kernelProgram, bodyProgram, dotLoop, reluSuffix]

#print axioms CodeAt.right
#print axioms rowSelection_correct
#print axioms kernel_correct

private def expectedSource : String :=
  let offset := (kernelSetup layerLaunch layerLayout).length + dotLoopStart
  let lines := (kernelProgram layerLaunch layerLayout).map
    (fun inst => inst.ptxText.replace s!"PC_{offset}" "DOT_LOOP")
  String.intercalate "\n" (lines.take offset ++ ["DOT_LOOP:"] ++ lines.drop offset)

def check (argv : List String) : IO Unit := do
  let source ← IO.FS.readFile (argv.headD "kernels/relu_layer.ptx")
  let normalized := "\n" ++ String.intercalate "\n"
    ((source.splitOn "\n").map (fun line => line.trimAscii.toString)) ++ "\n"
  let declarations := ".visible .entry relu_layer(\n" ++
    String.intercalate ",\n" ([Argument.weights, .input, .bias, .output].map
      (fun arg => s!".param .u64 {arg.symbol}")) ++ "\n)\n"
  for expected in [declarations,
      s!".reqntid {layerLaunch.threads}, 1, 1\n",
      s!"// Launch: grid=({layerLaunch.blocks},1,1), block=({layerLaunch.threads},1,1); arguments: W, x, b, y.\n"] do
    unless (normalized.splitOn expected).length == 2 do
      throw (IO.userError s!"Expected exactly one kernel declaration:\n{expected}")
  let body := ((source.splitOn "{").drop 1).headD ""
  let body := (body.splitOn "}").headD ""
  let lines := ((body.splitOn "\n").map (fun line => line.trimAscii.toString)).filter
    (fun line => !line.isEmpty && !(line.startsWith "//") && !(line.startsWith ".reg "))
  unless String.intercalate "\n" lines == expectedSource do
    throw (IO.userError "Complete kernel instruction list/branch label differs from the formal program.")
  -- Independent order-sensitive oracle for every thread of the small launch.
  for block in [:cfg.blocks] do
    for tid in [:cfg.threads] do
      let row := cfg.threads * block + tid
      let expected := 10 * (2 * row + 1 + 5) + (2 * row + 2 + 6) + (row + 7)
      match run (kernelProgram cfg layout) ⟨⟨BitVec.ofNat 32 tid, BitVec.ofNat 32 block⟩, env, ops⟩ (kernelSteps cfg layout) initial with
      | none => throw (IO.userError "Small-launch kernel execution failed.")
      | some final =>
        unless final.returned && final.pc == 31 && final.regs .row == BitVec.ofNat 32 row &&
            final.globalMemory (BitVec.ofNat 64 (64 + 4 * row)) == some (BitVec.ofNat 32 expected) do
          throw (IO.userError "Small-launch result did not match the independent ordered oracle.")
        for address in [:80] do
          if address != 64 + 4 * row then
            unless final.globalMemory (BitVec.ofNat 64 address) == initial.globalMemory (BitVec.ofNat 64 address) do
              throw (IO.userError "Kernel changed another word location.")
  -- Selected launch boundary rows and 1024 columns, using artificial word arithmetic.
  let sumOps : FP32.Arithmetic := ⟨fun w x acc => acc + w * x, fun a b => a + b⟩
  let weightBytes := layerLaunch.rows * layerLayout.rowStride
  let xBase := weightBytes
  let bBase := xBase + layerLayout.rowStride
  let yBase := bBase + layerLaunch.rows * fp32Bytes
  let fullArgs : Argument → Word64 := fun a => BitVec.ofNat 64 (1000 + (match a with
    | .weights => 0 | .input => xBase | .bias => bBase | .output => yBase))
  let fullEnv : Environment :=
    { params := fun symbol => if symbol = "W_ptr" then some (fullArgs .weights)
        else if symbol = "x_ptr" then some (fullArgs .input)
        else if symbol = "b_ptr" then some (fullArgs .bias)
        else if symbol = "y_ptr" then some (fullArgs .output) else none
      toGlobal := fun a => some (BitVec.ofNat 64 (a.toNat - 1000)) }
  let fullInitial : State Register := { initial with
    globalMemory := fun a =>
      if a.toNat < weightBytes then some 2
      else if a.toNat < bBase then some 3
      else if a.toNat < yBase then some 1 else some 0x3f800000 }
  for row in [0, 256, layerLaunch.rows - 1] do
    match run (kernelProgram layerLaunch layerLayout) ⟨⟨BitVec.ofNat 32 (row % layerLaunch.threads), BitVec.ofNat 32 (row / layerLaunch.threads)⟩, fullEnv, sumOps⟩ (kernelSteps layerLaunch layerLayout) fullInitial with
    | none => throw (IO.userError "Full-width kernel execution failed.")
    | some final =>
      let output := BitVec.ofNat 64 (yBase + row * fp32Bytes)
      unless final.pc == 31 && final.returned && final.regs .row == BitVec.ofNat 32 row &&
          final.regs .column == 1024 && final.regs .result == 6145 &&
          final.globalMemory output == some 6145 &&
          final.globalMemory (BitVec.ofNat 64 (bBase + row * fp32Bytes)) == some 1 &&
          final.globalMemory (output + 4) == some 0x3f800000 do
        throw (IO.userError "Full-width row/output/frame did not match independent oracles.")
  IO.println "Whole-kernel boundaries, absolute branch, all small-launch threads, 1024-column boundary rows, output/frame and exact PTX source pass."

end WholeKernelChecks

def main (argv : List String) : IO Unit := do
  ComputationChecks.check
  WholeKernelChecks.check argv
  IO.println "Kernel instruction, placement, memory, execution and complete source checks pass."

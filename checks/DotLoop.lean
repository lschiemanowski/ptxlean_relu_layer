import ReluLayer.PTX.DotLoop

open ReluLayer.PTX

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
  else if a = 24 then some 7 else if a = 28 then some 8 else none
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

-- Arbitrary scratch is overwritten; the first branch is taken, the last is not.
example : ((run (dotLoop 2) ⟨0, 0⟩ 10 initial Environment.empty (some ops)).map
    (fun s => (s.pc, (s.regs .column).toNat, (s.regs .accumulator).toNat,
      (s.regs64 .weightRead).toNat, (s.regs64 .inputRead).toNat, s.predicates .more))) =
    some (2, 1, 8, 12, 20, true) := by decide
example : ((run (dotLoop 2) ⟨0, 0⟩ 18 initial Environment.empty (some ops)).map
    (fun s => (s.pc, (s.regs .column).toNat, (s.regs .accumulator).toNat,
      (s.regs64 .weightRead).toNat, (s.regs64 .inputRead).toNat, s.predicates .more))) =
    some (10, 2, 90, 16, 24, false) := by decide
example : ((run (dotLoop 1) ⟨0, 0⟩ 10 initial Environment.empty (some ops)).map
    (fun s => (s.pc, (s.regs .accumulator).toNat, s.predicates .more))) =
    some (10, 8, false) := by decide
example : ((run (dotLoop 2) ⟨0, 0⟩ 18 initial Environment.empty (some ops)).map
    (fun s => ((s.regs .row).toNat, (s.regs64 .biasRead).toNat,
      (s.regs64 .outputWrite).toNat, s.globalMemory 8, s.predicates .row))) =
    some (1, 28, 36, some 3, true) := by decide

-- Unsigned comparison, finite-width addition, unavailable loads/operations, and FMA aliasing.
example : ((Instruction.setpLt Register.more (.imm 0x80000000) (.imm 1)).execute
    ⟨0, 0⟩ initial).map (fun s => s.predicates .more) = some false := by decide
example : ((Instruction.add32 Register.column (.imm 0xffffffff) (.imm 1)).execute
    ⟨0, 0⟩ initial).map (fun s => s.regs .column) = some 0 := by decide
example : (Instruction.loadF32 Register.weightValue (.imm 100)).execute ⟨0, 0⟩ initial = none := by decide
example : (Instruction.fmaRN Register.accumulator (.imm 1) (.imm 2) (.imm 3)).execute
    ⟨0, 0⟩ initial = none := by decide
example : ((Instruction.fmaRN Register.accumulator (.imm 1) (.imm 2) (.reg .accumulator)).execute
    ⟨0, 0⟩ initial Environment.empty (some ops)).map (fun s => s.regs .accumulator) = some 993 := by decide
example : ((Instruction.branchIf Register.more 7).execute ⟨0, 0⟩ initial).map (fun s => s.pc) = some 7 := by decide
example : ((Instruction.branchIf Register.more 7).execute ⟨0, 0⟩
    { initial with pc := 4, predicates := fun _ => false }).map (fun s => s.pc) = some 5 := by decide

-- The full-width invariant includes exit PC, counter, accumulator and memory facts.
example : ∃ final, run (dotLoop 2) ⟨0, 0⟩ 18 initial Environment.empty (some ops) = some final ∧
    LoopInvariant ops data buffers 1 initial 2 final :=
  dotLoop_correct ops data buffers 1 initial ⟨0, 0⟩ Environment.empty entry

#print axioms run_add
#print axioms prefixAccumulator_succ
#print axioms dotLoop_iteration
#print axioms dotLoop_prefix
#print axioms dotLoop_correct

-- This executable label table binds the PTX label to the actual loop-entry PC.
private def expectedSource : String :=
  let lines := (dotLoop layerLayout.inputWidth).map
    (fun inst => inst.ptxText.replace s!"PC_{dotLoopStart}" "DOT_LOOP")
  "\n" ++ String.intercalate "\n"
    (lines.take dotLoopStart ++ ["DOT_LOOP:"] ++ lines.drop dotLoopStart) ++ "\n"

def main (args : List String) : IO Unit := do
  let source ← IO.FS.readFile (args.headD "kernels/relu_layer.ptx")
  let lines := ((source.splitOn "\n").map (fun line => line.trimAscii.toString)).filter
    (fun line => !line.isEmpty && !(line.startsWith "//"))
  let normalized := "\n" ++ String.intercalate "\n" lines ++ "\n"
  unless (normalized.splitOn expectedSource).length == 2 do
    throw (IO.userError s!"Expected exactly one initialized dot-loop fragment:\n{expectedSource}")
  -- Full-size execution uses artificial word arithmetic, not a hardware floating-point claim.
  let sumOps : FP32.Arithmetic := ⟨fun w x acc => acc + w * x, fun a b => a + b⟩
  let fullInitial : State Register := { initial with
    regs64 := fun r => if r = .weightRead then 0 else if r = .inputRead then 8192 else 999
    globalMemory := fun a => if a.toNat < 4096 then some 2
      else if 8192 ≤ a.toNat && a.toNat < 12288 then some 3 else none }
  let result := run (dotLoop layerLayout.inputWidth) ⟨0, 0⟩
    (dotLoopStart + dotIterationSteps * layerLayout.inputWidth) fullInitial Environment.empty (some sumOps)
  match result with
  | none => throw (IO.userError "Full-width loop execution failed.")
  | some final =>
    unless final.pc == dotLoopExit && final.regs .column == 1024 &&
        final.regs .accumulator == 6144 && final.regs64 .weightRead == 4096 &&
        final.regs64 .inputRead == 12288 && !final.predicates .more do
      throw (IO.userError "Full-width loop trace did not match the independent oracle.")
  IO.println "Dot-loop semantics, 1024-column execution, and exact PTX instructions/label placement pass."

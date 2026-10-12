import ReluLayer.PTX.BodyExecution

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

private def executed : Option (State Register) :=
  run (bodyProgram 2) ⟨0, 0⟩ (bodySteps 2) initial Environment.empty (some ops)

-- The appended program crosses the loop/suffix boundary with no PC reset.
example : ((run (bodyProgram 2) ⟨0, 0⟩ 18 initial Environment.empty (some ops)).map
    (fun s => (s.pc, s.returned, (s.regs .accumulator).toNat))) = some (10, false, 90) := by decide
example : executed.map (fun s => (s.pc, s.returned)) = some (15, true) := by decide
example : executed.map (fun s => ((s.regs .column).toNat, (s.regs .result).toNat)) = some (2, 98) := by decide
example : executed.bind (fun s => s.globalMemory 36) = some 98 := by decide
example : executed.bind (fun s => s.globalMemory 28) = some 8 := by decide
example : executed.bind (fun s => s.globalMemory 32) = some 0x3f800000 := by decide
example : executed.map (fun s => ((s.regs .row).toNat, (s.regs64 .biasRead).toNat,
    (s.regs64 .outputWrite).toNat)) = some (1, 28, 36) := by decide
example : ∃ final, run (bodyProgram 2) ⟨0, 0⟩ (bodySteps 2) initial Environment.empty (some ops) = some final ∧
    BodyResult ops data buffers 1 initial final :=
  body_correct ops data buffers 1 initial ⟨0, 0⟩ Environment.empty entry

-- The generalized suffix theorem also applies after a non-loop prelude.
private def prelude : List (Instruction Register) := [.mov .row (.imm 1), .mov .column (.imm 2)]
private def suffixInitial : State Register :=
  { initial with pc := prelude.length, regs := fun r => if r = .accumulator then 90 else initial.regs r }
private theorem suffixEntry : SuffixEntry ops data buffers 1 suffixInitial prelude.length := by
  constructor <;> decide
example : ∃ final, run (prelude ++ reluSuffix) ⟨0, 0⟩ 5 suffixInitial Environment.empty (some ops) = some final ∧
    SuffixResult ops data buffers 1 suffixInitial final prelude.length :=
  reluSuffix_correct_at prelude ops data buffers 1 suffixInitial ⟨0, 0⟩ Environment.empty suffixEntry

#print axioms run_append_of_run
#print axioms reluSuffix_correct_at
#print axioms body_correct

private def expectedSource : String :=
  let lines := (bodyProgram layerLayout.inputWidth).map
    (fun inst => inst.ptxText.replace s!"PC_{dotLoopStart}" "DOT_LOOP")
  "\n" ++ String.intercalate "\n"
    (lines.take dotLoopStart ++ ["DOT_LOOP:"] ++ lines.drop dotLoopStart) ++ "\n"

def main (args : List String) : IO Unit := do
  let source ← IO.FS.readFile (args.headD "kernels/relu_layer.ptx")
  let lines := ((source.splitOn "\n").map (fun line => line.trimAscii.toString)).filter
    (fun line => !line.isEmpty && !(line.startsWith "//"))
  let normalized := "\n" ++ String.intercalate "\n" lines ++ "\n"
  unless (normalized.splitOn expectedSource).length == 2 do
    throw (IO.userError s!"Expected exactly one complete computational block:\n{expectedSource}")
  -- Artificial word arithmetic supplies independent full-size execution oracles.
  let sumOps : FP32.Arithmetic := ⟨fun w x acc => acc + w * x, fun a b => a + b⟩
  let fullInitial : State Register := { initial with
    regs64 := fun r => match r with
      | .weightRead => 0 | .inputRead => 8192 | .biasRead => 16380 | .outputWrite => 16384 | _ => 999
    globalMemory := fun a => if a.toNat < 4096 then some 2
      else if 8192 ≤ a.toNat && a.toNat < 12288 then some 3
      else if a = 16380 then some 1 else if a = 16388 then some 0x3f800000 else none }
  match run (bodyProgram layerLayout.inputWidth) ⟨0, 0⟩
      (bodySteps layerLayout.inputWidth) fullInitial Environment.empty (some sumOps) with
  | none => throw (IO.userError "Full-width body execution failed.")
  | some final =>
    unless final.pc == (bodyProgram layerLayout.inputWidth).length && final.returned &&
        final.regs .column == 1024 && final.regs .result == 6145 &&
        final.globalMemory 16384 == some 6145 && final.globalMemory 16380 == some 1 &&
        final.globalMemory 16388 == some 0x3f800000 do
      throw (IO.userError "Full-width body output/frame did not match independent oracles.")
  IO.println "Complete computational block, suffix offset, 1024-column output/frame, and exact PTX source pass."

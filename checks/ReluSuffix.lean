import ReluLayer.PTX.ReluSuffix

open ReluLayer.PTX

-- Arbitrary word arithmetic for integer-only instruction fixtures.
private def testArithmetic : FP32.Arithmetic :=
  ⟨fun w x a => w * x + a, fun a b => a + b⟩

private def layout : Layout := ⟨2, by decide, by decide⟩
private def buffers : Buffers
  | .weights => 0 | .input => 16 | .bias => 24 | .output => 32
private def data : TensorData layout 2 :=
  ⟨fun i j => BitVec.ofNat 32 (i.val * 2 + j.val + 1),
    fun j => BitVec.ofNat 32 (j.val + 5), fun i => BitVec.ofNat 32 (i.val + 7)⟩
-- Artificial arithmetic makes the independent expected dot/bias values transparent.
private def ops : FP32.Arithmetic := ⟨fun w x acc => acc * 10 + w + x, fun a b => a + b⟩
private def initial : State Register :=
  { regs := fun r => match r with
      | .row => 1 | .column => 2 | .accumulator => 90
      | .biasValue => 0x7fc00000 | .result => 0xdeadbeef | _ => 99
    pc := 0
    regs64 := fun r => if r = .biasRead then 28 else if r = .outputWrite then 36 else 999
    globalMemory := fun a => if a = 28 then some 8 else if a = 32 then some 0x3f800000 else none
    predicates := fun _ => true }
private theorem entry : SuffixEntry ops data buffers 1 initial := by constructor <;> decide

private def executed : Option (State Register) := run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 initial
example : executed.map (fun s => (s.pc, s.returned)) = some (5, true) := by decide
example : executed.map (fun s => ((s.regs .biasValue).toNat, (s.regs .accumulator).toNat,
    (s.regs .result).toNat)) = some (8, 98, 98) := by decide
example : executed.bind (fun s => s.globalMemory 36) = some 98 := by decide
example : initial.globalMemory 36 = none := by decide
example : executed.map (fun s => ((s.regs .row).toNat, (s.regs .column).toNat)) = some (1, 2) := by decide
example : executed.map (fun s => ((s.regs64 .biasRead).toNat, (s.regs64 .outputWrite).toNat)) =
    some (28, 36) := by decide
example : executed.bind (fun s => s.globalMemory 28) = some 8 := by decide
example : executed.bind (fun s => s.globalMemory 32) = some 0x3f800000 := by decide
example : executed.map (fun s => s.predicates .more) = some true := by decide
example : ∃ final, run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 initial = some final ∧
    SuffixResult ops data buffers 1 initial final :=
  reluSuffix_correct ⟨⟨0, 0⟩, Environment.empty, ops⟩ data buffers 1 initial entry

-- Feed chosen preactivation patterns through the actual max/store/return sequence.
private def preactivationOps (v : Word32) : FP32.Arithmetic := ⟨fun _ _ a => a, fun _ _ => v⟩
private def stored (v : Word32) : Option Word32 :=
  (run reluSuffix ⟨⟨0, 0⟩, Environment.empty, (preactivationOps v)⟩ 5 initial).bind
    (fun s => s.globalMemory 36)
example : stored 0xbf800000 = some 0 := by decide
example : stored 0x80000000 = some 0 := by decide
example : stored 0x7fc00001 = some 0 := by decide
example : stored 0x7f800001 = some 0 := by decide
example : stored 0x7f800000 = some 0x7f800000 := by decide
example : stored 1 = some 1 := by decide

-- Addition reads the old destination; a missing bias excludes execution.
example : ((Instruction.addF32RN Register.accumulator (.reg .accumulator) (.imm 8)).execute ⟨⟨0, 0⟩, Environment.empty, ops⟩ initial).map (fun s => s.regs .accumulator) = some 98 := by decide
example : run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 { initial with globalMemory := fun _ => none } = none := by decide
example : run reluSuffix ⟨⟨0, 0⟩, Environment.empty, ops⟩ 5 { initial with returned := true } = none := by decide

-- ret stops execution even if a valid store instruction follows it.
private def afterRet : List (Instruction Register) := [.ret, .storeF32 (.imm 36) (.imm 123)]
example : ((run afterRet ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 1 initial).map (fun s => (s.returned, s.globalMemory 36))) =
    some (true, none) := by decide
example : run afterRet ⟨⟨0, 0⟩, Environment.empty, testArithmetic⟩ 2 initial = none := by decide

#print axioms reluSuffix_correct
#print axioms step_after_return

private def expectedSource : String :=
  "\n" ++ String.intercalate "\n" (reluSuffix.map Instruction.ptxText) ++ "\n"

def main (args : List String) : IO Unit := do
  let source ← IO.FS.readFile (args.headD "kernels/relu_layer.ptx")
  let lines := ((source.splitOn "\n").map (fun line => line.trimAscii.toString)).filter
    (fun line => !line.isEmpty && !(line.startsWith "//"))
  let normalized := "\n" ++ String.intercalate "\n" lines ++ "\n"
  unless (normalized.splitOn expectedSource).length == 2 do
    throw (IO.userError s!"Expected exactly one bias/ReLU/store/return suffix:\n{expectedSource}")
  IO.println "Bias/ReLU/store/return semantics and five exact PTX instructions pass."

import ReluLayer.PTX.KernelExecution

open ReluLayer.PTX

private def cfg : Launch := ⟨2, 2, by decide, by decide⟩
private def layout : Layout := ⟨2, by decide, by decide⟩
private def args : Argument → Word64
  | .weights => 1000 | .input => 1032 | .bias => 1040 | .output => 1064
private def env : Environment :=
  { params := fun symbol => if symbol = "W_ptr" then some 1000
      else if symbol = "x_ptr" then some 1032 else if symbol = "b_ptr" then some 1040
      else if symbol = "y_ptr" then some 1064 else none
    space := ⟨⟨1000, 1000, by decide⟩, []⟩ }
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
  run (kernelProgram cfg layout) thread.identity fuel initial env (some ops)

private theorem entry : KernelEntry cfg layout env args data initial := by
  constructor
  · rfl
  · rfl
  · intro arg; cases arg <;> decide
  · intro arg; cases arg <;> decide
  · constructor <;> decide

-- Full execution reuses the theorem for arbitrary launch counts and input width.
example : ∃ final, executed (kernelSteps cfg layout) = some final ∧
    KernelResult cfg layout ops data (convertedBuffers env args) thread initial final :=
  kernel_correct cfg layout (by decide) ops data env args thread initial entry

-- The only store follows all reads: output/bias aliasing is valid for one thread.
private def aliasArgs : Argument → Word64 :=
  fun arg => if arg = .output then args .bias else args arg
private def aliasEnv : Environment :=
  { env with params := fun symbol =>
      if symbol = "y_ptr" then some (args .bias) else env.params symbol }
private theorem aliasEntry : KernelEntry cfg layout aliasEnv aliasArgs data initial := by
  constructor
  · rfl
  · rfl
  · intro arg; cases arg <;> decide
  · intro arg; cases arg <;> decide
  · constructor <;> decide
example : ¬ OutputsSeparated layout cfg.rows (convertedBuffers aliasEnv aliasArgs) := by
  intro h
  exact h.bias ⟨0, by decide⟩ ⟨0, by decide⟩ rfl
example : ∃ final, run (kernelProgram cfg layout) thread.identity (kernelSteps cfg layout)
    initial aliasEnv (some ops) = some final ∧
    KernelResult cfg layout ops data (convertedBuffers aliasEnv aliasArgs) thread initial final :=
  kernel_correct cfg layout (by decide) ops data aliasEnv aliasArgs thread initial aliasEntry
example : (run (kernelProgram cfg layout) thread.identity (kernelSteps cfg layout)
    initial aliasEnv (some ops)).bind (fun s => s.globalMemory 48) = some 121 := by decide
example : (run (kernelProgram cfg layout) thread.identity (kernelSteps cfg layout)
    initial aliasEnv (some ops)).bind (fun s => s.globalMemory 40) = some 7 := by decide

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
example : run (kernelProgram cfg layout) thread.identity 1 initial
    { env with params := fun _ => none } (some ops) = none := by decide
example : run (kernelProgram cfg layout) thread.identity 8 initial
    { env with space := Environment.empty.space } (some ops) = none := by decide
example : run (kernelProgram cfg layout) thread.identity 39 initial env none = none := by decide

#print axioms run_relocate
#print axioms rowSelection_correct
#print axioms kernel_correct
#print axioms layer_kernel_correct

private def expectedSource : String :=
  let offset := (kernelSetup layerLaunch layerLayout).length + dotLoopStart
  let lines := (kernelProgram layerLaunch layerLayout).map
    (fun inst => inst.ptxText.replace s!"PC_{offset}" "DOT_LOOP")
  String.intercalate "\n" (lines.take offset ++ ["DOT_LOOP:"] ++ lines.drop offset)

def main (argv : List String) : IO Unit := do
  let source ← IO.FS.readFile (argv.headD "kernels/relu_layer.ptx")
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
      match run (kernelProgram cfg layout) ⟨BitVec.ofNat 32 tid, BitVec.ofNat 32 block⟩
          (kernelSteps cfg layout) initial env (some ops) with
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
      space := ⟨⟨1000, 2 ^ 24, by decide⟩, []⟩ }
  let fullInitial : State Register := { initial with
    globalMemory := fun a =>
      if a.toNat < weightBytes then some 2
      else if a.toNat < bBase then some 3
      else if a.toNat < yBase then some 1 else some 0x3f800000 }
  for row in [0, 256, layerLaunch.rows - 1] do
    match run (kernelProgram layerLaunch layerLayout)
        ⟨BitVec.ofNat 32 (row % layerLaunch.threads), BitVec.ofNat 32 (row / layerLaunch.threads)⟩
        (kernelSteps layerLaunch layerLayout) fullInitial fullEnv (some sumOps) with
    | none => throw (IO.userError "Full-width kernel execution failed.")
    | some final =>
      let output := BitVec.ofNat 64 (yBase + row * fp32Bytes)
      unless final.pc == 31 && final.returned && final.regs .row == BitVec.ofNat 32 row &&
          final.regs .column == 1024 && final.regs .result == 6145 &&
          final.globalMemory output == some 6145 &&
          final.globalMemory (BitVec.ofNat 64 (bBase + row * fp32Bytes)) == some 1 &&
          final.globalMemory (output + 4) == some 0x3f800000 do
        throw (IO.userError "Full-width row/output/frame did not match independent oracles.")
  IO.println "Whole-kernel boundaries, relocated branch, all small-launch threads, 1024-column boundary rows, output/frame and exact PTX source pass."

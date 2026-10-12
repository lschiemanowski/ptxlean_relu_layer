import ReluLayer.PTX.BaseLoading
import ReluLayer.PTX.BodyExecution
import ReluLayer.PTX.ProgramComposition

namespace ReluLayer.PTX

/-- The three setup blocks retain their independent instruction lists. -/
def kernelSetup (cfg : Launch) (layout : Layout) : List (Instruction Register) :=
  baseLoading ++ rowSelection cfg.threads ++ addressCalculation layout.inputWidth

/-- The computational block's absolute branch targets include the setup offset. -/
def kernelProgram (cfg : Launch) (layout : Layout) : List (Instruction Register) :=
  kernelSetup cfg layout ++
    (bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length)

def kernelSteps (cfg : Launch) (layout : Layout) : Nat :=
  (kernelSetup cfg layout).length + bodySteps layout.inputWidth

/-- Global bases obtained from the four generic pointer arguments. -/
def convertedBuffers (env : Environment) (args : Argument → Word64) : Buffers :=
  fun arg => env.space.globalAddress (args arg)

structure KernelEntry (cfg : Launch) (layout : Layout) (env : Environment)
    (args : Argument → Word64) (data : TensorData layout cfg.rows)
    (initial : State Register) : Prop where
  pc : initial.pc = 0
  active : initial.returned = false
  parameters : ∀ arg, env.params arg.symbol = some (args arg)
  admitted : ∀ arg, env.space.admits (args arg) = true
  inputs : InputsStored layout cfg.rows (convertedBuffers env args) data initial.globalMemory

structure KernelResult (cfg : Launch) (layout : Layout) (ops : FP32.Arithmetic)
    (data : TensorData layout cfg.rows) (buffers : Buffers) (t : Thread cfg)
    (initial final : State Register) : Prop where
  pc : final.pc = (kernelProgram cfg layout).length
  returned : final.returned = true
  row : (final.regs .row).toNat = t.row
  column : final.regs .column = BitVec.ofNat 32 layout.inputWidth
  result : final.regs .result = rowResult ops data t.rowIndex
  memory : OutputStored ops data buffers t.rowIndex
    initial.globalMemory final.globalMemory

private theorem shift_reset (s : State Reg) (offset : Nat) (h : s.pc = offset) :
    State.shiftPC offset { s with pc := 0 } = s := by
  cases s; simp_all [State.shiftPC]

private theorem wideProduct_row (row : Word32) (stride : Nat) (h : stride < 2 ^ 32) :
    wideProduct row (BitVec.ofNat 32 stride) = BitVec.ofNat 64 (row.toNat * stride) := by
  simp [wideProduct, Nat.mod_eq_of_lt h]

/-- Each launched thread executes all four blocks and stores its specified row result. -/
theorem kernel_correct (cfg : Launch) (layout : Layout) (hfit : cfg.rows ≤ 2 ^ 32)
    (ops : FP32.Arithmetic) (data : TensorData layout cfg.rows)
    (env : Environment) (args : Argument → Word64) (t : Thread cfg)
    (initial : State Register) (hentry : KernelEntry cfg layout env args data initial) :
    ∃ final, run (kernelProgram cfg layout) t.identity (kernelSteps cfg layout)
        initial env (some ops) = some final ∧
      KernelResult cfg layout ops data (convertedBuffers env args) t initial final := by
  obtain ⟨bases, hbase, hb⟩ := baseLoading_correct env args t.identity initial
    hentry.pc hentry.active hentry.parameters hentry.admitted (some ops)
  let rowInitial := { bases with pc := 0 }
  obtain ⟨rowFinal, hrow, hrr⟩ :=
    rowSelection_correct cfg hfit t rowInitial rfl
      (hb.active.trans hentry.active) env (some ops)
  have hractive := hrr.active.trans (hb.active.trans hentry.active)
  let addressInitial := { rowFinal with pc := 0 }
  obtain ⟨addressFinal, haddress, ha⟩ := addressCalculation_exec layout.inputWidth
    t.identity addressInitial rfl env hractive (some ops)
  let bodyInitial := { addressFinal with pc := 0 }
  have hmemory : bodyInitial.globalMemory = initial.globalMemory :=
    ha.memory.trans (hrr.memory.trans hb.memory)
  have hselected : (addressInitial.regs .row).toNat = t.row := hrr.row
  have hweights : addressInitial.regs64 .weightBase = convertedBuffers env args .weights :=
    congrFun hrr.regs64 .weightBase |>.trans (hb.bases .weights)
  have hbias : addressInitial.regs64 .biasBase = convertedBuffers env args .bias :=
    congrFun hrr.regs64 .biasBase |>.trans (hb.bases .bias)
  have houtput : addressInitial.regs64 .outputBase = convertedBuffers env args .output :=
    congrFun hrr.regs64 .outputBase |>.trans (hb.bases .output)
  have hinput : addressInitial.regs64 .inputRead = convertedBuffers env args .input :=
    congrFun hrr.regs64 .inputRead |>.trans (hb.bases .input)
  have hbodyEntry : BodyEntry layout cfg.rows (convertedBuffers env args) data
      t.rowIndex bodyInitial := by
    refine ⟨?_, rfl, ha.active.trans hractive, ?_, ?_, ?_, ?_, ?_⟩
    · simpa only [hmemory] using hentry.inputs
    · exact (congrArg (fun regs => (regs .row).toNat) ha.regs32).trans hrr.row
    · change addressFinal.regs64 .weightRead = _
      rw [ha.weights, hweights, wideProduct_row _ _ layout.stride_fits, hselected]
      simp [weightAddress, Layout.rowStride]
    · exact ha.inputRead.trans hinput
    · change addressFinal.regs64 .biasRead = _
      rw [ha.bias, hbias, wideProduct_row _ _ (by decide : fp32Bytes < 2 ^ 32), hselected]
      rfl
    · change addressFinal.regs64 .outputWrite = _
      rw [ha.output, houtput, wideProduct_row _ _ (by decide : fp32Bytes < 2 ^ 32), hselected]
      rfl
  obtain ⟨bodyFinal, hbody, hbr⟩ := body_correct ops data (convertedBuffers env args)
    t.rowIndex bodyInitial t.identity env hbodyEntry
  -- Lift each existing block execution to its actual PC in the concatenated kernel.
  have hrowAt := run_relocate baseLoading (rowSelection cfg.threads) t.identity 3
    rowInitial rowFinal env (some ops) hrow
  rw [shift_reset bases baseLoading.length (by simpa [baseLoading] using hb.pc)] at hrowAt
  have hrowFull := run_append_of_run
    (baseLoading ++ rowSelection cfg.threads)
    (addressCalculation layout.inputWidth ++
      (bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length))
    t.identity 3 bases (rowFinal.shiftPC baseLoading.length) env (some ops) hrowAt
  have haddressAt := run_relocate (baseLoading ++ rowSelection cfg.threads)
    (addressCalculation layout.inputWidth) t.identity 5 addressInitial addressFinal env (some ops) haddress
  have haddressStart : addressInitial.shiftPC (baseLoading ++ rowSelection cfg.threads).length =
      rowFinal.shiftPC baseLoading.length := by
    simp [addressInitial, State.shiftPC, baseLoading, rowSelection, hrr.pc]
  rw [haddressStart] at haddressAt
  have haddressFull := run_append_of_run (kernelSetup cfg layout)
    ((bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length))
    t.identity 5 (rowFinal.shiftPC baseLoading.length)
    (addressFinal.shiftPC (baseLoading ++ rowSelection cfg.threads).length) env (some ops) haddressAt
  change run (kernelProgram cfg layout) t.identity 5
    (rowFinal.shiftPC baseLoading.length) env (some ops) = _ at haddressFull
  have hbodyAt := run_relocate (kernelSetup cfg layout) (bodyProgram layout.inputWidth)
    t.identity (bodySteps layout.inputWidth) bodyInitial bodyFinal env (some ops) hbody
  have hbodyStart : bodyInitial.shiftPC (kernelSetup cfg layout).length =
      addressFinal.shiftPC (baseLoading ++ rowSelection cfg.threads).length := by
    simp [bodyInitial, State.shiftPC, kernelSetup, baseLoading, rowSelection,
      addressCalculation, ha.pc]
  rw [hbodyStart] at hbodyAt
  have hbaseFull := run_append_of_run baseLoading
    (rowSelection cfg.threads ++ addressCalculation layout.inputWidth ++
      (bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length))
    t.identity 8 initial bases env (some ops) hbase
  have hbfull : run (kernelProgram cfg layout) t.identity 8 initial env (some ops) = some bases := by
    simpa only [kernelProgram, kernelSetup, List.append_assoc] using hbaseFull
  have hrfull : run (kernelProgram cfg layout) t.identity 3 bases env (some ops) =
      some (rowFinal.shiftPC baseLoading.length) := by
    simpa only [kernelProgram, kernelSetup, List.append_assoc] using hrowFull
  refine ⟨bodyFinal.shiftPC (kernelSetup cfg layout).length, ?_, ?_⟩
  · have hsteps : kernelSteps cfg layout = 8 + (3 + (5 + bodySteps layout.inputWidth)) := by
      simp [kernelSteps, kernelSetup, baseLoading, rowSelection, addressCalculation]; omega
    rw [hsteps, run_add, hbfull, Option.bind_some, run_add, hrfull, Option.bind_some,
      run_add, haddressFull, Option.bind_some]
    exact hbodyAt
  · constructor
    · simpa [State.shiftPC, kernelProgram] using congrArg
        (fun pc => (kernelSetup cfg layout).length + pc) hbr.pc
    · exact hbr.returned
    · exact (congrArg BitVec.toNat (hbr.other32 .row (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide))).trans hbodyEntry.row
    · exact hbr.column
    · exact hbr.result
    · simpa only [State.shiftPC, hmemory] using hbr.memory

/-- Specialization to the selected 1024-input, 3072-output kernel. -/
theorem layer_kernel_correct (ops : FP32.Arithmetic) (data : TensorData layerLayout layerLaunch.rows)
    (env : Environment) (args : Argument → Word64) (t : Thread layerLaunch)
    (initial : State Register) (hentry : KernelEntry layerLaunch layerLayout env args data initial) :
    ∃ final, run (kernelProgram layerLaunch layerLayout) t.identity
        (kernelSteps layerLaunch layerLayout) initial env (some ops) = some final ∧
      KernelResult layerLaunch layerLayout ops data (convertedBuffers env args) t initial final :=
  kernel_correct layerLaunch layerLayout (by decide) ops data env args t initial hentry

end ReluLayer.PTX

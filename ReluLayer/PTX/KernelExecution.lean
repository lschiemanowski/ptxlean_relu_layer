import ReluLayer.PTX.BaseLoading
import ReluLayer.PTX.BodyExecution
import ReluLayer.PTX.ProgramComposition

namespace ReluLayer.PTX

/-- The three setup blocks retain their independent instruction lists. -/
def kernelSetup (cfg : Launch) (layout : Layout) : List (Instruction Register) :=
  baseLoading ++ rowSelection cfg.threads ++ addressCalculation layout.inputWidth

/-- Construct the computational block with its actual entry offset and branch target. -/
def kernelProgram (cfg : Launch) (layout : Layout) : List (Instruction Register) :=
  kernelSetup cfg layout ++
    bodyProgram layout.inputWidth (kernelSetup cfg layout).length

def kernelSteps (cfg : Launch) (layout : Layout) : Nat :=
  (kernelSetup cfg layout).length + bodySteps layout.inputWidth

/-- Supplied parameters, successful address conversions and initialized input words. -/
structure KernelEntry (cfg : Launch) (layout : Layout) (env : Environment)
    (args : Argument → Word64) (buffers : Buffers) (data : TensorData layout cfg.rows)
    (initial : State Register) : Prop where
  pc : initial.pc = 0
  active : initial.returned = false
  parameters : ∀ arg, env.params arg.symbol = some (args arg)
  conversions : ∀ arg, env.toGlobal (args arg) = some (buffers arg)
  inputs : InputsStored layout cfg.rows buffers data initial.globalMemory

/-- The final stored result, with internal register/PC bookkeeping left to block contracts. -/
structure KernelResult (cfg : Launch) (layout : Layout) (ops : FP32.Arithmetic)
    (data : TensorData layout cfg.rows) (buffers : Buffers) (t : Thread cfg)
    (initial final : State Register) : Prop where
  returned : final.returned = true
  result : final.regs .result = rowResult ops data t.rowIndex
  memory : OutputStored ops data buffers t.rowIndex initial.globalMemory final.globalMemory

private theorem wideProduct_row (row : Word32) (stride : Nat) (h : stride < 2 ^ 32) :
    wideProduct row (BitVec.ofNat 32 stride) = BitVec.ofNat 64 (row.toNat * stride) := by
  simp [wideProduct, Nat.mod_eq_of_lt h]

/-- Each launched thread executes all four blocks and stores its specified row result. -/
theorem kernel_correct (cfg : Launch) (layout : Layout) (hfit : cfg.rows ≤ 2 ^ 32)
    (ctx : ExecutionContext) (data : TensorData layout cfg.rows)
    (args : Argument → Word64) (buffers : Buffers) (t : Thread cfg)
    (hid : ctx.id = t.identity)
    (initial : State Register) (hentry : KernelEntry cfg layout ctx.env args buffers data initial) :
    ∃ final, run (kernelProgram cfg layout) ctx (kernelSteps cfg layout) initial = some final ∧
      KernelResult cfg layout ctx.ops data buffers t initial final := by
  have hbaseCode : CodeAt (kernelProgram cfg layout) initial.pc baseLoading := by
    rw [hentry.pc]
    simpa only [kernelProgram, kernelSetup, List.append_assoc, List.nil_append, List.length_nil] using
      codeAt_append [] baseLoading (rowSelection cfg.threads ++ addressCalculation layout.inputWidth ++
        bodyProgram layout.inputWidth (kernelSetup cfg layout).length)
  obtain ⟨bases, hbase, hb⟩ := baseLoading_correct (kernelProgram cfg layout) ctx args buffers initial
    hentry.active hentry.parameters hentry.conversions hbaseCode
  have hrowCode : CodeAt (kernelProgram cfg layout) bases.pc (rowSelection cfg.threads) := by
    rw [hb.pc, hentry.pc, Nat.zero_add]
    simpa only [kernelProgram, kernelSetup, List.append_assoc] using
      codeAt_append baseLoading (rowSelection cfg.threads)
        (addressCalculation layout.inputWidth ++
          bodyProgram layout.inputWidth (kernelSetup cfg layout).length)
  obtain ⟨rowFinal, hrow, hrr⟩ :=
    rowSelection_correct (kernelProgram cfg layout) cfg hfit t bases
      (hb.active.trans hentry.active) ctx hid hrowCode
  have hractive := hrr.active.trans (hb.active.trans hentry.active)
  have haddressCode : CodeAt (kernelProgram cfg layout) rowFinal.pc
      (addressCalculation layout.inputWidth) := by
    rw [hrr.pc, hb.pc, hentry.pc, Nat.zero_add]
    simpa only [kernelProgram, kernelSetup, List.length_append] using
      codeAt_append (baseLoading ++ rowSelection cfg.threads) (addressCalculation layout.inputWidth)
        (bodyProgram layout.inputWidth (kernelSetup cfg layout).length)
  obtain ⟨addressFinal, haddress, ha⟩ := addressCalculation_exec (kernelProgram cfg layout)
    layout.inputWidth ctx rowFinal hractive haddressCode
  have hmemory : addressFinal.globalMemory = initial.globalMemory :=
    ha.memory.trans (hrr.memory.trans hb.memory)
  have hselected : (rowFinal.regs .row).toNat = t.row := hrr.row
  have hweights : rowFinal.regs64 .weightBase = buffers .weights :=
    congrFun hrr.regs64 .weightBase |>.trans (hb.bases .weights)
  have hbias : rowFinal.regs64 .biasBase = buffers .bias :=
    congrFun hrr.regs64 .biasBase |>.trans (hb.bases .bias)
  have houtput : rowFinal.regs64 .outputBase = buffers .output :=
    congrFun hrr.regs64 .outputBase |>.trans (hb.bases .output)
  have hinput : rowFinal.regs64 .inputRead = buffers .input :=
    congrFun hrr.regs64 .inputRead |>.trans (hb.bases .input)
  have hbodyEntry : BodyEntry layout cfg.rows buffers data
      t.rowIndex addressFinal := by
    refine ⟨?_, ha.active.trans hractive, ?_, ?_, ?_, ?_⟩
    · simpa only [hmemory] using hentry.inputs
    · rw [ha.weights, hweights, wideProduct_row _ _ layout.stride_fits, hselected]
      simp [weightAddress, Layout.rowStride]
    · exact ha.inputRead.trans hinput
    · rw [ha.bias, hbias, wideProduct_row _ _ (by decide : fp32Bytes < 2 ^ 32), hselected]
      rfl
    · rw [ha.output, houtput, wideProduct_row _ _ (by decide : fp32Bytes < 2 ^ 32), hselected]
      rfl
  have hbodyCode : CodeAt (kernelProgram cfg layout) addressFinal.pc
      (bodyProgram layout.inputWidth addressFinal.pc) := by
    have hpc : addressFinal.pc = (kernelSetup cfg layout).length := by
      simpa only [kernelSetup, List.length_append, hrr.pc, hb.pc, hentry.pc, Nat.zero_add] using ha.pc
    rw [hpc]
    simpa only [kernelProgram, List.append_nil] using
      codeAt_append (kernelSetup cfg layout)
        (bodyProgram layout.inputWidth (kernelSetup cfg layout).length) []
  obtain ⟨bodyFinal, hbody, hbr⟩ := body_correct (kernelProgram cfg layout) ctx data buffers
    t.rowIndex addressFinal hbodyEntry hbodyCode
  refine ⟨bodyFinal, ?_, ?_⟩
  · have hsteps : kernelSteps cfg layout = baseLoading.length +
        ((rowSelection cfg.threads).length +
          ((addressCalculation layout.inputWidth).length + bodySteps layout.inputWidth)) := by
      simp [kernelSteps, kernelSetup, baseLoading, rowSelection, addressCalculation]; omega
    rw [hsteps, run_add, hbase, Option.bind_some, run_add, hrow, Option.bind_some,
      run_add, haddress, Option.bind_some]
    exact hbody
  · constructor
    · exact hbr.returned
    · exact hbr.result
    · simpa only [hmemory] using hbr.memory

end ReluLayer.PTX

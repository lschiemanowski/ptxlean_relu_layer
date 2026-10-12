import ReluLayer.PTX.BaseLoading
import ReluLayer.PTX.BodyExecution
import ReluLayer.PTX.ProgramComposition

namespace ReluLayer.PTX

variable {Reg : Type}

/-- The three setup blocks retain their independent instruction lists. -/
def kernelSetup (cfg : Launch) (layout : Layout) : List (Instruction Register) :=
  baseLoading ++ rowSelection cfg.threads ++ addressCalculation layout.inputWidth

/-- The computational block's absolute branch targets include the setup offset. -/
def kernelProgram (cfg : Launch) (layout : Layout) : List (Instruction Register) :=
  kernelSetup cfg layout ++
    (bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length)

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

private theorem shift_reset (s : State Reg) (offset : Nat) (h : s.pc = offset) :
    State.shiftPC offset { s with pc := 0 } = s := by
  cases s; simp_all [State.shiftPC]

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
  obtain ⟨bases, hbase, hb⟩ := baseLoading_correct ctx args buffers initial
    hentry.pc hentry.active hentry.parameters hentry.conversions
  let rowInitial := { bases with pc := 0 }
  obtain ⟨rowFinal, hrow, hrr⟩ :=
    rowSelection_correct cfg hfit t rowInitial rfl
      (hb.active.trans hentry.active) ctx hid
  have hractive := hrr.active.trans (hb.active.trans hentry.active)
  let addressInitial := { rowFinal with pc := 0 }
  obtain ⟨addressFinal, haddress, ha⟩ := addressCalculation_exec layout.inputWidth
    ctx addressInitial rfl hractive
  let bodyInitial := { addressFinal with pc := 0 }
  have hmemory : bodyInitial.globalMemory = initial.globalMemory :=
    ha.memory.trans (hrr.memory.trans hb.memory)
  have hselected : (addressInitial.regs .row).toNat = t.row := hrr.row
  have hweights : addressInitial.regs64 .weightBase = buffers .weights :=
    congrFun hrr.regs64 .weightBase |>.trans (hb.bases .weights)
  have hbias : addressInitial.regs64 .biasBase = buffers .bias :=
    congrFun hrr.regs64 .biasBase |>.trans (hb.bases .bias)
  have houtput : addressInitial.regs64 .outputBase = buffers .output :=
    congrFun hrr.regs64 .outputBase |>.trans (hb.bases .output)
  have hinput : addressInitial.regs64 .inputRead = buffers .input :=
    congrFun hrr.regs64 .inputRead |>.trans (hb.bases .input)
  have hbodyEntry : BodyEntry layout cfg.rows buffers data
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
  obtain ⟨bodyFinal, hbody, hbr⟩ := body_correct ctx data buffers
    t.rowIndex bodyInitial hbodyEntry
  -- Lift each existing block execution to its actual PC in the concatenated kernel.
  have hrowAt := run_relocate baseLoading (rowSelection cfg.threads) ctx (rowSelection
      cfg.threads).length rowInitial rowFinal hrow
  rw [shift_reset bases baseLoading.length (by simpa [baseLoading] using hb.pc)] at hrowAt
  have hrowFull := run_append_of_run (baseLoading ++ rowSelection cfg.threads)
      (addressCalculation layout.inputWidth ++
      (bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length))
          ctx (rowSelection cfg.threads).length bases (rowFinal.shiftPC baseLoading.length) hrowAt
  have haddressAt := run_relocate (baseLoading ++ rowSelection cfg.threads) (addressCalculation
      layout.inputWidth) ctx (addressCalculation layout.inputWidth).length addressInitial
      addressFinal haddress
  have haddressStart : addressInitial.shiftPC (baseLoading ++ rowSelection cfg.threads).length =
      rowFinal.shiftPC baseLoading.length := by
    simp [addressInitial, State.shiftPC, baseLoading, rowSelection, hrr.pc]
  rw [haddressStart] at haddressAt
  have haddressFull := run_append_of_run (kernelSetup cfg layout) ((bodyProgram
      layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length)) ctx
      (addressCalculation layout.inputWidth).length (rowFinal.shiftPC baseLoading.length)
      (addressFinal.shiftPC (baseLoading ++ rowSelection cfg.threads).length) haddressAt
  change run (kernelProgram cfg layout) ctx (addressCalculation layout.inputWidth).length
      (rowFinal.shiftPC baseLoading.length) = _ at haddressFull
  have hbodyAt := run_relocate (kernelSetup cfg layout) (bodyProgram layout.inputWidth) ctx
      (bodySteps layout.inputWidth) bodyInitial bodyFinal hbody
  have hbodyStart : bodyInitial.shiftPC (kernelSetup cfg layout).length =
      addressFinal.shiftPC (baseLoading ++ rowSelection cfg.threads).length := by
    simp [bodyInitial, State.shiftPC, kernelSetup, baseLoading, rowSelection,
      addressCalculation, ha.pc]
  rw [hbodyStart] at hbodyAt
  have hbaseFull := run_append_of_run baseLoading (rowSelection cfg.threads ++ addressCalculation
      layout.inputWidth ++
      (bodyProgram layout.inputWidth).map (Instruction.relocate (kernelSetup cfg layout).length))
          ctx baseLoading.length initial bases hbase
  have hbfull : run (kernelProgram cfg layout) ctx baseLoading.length initial = some bases := by
    simpa only [kernelProgram, kernelSetup, List.append_assoc] using hbaseFull
  have hrfull : run (kernelProgram cfg layout) ctx (rowSelection cfg.threads).length bases =
      some (rowFinal.shiftPC baseLoading.length) := by
    simpa only [kernelProgram, kernelSetup, List.append_assoc] using hrowFull
  refine ⟨bodyFinal.shiftPC (kernelSetup cfg layout).length, ?_, ?_⟩
  · have hsteps : kernelSteps cfg layout = baseLoading.length +
        ((rowSelection cfg.threads).length +
          ((addressCalculation layout.inputWidth).length + bodySteps layout.inputWidth)) := by
      simp [kernelSteps, kernelSetup, baseLoading, rowSelection, addressCalculation]; omega
    rw [hsteps, run_add, hbfull, Option.bind_some, run_add, hrfull, Option.bind_some,
      run_add, haddressFull, Option.bind_some]
    exact hbodyAt
  · constructor
    · exact hbr.returned
    · exact hbr.result
    · simpa only [State.shiftPC, hmemory] using hbr.memory

end ReluLayer.PTX

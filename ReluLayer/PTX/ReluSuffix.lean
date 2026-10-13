import ReluLayer.PTX.BodyContract

namespace ReluLayer.PTX

variable {layout : Layout} {rows : Nat}

/-- Exactly PTX lines 58--62; these instructions contain no branch targets. -/
def reluSuffix : List (Instruction Register) :=
  [ .loadF32 .biasValue (.reg .biasRead),
    .addF32RN .accumulator (.reg .accumulator) (.reg .biasValue),
    .maxPositiveZero .result (.reg .accumulator),
    .storeF32 (.reg .outputWrite) (.reg .result),
    .ret ]

/-- The suffix consumes a completed accumulator and the selected row's bias/output pointers. -/
structure SuffixEntry (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register) : Prop where
  active : initial.returned = false
  accumulator : initial.regs .accumulator = prefixAccumulator ops data i layout.inputWidth
  biasRead : initial.regs64 .biasRead = elementAddress buffers .bias i.val
  biasStored : initial.globalMemory (elementAddress buffers .bias i.val) = some (data.bias i)
  outputWrite : initial.regs64 .outputWrite = elementAddress buffers .output i.val

/-- Exact bias/add/ReLU/store/return effect on the incoming state. -/
def reluStored (ops : FP32.Arithmetic) (bias : Word32)
    (initial : State Register) : State Register :=
  let accumulator := ops.addRN (initial.regs .accumulator) bias
  let result := FP32.maxPositiveZero accumulator
  { initial with
    regs := fun r => match r with
      | .biasValue => bias | .accumulator => accumulator | .result => result
      | _ => initial.regs r
    globalMemory := fun a => if a = initial.regs64 .outputWrite then some result
      else initial.globalMemory a
    pc := initial.pc + reluSuffix.length
    returned := true }

/-- A completed accumulator makes the result register contain the ordered row result. -/
theorem reluStored_result (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : SuffixEntry ops data buffers i initial) :
    (reluStored ops (data.bias i) initial).regs .result = rowResult ops data i := by
  simp [reluStored, hentry.accumulator, rowResult, preactivation]

/-- Store the specified output and preserve every other word address. -/
theorem reluStored_output (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : SuffixEntry ops data buffers i initial) :
    OutputStored ops data buffers i initial.globalMemory
      (reluStored ops (data.bias i) initial).globalMemory := by
  constructor
  · simp [reluStored, hentry.outputWrite, hentry.accumulator, rowResult, preactivation]
  · intro address h
    simp [reluStored, hentry.outputWrite, h]

/-- Execute the suffix at the current PC of an enclosing program. -/
theorem reluSuffix_correct (program : List (Instruction Register)) (ctx : ExecutionContext)
    (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : SuffixEntry ctx.ops data buffers i initial)
    (hcode : CodeAt program initial.pc reluSuffix) :
    run program ctx reluSuffix.length initial =
      some (reluStored ctx.ops (data.bias i) initial) := by
  have hb : initial.globalMemory (initial.regs64 .biasRead) = some (data.bias i) := by
    rw [hentry.biasRead]; exact hentry.biasStored
  have h0 := hcode 0 (by simp [reluSuffix])
  have h1 := hcode 1 (by simp [reluSuffix])
  have h2 := hcode 2 (by simp [reluSuffix])
  have h3 := hcode 3 (by simp [reluSuffix])
  have h4 := hcode 4 (by simp [reluSuffix])
  simp [reluSuffix] at h0 h1 h2 h3 h4
  simp [run, step, reluSuffix, h0, h1, h2, h3, h4, Instruction.execute, Operand.read,
    Operand64.read, State.writeNext, reluStored, hentry.active, hb, Nat.add_assoc]
  funext r
  cases r <;> rfl

end ReluLayer.PTX

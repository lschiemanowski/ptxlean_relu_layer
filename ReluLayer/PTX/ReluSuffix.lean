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

private def finalState (ops : FP32.Arithmetic) (s : State Register) (bias : Word32) : State
    Register :=
  let s := s.writeNext .biasValue bias
  let s := s.writeNext .accumulator (ops.addRN (s.regs .accumulator) (s.regs .biasValue))
  let s := s.writeNext .result (FP32.maxPositiveZero (s.regs .accumulator))
  let s : State Register := { s with
    globalMemory := fun a => if a = s.regs64 .outputWrite then some (s.regs .result) else
        s.globalMemory a
    pc := s.pc + 1 }
  { s with returned := true, pc := s.pc + 1 }

/-- Return with the specified stored result and all unrelated memory/register values preserved. -/
structure SuffixResult (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial final : State Register) : Prop where
  pc : final.pc = initial.pc + reluSuffix.length
  returned : final.returned = true
  bias : final.regs .biasValue = data.bias i
  accumulator : final.regs .accumulator =
    ops.addRN (prefixAccumulator ops data i layout.inputWidth) (data.bias i)
  result : final.regs .result = rowResult ops data i
  memory : OutputStored ops data buffers i initial.globalMemory final.globalMemory
  other32 : ∀ r, r ≠ .biasValue → r ≠ .accumulator → r ≠ .result → final.regs r = initial.regs r
  regs64 : final.regs64 = initial.regs64
  predicates : final.predicates = initial.predicates

/-- Execute the suffix at the current PC of an enclosing program. -/
theorem reluSuffix_correct (program : List (Instruction Register)) (ctx : ExecutionContext)
    (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : SuffixEntry ctx.ops data buffers i initial)
    (hcode : CodeAt program initial.pc reluSuffix) :
    ∃ final, run program ctx reluSuffix.length initial = some final ∧
      SuffixResult ctx.ops data buffers i initial final := by
  have hb : initial.globalMemory (initial.regs64 .biasRead) = some (data.bias i) := by
    rw [hentry.biasRead]; exact hentry.biasStored
  have h0 := hcode 0 (by simp [reluSuffix])
  have h1 := hcode 1 (by simp [reluSuffix])
  have h2 := hcode 2 (by simp [reluSuffix])
  have h3 := hcode 3 (by simp [reluSuffix])
  have h4 := hcode 4 (by simp [reluSuffix])
  simp [reluSuffix] at h0 h1 h2 h3 h4
  refine ⟨finalState ctx.ops initial (data.bias i), ?_, ?_⟩
  · simp [run, step, reluSuffix, h0, h1, h2, h3, h4, Instruction.execute, Operand.read,
      Operand64.read, State.writeNext, finalState, hentry.active, hb, Nat.add_assoc]
  · constructor
    · simp [finalState, State.writeNext, reluSuffix, Nat.add_assoc]
    · rfl
    · simp [finalState, State.writeNext]
    · simp [finalState, State.writeNext, hentry.accumulator]
    · simp [finalState, State.writeNext, rowResult, preactivation, hentry.accumulator]
    · constructor
      · simp [finalState, State.writeNext, hentry.outputWrite, hentry.accumulator, rowResult,
          preactivation]
      · intro address h
        simp [finalState, State.writeNext, hentry.outputWrite, h]
    · intro r hb ha hr; simp [finalState, State.writeNext, hb, ha, hr]
    · rfl
    · rfl

end ReluLayer.PTX

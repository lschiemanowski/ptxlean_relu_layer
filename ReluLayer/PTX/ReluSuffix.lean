import ReluLayer.PTX.BodyContract

namespace ReluLayer.PTX

variable {layout : Layout} {rows : Nat}

/-- Exactly PTX lines 58--62, with a fragment-relative entry PC of zero. -/
def reluSuffix : List (Instruction Register) :=
  [ .loadF32 .biasValue (.reg .biasRead),
    .addF32RN .accumulator (.reg .accumulator) (.reg .biasValue),
    .maxPositiveZero .result (.reg .accumulator),
    .storeF32 (.reg .outputWrite) (.reg .result),
    .ret ]

/-- The suffix consumes a completed accumulator and the selected row's bias/output pointers. -/
structure SuffixEntry (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register) (start : Nat := 0) : Prop where
  pc : initial.pc = start
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
    (buffers : Buffers) (i : Fin rows) (initial final : State Register) (start : Nat := 0) : Prop
        where
  pc : final.pc = start + reluSuffix.length
  returned : final.returned = true
  bias : final.regs .biasValue = data.bias i
  accumulator : final.regs .accumulator =
    ops.addRN (prefixAccumulator ops data i layout.inputWidth) (data.bias i)
  result : final.regs .result = rowResult ops data i
  memory : OutputStored ops data buffers i initial.globalMemory final.globalMemory
  other32 : ∀ r, r ≠ .biasValue → r ≠ .accumulator → r ≠ .result → final.regs r = initial.regs r
  regs64 : final.regs64 = initial.regs64
  predicates : final.predicates = initial.predicates

/-- The same suffix theorem at its actual offset after any instruction prefix. -/
theorem reluSuffix_correct_at (prelude : List (Instruction Register)) (ctx : ExecutionContext)
    (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : SuffixEntry ctx.ops data buffers i initial prelude.length) :
    ∃ final, run (prelude ++ reluSuffix) ctx reluSuffix.length initial = some final ∧
      SuffixResult ctx.ops data buffers i initial final prelude.length := by
  have hb : initial.globalMemory (initial.regs64 .biasRead) = some (data.bias i) := by
    rw [hentry.biasRead]; exact hentry.biasStored
  have hoff (j : Nat) : ¬ prelude.length + j < prelude.length := by omega
  refine ⟨finalState ctx.ops initial (data.bias i), ?_, ?_⟩
  · simp [run, step, reluSuffix, Instruction.execute, Operand.read, Operand64.read,
      State.writeNext, finalState, hentry.pc, hentry.active, hb, List.getElem?_append,
          Nat.add_assoc, hoff]
  · constructor
    · simp [finalState, State.writeNext, reluSuffix, hentry.pc, Nat.add_assoc]
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

/-- Execute the suffix independently at fragment-relative PC zero. -/
theorem reluSuffix_correct (ctx : ExecutionContext) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : SuffixEntry ctx.ops data buffers i initial) :
    ∃ final, run reluSuffix ctx reluSuffix.length initial = some final ∧
      SuffixResult ctx.ops data buffers i initial final := by
  simpa using reluSuffix_correct_at [] ctx data buffers i initial hentry

end ReluLayer.PTX

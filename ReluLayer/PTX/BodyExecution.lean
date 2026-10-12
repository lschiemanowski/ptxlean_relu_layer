import ReluLayer.PTX.DotLoop
import ReluLayer.PTX.ReluSuffix

namespace ReluLayer.PTX

/-- Block four: initialized dot loop followed by bias/ReLU/store/return. -/
def bodyProgram (width : Nat) : List (Instruction Register) := dotLoop width ++ reluSuffix

def bodySteps (width : Nat) : Nat := dotLoopStart + dotIterationSteps * width + reluSuffix.length

/-- The complete computational block returns with the specified output and preserved frame. -/
structure BodyResult (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial final : State Register) : Prop where
  pc : final.pc = (bodyProgram layout.inputWidth).length
  returned : final.returned = true
  column : final.regs .column = BitVec.ofNat 32 layout.inputWidth
  result : final.regs .result = rowResult ops data i
  memory : OutputStored ops data buffers i initial.globalMemory final.globalMemory
  other32 : ∀ r, r ≠ .column → r ≠ .weightValue → r ≠ .inputValue →
    r ≠ .accumulator → r ≠ .biasValue → r ≠ .result → final.regs r = initial.regs r
  other64 : ∀ r, r ≠ .weightRead → r ≠ .inputRead → final.regs64 r = initial.regs64 r
  otherPredicates : ∀ r, r ≠ .more → final.predicates r = initial.predicates r

/-- Actual concatenated-program execution, reusing the loop and suffix theorems. -/
theorem body_correct (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register) (id : Identity)
    (env : Environment) (hentry : BodyEntry layout rows buffers data i initial) :
    ∃ final, run (bodyProgram layout.inputWidth) id (bodySteps layout.inputWidth)
        initial env (some ops) = some final ∧ BodyResult ops data buffers i initial final := by
  obtain ⟨middle, hloop, hinv⟩ :=
    dotLoop_correct ops data buffers i initial id env hentry
  have hb : middle.regs64 .biasRead = elementAddress buffers .bias i.val :=
    (hinv.other64 .biasRead (by decide) (by decide)).trans hentry.biasRead
  have hy : middle.regs64 .outputWrite = elementAddress buffers .output i.val :=
    (hinv.other64 .outputWrite (by decide) (by decide)).trans hentry.outputWrite
  have hsuffix : SuffixEntry ops data buffers i middle (dotLoop layout.inputWidth).length := by
    refine ⟨?_, hinv.active, hinv.accumulator, hb, ?_, hy⟩
    · simpa [dotLoopExit, dotLoop] using hinv.pc
    · rw [hinv.memory]; exact hentry.inputs.bias i
  obtain ⟨final, hs, hr⟩ := reluSuffix_correct_at (dotLoop layout.inputWidth)
    ops data buffers i middle id env hsuffix
  have hl := run_append_of_run (dotLoop layout.inputWidth) reluSuffix id
    (dotLoopStart + dotIterationSteps * layout.inputWidth) initial middle env (some ops) hloop
  refine ⟨final, ?_, ?_⟩
  · unfold bodyProgram bodySteps
    rw [run_add, hl]
    exact hs
  · constructor
    · simpa [bodyProgram] using hr.pc
    · exact hr.returned
    · exact (hr.other32 .column (by decide) (by decide) (by decide)).trans hinv.column
    · exact hr.result
    · simpa only [hinv.memory] using hr.memory
    · intro r hc hw hx ha hb hresult
      exact (hr.other32 r hb ha hresult).trans (hinv.other32 r hc hw hx ha)
    · intro r hw hx
      rw [hr.regs64]; exact hinv.other64 r hw hx
    · intro r hmore
      rw [hr.predicates]; exact hinv.otherPredicates r hmore

end ReluLayer.PTX

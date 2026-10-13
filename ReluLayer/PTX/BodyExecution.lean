import ReluLayer.PTX.DotLoop
import ReluLayer.PTX.ReluSuffix

namespace ReluLayer.PTX

variable {layout : Layout} {rows : Nat}

/-- Block four: initialized dot loop followed by bias/ReLU/store/return. -/
def bodyProgram (width start : Nat) : List (Instruction Register) :=
  dotLoop width start ++ reluSuffix

def bodySteps (width : Nat) : Nat := dotLoopStart + dotIterationSteps * width + reluSuffix.length

/-- The complete computational block returns with the specified output and preserved frame. -/
structure BodyResult (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial final : State Register) : Prop where
  returned : final.returned = true
  result : final.regs .result = rowResult ops data i
  memory : OutputStored ops data buffers i initial.globalMemory final.globalMemory

/-- Actual concatenated-program execution, reusing the loop and suffix theorems. -/
theorem body_correct (program : List (Instruction Register)) (ctx : ExecutionContext) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : BodyEntry layout rows buffers data i initial)
    (hcode : CodeAt program initial.pc (bodyProgram layout.inputWidth initial.pc)) :
    ∃ final, run program ctx (bodySteps layout.inputWidth) initial = some
        final ∧ BodyResult ctx.ops data buffers i initial final := by
  obtain ⟨middle, hloop, hinv⟩ :=
    dotLoop_correct program ctx data buffers i initial hentry hcode.left
  have hb : middle.regs64 .biasRead = elementAddress buffers .bias i.val :=
    (hinv.other64 .biasRead (by decide) (by decide)).trans hentry.biasRead
  have hy : middle.regs64 .outputWrite = elementAddress buffers .output i.val :=
    (hinv.other64 .outputWrite (by decide) (by decide)).trans hentry.outputWrite
  have hsuffix : SuffixEntry ctx.ops data buffers i middle := by
    refine ⟨hinv.active, hinv.accumulator, hb, ?_, hy⟩
    rw [hinv.memory]; exact hentry.inputs.bias i
  have hpc : middle.pc = initial.pc + (dotLoop layout.inputWidth initial.pc).length := by
    simpa [dotLoopExit, dotLoop] using hinv.pc
  have hsuffixCode : CodeAt program middle.pc reluSuffix := by
    rw [hpc]
    exact hcode.right
  obtain ⟨final, hs, hr⟩ := reluSuffix_correct program ctx data buffers i middle hsuffix hsuffixCode
  refine ⟨final, ?_, ?_⟩
  · unfold bodySteps
    rw [run_add, hloop]
    exact hs
  · constructor
    · exact hr.returned
    · exact hr.result
    · simpa only [hinv.memory] using hr.memory

end ReluLayer.PTX

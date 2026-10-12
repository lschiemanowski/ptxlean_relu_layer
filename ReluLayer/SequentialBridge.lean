import ReluLayer.RealForward
import ReluLayer.PTX.RoundingError

namespace ReluLayer.PTX

open TorchLean TorchLean.Tensor

noncomputable section

/-- Real tensors use exactly the values represented by the kernel's input words. -/
def decodedWeights (data : TensorData layerLayout layerLaunch.rows) : Tensor ℝ [outputSize,
    inputSize] :=
  Tensor.dim fun i => Tensor.dim fun j => Tensor.scalar (FP32.toReal (data.weights i j))

def decodedBias (data : TensorData layerLayout layerLaunch.rows) : Tensor ℝ [outputSize] :=
  Tensor.dim fun i => Tensor.scalar (FP32.toReal (data.bias i))

def decodedInput (data : TensorData layerLayout layerLaunch.rows) : Tensor ℝ [inputSize] :=
  Tensor.dim fun j => Tensor.scalar (FP32.toReal (data.input j))

/-- The numerical target is a coordinate of the actual upstream sequential forward output. -/
theorem exactRowResult_eq_forward (data : TensorData layerLayout layerLaunch.rows)
    (i : Fin outputSize) :
    exactRowResult data i =
      getScalar (ReluLayer.forward (decodedWeights data) (decodedBias data) (decodedInput data))
          i := by
  rw [ReluLayer.forward_coordinate]
  simp [exactRowResult, decodedWeights, decodedBias, decodedInput,
    Spec.get2_eq_apply, getScalar_eq_apply, Tensor.dim,
    Tensor.Internal.Rep.stack_apply, Tensor.scalar_apply]
  rfl

/-- Concrete binary32 context for the selected kernel thread. -/
def layerContext (t : Thread layerLaunch) (env : Environment) : ExecutionContext :=
  t.context env FP32.floatLibArithmetic

/-- Return and store a finite word close to the actual Sequential forward coordinate. -/
theorem layer_kernel_forward_abs_error_le (data : TensorData layerLayout layerLaunch.rows)
    (env : Environment) (args : Argument → Word64) (buffers : Buffers) (t : Thread layerLaunch)
    (initial : State Register)
    (hentry : KernelEntry layerLaunch layerLayout env args buffers data initial)
    (hfinite : FiniteRowTrace data t.rowIndex) :
    ∃ final word,
      Returns (kernelProgram layerLaunch layerLayout) (layerContext t env) initial final ∧
      final.globalMemory (t.outputAddress buffers) = some word ∧
      (∀ address, address ≠ t.outputAddress buffers →
        final.globalMemory address = initial.globalMemory address) ∧
      FP32.isFinite word = true ∧
      |FP32.toReal word -
        getScalar (ReluLayer.forward (decodedWeights data) (decodedBias data) (decodedInput data))
          t.rowIndex| ≤ traceErrorBudget data t.rowIndex := by
  obtain ⟨final, hexec, hr, hf, he⟩ := kernel_abs_error_le layerLaunch layerLayout
    (by decide) data env args buffers t initial hentry hfinite
  refine ⟨final, final.regs .result, ⟨⟨kernelSteps layerLaunch layerLayout, hexec⟩,
    hr.returned⟩, ?_, hr.memory.frame, hf, ?_⟩
  · rw [hr.result]
    exact hr.memory.output
  · have htarget := exactRowResult_eq_forward data t.rowIndex
    calc
      _ = |FP32.toReal (final.regs .result) - exactRowResult data t.rowIndex| :=
        (congrArg (fun value => |FP32.toReal (final.regs .result) - value|) htarget).symm
      _ ≤ traceErrorBudget data t.rowIndex := he

end
end ReluLayer.PTX

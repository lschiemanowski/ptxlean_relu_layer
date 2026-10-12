import ReluLayer.PTX.Layer
import ReluLayer.SequentialBridge

open ReluLayer.PTX TorchLean TorchLean.Tensor

-- Probe tensor coordinates independently of the forward formula, including the
-- explicit parameter-state slots and the last row/column of the concrete shapes.
private def indexedData : TensorData layerLayout layerLaunch.rows :=
  { weights := fun i j => BitVec.ofNat 32 (i.val * layerLayout.inputWidth + j.val)
    bias := fun i => BitVec.ofNat 32 i.val
    input := fun j => BitVec.ofNat 32 j.val }

example : Spec.get2 (decodedWeights indexedData) 1 2 = FP32.toReal 1026 := by
  simp [decodedWeights, indexedData, Spec.get2_eq_apply, Tensor.dim, layerLayout, ReluLayer.inputSize]
example : Spec.get2 (decodedWeights indexedData) 3071 1023 = FP32.toReal 3145727 := by
  simp [decodedWeights, indexedData, Spec.get2_eq_apply, Tensor.dim, layerLayout, ReluLayer.inputSize]
example : getScalar (decodedBias indexedData) 3071 = FP32.toReal 3071 := by
  simp [decodedBias, indexedData, getScalar_eq_apply, Tensor.dim]
example : getScalar (decodedInput indexedData) 1023 = FP32.toReal 1023 := by
  simp [decodedInput, indexedData, getScalar_eq_apply, Tensor.dim]

example (data : TensorData layerLayout layerLaunch.rows) (i : Fin 3072) (j : Fin 1024) :
    Spec.get2 ((ReluLayer.parameters (decodedWeights data) (decodedBias data)).get
      ⟨0, by decide⟩) i j = FP32.toReal (data.weights i j) := by
  change Spec.get2 (decodedWeights data) i j = _
  simp [decodedWeights, Spec.get2_eq_apply, Tensor.dim]
example (data : TensorData layerLayout layerLaunch.rows) (i : Fin 3072) :
    getScalar ((ReluLayer.parameters (decodedWeights data) (decodedBias data)).get
      ⟨1, by decide⟩) i = FP32.toReal (data.bias i) := by
  change getScalar (decodedBias data) i = _
  simp [decodedBias, getScalar_eq_apply, Tensor.dim]

#print axioms exactRowResult_eq_forward
#print axioms layer_kernel_forward_abs_error_le

def main : IO Unit :=
  IO.println "Decoded tensor row/column and endpoint indices, explicit weight/bias state slots, actual Sequential target and stored-output theorem axioms pass."

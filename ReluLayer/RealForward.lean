import ReluLayer.Specification
import NN.Verification.Builtin.SpecEval
import NN.Spec.Core.Context.Real
import NN.Proofs.Tensor.Basic.Folds

namespace ReluLayer

open TorchLean TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- Interpret the actual sequential forward program on the supplied real parameters. -/
def evaluate (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) : Except String (Tensor ℝ [outputSize]) :=
  let state := parameters W b
  forwardProgram (α := ℝ) (m := NN.Verification.Builtin.SpecM)
    (state.get ⟨0, by decide⟩) (state.get ⟨1, by decide⟩) x

/-- The row-vector implementation used by the upstream linear layer. -/
private def rawAffine (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) : Tensor ℝ [outputSize] :=
  reshapeSpec
    (addSpec
      (matmulSpec (batchA := []) (batchB := []) (batch := [])
        (m := 1) (n := inputSize) (p := outputSize)
        Spec.Shape.BroadcastTo.proof Spec.Shape.BroadcastTo.proof
        (reshapeSpec x (by decide)) (swapAdjacentAxes W 0))
      (broadcastTo (s₁ := [outputSize]) (s₂ := [1, outputSize]) Spec.Shape.BroadcastTo.proof b))
    (by decide)

private theorem reshape_to_row {n : Nat} (v : Tensor ℝ [n]) :
    reshapeSpec (target := [1, n]) v (by simp [Spec.Shape.size]) =
      Tensor.dim (fun _ => v) := by
  exact TorchLean.Tensor.Internal.Rep.reshape_one_cons _ v

private theorem reshape_from_row {n : Nat} (v : Tensor ℝ [1, n]) :
    reshapeSpec (target := [n]) v (by simp [Spec.Shape.size]) =
      Tensor.unstack v 0 := by
  have h : v = Tensor.dim (fun (_ : Fin 1) => Tensor.unstack v 0) := by
    rw [← Tensor.dim_unstack v]
    congr 1
    funext i
    simpa only [Tensor.unstack_dim] using
      congrArg (Tensor.unstack v) (Subsingleton.elim i 0)
  conv_lhs => rw [h, ← reshape_to_row]
  exact reshapeSpec_roundtrip _ _

private theorem rawAffine_eq (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) : rawAffine W b x = addSpec (matVecMulSpec W x) b := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [rawAffine, reshape_from_row, matmulSpec, broadcastTo_self,
    LinearAlgebra.Internal.matmulCommonBatchSpec_scalar, reshape_to_row]
  rw [broadcastTo_expand (by decide)]
  simp only [broadcastTo_self]
  rcases coordinate with ⟨i, ⟨⟩⟩
  simp only [addSpec, map2Spec, Tensor.unstack,
    Tensor.Internal.Rep.unstack_apply, Tensor.Internal.Rep.zipWith_apply,
    Tensor.dim, Tensor.Internal.Rep.stack_apply]
  simp only [matMulSpec, matVecMulSpec, Tensor.Internal.Rep.get_ofFn,
    Spec.get2_eq_apply, getScalar_eq_apply, Tensor.Internal.Rep.stack_apply,
    swapAdjacentAxes_apply, Internal.swapAdjacentAxesCoordinate, mul_comm]

/-- The actual sequential evaluation always succeeds and computes the tensor formula. -/
theorem evaluate_eq (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) :
    evaluate W b x = .ok (Activation.reluSpec (addSpec (matVecMulSpec W x) b)) := by
  calc
    evaluate W b x = .ok (Activation.reluSpec (rawAffine W b x)) := by rfl
    _ = _ := by rw [rawAffine_eq]

/-- Extract the real output using the proof that sequential evaluation succeeds. -/
def forward (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) : Tensor ℝ [outputSize] :=
  (evaluate W b x).toOption.get (by simp [evaluate_eq, Except.toOption])

theorem forward_eq (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) :
    forward W b x = Activation.reluSpec (addSpec (matVecMulSpec W x) b) := by
  simp [forward, evaluate_eq, Except.toOption]

/-- Each coordinate of the sequential model is exactly the dense ReLU formula. -/
theorem forward_coordinate (W : Tensor ℝ [outputSize, inputSize]) (b : Tensor ℝ [outputSize])
    (x : Tensor ℝ [inputSize]) (i : Fin outputSize) :
    getScalar (forward W b x) i =
      max ((∑ j : Fin inputSize, Spec.get2 W i j * getScalar x j) + getScalar b i) 0 := by
  rw [forward_eq]
  simp [Activation.reluSpec, mapSpec, Spec.getScalar_add_spec,
    Spec.getScalar_mat_vec_mul_spec]

end
end ReluLayer

import ReluLayer.PTX.KernelExecution
import ReluLayer.PTX.Layer
import ReluLayer.PTX.RealDecode
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error

namespace ReluLayer.PTX

variable {layout : Layout} {rows : Nat}

open FloatLib.Floats.Formats.BinaryInterchange
open scoped BigOperators

/-- Exact sum of the first k represented products, with no input-conversion error. -/
private noncomputable def exactPrefix (data : TensorData layout rows) (i : Fin rows) (k : Nat) :
    ℝ :=
  (((List.ofFn (fun j => (data.weights i j, data.input j))).take k).map (fun pair => FP32.toReal
      pair.1 * FP32.toReal pair.2)).sum

/-- Exact real argument rounded at this FMA site, using the actual incoming accumulator. -/
noncomputable def fmaSite (data : TensorData layout rows) (i : Fin rows)
    (j : Fin layout.inputWidth) : ℝ :=
  FP32.toReal (data.weights i j) * FP32.toReal (data.input j) +
    FP32.toReal (prefixAccumulator FP32.floatLibArithmetic data i j.val)

/-- The first k local half-ULP budgets, including the subnormal grid. -/
private noncomputable def prefixErrorBudget (data : TensorData layout rows) (i : Fin rows) (k :
    Nat) : ℝ :=
  ((List.ofFn (fun j => Model.epsilonAt FloatFormat.binary32 (fmaSite data i j))).take k).sum

noncomputable def traceErrorBudget (data : TensorData layout rows) (i : Fin rows) : ℝ :=
  (∑ j : Fin layout.inputWidth, Model.epsilonAt FloatFormat.binary32 (fmaSite data i j)) +
    Model.epsilonAt FloatFormat.binary32
      (FP32.toReal (prefixAccumulator FP32.floatLibArithmetic data i layout.inputWidth) +
        FP32.toReal (data.bias i))

noncomputable def exactRowResult (data : TensorData layout rows) (i : Fin rows) : ℝ :=
  max ((∑ j : Fin layout.inputWidth,
    FP32.toReal (data.weights i j) * FP32.toReal (data.input j)) + FP32.toReal (data.bias i)) 0

/-- Finite inputs alone do not prevent overflow: every executed rounded result is required finite. -/
structure FiniteRowTrace (data : TensorData layout rows) (i : Fin rows) : Prop where
  weights : ∀ j, FP32.isFinite (data.weights i j) = true
  input : ∀ j, FP32.isFinite (data.input j) = true
  bias : FP32.isFinite (data.bias i) = true
  accumulators : ∀ k : Fin (layout.inputWidth + 1),
    FP32.isFinite (prefixAccumulator FP32.floatLibArithmetic data i k.val) = true
  preactivation : FP32.isFinite (preactivation FP32.floatLibArithmetic data i) = true

private theorem exactPrefix_succ (data : TensorData layout rows) (i : Fin rows)
    (k : Nat) (hk : k < layout.inputWidth) :
    exactPrefix data i (k + 1) = exactPrefix data i k +
      FP32.toReal (data.weights i ⟨k, hk⟩) * FP32.toReal (data.input ⟨k, hk⟩) := by
  have hlen : k < (List.ofFn (fun j => (data.weights i j, data.input j))).length := by simpa 
      using hk
  unfold exactPrefix
  rw [← List.take_concat_get hlen, List.concat_eq_append, List.map_append, List.sum_append]
  simp 

private theorem prefixErrorBudget_succ (data : TensorData layout rows) (i : Fin rows)
    (k : Nat) (hk : k < layout.inputWidth) :
    prefixErrorBudget data i (k + 1) = prefixErrorBudget data i k +
      Model.epsilonAt FloatFormat.binary32 (fmaSite data i ⟨k, hk⟩) := by
  have hlen : k < (List.ofFn (fun j => Model.epsilonAt FloatFormat.binary32 (fmaSite data i
      j))).length := by
    simpa using hk
  unfold prefixErrorBudget
  rw [← List.take_concat_get hlen, List.concat_eq_append, List.sum_append]
  simp

private theorem prefixErrorBudget_full (data : TensorData layout rows) (i : Fin rows) :
    prefixErrorBudget data i layout.inputWidth =
      ∑ j : Fin layout.inputWidth, Model.epsilonAt FloatFormat.binary32 (fmaSite data i j) := by
  unfold prefixErrorBudget
  rw [List.take_of_length_le (by simp), List.sum_ofFn]

private theorem fma_abs_error_le (w x a : FP32.Bits)
    (hw : FP32.isFinite w = true) (hx : FP32.isFinite x = true) (ha : FP32.isFinite a = true)
    (hout : FP32.isFinite (FP32.floatLibArithmetic.fmaRN w x a) = true) :
    |FP32.toReal (FP32.floatLibArithmetic.fmaRN w x a) -
        (FP32.toReal w * FP32.toReal x + FP32.toReal a)| ≤
      Model.epsilonAt FloatFormat.binary32 (FP32.toReal w * FP32.toReal x + FP32.toReal a) :=
  Model.abs_toReal_fma_sub_le (FP32.decode w) (FP32.decode x) (FP32.decode a)
    (by decide) hw hx ha hout

/-- The ordered rounded prefix differs from the exact prefix by at most its accumulated budget. -/
private theorem prefixAccumulator_abs_error_le (data : TensorData layout rows) (i : Fin rows)
    (hfinite : FiniteRowTrace data i) (k : Nat) (hk : k ≤ layout.inputWidth) :
    |FP32.toReal (prefixAccumulator FP32.floatLibArithmetic data i k) - exactPrefix data i k| ≤
      prefixErrorBudget data i k := by
  induction k with
  | zero => simp [exactPrefix, prefixErrorBudget]
  | succ k ih =>
    have hlt : k < layout.inputWidth := by omega
    have hprev := ih (by omega)
    have hlocal := fma_abs_error_le
      (data.weights i ⟨k, hlt⟩) (data.input ⟨k, hlt⟩)
      (prefixAccumulator FP32.floatLibArithmetic data i k)
      (hfinite.weights _) (hfinite.input _) (hfinite.accumulators ⟨k, by omega⟩)
      (by rw [← prefixAccumulator_succ _ _ _ _ hlt]; exact hfinite.accumulators ⟨k + 1, by omega⟩)
    rw [prefixAccumulator_succ _ _ _ _ hlt, exactPrefix_succ _ _ _ hlt,
      prefixErrorBudget_succ _ _ _ hlt]
    unfold fmaSite
    rw [add_comm (exactPrefix data i k)]
    have htriangle := abs_sub_le
      (FP32.toReal (FP32.floatLibArithmetic.fmaRN (data.weights i ⟨k, hlt⟩)
        (data.input ⟨k, hlt⟩) (prefixAccumulator FP32.floatLibArithmetic data i k)))
      (fmaSite data i ⟨k, hlt⟩)
      (exactPrefix data i k +
        FP32.toReal (data.weights i ⟨k, hlt⟩) * FP32.toReal (data.input ⟨k, hlt⟩))
    rw [fmaSite, add_comm (exactPrefix data i k), add_sub_add_left_eq_sub] at htriangle
    exact htriangle.trans (by linarith)

/-- Bias addition contributes one further half-ULP bound. -/
theorem preactivation_abs_error_le (data : TensorData layout rows) (i : Fin rows)
    (hfinite : FiniteRowTrace data i) :
    |FP32.toReal (preactivation FP32.floatLibArithmetic data i) -
        (exactPrefix data i layout.inputWidth + FP32.toReal (data.bias i))| ≤
      traceErrorBudget data i := by
  have hdot := prefixAccumulator_abs_error_le data i hfinite layout.inputWidth (by omega)
  have hbias := Model.abs_toReal_add_sub_le
    (FP32.decode (prefixAccumulator FP32.floatLibArithmetic data i layout.inputWidth))
    (FP32.decode (data.bias i)) (by decide)
    (hfinite.accumulators ⟨layout.inputWidth, by omega⟩) hfinite.bias hfinite.preactivation
  change |FP32.toReal (preactivation FP32.floatLibArithmetic data i) -
    (FP32.toReal (prefixAccumulator FP32.floatLibArithmetic data i layout.inputWidth) +
      FP32.toReal (data.bias i))| ≤
    Model.epsilonAt FloatFormat.binary32
      (FP32.toReal (prefixAccumulator FP32.floatLibArithmetic data i layout.inputWidth) +
        FP32.toReal (data.bias i)) at hbias
  have htriangle := abs_sub_le
    (FP32.toReal (preactivation FP32.floatLibArithmetic data i))
    (FP32.toReal (prefixAccumulator FP32.floatLibArithmetic data i layout.inputWidth) +
      FP32.toReal (data.bias i))
    (exactPrefix data i layout.inputWidth + FP32.toReal (data.bias i))
  rw [add_sub_add_right_eq_sub] at htriangle
  exact htriangle.trans (by unfold traceErrorBudget; rw [← prefixErrorBudget_full]; linarith)

private theorem exactPrefix_full (data : TensorData layout rows) (i : Fin rows) :
    exactPrefix data i layout.inputWidth =
      ∑ j : Fin layout.inputWidth, FP32.toReal (data.weights i j) * FP32.toReal (data.input j) := by
  unfold exactPrefix
  rw [List.take_of_length_le (by simp), List.map_ofFn, List.sum_ofFn]
  rfl

/-- ReLU adds no error budget to the FMA loop and bias addition. -/
theorem rowResult_abs_error_le (data : TensorData layout rows) (i : Fin rows)
    (hfinite : FiniteRowTrace data i) :
    FP32.isFinite (rowResult FP32.floatLibArithmetic data i) = true ∧
    |FP32.toReal (rowResult FP32.floatLibArithmetic data i) - exactRowResult data i| ≤
      traceErrorBudget data i := by
  constructor
  · exact FP32.isFinite_maxPositiveZero (preactivation FP32.floatLibArithmetic data i)
      hfinite.preactivation
  · change |FP32.toReal (FP32.maxPositiveZero (preactivation FP32.floatLibArithmetic data i)) -
      _| ≤ _
    unfold exactRowResult
    rw [← exactPrefix_full]
    exact (FP32.maxPositiveZero_abs_error_le _ hfinite.preactivation _).trans
      (preactivation_abs_error_le data i hfinite)

/-- Concrete kernel execution stores the numerically bounded result and preserves the memory frame. -/
theorem kernel_abs_error_le (cfg : Launch) (layout : Layout) (hfit : cfg.rows ≤ 2 ^ 32)
    (data : TensorData layout cfg.rows) (env : Environment) (args : Argument → Word64) (buffers :
        Buffers)
    (t : Thread cfg) (initial : State Register) (hentry : KernelEntry cfg layout env args buffers
        data initial)
    (hfinite : FiniteRowTrace data t.rowIndex) :
    ∃ final, run (kernelProgram cfg layout) (t.context env FP32.floatLibArithmetic) (kernelSteps
        cfg layout) initial = some final ∧
      KernelResult cfg layout FP32.floatLibArithmetic data buffers t initial final ∧
      FP32.isFinite (final.regs .result) = true ∧
      |FP32.toReal (final.regs .result) - exactRowResult data t.rowIndex| ≤
        traceErrorBudget data t.rowIndex := by
  obtain ⟨final, hexec, hr⟩ := kernel_correct cfg layout hfit (t.context env
      FP32.floatLibArithmetic) data
    args buffers t rfl initial hentry
  refine ⟨final, hexec, hr, ?_⟩
  rw [hr.result]
  exact rowResult_abs_error_le data _ hfinite

end ReluLayer.PTX

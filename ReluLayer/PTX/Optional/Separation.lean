import ReluLayer.PTX.TensorMemory

namespace ReluLayer.PTX

/-- Optional launch-level separation; single-thread correctness does not require it. -/
structure OutputsSeparated (layout : Layout) (rows : Nat) (buffers : Buffers) : Prop where
  weights : ∀ (i r : Fin rows) (j : Fin layout.inputWidth),
    elementAddress buffers .output i.val ≠ weightAddress layout buffers r.val j.val
  input : ∀ (i : Fin rows) (j : Fin layout.inputWidth),
    elementAddress buffers .output i.val ≠ elementAddress buffers .input j.val
  bias : ∀ (i r : Fin rows),
    elementAddress buffers .output i.val ≠ elementAddress buffers .bias r.val
  outputs : ∀ (i r : Fin rows), i ≠ r →
    elementAddress buffers .output i.val ≠ elementAddress buffers .output r.val

end ReluLayer.PTX

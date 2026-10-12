import ReluLayer.PTX.AddressCalculation
import ReluLayer.PTX.BaseLoading
import ReluLayer.PTX.FP32

namespace ReluLayer.PTX

abbrev Buffers := Argument → Word64

structure TensorData (layout : Layout) (rows : Nat) where
  weights : Fin rows → Fin layout.inputWidth → FP32.Bits
  input : Fin layout.inputWidth → FP32.Bits
  bias : Fin rows → FP32.Bits

/-- Addresses retain byte units even though memory stores complete FP32 words. -/
def weightAddress (layout : Layout) (buffers : Buffers) (i j : Nat) : Word64 :=
  buffers .weights + BitVec.ofNat 64 (i * layout.rowStride + j * fp32Bytes)

def elementAddress (buffers : Buffers) (arg : Argument) (i : Nat) : Word64 :=
  buffers arg + BitVec.ofNat 64 (i * fp32Bytes)

/-- Correct input words at the addresses used by the computation. -/
structure InputsStored (layout : Layout) (rows : Nat) (buffers : Buffers)
    (data : TensorData layout rows) (memory : GlobalMemory) : Prop where
  weights : ∀ i j, memory (weightAddress layout buffers i.val j.val) = some (data.weights i j)
  input : ∀ j, memory (elementAddress buffers .input j.val) = some (data.input j)
  bias : ∀ i, memory (elementAddress buffers .bias i.val) = some (data.bias i)

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

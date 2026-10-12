import ReluLayer.PTX.AddressCalculation
import ReluLayer.PTX.BaseLoading
import ReluLayer.PTX.FP32

namespace ReluLayer.PTX

variable {cfg : Launch}

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

/-- The byte address owned by this thread's output store. -/
def Thread.outputAddress (t : Thread cfg) (buffers : Buffers) : Word64 :=
  elementAddress buffers .output t.row

end ReluLayer.PTX

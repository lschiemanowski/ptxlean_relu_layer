import ReluLayer.PTX.TensorMemory

namespace ReluLayer.PTX

variable {layout : Layout} {rows : Nat}

/-- Ordered FMA recurrence; after the last column the accumulator stays unchanged. -/
def prefixAccumulator (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (i : Fin rows) : Nat → FP32.Bits
  | 0 => FP32.positiveZero
  | k + 1 =>
      if hk : k < layout.inputWidth then
        ops.fmaRN (data.weights i ⟨k, hk⟩) (data.input ⟨k, hk⟩)
          (prefixAccumulator ops data i k)
      else prefixAccumulator ops data i k

@[simp] theorem prefixAccumulator_zero (ops : FP32.Arithmetic)
    (data : TensorData layout rows) (i : Fin rows) :
    prefixAccumulator ops data i 0 = FP32.positiveZero := rfl

/-- Rounded bias addition, shared by the bit-level result and numerical analysis. -/
def preactivation (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (i : Fin rows) : FP32.Bits :=
  ops.addRN (prefixAccumulator ops data i layout.inputWidth) (data.bias i)

def rowResult (ops : FP32.Arithmetic) (data : TensorData layout rows) (i : Fin rows) : FP32.Bits :=
  FP32.maxPositiveZero (preactivation ops data i)

/-- Entry to the independent fragment at PTX line 43; scratch contents are unrestricted. -/
structure BodyEntry (layout : Layout) (rows : Nat)
    (buffers : Buffers) (data : TensorData layout rows)
    (i : Fin rows) (initial : State Register) : Prop where
  inputs : InputsStored layout rows buffers data initial.globalMemory
  pc : initial.pc = 0
  active : initial.returned = false
  row : (initial.regs .row).toNat = i.val
  weightRead : initial.regs64 .weightRead = weightAddress layout buffers i.val 0
  inputRead : initial.regs64 .inputRead = buffers .input
  biasRead : initial.regs64 .biasRead = elementAddress buffers .bias i.val
  outputWrite : initial.regs64 .outputWrite = elementAddress buffers .output i.val

/-- Store the selected row result and preserve every other word address. -/
structure OutputStored (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial final : GlobalMemory) : Prop where
  output : final (elementAddress buffers .output i.val) = some (rowResult ops data i)
  frame : ∀ address, address ≠ elementAddress buffers .output i.val →
    final address = initial address

end ReluLayer.PTX

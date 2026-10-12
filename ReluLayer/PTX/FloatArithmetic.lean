import ReluLayer.PTX.FP32
import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Proof

namespace ReluLayer.PTX.FP32

open FloatLib.Floats.Formats.BinaryInterchange

abbrev Binary32 := Model FloatFormat.binary32

/-- FloatLib's bit-level binary32 carrier preserves every input encoding. -/
def decode (word : Bits) : Binary32 := Model.ofBits word

def encode (value : Binary32) : Bits := value.toBits

@[simp] theorem encode_decode (word : Bits) : encode (decode word) = word := rfl

@[simp] theorem decode_encode (value : Binary32) : decode (encode value) = value := by
  cases value; rfl

/-- Certified FloatLib dispatchers: binary32 nearest-even, with one rounding for FMA. -/
def floatLibArithmetic : Arithmetic where
  fmaRN a b c := encode (Model.fma (decode a) (decode b) (decode c))
  addRN a b := encode (Model.add (decode a) (decode b))

/-- The executable operation agrees with exact dyadic addition followed by reference rounding. -/
theorem addRN_eq_reference (a b : Bits) :
    floatLibArithmetic.addRN a b = encode (Model.Spec.add (decode a) (decode b)) :=
  congrArg encode (Model.Proof.add_eq_spec (decode a) (decode b))

/-- The executable FMA agrees with exact dyadic product-plus-addend and one reference rounding. -/
theorem fmaRN_eq_reference (a b c : Bits) :
    floatLibArithmetic.fmaRN a b c = encode (Model.Spec.fma (decode a) (decode b) (decode c)) :=
  congrArg encode (Model.Proof.fma_eq_spec (decode a) (decode b) (decode c))

end ReluLayer.PTX.FP32

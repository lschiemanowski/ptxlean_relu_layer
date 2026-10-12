import ReluLayer.PTX.KernelExecution
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

/-- Exact dyadic intermediates followed by nearest-even binary32 rounding. -/
def referenceArithmetic : Arithmetic where
  fmaRN a b c := encode (Model.Spec.fma (decode a) (decode b) (decode c))
  addRN a b := encode (Model.Spec.add (decode a) (decode b))

theorem addRN_eq_reference (a b : Bits) :
    floatLibArithmetic.addRN a b = referenceArithmetic.addRN a b :=
  congrArg encode (Model.Proof.add_eq_spec (decode a) (decode b))

theorem fmaRN_eq_reference (a b c : Bits) :
    floatLibArithmetic.fmaRN a b c = referenceArithmetic.fmaRN a b c :=
  congrArg encode (Model.Proof.fma_eq_spec (decode a) (decode b) (decode c))

theorem floatLibArithmetic_eq_reference : floatLibArithmetic = referenceArithmetic := by
  apply congrArg₂ Arithmetic.mk
  · funext a b c; exact fmaRN_eq_reference a b c
  · funext a b; exact addRN_eq_reference a b

end ReluLayer.PTX.FP32

namespace ReluLayer.PTX

/-- Whole-kernel execution with concrete FloatLib arithmetic, stated using its reference result. -/
theorem layer_kernel_floatLib_correct (data : TensorData layerLayout layerLaunch.rows)
    (env : Environment) (args : Argument → Word64) (t : Thread layerLaunch)
    (initial : State Register) (hentry : KernelEntry layerLaunch layerLayout env args data initial) :
    ∃ final, run (kernelProgram layerLaunch layerLayout) t.identity
        (kernelSteps layerLaunch layerLayout) initial env (some FP32.floatLibArithmetic) = some final ∧
      KernelResult layerLaunch layerLayout FP32.referenceArithmetic data
        (convertedBuffers env args) t initial final := by
  simpa only [FP32.floatLibArithmetic_eq_reference] using
    layer_kernel_correct FP32.floatLibArithmetic data env args t initial hentry

end ReluLayer.PTX

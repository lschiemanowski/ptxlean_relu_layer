import ReluLayer.Dimensions
import ReluLayer.PTX.AddressCalculation

namespace ReluLayer.PTX

variable {cfg : Launch}

/-- Concrete configuration of the selected source kernel. -/
def layerLaunch : Launch := ⟨12, 256, by decide, by decide⟩
def layerLayout : Layout := ⟨inputSize, by decide, by decide⟩

theorem layerLaunch_rows : layerLaunch.rows = outputSize := rfl

/-- The required context for a configured thread, with explicit environment and arithmetic. -/
def Thread.context (t : Thread cfg) (env : Environment) (ops : FP32.Arithmetic) :
    ExecutionContext :=
  ⟨t.identity, env, ops⟩

end ReluLayer.PTX

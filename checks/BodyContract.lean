import ReluLayer.PTX.BodyContract

open ReluLayer.PTX

-- Bit-pattern oracles cover both signs, infinities, NaNs, and subnormals.
example : FP32.maxPositiveZero 0x80000000 = 0 := by decide
example : FP32.maxPositiveZero 0xbf800000 = 0 := by decide
example : FP32.maxPositiveZero 0xff800000 = 0 := by decide
example : FP32.maxPositiveZero 0x7fc00001 = 0 := by decide
example : FP32.maxPositiveZero 0x7f800001 = 0 := by decide
example : FP32.maxPositiveZero 0xffc00001 = 0 := by decide
example : FP32.maxPositiveZero 0x7f800000 = 0x7f800000 := by decide
example : FP32.maxPositiveZero 1 = 1 := by decide
example : FP32.maxPositiveZero 0x80000001 = 0 := by decide
example : FP32.maxPositiveZero 0 = 0 := by decide
example : FP32.maxPositiveZero 0x3f800000 = 0x3f800000 := by decide

private def layout : Layout := ⟨2, by decide, by decide⟩
private def buffers : Buffers
  | .weights => 0
  | .input => 16
  | .bias => 24
  | .output => 32
private def data : TensorData layout 2 :=
  ⟨fun i j => BitVec.ofNat 32 (i.val * 2 + j.val + 1),
    fun j => BitVec.ofNat 32 (j.val + 5), fun i => BitVec.ofNat 32 (i.val + 7)⟩
private def memory : GlobalMemory := fun a =>
  if a = 0 then some 1 else if a = 4 then some 2
  else if a = 8 then some 3 else if a = 12 then some 4
  else if a = 16 then some 5 else if a = 20 then some 6
  else if a = 24 then some 7 else if a = 28 then some 8
  else if a = 36 then some 0x3f800000 else none

private theorem inputsStored : InputsStored layout 2 buffers data memory := by
  constructor <;> decide
private theorem separated : OutputsSeparated layout 2 buffers := by
  constructor <;> decide

-- No output initialization is required. Missing/wrong input words are rejected.
example : memory 32 = none := by decide
example : ¬ InputsStored layout 2 buffers data (fun _ => none) := by
  intro h
  have hh := h.weights (0 : Fin 2) ⟨0, by decide⟩
  contradiction
example : ¬ InputsStored layout 2 buffers data (fun a => if a = 0 then some 9 else memory a) := by
  intro h
  have hh := h.weights (0 : Fin 2) ⟨0, by decide⟩
  exact (by decide : (some (9 : Word32)) ≠ some 1) hh

-- The optional launch-level contract excludes output/input and output/output aliasing.
example : ¬ OutputsSeparated layout 2 (fun arg => if arg = .output then 16 else buffers arg) := by
  intro h
  exact h.input (0 : Fin 2) ⟨0, by decide⟩ rfl
example : ¬ OutputsSeparated layout (2 ^ 62 + 1) buffers := by
  intro h
  exact h.outputs ⟨0, by decide⟩ ⟨2 ^ 62, by decide⟩ (by decide) (by decide)

-- Read-only inputs can share addresses when their specified values agree.
private def sharedBuffers : Buffers := fun arg => if arg = .input then 0 else buffers arg
private def sharedData : TensorData layout 2 :=
  { data with input := fun j => BitVec.ofNat 32 (j.val + 1) }
example : InputsStored layout 2 sharedBuffers sharedData memory := by constructor <;> decide
example : OutputsSeparated layout 2 sharedBuffers := by constructor <;> decide

private def initial : State Register :=
  { regs := fun r => if r = .row then 0 else 77
    pc := 0
    regs64 := fun r => match r with
      | .weightRead => 0
      | .inputRead => 16
      | .biasRead => 24
      | .outputWrite => 32
      | _ => 999
    globalMemory := memory }
example : BodyEntry layout 2 buffers data 0 initial :=
  ⟨inputsStored, rfl, rfl, by decide, by decide, by decide, by decide, by decide⟩

-- Artificial operations expose order/arity; they are not rounded FP32 arithmetic.
private def traceOps : FP32.Arithmetic := ⟨fun w x acc => acc * 10 + w + x, fun a b => a + b⟩
example : prefixAccumulator traceOps data 0 0 = 0 := by decide
example : prefixAccumulator traceOps data 0 1 = 6 := by decide
example : prefixAccumulator traceOps data 0 2 = 68 := by decide
example : rowResult traceOps data 0 = 75 := by decide
example : rowResult traceOps data 1 = 98 := by decide

private def written : GlobalMemory := fun a => if a = 32 then some 75 else memory a
example : OutputStored traceOps data buffers 0 memory written := by
  constructor
  · decide
  · intro address h
    change address ≠ (32 : Word64) at h
    exact ite_eq_right h
example : written 0 = memory 0 ∧ written 20 = memory 20 ∧ written 36 = some 0x3f800000 := by decide

#print axioms prefixAccumulator_zero

def main : IO Unit := IO.println "Body contract: word inputs, output separation/frame, FP32 maximum, and ordered-fold checks pass."

import ReluLayer.PTX.FloatArithmetic

open ReluLayer.PTX

-- Exact bit preservation includes NaN payloads and both signed zeros.
example (word : FP32.Bits) : FP32.encode (FP32.decode word) = word := FP32.encode_decode word
example (a b : FP32.Bits) :
    FP32.floatLibArithmetic.addRN a b = FP32.referenceArithmetic.addRN a b :=
  FP32.addRN_eq_reference a b
example (a b c : FP32.Bits) :
    FP32.floatLibArithmetic.fmaRN a b c = FP32.referenceArithmetic.fmaRN a b c :=
  FP32.fmaRN_eq_reference a b c

#print axioms FP32.encode_decode
#print axioms FP32.addRN_eq_reference
#print axioms FP32.fmaRN_eq_reference
#print axioms layer_kernel_floatLib_correct

private def assertWord (name : String) (actual expected : FP32.Bits) : IO Unit :=
  unless actual == expected do
    throw (IO.userError s!"{name}: got {actual.toNat}, expected {expected.toNat}")

def main : IO Unit := do
  let op := FP32.floatLibArithmetic
  assertWord "even halfway addition" (op.addRN 0x3f800000 0x33800000) 0x3f800000
  assertWord "odd halfway addition" (op.addRN 0x3f800001 0x33800000) 0x3f800002
  assertWord "subnormal addition" (op.addRN 1 1) 2
  assertWord "negative zeros" (op.addRN 0x80000000 0x80000000) 0x80000000
  assertWord "opposite zeros" (op.addRN 0 0x80000000) 0
  assertWord "overflow" (op.addRN 0x7f7fffff 0x7f7fffff) 0x7f800000
  assertWord "fused cancellation" (op.fmaRN 0x3f800001 0x3f7ffffe 0xbf800000) 0xa8800000
  assertWord "normal-to-subnormal FMA" (op.fmaRN 0x00800000 0x3f000000 0) 0x00400000
  assertWord "underflow halfway to zero" (op.fmaRN 1 0x3f000000 0) 0
  assertWord "underflow halfway to even" (op.fmaRN 3 0x3f000000 0) 2
  unless FP32.isNaN (op.addRN 0x7f800000 0xff800000) &&
      FP32.isNaN (op.fmaRN 0 0x7f800000 0) &&
      FP32.isNaN (op.fmaRN 0x7fc01234 0x3f800000 0) do
    throw (IO.userError "Exceptional arithmetic did not return NaN.")
  for bits in [0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0xff801234] do
    assertWord "word roundtrip" (FP32.encode (FP32.decode bits)) bits
  -- Lean 4.35 native execution is a comparison oracle, not a GPU refinement proof.
  for (a, b, c) in [(0x3f800001, 0x3f7ffffe, 0xbf800000),
      (0x00800000, 0x3f000000, 0), (1, 0x3f000000, 0), (3, 0x3f000000, 0)] do
    let fa := Float32.ofBits (UInt32.ofBitVec a)
    let fb := Float32.ofBits (UInt32.ofBitVec b)
    let fc := Float32.ofBits (UInt32.ofBitVec c)
    assertWord "Lean 4.35 FMA comparison" (op.fmaRN a b c) (fa.fma fb fc).toBits.toBitVec
  -- Concrete arithmetic in the full 1024-column, 31-instruction kernel.
  let weightBytes := layerLaunch.rows * layerLayout.rowStride
  let xBase := weightBytes
  let bBase := xBase + layerLayout.rowStride
  let yBase := bBase + layerLaunch.rows * fp32Bytes
  let arg : Argument → Word64 := fun a => BitVec.ofNat 64 (1000 + (match a with
    | .weights => 0 | .input => xBase | .bias => bBase | .output => yBase))
  let env : Environment :=
    { params := fun symbol => if symbol = "W_ptr" then some (arg .weights)
        else if symbol = "x_ptr" then some (arg .input)
        else if symbol = "b_ptr" then some (arg .bias)
        else if symbol = "y_ptr" then some (arg .output) else none
      space := ⟨⟨1000, 2 ^ 24, by decide⟩, []⟩ }
  let initial : State Register :=
    { regs := fun _ => 99, regs64 := fun _ => 999, pc := 0, predicates := fun _ => true
      globalMemory := fun a =>
        if a.toNat < weightBytes then some 0x3f800000 -- W = 1
        else if a.toNat < bBase then some 0x3f000000 -- x = 1/2
        else if a.toNat < yBase then some 0x3e800000 -- b = 1/4
        else some 0x42c60000 }
  for row in [0, layerLaunch.rows - 1] do
    match run (kernelProgram layerLaunch layerLayout)
        ⟨BitVec.ofNat 32 (row % layerLaunch.threads), BitVec.ofNat 32 (row / layerLaunch.threads)⟩
        (kernelSteps layerLaunch layerLayout) initial env (some op) with
    | none => throw (IO.userError "Concrete binary32 kernel execution failed.")
    | some final =>
      let output := BitVec.ofNat 64 (yBase + row * fp32Bytes)
      assertWord "kernel result 512.25" (final.regs .result) 0x44001000
      unless final.returned && final.pc == 31 && final.globalMemory output == some 0x44001000 &&
          final.globalMemory (output + 4) == some 0x42c60000 &&
          final.globalMemory (BitVec.ofNat 64 bBase) == some 0x3e800000 do
        throw (IO.userError "Concrete binary32 kernel output/return/frame failed.")
  IO.println "FloatLib refinement audits, encoding, rounding/FMA/subnormal/exception cases, Lean 4.35 comparison and concrete full-width kernel execution pass."

import ReluLayer.PTX.Environment

namespace ReluLayer.PTX.FP32

/-- Binary32 storage, retaining signed zeros, infinities and NaN payloads. -/
abbrev Bits := Word32

def positiveZero : Bits := 0
def fractionBits : Nat := 23
def exponentBits : Nat := 8

def fraction (v : Bits) : Nat := v.toNat % 2 ^ fractionBits
def exponent (v : Bits) : Nat := v.toNat / 2 ^ fractionBits % 2 ^ exponentBits
def negative (v : Bits) : Bool := decide (2 ^ (fractionBits + exponentBits) ≤ v.toNat)
def isNaN (v : Bits) : Bool :=
  decide (exponent v = 2 ^ exponentBits - 1 ∧ fraction v ≠ 0)

/-- The selected max.f32 with a fixed positive-zero second operand, without .ftz. -/
def maxPositiveZero (v : Bits) : Bits :=
  if isNaN v || negative v then positiveZero else v

/-- An operation interface, not a certified implementation of rounded arithmetic. -/
structure Arithmetic where
  fmaRN : Bits → Bits → Bits → Bits
  addRN : Bits → Bits → Bits

end ReluLayer.PTX.FP32

namespace ReluLayer.PTX

abbrev Word32 := BitVec 32
abbrev Word64 := BitVec 64

/-- Abstract FP32 words at byte addresses; valid word accesses are assumed. -/
abbrev GlobalMemory := Word64 → Option Word32

/-- Read-only parameter slots and a supplied partial generic-to-global conversion. -/
structure Environment where
  params : String → Option Word64
  toGlobal : Word64 → Option Word64

def Environment.empty : Environment := ⟨fun _ => none, fun _ => none⟩

end ReluLayer.PTX

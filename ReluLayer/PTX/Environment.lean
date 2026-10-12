import Mathlib.Data.Nat.Basic

namespace ReluLayer.PTX

abbrev Word32 := BitVec 32
abbrev Word64 := BitVec 64

/-- Abstract FP32 words at byte addresses; valid word accesses are assumed. -/
abbrev GlobalMemory := Word64 → Option Word32

/-- A nonwrapping window in the generic 64-bit address space. -/
structure Window where
  base : Nat
  size : Nat
  end_le : base + size ≤ 2 ^ 64

def Window.contains (w : Window) (a : Word64) : Bool :=
  decide (w.base ≤ a.toNat ∧ a.toNat < w.base + w.size)

/-- Excluded windows represent non-global spaces; kernel-param windows need not be excluded. -/
structure AddressSpace where
  globalWindow : Window
  excluded : List Window

def AddressSpace.admits (space : AddressSpace) (a : Word64) : Bool :=
  space.globalWindow.contains a && !(space.excluded.any (fun w => w.contains a))

def AddressSpace.globalAddress (space : AddressSpace) (a : Word64) : Word64 :=
  BitVec.ofNat 64 (a.toNat - space.globalWindow.base)

/-- Partial semantics deliberately excludes conversions outside the admitted region. -/
def AddressSpace.toGlobal (space : AddressSpace) (a : Word64) : Option Word64 :=
  if space.admits a then some (space.globalAddress a) else none

theorem AddressSpace.toGlobal_of_admits (space : AddressSpace) (a : Word64)
    (h : space.admits a = true) : space.toGlobal a = some (space.globalAddress a) := by
  simp [AddressSpace.toGlobal, h]

theorem AddressSpace.globalAddress_toNat (space : AddressSpace) (a : Word64) :
    (space.globalAddress a).toNat = a.toNat - space.globalWindow.base := by
  exact Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.sub_le _ _) a.isLt)

theorem AddressSpace.admitted_bounds (space : AddressSpace) (a : Word64)
    (h : space.admits a = true) :
    space.globalWindow.base ≤ a.toNat ∧
      (space.globalAddress a).toNat < space.globalWindow.size := by
  have hw : space.globalWindow.contains a = true := (Bool.and_eq_true_iff.mp h).1
  have hb : space.globalWindow.base ≤ a.toNat ∧
      a.toNat < space.globalWindow.base + space.globalWindow.size := by
    simpa [Window.contains] using hw
  rw [AddressSpace.globalAddress_toNat]
  omega

/-- Declared, naturally aligned u64 parameter slots, not serialized host ABI bytes. -/
structure Environment where
  params : String → Option Word64
  space : AddressSpace

def Environment.empty : Environment :=
  ⟨fun _ => none, ⟨⟨0, 0, by decide⟩, []⟩⟩

end ReluLayer.PTX

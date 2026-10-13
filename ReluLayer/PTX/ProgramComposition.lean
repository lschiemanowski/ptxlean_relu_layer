import ReluLayer.PTX.Machine

namespace ReluLayer.PTX

variable {Reg : Type}

/-- The enclosing program contains this block at the given instruction offset. -/
def CodeAt (program : List (Instruction Reg)) (offset : Nat)
    (block : List (Instruction Reg)) : Prop :=
  ∀ i, i < block.length → program[offset + i]? = block[i]?

/-- Concatenation places a block immediately after its prefix. -/
theorem codeAt_append (pre block post : List (Instruction Reg)) :
    CodeAt (pre ++ block ++ post) pre.length block := by
  intro i hi
  have hpre : ¬ pre.length + i < pre.length := by omega
  simp [List.append_assoc, List.getElem?_append, hpre, hi]

/-- A placed concatenation contains its first block at the same offset. -/
theorem CodeAt.left {program pre post : List (Instruction Reg)} {offset : Nat}
    (h : CodeAt program offset (pre ++ post)) : CodeAt program offset pre := by
  intro i hi
  simpa only [List.getElem?_append_left hi] using h i (by simp; omega)

/-- Its second block starts after the first block's static instruction length. -/
theorem CodeAt.right {program pre post : List (Instruction Reg)} {offset : Nat}
    (h : CodeAt program offset (pre ++ post)) :
    CodeAt program (offset + pre.length) post := by
  intro i hi
  have hpre : ¬ pre.length + i < pre.length := by omega
  simpa [List.getElem?_append, hpre, Nat.add_assoc] using
    h (pre.length + i) (by simp; omega)

end ReluLayer.PTX

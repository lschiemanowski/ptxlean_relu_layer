import ReluLayer.PTX.Machine

namespace ReluLayer.PTX

/-- Register names shared by all selected kernel blocks. -/
inductive Register where
  | thread | block | row
  | weightBase | inputRead | biasBase | outputBase
  | rowOffset | elementOffset | weightRead | biasRead | outputWrite
  | column | weightValue | inputValue | accumulator | more
  | biasValue | result
  deriving DecidableEq

def Register.ptxName : Register → String
  | .thread => "%thread"
  | .block => "%block"
  | .row => "%row"
  | .weightBase => "%W_base"
  | .inputRead => "%x_read"
  | .biasBase => "%b_base"
  | .outputBase => "%y_base"
  | .rowOffset => "%row_offset"
  | .elementOffset => "%element_offset"
  | .weightRead => "%W_read"
  | .biasRead => "%b_read"
  | .outputWrite => "%y_write"
  | .column => "%j"
  | .weightValue => "%w"
  | .inputValue => "%x"
  | .accumulator => "%acc"
  | .more => "%more"
  | .biasValue => "%bias"
  | .result => "%result"

private def operandText : Operand Register → String
  | .reg r => r.ptxName
  | .imm v => toString v.toNat
  | .tidX => "%tid.x"
  | .ctaidX => "%ctaid.x"

private def operand64Text : Operand64 Register → String
  | .reg r => r.ptxName
  | .imm v => toString v.toNat

/-- Render the selected instruction forms for executable source-fragment comparisons. -/
def Instruction.ptxText : Instruction Register → String
  | .mov dst src => s!"mov.u32 {dst.ptxName}, {operandText src};"
  | .madLo dst a b c =>
      s!"mad.lo.u32 {dst.ptxName}, {operandText a}, {operandText b}, {operandText c};"
  | .mulWide dst a b => s!"mul.wide.u32 {dst.ptxName}, {operandText a}, {operandText b};"
  | .add64 dst a b => s!"add.u64 {dst.ptxName}, {operand64Text a}, {operand64Text b};"
  | .ldParam64 dst symbol => s!"ld.param.u64 {dst.ptxName}, [{symbol}];"
  | .cvtaGlobal64 dst src => s!"cvta.to.global.u64 {dst.ptxName}, {operand64Text src};"
  | .movF32 dst value =>
      let digits := Nat.toDigits 16 value.toNat
      s!"mov.f32 {dst.ptxName}, 0f{String.ofList (List.replicate (8 - digits.length) '0' ++ digits)};"
  | .loadF32 dst address => s!"ld.global.f32 {dst.ptxName}, [{operand64Text address}];"
  | .fmaRN dst a b c => s!"fma.rn.f32 {dst.ptxName}, {operandText a}, {operandText b}, {operandText c};"
  | .add32 dst a b => s!"add.u32 {dst.ptxName}, {operandText a}, {operandText b};"
  | .setpLt dst a b => s!"setp.lt.u32 {dst.ptxName}, {operandText a}, {operandText b};"
  | .branchIf predicate target => s!"@{predicate.ptxName} bra PC_{target};"
  | .addF32RN dst a b => s!"add.rn.f32 {dst.ptxName}, {operandText a}, {operandText b};"
  | .maxPositiveZero dst src => s!"max.f32 {dst.ptxName}, {operandText src}, 0f00000000;"
  | .storeF32 address src => s!"st.global.f32 [{operand64Text address}], {operandText src};"
  | .ret => "ret;"

end ReluLayer.PTX

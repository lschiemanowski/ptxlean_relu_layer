import ReluLayer.PTX.BaseLoading

open ReluLayer.PTX

private def args : Argument → Word64
  | .weights => 1024
  | .input => 1040
  | .bias => 1064
  | .output => 1080

private def env : Environment :=
  { params := fun symbol =>
      if symbol = "W_ptr" then some (args .weights)
      else if symbol = "x_ptr" then some (args .input)
      else if symbol = "b_ptr" then some (args .bias)
      else if symbol = "y_ptr" then some (args .output)
      else none
    space := ⟨⟨1000, 200, by decide⟩, [⟨1100, 16, by decide⟩]⟩ }

private def initial : State Register :=
  ⟨fun _ => 77, 0, fun _ => 999, fun address => if address = 24 then some 42 else none, fun _ => false, false⟩

-- Nonzero window bases exercise a real conversion rather than an identity operation.
example : ((run baseLoading ⟨0, 0⟩ 8 initial env).map (fun s =>
    ((s.regs64 .weightBase).toNat, (s.regs64 .inputRead).toNat,
      (s.regs64 .biasBase).toNat, (s.regs64 .outputBase).toNat))) =
    some (24, 40, 64, 80) := by decide

example : ∃ final, run baseLoading ⟨0, 0⟩ 8 initial env = some final ∧
    BaseResult env args initial final :=
  baseLoading_correct env args ⟨0, 0⟩ initial (by decide) (by decide)
    (fun arg => by cases arg <;> decide) (fun arg => by cases arg <;> decide)

-- Admitted-region edges, excluded windows, and absent parameters.
example : env.space.toGlobal 1000 = some 0 := by decide
example : env.space.toGlobal 1199 = some 199 := by decide
example : env.space.toGlobal 1200 = none := by decide
example : env.space.toGlobal 999 = none := by decide
example : env.space.toGlobal 1100 = none := by decide
example : env.space.toGlobal 1115 = none := by decide
example : env.space.toGlobal 1116 = some 116 := by decide
example : (Instruction.ldParam64 Register.weightBase "missing").execute
    ⟨0, 0⟩ initial env = none := by decide
example : run baseLoading ⟨0, 0⟩ 8 initial Environment.empty = none := by decide

-- Conversion reads the old source word when destination and source coincide.
private def aliasInitial : State Register := { initial with regs64 := fun _ => 1024 }
example : ((Instruction.cvtaGlobal64 Register.weightBase (.reg .weightBase)).execute
    ⟨0, 0⟩ aliasInitial env).map (fun s => (s.regs64 .weightBase).toNat) = some 24 := by decide
example : ((run baseLoading ⟨0, 0⟩ 8 initial env).map (fun s =>
    ((s.regs .row).toNat, (s.regs64 .weightRead).toNat, s.globalMemory 24, s.pc))) =
    some (77, 999, some 42, 8) := by decide

#print axioms AddressSpace.globalAddress_toNat
#print axioms AddressSpace.admitted_bounds
#print axioms baseLoading_correct

private def instructionText : Instruction Register → String := Instruction.ptxText

def main (arguments : List String) : IO Unit := do
  let source ← IO.FS.readFile (arguments.headD "kernels/relu_layer.ptx")
  let normalized := "\n" ++ String.intercalate "\n"
    ((source.splitOn "\n").map (fun line => line.trimAscii.toString))
  let expected := "\n" ++ String.intercalate "\n" (baseLoading.map instructionText) ++ "\n"
  unless (normalized.splitOn expected).length == 2 do
    throw (IO.userError s!"Expected exactly one base-loading fragment:\n{expected}")
  let declarations := ".visible .entry relu_layer(\n" ++
    String.intercalate ",\n" ([Argument.weights, .input, .bias, .output].map
      (fun arg => s!".param .u64 {arg.symbol}")) ++ "\n)\n"
  unless (normalized.splitOn declarations).length == 2 do
    throw (IO.userError "The kernel's four u64 parameter declarations do not match.")
  IO.println "Eight base-loading instructions and four parameter declarations match the formal block."

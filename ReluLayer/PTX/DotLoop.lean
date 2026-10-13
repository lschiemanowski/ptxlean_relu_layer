import ReluLayer.PTX.BodyContract

namespace ReluLayer.PTX

variable {layout : Layout} {rows : Nat}

/-- Fragment-relative loop entry and exit, determined by the instruction layout below. -/
def dotLoopStart : Nat := 2
def dotLoopExit : Nat := 10
def dotIterationSteps : Nat := 8

/-- PTX lines 43--55, including counter/accumulator initialization. -/
def dotLoop (width start : Nat) : List (Instruction Register) :=
  [ .mov .column (.imm 0), .movF32 .accumulator FP32.positiveZero,
    .loadF32 .weightValue (.reg .weightRead),
    .loadF32 .inputValue (.reg .inputRead),
    .fmaRN .accumulator (.reg .weightValue) (.reg .inputValue) (.reg .accumulator),
    .add64 .weightRead (.reg .weightRead) (.imm (BitVec.ofNat 64 fp32Bytes)),
    .add64 .inputRead (.reg .inputRead) (.imm (BitVec.ofNat 64 fp32Bytes)),
    .add32 .column (.reg .column) (.imm 1),
    .setpLt .more (.reg .column) (.imm (BitVec.ofNat 32 width)),
    .branchIf .more (start + dotLoopStart) ]

theorem prefixAccumulator_succ (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (i : Fin rows) (k : Nat) (hk : k < layout.inputWidth) :
    prefixAccumulator ops data i (k + 1) =
      ops.fmaRN (data.weights i ⟨k, hk⟩) (data.input ⟨k, hk⟩)
        (prefixAccumulator ops data i k) := by
  simp [prefixAccumulator, hk]

private theorem weight_next (layout : Layout) (buffers : Buffers) (i k : Nat) :
    weightAddress layout buffers i k + BitVec.ofNat 64 fp32Bytes =
      weightAddress layout buffers i (k + 1) := by
  simp [weightAddress, Nat.add_mul, BitVec.ofNat_add, BitVec.add_assoc]

private theorem input_next (buffers : Buffers) (k : Nat) :
    elementAddress buffers .input k + BitVec.ofNat 64 fp32Bytes =
      elementAddress buffers .input (k + 1) := by
  simp [elementAddress, Nat.add_mul, BitVec.ofNat_add, BitVec.add_assoc]

/-- At a loop boundary, all k processed columns agree with the ordered specification. -/
structure LoopInvariant (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register) (k : Nat)
    (s : State Register) : Prop where
  pc : s.pc = if k < layout.inputWidth then initial.pc + dotLoopStart
    else initial.pc + dotLoopExit
  column : s.regs .column = BitVec.ofNat 32 k
  accumulator : s.regs .accumulator = prefixAccumulator ops data i k
  weightRead : s.regs64 .weightRead = weightAddress layout buffers i.val k
  inputRead : s.regs64 .inputRead = elementAddress buffers .input k
  memory : s.globalMemory = initial.globalMemory
  other32 : ∀ r, r ≠ .column → r ≠ .weightValue → r ≠ .inputValue → r ≠ .accumulator →
    s.regs r = initial.regs r
  other64 : ∀ r, r ≠ .weightRead → r ≠ .inputRead → s.regs64 r = initial.regs64 r
  otherPredicates : ∀ r, r ≠ .more → s.predicates r = initial.predicates r
  active : s.returned = false

private def iterationFinal (ops : FP32.Arithmetic) (s : State Register)
    (w x : Word32) (width start : Nat) : State Register :=
  let s := s.writeNext .weightValue w
  let s := s.writeNext .inputValue x
  let s := s.writeNext .accumulator (ops.fmaRN (s.regs .weightValue) (s.regs .inputValue) (s.regs
      .accumulator))
  let s := s.write64Next .weightRead (s.regs64 .weightRead + BitVec.ofNat 64 fp32Bytes)
  let s := s.write64Next .inputRead (s.regs64 .inputRead + BitVec.ofNat 64 fp32Bytes)
  let s := s.writeNext .column (s.regs .column + 1)
  let s : State Register := { s with
    predicates := fun r => if r = .more then decide ((s.regs .column).toNat < width) else
        s.predicates r
    pc := s.pc + 1 }
  { s with pc := if s.predicates .more then start + dotLoopStart else s.pc + 1 }

/-- One actual eight-instruction iteration extends the invariant by one column. -/
theorem dotLoop_iteration (program : List (Instruction Register)) (ctx : ExecutionContext) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial s : State Register)
    (k : Nat) (hk : k < layout.inputWidth)
    (hinputs : InputsStored layout rows buffers data initial.globalMemory)
    (hinv : LoopInvariant ctx.ops data buffers i initial k s)
    (hcode : CodeAt program initial.pc (dotLoop layout.inputWidth initial.pc)) :
    ∃ next, run program ctx dotIterationSteps s = some next ∧
      LoopInvariant ctx.ops data buffers i initial (k + 1) next := by
  have hn : layout.inputWidth < 2 ^ 32 := by
    have h := layout.stride_fits
    simp only [fp32Bytes] at h
    omega
  have hnext : k + 1 < 2 ^ 32 := by omega
  have hw : s.globalMemory (s.regs64 .weightRead) = some (data.weights i ⟨k, hk⟩) := by
    rw [hinv.memory, hinv.weightRead]; exact hinputs.weights i ⟨k, hk⟩
  have hx : s.globalMemory (s.regs64 .inputRead) = some (data.input ⟨k, hk⟩) := by
    rw [hinv.memory, hinv.inputRead]; exact hinputs.input ⟨k, hk⟩
  have hpc : s.pc = initial.pc + dotLoopStart := by simpa [hk] using hinv.pc
  have h2 := hcode 2 (by simp [dotLoop])
  have h3 := hcode 3 (by simp [dotLoop])
  have h4 := hcode 4 (by simp [dotLoop])
  have h5 := hcode 5 (by simp [dotLoop])
  have h6 := hcode 6 (by simp [dotLoop])
  have h7 := hcode 7 (by simp [dotLoop])
  have h8 := hcode 8 (by simp [dotLoop])
  have h9 := hcode 9 (by simp [dotLoop])
  simp [dotLoop, dotLoopStart] at h2 h3 h4 h5 h6 h7 h8 h9
  refine ⟨iterationFinal ctx.ops s (data.weights i ⟨k, hk⟩) (data.input ⟨k, hk⟩)
      layout.inputWidth initial.pc, ?_, ?_⟩
  · simp [run, step, dotIterationSteps, dotLoopStart, h2, h3, h4, h5, h6, h7, h8, h9,
      Instruction.execute,
      Operand.read, Operand64.read, State.writeNext, State.write64Next, iterationFinal,
      hpc, hinv.active, hw, hx, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hn, Nat.add_assoc]
  · constructor
    · simp [iterationFinal, State.writeNext, State.write64Next, hpc, dotLoopStart, dotLoopExit,
        hinv.column,
        BitVec.ofNat_add_ofNat, Nat.mod_eq_of_lt hnext, Nat.add_assoc]
    · simp [iterationFinal, State.writeNext, State.write64Next, hinv.column, BitVec.ofNat_add_ofNat]
    · simp [iterationFinal, State.writeNext, State.write64Next, hinv.accumulator,
        prefixAccumulator_succ ctx.ops data i k hk]
    · simpa [iterationFinal, State.writeNext, State.write64Next, hinv.weightRead] using
        weight_next layout buffers i.val k
    · simpa [iterationFinal, State.writeNext, State.write64Next, hinv.inputRead] using input_next
        buffers k
    · simpa [iterationFinal, State.writeNext, State.write64Next] using hinv.memory
    · intro r hj hw hx ha
      simpa [iterationFinal, State.writeNext, State.write64Next, hj, hw, hx, ha] using
          hinv.other32 r hj hw hx ha
    · intro r hw hx
      simpa [iterationFinal, State.writeNext, State.write64Next, hw, hx] using hinv.other64 r hw hx
    · intro r hr
      simpa [iterationFinal, State.writeNext, State.write64Next, hr] using hinv.otherPredicates r hr
    · simpa [iterationFinal, State.writeNext, State.write64Next] using hinv.active

private def initialized (s : State Register) : State Register :=
  (s.writeNext .column 0).writeNext .accumulator FP32.positiveZero

/-- Every prefix follows actual execution, including the final untaken branch. -/
theorem dotLoop_prefix (program : List (Instruction Register)) (ctx : ExecutionContext) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : BodyEntry layout rows buffers data i initial)
    (hcode : CodeAt program initial.pc (dotLoop layout.inputWidth initial.pc))
    (k : Nat) (hk : k ≤ layout.inputWidth) :
    ∃ final, run program ctx (dotLoopStart + dotIterationSteps * k) initial =
        some final ∧ LoopInvariant ctx.ops data buffers i initial k final := by
  have h0 := hcode 0 (by simp [dotLoop])
  have h1 := hcode 1 (by simp [dotLoop])
  simp [dotLoop] at h0 h1
  induction k with
  | zero =>
    refine ⟨initialized initial, ?_, ?_⟩
    · simp [run, step, dotLoopStart, h0, h1, Instruction.execute, Operand.read,
        State.writeNext, initialized, hentry.active, Nat.add_assoc]
    · constructor
      · simp [initialized, State.writeNext, layout.inputWidth_pos, dotLoopStart, Nat.add_assoc]
      · simp [initialized, State.writeNext]
      · simp [initialized, State.writeNext, FP32.positiveZero]
      · simpa [initialized, State.writeNext] using hentry.weightRead
      · simpa [initialized, State.writeNext, elementAddress] using hentry.inputRead
      · rfl
      · intro r hj hw hx ha; simp [initialized, State.writeNext, hj, ha]
      · intro r hw hx; rfl
      · intro r hr; rfl
      · simpa [initialized, State.writeNext] using hentry.active
  | succ k ih =>
    have hlt : k < layout.inputWidth := by omega
    obtain ⟨s, hexec, hinv⟩ := ih (by omega)
    obtain ⟨next, hstep, hnext⟩ := dotLoop_iteration program ctx data buffers i initial s k hlt
        hentry.inputs hinv hcode
    refine ⟨next, ?_, hnext⟩
    have hfuel : dotLoopStart + dotIterationSteps * (k + 1) =
        (dotLoopStart + dotIterationSteps * k) + dotIterationSteps := by simp [Nat.mul_add,
            Nat.add_assoc]
    rw [hfuel, run_add, hexec]
    exact hstep

/-- The complete loop exits with its invariant instantiated at the input width. -/
theorem dotLoop_correct (program : List (Instruction Register)) (ctx : ExecutionContext) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register)
    (hentry : BodyEntry layout rows buffers data i initial)
    (hcode : CodeAt program initial.pc (dotLoop layout.inputWidth initial.pc)) :
    ∃ final, run program ctx (dotLoopStart + dotIterationSteps *
        layout.inputWidth) initial = some final ∧
      LoopInvariant ctx.ops data buffers i initial layout.inputWidth final :=
  dotLoop_prefix program ctx data buffers i initial hentry hcode layout.inputWidth (Nat.le_refl _)

end ReluLayer.PTX

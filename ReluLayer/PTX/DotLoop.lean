import ReluLayer.PTX.BodyContract

namespace ReluLayer.PTX

/-- Fragment-relative loop entry and exit, determined by the instruction layout below. -/
def dotLoopStart : Nat := 2
def dotLoopExit : Nat := 10
def dotIterationSteps : Nat := 8

/-- PTX lines 43--55, including counter/accumulator initialization. -/
def dotLoop (width : Nat) : List (Instruction Register) :=
  [ .mov .column (.imm 0), .movF32 .accumulator FP32.positiveZero,
    .loadF32 .weightValue (.reg .weightRead),
    .loadF32 .inputValue (.reg .inputRead),
    .fmaRN .accumulator (.reg .weightValue) (.reg .inputValue) (.reg .accumulator),
    .add64 .weightRead (.reg .weightRead) (.imm (BitVec.ofNat 64 fp32Bytes)),
    .add64 .inputRead (.reg .inputRead) (.imm (BitVec.ofNat 64 fp32Bytes)),
    .add32 .column (.reg .column) (.imm 1),
    .setpLt .more (.reg .column) (.imm (BitVec.ofNat 32 width)),
    .branchIf .more dotLoopStart ]

theorem prefixAccumulator_succ (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (i : Fin rows) (k : Nat) (hk : k < layout.inputWidth) :
    prefixAccumulator ops data i (k + 1) =
      ops.fmaRN (data.weights i ⟨k, hk⟩) (data.input ⟨k, hk⟩)
        (prefixAccumulator ops data i k) := by
  have hlen : k < (rowTerms data i).length := by simpa [rowTerms] using hk
  unfold prefixAccumulator
  rw [← List.take_concat_get hlen, List.concat_eq_append, List.foldl_append]
  simp [rowTerms]

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
  pc : s.pc = if k < layout.inputWidth then dotLoopStart else dotLoopExit
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
    (w x : Word32) (width : Nat) : State Register :=
  let s := s.write .weightValue w
  let s := s.write .inputValue x
  let s := s.write .accumulator (ops.fmaRN (s.regs .weightValue) (s.regs .inputValue) (s.regs .accumulator))
  let s := s.write64 .weightRead (s.regs64 .weightRead + BitVec.ofNat 64 fp32Bytes)
  let s := s.write64 .inputRead (s.regs64 .inputRead + BitVec.ofNat 64 fp32Bytes)
  let s := s.write .column (s.regs .column + 1)
  let s : State Register := { s with
    predicates := fun r => if r = .more then decide ((s.regs .column).toNat < width) else s.predicates r
    pc := s.pc + 1 }
  { s with pc := if s.predicates .more then dotLoopStart else s.pc + 1 }

/-- One actual eight-instruction iteration extends the invariant by one column. -/
theorem dotLoop_iteration (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial s : State Register) (id : Identity)
    (env : Environment) (k : Nat) (hk : k < layout.inputWidth)
    (hinputs : InputsStored layout rows buffers data initial.globalMemory)
    (hinv : LoopInvariant ops data buffers i initial k s) :
    ∃ next, run (dotLoop layout.inputWidth) id dotIterationSteps s env (some ops) = some next ∧
      LoopInvariant ops data buffers i initial (k + 1) next := by
  have hn : layout.inputWidth < 2 ^ 32 := by
    have h := layout.stride_fits
    simp only [fp32Bytes] at h
    omega
  have hnext : k + 1 < 2 ^ 32 := by omega
  have hw : s.globalMemory (s.regs64 .weightRead) = some (data.weights i ⟨k, hk⟩) := by
    rw [hinv.memory, hinv.weightRead]; exact hinputs.weights i ⟨k, hk⟩
  have hx : s.globalMemory (s.regs64 .inputRead) = some (data.input ⟨k, hk⟩) := by
    rw [hinv.memory, hinv.inputRead]; exact hinputs.input ⟨k, hk⟩
  have hpc : s.pc = dotLoopStart := by simpa [hk] using hinv.pc
  refine ⟨iterationFinal ops s (data.weights i ⟨k, hk⟩) (data.input ⟨k, hk⟩) layout.inputWidth, ?_, ?_⟩
  · simp [run, step, dotLoop, dotIterationSteps, dotLoopStart, Instruction.execute,
      Operand.read, Operand64.read, State.write, State.write64, iterationFinal,
      hpc, hinv.active, hw, hx, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hn]
  · constructor
    · simp [iterationFinal, State.write, State.write64, hpc, dotLoopStart, dotLoopExit, hinv.column,
        BitVec.ofNat_add_ofNat, Nat.mod_eq_of_lt hnext]
    · simp [iterationFinal, State.write, State.write64, hinv.column, BitVec.ofNat_add_ofNat]
    · simp [iterationFinal, State.write, State.write64, hinv.accumulator,
        prefixAccumulator_succ ops data i k hk]
    · simpa [iterationFinal, State.write, State.write64, hinv.weightRead] using
        weight_next layout buffers i.val k
    · simpa [iterationFinal, State.write, State.write64, hinv.inputRead] using input_next buffers k
    · simpa [iterationFinal, State.write, State.write64] using hinv.memory
    · intro r hj hw hx ha
      simpa [iterationFinal, State.write, State.write64, hj, hw, hx, ha] using hinv.other32 r hj hw hx ha
    · intro r hw hx
      simpa [iterationFinal, State.write, State.write64, hw, hx] using hinv.other64 r hw hx
    · intro r hr
      simpa [iterationFinal, State.write, State.write64, hr] using hinv.otherPredicates r hr
    · simpa [iterationFinal, State.write, State.write64] using hinv.active

private def initialized (s : State Register) : State Register :=
  (s.write .column 0).write .accumulator FP32.positiveZero

/-- Every prefix follows actual execution, including the final untaken branch. -/
theorem dotLoop_prefix (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register) (id : Identity)
    (env : Environment) (hentry : BodyEntry layout rows buffers data i initial)
    (k : Nat) (hk : k ≤ layout.inputWidth) :
    ∃ final, run (dotLoop layout.inputWidth) id (dotLoopStart + dotIterationSteps * k)
        initial env (some ops) = some final ∧ LoopInvariant ops data buffers i initial k final := by
  induction k with
  | zero =>
    refine ⟨initialized initial, ?_, ?_⟩
    · simp [run, step, dotLoop, dotLoopStart, Instruction.execute, Operand.read,
        State.write, initialized, hentry.pc, hentry.active]
    · constructor
      · simp [initialized, State.write, hentry.pc, layout.inputWidth_pos, dotLoopStart]
      · simp [initialized, State.write]
      · simp [initialized, State.write, FP32.positiveZero]
      · simpa [initialized, State.write] using hentry.weightRead
      · simpa [initialized, State.write, elementAddress] using hentry.inputRead
      · rfl
      · intro r hj hw hx ha; simp [initialized, State.write, hj, ha]
      · intro r hw hx; rfl
      · intro r hr; rfl
      · simpa [initialized, State.write] using hentry.active
  | succ k ih =>
    have hlt : k < layout.inputWidth := by omega
    obtain ⟨s, hexec, hinv⟩ := ih (by omega)
    obtain ⟨next, hstep, hnext⟩ := dotLoop_iteration ops data buffers i initial s id env k hlt hentry.inputs hinv
    refine ⟨next, ?_, hnext⟩
    have hfuel : dotLoopStart + dotIterationSteps * (k + 1) =
        (dotLoopStart + dotIterationSteps * k) + dotIterationSteps := by simp [Nat.mul_add, Nat.add_assoc]
    rw [hfuel, run_add, hexec]
    exact hstep

/-- The complete loop exits with its invariant instantiated at the input width. -/
theorem dotLoop_correct (ops : FP32.Arithmetic) (data : TensorData layout rows)
    (buffers : Buffers) (i : Fin rows) (initial : State Register) (id : Identity)
    (env : Environment) (hentry : BodyEntry layout rows buffers data i initial) :
    ∃ final, run (dotLoop layout.inputWidth) id
        (dotLoopStart + dotIterationSteps * layout.inputWidth) initial env (some ops) = some final ∧
      LoopInvariant ops data buffers i initial layout.inputWidth final :=
  dotLoop_prefix ops data buffers i initial id env hentry layout.inputWidth (Nat.le_refl _)

end ReluLayer.PTX

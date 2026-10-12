import ReluLayer.PTX.Layer

namespace ReluLayer.PTX

private theorem pointer_value (base : Word64) (row : Word32) (stride : Nat)
    (hstride : stride < 2 ^ 32) (haddr : base.toNat + row.toNat * stride < 2 ^ 64) :
    (base + wideProduct row (BitVec.ofNat 32 stride)).toNat =
      base.toNat + row.toNat * stride := by
  simp only [BitVec.toNat_add, wideProduct_toNat, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt hstride, Nat.mod_eq_of_lt haddr]

/-- A four-byte element fits when its row is in range and the stride is at least four. -/
private theorem element_end_le (base row rows stride : Nat)
    (hrow : row < rows) (hstride : fp32Bytes ≤ stride) :
    base + row * stride + fp32Bytes ≤ base + rows * stride := by
  have h := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt hrow)
  simp only [Nat.succ_mul] at h
  omega

/-- Natural-number pointer equations and room for one full FP32 element. -/
structure AddressResult (layout : Layout) (rows : Nat)
    (initial final : State Register) : Prop where
  words : AddressWords layout.inputWidth initial final
  weights : (final.regs64 .weightRead).toNat =
    (initial.regs64 .weightBase).toNat + (initial.regs .row).toNat * layout.rowStride
  bias : (final.regs64 .biasRead).toNat =
    (initial.regs64 .biasBase).toNat + (initial.regs .row).toNat * fp32Bytes
  output : (final.regs64 .outputWrite).toNat =
    (initial.regs64 .outputBase).toNat + (initial.regs .row).toNat * fp32Bytes
  weights_in_bounds : (initial.regs64 .weightBase).toNat ≤ (final.regs64 .weightRead).toNat ∧
    (final.regs64 .weightRead).toNat + fp32Bytes ≤
      (initial.regs64 .weightBase).toNat + rows * layout.rowStride
  bias_in_bounds : (initial.regs64 .biasBase).toNat ≤ (final.regs64 .biasRead).toNat ∧
    (final.regs64 .biasRead).toNat + fp32Bytes ≤
      (initial.regs64 .biasBase).toNat + rows * fp32Bytes
  output_in_bounds : (initial.regs64 .outputBase).toNat ≤ (final.regs64 .outputWrite).toNat ∧
    (final.regs64 .outputWrite).toNat + fp32Bytes ≤
      (initial.regs64 .outputBase).toNat + rows * fp32Bytes

/-- Buffer-end bounds imply exact addresses, without 64-bit wraparound. -/
theorem addressCalculation_correct (layout : Layout) (rows : Nat) (ctx : ExecutionContext)
    (initial : State Register) (hpc : initial.pc = 0)
    (hrow : (initial.regs .row).toNat < rows)
    (hW : (initial.regs64 .weightBase).toNat + rows * layout.rowStride < 2 ^ 64)
    (hb : (initial.regs64 .biasBase).toNat + rows * fp32Bytes < 2 ^ 64)
    (hy : (initial.regs64 .outputBase).toNat + rows * fp32Bytes < 2 ^ 64)
    (hactive : initial.returned = false) :
    ∃ final, run (addressCalculation layout.inputWidth) ctx 5 initial = some final ∧
      AddressResult layout rows initial final := by
  obtain ⟨final, hexec, hwords⟩ := addressCalculation_exec layout.inputWidth ctx initial hpc hactive
  have hstride : fp32Bytes ≤ layout.rowStride := by
    have hwidth : 1 ≤ layout.inputWidth := layout.inputWidth_pos
    simpa [Layout.rowStride] using Nat.mul_le_mul_right fp32Bytes hwidth
  have hEW := element_end_le (initial.regs64 .weightBase).toNat _ _ _ hrow hstride
  have hEb := element_end_le (initial.regs64 .biasBase).toNat _ _ _ hrow (Nat.le_refl fp32Bytes)
  have hEy := element_end_le (initial.regs64 .outputBase).toNat _ _ _ hrow (Nat.le_refl fp32Bytes)
  have hWA : (initial.regs64 .weightBase).toNat +
      (initial.regs .row).toNat * layout.rowStride < 2 ^ 64 := by omega
  have hbA : (initial.regs64 .biasBase).toNat +
      (initial.regs .row).toNat * fp32Bytes < 2 ^ 64 := by omega
  have hyA : (initial.regs64 .outputBase).toNat +
      (initial.regs .row).toNat * fp32Bytes < 2 ^ 64 := by omega
  have hweq := pointer_value (initial.regs64 .weightBase) (initial.regs .row)
    layout.rowStride layout.stride_fits hWA
  have hbeq := pointer_value (initial.regs64 .biasBase) (initial.regs .row)
    fp32Bytes (by decide) hbA
  have hyeq := pointer_value (initial.regs64 .outputBase) (initial.regs .row)
    fp32Bytes (by decide) hyA
  simp only [Layout.rowStride] at hweq
  rw [← hwords.weights] at hweq
  rw [← hwords.bias] at hbeq
  rw [← hwords.output] at hyeq
  refine ⟨final, hexec, hwords, hweq, hbeq, hyeq, ?_, ?_, ?_⟩
  · rw [hweq]; exact ⟨Nat.le_add_right _ _, hEW⟩
  · rw [hbeq]; exact ⟨Nat.le_add_right _ _, hEb⟩
  · rw [hyeq]; exact ⟨Nat.le_add_right _ _, hEy⟩

theorem layer_addressCalculation_correct (ctx : ExecutionContext) (initial : State Register)
    (hpc : initial.pc = 0) (hrow : (initial.regs .row).toNat < layerLaunch.rows)
    (hW : (initial.regs64 .weightBase).toNat + layerLaunch.rows * layerLayout.rowStride < 2 ^ 64)
    (hb : (initial.regs64 .biasBase).toNat + layerLaunch.rows * fp32Bytes < 2 ^ 64)
    (hy : (initial.regs64 .outputBase).toNat + layerLaunch.rows * fp32Bytes < 2 ^ 64)
    (hactive : initial.returned = false) :
    ∃ final, run (addressCalculation layerLayout.inputWidth) ctx 5 initial = some final ∧
      AddressResult layerLayout layerLaunch.rows initial final :=
  addressCalculation_correct layerLayout layerLaunch.rows ctx initial hpc hrow hW hb hy hactive

end ReluLayer.PTX

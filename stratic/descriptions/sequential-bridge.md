# Kernel connection to the sequential specification

The selected kernel's FP32 weights, bias, and input become real TorchLean tensors
of shapes `[3072, 1024]`, `[3072]`, and `[1024]`. Each tensor coordinate is the
FloatLib-decoded value of the corresponding input word. The weight indices
retain row-then-column order; no transpose or parameter initialization is
introduced.

These tensors are supplied to the existing evaluation-mode forward function of
the upstream `nn.Sequential![nn.linear 1024 3072, nn.relu]` model. Its parameter
state uses the decoded weights and bias. For every row, the existing exact real
row formula equals that coordinate of the actual sequential forward output.

For a valid selected thread, the kernel entry contract and finite-row-trace contract
ensure the complete modeled kernel returns and stores its result word at the
thread's output address. Every other modeled word location retains its initial
contents. The stored word is finite, and its decoded value satisfies

$$
\left|\operatorname{toReal}(y_i^{\mathrm{kernel}})
-\operatorname{forward}(W^{\mathbb R},b^{\mathbb R},x^{\mathbb R})_i\right|
\le E_i.
$$

Here `forward` evaluates the actual sequential model over real numbers, and
`E_i` is the already-established accumulated FMA and bias rounding budget.
This connection introduces no new numerical estimate or input-conversion error.
The public conclusion quantifies over the stored word directly. It states
`Returns`, storage at the thread's named output address, preservation of every
other memory address, finiteness, and the Sequential-coordinate discrepancy.
The concrete execution context selects FloatLib arithmetic. Global buffer
addresses are supplied directly, with successful generic-to-global conversion
recorded in the entry contract. Internal
PC and register facts remain in the block-composition contracts.

It requires the same finite inputs and finite rounded intermediates for the
selected row as the existing numerical theorem. Output separation is not
required for this single-thread conclusion.

Tensor construction uses the existing total decoder, whose value on NaN and
infinity is zero by convention. The finite-row hypotheses exclude those
encodings from the selected row's comparison. The theorem covers one thread's
formal execution with concrete FloatLib arithmetic and abstract word memory;
PTX arithmetic adequacy, concurrent device execution and GPU hardware
correspondence remain separate.

# Dot-product and ReLU contract

The final block starts with pointers to weight row `i`, the input vector, bias
element `i`, and output element `i`. It initializes its counter and accumulator,
reads columns in ascending order, adds the bias, applies the maximum with positive
zero, stores one FP32 output, and returns. This is PTX lines 43–62.

Memory maps 64-bit addresses directly to optional FP32 bit patterns. Addresses
retain byte units: consecutive tensor words are four bytes apart, and the weight
row stride comes from the input width in `Layout`. The output row count is a
parameter. Loads and stores operate on complete words in this abstract model.
Valid FP32 accesses are assumed; allocation, permissions, alignment, byte
encoding, and CUDA memory correspondence are outside this contract.

The input memory contains the supplied weight, input, and bias words at their
computed addresses. The single-thread theorem requires no output separation:
all input reads finish before its only store. Exactly the selected output word
is updated, and every other address keeps its initial contents. Read-only
inputs may overlap when their specified values agree. No initial contents are
required for a separate output location.

The ordered result starts from positive zero and recursively applies an FMA over columns
`0` through `n - 1`, then performs one bias addition and a maximum with positive
zero. The prefix accumulator records the same recurrence over the first `k` columns.
Beyond the input width it stays at the complete accumulator. Generic
`preactivation` adds the bias once and is shared with the numerical analysis:

$$
a_0=+0,\qquad a_{j+1}=\operatorname{fmaRN}(W_{ij},x_j,a_j),
\qquad y_i=\operatorname{maxPositiveZero}(\operatorname{addRN}(a_n,b_i)).
$$

FMA and addition are explicit operation parameters on binary32 bit patterns.
The concrete FloatLib adapter supplies nearest-even binary32 addition and
single-rounding FMA, and proves agreement with FloatLib reference operations.
The generic interface itself imposes no arithmetic laws. The result definition
does not substitute real-number arithmetic or claim equality with TorchLean's
real-valued forward map. PTX arithmetic correspondence remains separate.

The maximum with positive zero is defined directly from binary32 sign, exponent,
and fraction bits. It maps NaNs and negative values, including negative zero,
to positive zero, preserves positive infinity and positive subnormal values,
and leaves positive zero unchanged. This follows the selected unmodified
[`max.f32` rule](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#floating-point-instructions-max).

The entry contract combines the supplied input words and the four actual
read/write pointers for the selected tensor row. The row register is
unrestricted: the computation uses the pointers and never reads that register. Scratch register
contents and entry PC are unrestricted. The thread is active. The execution
theorem separately requires `CodeAt` for the complete computation at that PC,
with the loop branch already targeting its actual iteration entry.

The memory result requires output location `i` to contain the ordered result,
with every other word location unchanged. The complete block theorem connects
this postcondition to actual execution of initialization, the dot-product loop,
and the bias/ReLU/store/return suffix. It proves the thread returns with the
result register containing `rowResult`. The loop theorem separately records
the completed counter and pointer progress.

Finite fixtures exercise multiple rows and columns, uninitialized output,
rejection of missing or incorrect input words, successful output/bias aliasing,
and preservation of every address other than the selected output. Artificial arithmetic
operations check column order and accumulator placement without pretending to
implement FP32 FMA or addition. Bit-pattern cases cover NaNs, infinities, signed
zeros, and subnormal values in the maximum.

The complete program appends the suffix to the initialized loop. Both theorems
execute in the same enclosing program; the loop's final PC points to the suffix.
The suffix theorem applies at that offset, using the loop's
completed accumulator, unchanged input memory, and preserved bias/output
pointers. The combined instruction count is `2 + 8 * inputWidth + 5`.

The block's public result records return, the result register, and the stored
output with every other memory location preserved. Register and PC facts stay
in the loop and suffix contracts where they support their handoff. The proof
reuses those execution theorems. The whole-kernel theorem composes this result with the three setup blocks.
The concrete FloatLib instance agrees with its reference arithmetic. The
finite-row bound connects its result to the actual real-valued Sequential
forward coordinate on decoded inputs. PTX arithmetic correspondence remains
separate.

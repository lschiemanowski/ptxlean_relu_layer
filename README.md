# ptxlean_relu_layer

A minimal Lean formalization connecting a scalar FP32 PTX kernel to a
TorchLean `nn.Sequential` specification of one dense ReLU layer:
1024 inputs, 3072 outputs, and arbitrary weights and bias.

This is a focused companion to
[PTXLean](https://github.com/lschiemanowski/ptxlean), the more ambitious project
covering broader PTX semantics and connections to TorchLean programs. PTXLean
is currently somewhat disorganized; this project isolates one layer to make its
specification, execution model and correctness statement easier to follow.

## 1. From Torch to the TorchLean specification

The familiar Torch implementation is:

```python
import torch
from torch import nn

layer = nn.Sequential(
    nn.Linear(1024, 3072, bias=True),
    nn.ReLU(),
)
x = torch.randn(1024)
y = layer(x)  # shape: [3072]
```

The first module holds `W = layer[0].weight`, of shape `[3072, 1024]`,
and `b = layer[0].bias`, of shape `[3072]`. For one input vector, the intended
mathematical computation is

$$
y_i = \max\left(\sum_{j=0}^{1023} W_{ij}x_j + b_i,\;0\right),
\qquad 0 \leq i < 3072.
$$

[Specification.lean](ReluLayer/Specification.lean) expresses the architecture
using upstream TorchLean's Sequential builder:

```lean
def architecture : nn.Builder (nn.Sequential [inputSize] [outputSize]) :=
  nn.Sequential![
    nn.linear inputSize outputSize,
    nn.relu
  ]
```

The shared dimensions `inputSize = 1024` and `outputSize = 3072` live in
[Dimensions.lean](ReluLayer/Dimensions.lean). `model` builds this architecture;
`parameters W b` supplies explicit weight and bias tensors as its state, and
`forwardProgram` is the actual Sequential forward program in evaluation mode.
The theorems cover arbitrary supplied parameters rather than just the model's
initialized state.

[RealForward.lean](ReluLayer/RealForward.lean) interprets that program over real
numbers. `evaluate_eq` proves that evaluation succeeds and computes
`ReLU(Wx + b)`. `forward` extracts the successful output, and
`forward_coordinate` proves the coordinate formula above. This real-valued
Sequential output is the target of the PTX correctness theorem. The Python
snippet illustrates the architecture; the theorem does not compare against a
particular PyTorch floating-point backend.

## 2. The PTX kernel, block by block

[kernels/relu_layer.ptx](kernels/relu_layer.ptx) assigns one output row to each
thread. The launch has 12 blocks of 256 threads, covering all 3072 rows.
Weights are row-major FP32 words; the four pointer arguments are `W`, `x`, `b`
and `y`. Each thread reads its weight row and the input vector, then writes one
output word. The following excerpts contain every executable instruction in
the kernel, in source order.

### Block 1: load base addresses (lines 22–29)

```ptx
    ld.param.u64 %W_base, [W_ptr];
    ld.param.u64 %x_read, [x_ptr];
    ld.param.u64 %b_base, [b_ptr];
    ld.param.u64 %y_base, [y_ptr];
    cvta.to.global.u64 %W_base, %W_base;
    cvta.to.global.u64 %x_read, %x_read;
    cvta.to.global.u64 %b_base, %b_base;
    cvta.to.global.u64 %y_base, %y_base;
```

`ld.param.u64` reads the four pointer arguments from parameter slots into
64-bit registers. `cvta.to.global.u64` converts each generic pointer to an
address in global memory. After this block, `%W_base`, `%x_read`, `%b_base`
and `%y_base` address the beginnings of their respective buffers. `%x_read`
will advance through the input vector during the loop.

### Block 2: select the output row (lines 32–34)

```ptx
    mov.u32 %thread, %tid.x;
    mov.u32 %block, %ctaid.x;
    mad.lo.u32 %row, %block, 256, %thread;
```

`%tid.x` is the thread's index within its block, and `%ctaid.x` is the block's
index within the grid. `mad.lo.u32` multiplies the block index by 256 and adds
the thread index, keeping the low 32 bits. For this launch the result fits:
`i = block * 256 + tid` ranges from 0 to 3071. `%row` therefore identifies the
single output coordinate owned by this thread.

### Block 3: compute data addresses (lines 37–41)

```ptx
    mul.wide.u32 %row_offset, %row, 4096;
    mul.wide.u32 %element_offset, %row, 4;
    add.u64 %W_read, %W_base, %row_offset;
    add.u64 %b_read, %b_base, %element_offset;
    add.u64 %y_write, %y_base, %element_offset;
```

Each FP32 element occupies four bytes, so a weight row occupies
`1024 * 4 = 4096` bytes. `mul.wide.u32` produces the 64-bit offsets
`i * 4096` and `i * 4`. Adding these to the buffer bases places `%W_read`
at `W[i,0]`, `%b_read` at `b[i]`, and `%y_write` at `y[i]`. `%x_read` already
points to `x[0]` and needs no row offset.

### Block 4a: compute the dot product (lines 43–55)

```ptx
    mov.u32 %j, 0;
    mov.f32 %acc, 0f00000000;
```

The loop starts at column zero with a positive-zero accumulator.
`0f00000000` is the FP32 bit-pattern literal for positive zero.

```ptx
DOT_LOOP:
    // Read column j, then perform exactly one rounded fused multiply-add.
    ld.global.f32 %w, [%W_read];
    ld.global.f32 %x, [%x_read];
    fma.rn.f32 %acc, %w, %x, %acc;
    add.u64 %W_read, %W_read, 4;
    add.u64 %x_read, %x_read, 4;
    add.u32 %j, %j, 1;
    setp.lt.u32 %more, %j, 1024;
    @%more bra DOT_LOOP;
```

The two loads read the current weight and input element. `fma.rn.f32`
computes `W[i,j] * x[j] + acc` with one final rounding to nearest, ties to even;
it does not round the product separately. The pointer increments move to the
next FP32 elements, and `%j` advances by one. `setp.lt.u32` sets `%more`
when the updated column is below 1024; `@%more bra` jumps back to `DOT_LOOP`
only when that predicate is true.

The loop visits columns 0 through 1023 in order. On exit, `%j` is 1024 and
`%acc` holds the ordered rounded dot product. No memory has been written.

### Block 4b: add bias, apply ReLU and store (lines 58–62)

```ptx
    ld.global.f32 %bias, [%b_read];
    add.rn.f32 %acc, %acc, %bias;
    max.f32 %result, %acc, 0f00000000;
    st.global.f32 [%y_write], %result;
    ret;
```

The bias load reads `b[i]` from the address computed in block 3.
`add.rn.f32` adds it to the completed accumulator, again rounding to nearest,
ties to even. `max.f32` takes the maximum with positive zero, implementing
ReLU for the finite values covered by the main theorem. The store writes
that result to `y[i]`, and `ret` ends this thread's execution.

The loop and this suffix have separate theorems. Together they establish the
precise computation: an ordered sequence of rounded FMAs, a rounded bias
addition, and a positive-zero maximum. The next section connects the stored
result to the real-valued Sequential specification with an error bound.

## 3. The main theorem: stored output versus Sequential

[`layer_kernel_forward_abs_error_le`](ReluLayer/SequentialBridge.lean)
connects the complete modeled kernel to the actual Sequential forward program.
For any selected thread satisfying the entry and finite-trace assumptions, it
proves that:

- The kernel executes through `ret` and stores a finite FP32 word at that
  thread's output address.
- Every other memory word retains its initial contents.
- The stored word, decoded to a real number, differs from the corresponding
  Sequential output coordinate by at most `traceErrorBudget`.

Here is the full statement:

```lean
theorem layer_kernel_forward_abs_error_le (data : TensorData layerLayout layerLaunch.rows)
    (env : Environment) (args : Argument → Word64) (buffers : Buffers) (t : Thread layerLaunch)
    (initial : State Register)
    (hentry : KernelEntry layerLaunch layerLayout env args buffers data initial)
    (hfinite : FiniteRowTrace data t.rowIndex) :
    ∃ final word,
      Returns (kernelProgram layerLaunch layerLayout) (layerContext t env) initial final ∧
      final.globalMemory (t.outputAddress buffers) = some word ∧
      (∀ address, address ≠ t.outputAddress buffers →
        final.globalMemory address = initial.globalMemory address) ∧
      FP32.isFinite word = true ∧
      |FP32.toReal word -
        getScalar (ReluLayer.forward (decodedWeights data) (decodedBias data) (decodedInput data))
          t.rowIndex| ≤ traceErrorBudget data t.rowIndex
```

`decodedWeights`, `decodedBias` and `decodedInput` use the real values represented
by the input FP32 words. Thus the comparison uses the same represented inputs
on both sides; it does not include an initial rounding error from encoding
arbitrary real tensors into FP32. `t.rowIndex` is the selected row with its
range proof, and `t.outputAddress buffers` is its output address.

The two hypotheses mean:

- **`KernelEntry`:** begin at PC zero in a thread that has not returned; supply
  the four pointer parameters and their successful conversions to the supplied
  global buffers; place the specified weight, input and bias words at their
  computed addresses. Scratch registers and predicates may start arbitrarily.
- **`FiniteRowTrace`:** the selected row's weights and bias, the input vector,
  every rounded accumulator prefix and the rounded bias-addition result are
  finite. Signed zeros and subnormals are included; absence of overflow is
  assumed.

`traceErrorBudget` sums the local half-ULP rounding bounds for the FMA steps and
bias addition using the actual rounded accumulators. ReLU adds no further
error term. The project proves this trace-dependent budget; it does not yet
prove an explicit constant or an input-only bound.

This is a **per-thread theorem about the formal instruction model**. Executable
checks compare the PTX source with the formal instruction list, but there is no
verified PTX parser. Agreement with documented PTX arithmetic, CUDA memory,
concurrent execution of the whole launch, and GPU hardware remains unproved.

## 4. How instructions, memory and execution are formalized

[Machine.lean](ReluLayer/PTX/Machine.lean) defines just the 16 instruction forms
used by this kernel. An instruction is a Lean constructor with explicit
operands and destinations; `Instruction.execute` gives its state transition.
Integer registers are 32- or 64-bit words, with modular arithmetic. FP32 values
are represented by their 32-bit encodings.

A `State` holds one thread's private registers, predicate registers, program
counter and returned flag, together with a global-memory snapshot. An
`ExecutionContext` supplies the immutable thread/block identity, environment
and floating-point operations. The main theorem's `layerContext` selects
FloatLib binary32 round-to-nearest-even addition and single-rounding FMA.
[FloatArithmetic.lean](ReluLayer/PTX/FloatArithmetic.lean) proves that these
operations agree with FloatLib's reference operations.

[Environment.lean](ReluLayer/PTX/Environment.lean) represents global memory as
`Word64 → Option Word32`: byte addresses map directly to complete FP32 words.
It also supplies parameter slots and a partial generic-to-global address
conversion. A missing fetch, parameter, input word or conversion gives no next
state. Valid word accesses are assumed; allocation, permissions, alignment and
byte encoding are outside this model.

A program is a list of instructions. `step` fetches the instruction at the PC;
`run` executes a specified number of steps. `Returns` states that some finite
execution succeeds and reaches a returned state. The full `kernelProgram`
concatenates the independent blocks and constructs the loop with its actual
entry offset, so its branch targets the correct instruction in that program.

## 5. The block theorems and where to find them

Each theorem executes its block under an entry contract, establishes the next
block's required values, and records what remains unchanged. Internal PC and
register facts support composition; the main theorem exposes the stored output.
The setup, loop and suffix proofs are parameterized by launch or layout where
needed; [Layer.lean](ReluLayer/PTX/Layer.lean) supplies this layer's concrete
configuration. Every block is proved directly in the enclosing program. `CodeAt program initial.pc block`
states that the block occurs at the current PC; the loop is constructed with
that PC as its entry offset. The contracts record actual PC progress along with
the computed values and preserved state.

| Block | Theorem | File | What it establishes |
| --- | --- | --- | --- |
| 1. Base addresses | `baseLoading_correct` | [BaseLoading.lean](ReluLayer/PTX/BaseLoading.lean) | Execution produces exactly `basesLoaded buffers initial`: four converted base registers, advanced PC, and every other field preserved. |
| 2. Row selection | `rowSelection_correct`, `rowSelected_row` | [RowSelection.lean](ReluLayer/PTX/RowSelection.lean) | Execution produces exactly the three register updates in `rowSelected`; a fitting launch makes the row word equal the mathematical owned row. |
| 3. Data addresses | `addressCalculation_exec` | [AddressCalculation.lean](ReluLayer/PTX/AddressCalculation.lean) | Execution produces exactly `addressesComputed width initial`, including the modular offsets, three pointers, and preserved state. |
| 4a. Dot-product loop | `dotLoop_correct` | [DotLoop.lean](ReluLayer/PTX/DotLoop.lean) | The loop exits after the input width, with the accumulator equal to the ordered rounded FMA recurrence and memory preserved. |
| 4b. Bias/ReLU/store suffix | `reluSuffix_correct`, `reluStored_result`, `reluStored_output` | [ReluSuffix.lean](ReluLayer/PTX/ReluSuffix.lean) | Execution produces exactly `reluStored`; the semantic lemmas identify the row result, its store and preserved memory frame. |

The four straight-line blocks use exact state equations, with public state
functions `basesLoaded`, `rowSelected`, `addressesComputed`, and `reluStored`.
These functions specify updates and preserved fields together. Separate lemmas
identify the mathematical row and the suffix's ordered result and stored output.
The dot-product loop keeps its invariant to describe repeated execution.

`body_correct` in [BodyExecution.lean](ReluLayer/PTX/BodyExecution.lean) combines
4a and 4b. `kernel_correct` in
[KernelExecution.lean](ReluLayer/PTX/KernelExecution.lean) combines all four
blocks. [RealDecode.lean](ReluLayer/PTX/RealDecode.lean) connects the finite
positive-zero maximum to real ReLU, and
[RoundingError.lean](ReluLayer/PTX/RoundingError.lean) supplies the error bound.
Finally, [SequentialBridge.lean](ReluLayer/SequentialBridge.lean) identifies the
real target with the Sequential output and proves the main theorem.

The main proof needs no output-separation assumption for one thread: its only
store follows every input read, so its output may alias an input word.

## Build and checks

With [elan](https://github.com/leanprover/elan) and Python 3 installed, check the
main theorem with one command from the repository root:

```sh
python3 scripts/check_main_theorem.py
```

The script prepares the pinned dependencies, builds the proofs, and runs three
check groups: kernel instructions/execution and the complete PTX source,
FP32 arithmetic/decoding/error probes, and the Sequential bridge. The checks
also print theorem axiom audits, including the main theorem's axioms.
The project uses Lean 4.35.0-rc4. No CUDA toolkit or GPU is required.

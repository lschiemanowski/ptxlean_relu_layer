# ptxlean_relu_layer

A minimal Lean formalization connecting a scalar FP32 PTX kernel to a
TorchLean `nn.Sequential` specification of one dense ReLU layer:
1024 inputs, 3072 outputs, and arbitrary weights and bias.

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
output word.

| Block | PTX lines | Instructions and purpose |
| --- | --- | --- |
| 1. Load base addresses | 22–29 | Four `ld.param.u64` loads followed by four `cvta.to.global.u64` conversions supply the global buffer bases. |
| 2. Select the row | 32–34 | Two `mov.u32` instructions read `%tid.x` and `%ctaid.x`; `mad.lo.u32` computes `i = block * 256 + tid`. |
| 3. Compute data addresses | 37–41 | Two `mul.wide.u32` instructions compute the weight-row offset `i * 4096` and element offset `i * 4`; three `add.u64` instructions form the row, bias and output pointers. The input pointer already addresses `x[0]`. |
| 4a. Compute the dot product | 43–55 | Initialize column zero and a positive-zero accumulator. Each iteration loads `W[i,j]` and `x[j]`, performs one `fma.rn.f32`, advances both pointers and the column, then branches while `j < 1024`. |
| 4b. Add bias, apply ReLU and store | 58–62 | `ld.global.f32` loads `b[i]`; `add.rn.f32` adds it; `max.f32` takes the maximum with positive zero; `st.global.f32` writes `y[i]`; `ret` ends execution. |

Block 4 is split into the loop and its suffix so that the two computations can
be understood and proved separately. The accumulation order is explicit:
start at positive zero, use one rounded fused multiply-add per column in order,
then one rounded bias addition. The real-valued specification and this rounded
computation are connected by an error bound.

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
concatenates the independent blocks and relocates the loop's branch target to
its position in that combined program.

## 5. The block theorems and where to find them

Each theorem executes its block under an entry contract, establishes the next
block's required values, and records what remains unchanged. Internal PC and
register facts support composition; the main theorem exposes the stored output.
The setup, loop and suffix proofs are parameterized by launch or layout where
needed; [Layer.lean](ReluLayer/PTX/Layer.lean) supplies this layer's concrete
configuration.

| Block | Theorem | File | What it establishes |
| --- | --- | --- | --- |
| 1. Base addresses | `baseLoading_correct` | [BaseLoading.lean](ReluLayer/PTX/BaseLoading.lean) | The four base registers contain the supplied converted global addresses; memory and unrelated state are preserved. |
| 2. Row selection | `rowSelection_correct` | [RowSelection.lean](ReluLayer/PTX/RowSelection.lean) | The row register contains the thread's mathematical row, without 32-bit wrap for a fitting launch; unrelated state is preserved. |
| 3. Data addresses | `addressCalculation_exec` | [AddressCalculation.lean](ReluLayer/PTX/AddressCalculation.lean) | The pointer registers contain the row, bias and output addresses computed by the word-level PTX arithmetic; memory is preserved. |
| 4a. Dot-product loop | `dotLoop_correct` | [DotLoop.lean](ReluLayer/PTX/DotLoop.lean) | The loop exits after the input width, with the accumulator equal to the ordered rounded FMA recurrence and memory preserved. |
| 4b. Bias/ReLU/store suffix | `reluSuffix_correct` | [ReluSuffix.lean](ReluLayer/PTX/ReluSuffix.lean) | The completed accumulator is combined with bias and positive-zero maximum, the resulting word is stored, every other memory word is preserved, and the thread returns. |

`body_correct` in [BodyExecution.lean](ReluLayer/PTX/BodyExecution.lean) combines
4a and 4b. `kernel_correct` in
[KernelExecution.lean](ReluLayer/PTX/KernelExecution.lean) combines all four
blocks. [RealDecode.lean](ReluLayer/PTX/RealDecode.lean) connects the finite
positive-zero maximum to real ReLU, and
[RoundingError.lean](ReluLayer/PTX/RoundingError.lean) supplies the error bound.
Finally, [SequentialBridge.lean](ReluLayer/SequentialBridge.lean) identifies the
real target with the Sequential output and proves the main theorem.

The main proof needs no output-separation assumption for one thread: its only
store follows every input read, so its output may alias an input word. Optional
launch-level separation contracts live in
[Optional/Separation.lean](ReluLayer/PTX/Optional/Separation.lean). Optional
nonwrapping natural-address results live in
[Optional/AddressBounds.lean](ReluLayer/PTX/Optional/AddressBounds.lean). Both
modules are outside the main import chain.

## Build and checks

With [elan](https://github.com/leanprover/elan) installed, build with:

```sh
lake update
python3 scripts/prepare_dependencies.py
LEAN_NUM_THREADS=4 lake build
```

The project pins Lean 4.35.0-rc4, TorchLean, FloatLib and matching transitive
packages. `prepare_dependencies.py` applies four checked compatibility edits to
isolated package checkouts; it is idempotent, verifies exact revisions and source
contents, and preserves arithmetic definitions. Project contracts and exact
source links are maintained in [Stratic](stratic/descriptions/root.md).

Run the theorem audits, contract edge cases, arithmetic oracles and source
correspondence checks with:

```sh
lake build ReluLayer.PTX.Optional.AddressBounds ReluLayer.PTX.Optional.Separation
lake env lean --run checks/RowSelection.lean
lake env lean --run checks/AddressCalculation.lean
lake env lean --run checks/BaseLoading.lean
lake env lean --run checks/BodyContract.lean
lake env lean --run checks/DotLoop.lean
lake env lean --run checks/ReluSuffix.lean
lake env lean --run checks/BodyExecution.lean
lake env lean --run checks/KernelExecution.lean
lake env lean --run checks/FloatArithmetic.lean
lake env lean --run checks/RealDecode.lean
lake env lean --run checks/RoundingError.lean
lake env lean --run checks/SequentialBridge.lean
```

With the CUDA toolkit installed, `ptxas -arch=sm_80 -v
kernels/relu_layer.ptx -o /tmp/relu-layer-sm80.cubin` checks assembly for the
declared target. It does not execute the kernel.

# ptxlean_relu_layer

A minimal Lean demonstration connecting a scalar FP32 PTX kernel for
`ReLU(Wx + b)` to upstream TorchLean's `nn.Sequential`. The shared dimensions
are 1024 inputs and 3072 outputs. The selected kernel uses 12 blocks of 256
threads, one thread per output row.

The public result is
[`layer_kernel_forward_abs_error_le`](ReluLayer/SequentialBridge.lean).
For any valid selected thread, under the entry and finite-trace contracts, the
modeled kernel returns and stores a finite word at that thread's output address.
Every other word address retains its initial contents. The stored value differs
from the actual real-valued Sequential forward coordinate on decoded inputs by
at most the existing accumulated rounding budget.

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

`Thread.rowIndex` packages the owned row and its range proof. `word` is the
value in output memory, so the public conclusion needs no PC, counter or scratch
register facts. The block proofs retain necessary internal facts for composition. `Returns`
asserts a successful finite execution and a returned final state. The concrete
`layerContext` selects the thread identity, environment and FloatLib arithmetic.
`traceErrorBudget` sums the local half-ULP budgets of the ordered FMA loop and
bias addition using the actual rounded accumulators. ReLU contributes no
additional error term. No explicit constant or input-only error estimate is
claimed.

To inspect the proof chain:

1. [Dimensions](ReluLayer/Dimensions.lean) supplies the shared shapes, and
   [Layer](ReluLayer/PTX/Layer.lean) supplies the concrete launch and layout.
   [Specification](ReluLayer/Specification.lean) defines the upstream Sequential
   architecture and explicit weight-then-bias state.
   [RealForward](ReluLayer/RealForward.lean) proves evaluation succeeds and each
   coordinate equals the real dense-ReLU formula.
2. [Machine](ReluLayer/PTX/Machine.lean) defines instruction execution over
   private registers, predicates and PC with abstract global word memory.
   [Environment](ReluLayer/PTX/Environment.lean) supplies pointer parameters and
   the supplied partial generic-to-global address conversion. [FloatArithmetic](ReluLayer/PTX/FloatArithmetic.lean)
   supplies FloatLib binary32 nearest-even addition and single-rounding FMA,
   with proofs of agreement with FloatLib reference arithmetic.
3. [BaseLoading](ReluLayer/PTX/BaseLoading.lean),
   [RowSelection](ReluLayer/PTX/RowSelection.lean) and
   [AddressCalculation](ReluLayer/PTX/AddressCalculation.lean) prove the setup
   blocks independently. [DotLoop](ReluLayer/PTX/DotLoop.lean) and
   [ReluSuffix](ReluLayer/PTX/ReluSuffix.lean) prove the computation.
   [BodyExecution](ReluLayer/PTX/BodyExecution.lean) and
   [KernelExecution](ReluLayer/PTX/KernelExecution.lean) compose these on the
   concatenated instruction program, including relocation of its branch target.
4. [RealDecode](ReluLayer/PTX/RealDecode.lean) proves finite decoding turns the
   maximum with positive zero into real ReLU.
   [RoundingError](ReluLayer/PTX/RoundingError.lean) bounds the ordered rounded
   computation. [SequentialBridge](ReluLayer/SequentialBridge.lean) identifies
   its real target with the actual Sequential output and proves the headline.

The two public hypotheses have concrete responsibilities:

- `KernelEntry`: start at PC zero in an active state; supply the four pointer
  parameters and their successful conversions to the supplied global buffers; put the specified input words at
  their computed addresses. Scratch registers and predicates are arbitrary.
- `FiniteRowTrace`: the selected row's weights and bias, the input vector, every
  accumulator prefix and the rounded bias-addition result are finite. Signed
  zeros and subnormals are included. Absence of overflow is assumed.

One thread needs no output-separation hypothesis: its only store follows every
input read. The output may alias an input, in which case that address is updated;
only *other addresses* are preserved. `OutputsSeparated` lives in
[Optional/Separation](ReluLayer/PTX/Optional/Separation.lean) as a launch-level
contract for future composition of different threads.

The model maps byte addresses directly to optional complete FP32 words. Valid
word accesses are assumed. Allocation, permissions, alignment and byte encoding
are outside it. Optional nonwrapping natural-address results live in
[AddressBounds](ReluLayer/PTX/Optional/AddressBounds.lean); the main kernel proof
uses modular word addresses and does not import that module.

This is a per-thread theorem about the formal instruction model and FloatLib
arithmetic. [The PTX source](kernels/relu_layer.ptx) is compared with the formal
instruction list by executable checks. Those checks are not a verified parser.
Agreement with documented PTX arithmetic, CUDA memory, all-thread execution or
arbitrary GPU interleavings, and GPU hardware remains unproved. Reviewing the
meaning of the machine and arithmetic definitions is therefore essential in
addition to checking the Lean proofs.

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

# Bias, ReLU, store, and return

Five instructions load the selected row's bias, add it to the completed
accumulator, take the maximum with positive zero, store one output word, and
return from the kernel entry thread. This is PTX lines 58–62, separate from
the dot-product loop.

The entry contract requires an active thread, fragment-relative PC zero, the
completed ordered dot-product accumulator, the supplied bias word at the bias
pointer, and the output pointer for the same row. Initial bias/result scratch
registers and output contents are unrestricted. Dimensions come from the tensor
specification; the suffix has no independent width or row-count constants.

The execution theorem proves the thread returns after these five instructions,
with the bias loaded, the accumulator containing the supplied rounded bias
addition, the result register and output location containing `rowResult`, and
every other word-memory location preserved. Pointer registers, predicates, and
unrelated 32-bit registers remain unchanged. The returned flag makes subsequent
machine steps unavailable, including when further instructions follow `ret`.

The result uses the existing bit-pattern maximum with positive zero and the
explicitly supplied addition operation. Allocation, permissions, byte encoding,
concrete rounded-arithmetic correctness, and real-valued TorchLean equality
remain outside this theorem. The suffix and loop proofs remain independent;
the complete block theorem composes them. Whole-kernel composition reuses the
complete computational-block result.

The return rule is restricted to this top-level kernel entry thread. NVIDIA's
[`ret` description](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#control-flow-instructions-ret)
states that returning from a top-level entry routine terminates the thread.
Function-call stacks and divergent-return scheduling are not modeled here.

The executable checker compares all five instructions with the PTX suffix and
exercises nonzero row selection, bias addition, uninitialized output, preserved
inputs/other outputs, negative and NaN preactivation, positive infinity and
subnormals, and return with a valid instruction still present afterward.
Arithmetic fixtures are explicitly artificial operation parameters.

The same suffix theorem also runs at the offset immediately after any supplied
instruction prefix, without executing that prefix or resetting the PC. Its
standalone version specializes to an empty prefix and entry PC zero. This offset
form allows the complete block theorem to use the existing suffix proof.

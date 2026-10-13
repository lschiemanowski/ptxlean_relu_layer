# Row selection

A thread selects its output row by executing two `mov.u32` instructions followed
by `mad.lo.u32`. Its immutable identity supplies `%tid.x` and `%ctaid.x`; its
mutable state contains private 32-bit and 64-bit register banks, Boolean
predicates, a program counter, and a global-memory snapshot. Register names and
operands are explicit. Each of these three instructions reads the incoming
register state, writes only its destination, and advances the program counter.

`mov.u32` copies a 32-bit operand. `mad.lo.u32` multiplies two unsigned 32-bit
operands, keeps the low 32 bits, and adds its third operand modulo `2^32`.
These rules apply to arbitrary operand values and destination registers,
including a destination that is also a source. The definitions follow NVIDIA's
[move](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov) and
[integer multiply-add](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad)
instruction descriptions.

The launch configuration supplies positive block and thread counts. A thread's
block index is below the block count and its local thread index is below the
thread count. The row-selection program uses the configured thread count as
its multiply-add immediate. It can start at any program counter where the enclosing program contains
these three instructions. In the kernel, this is immediately after the
pointer-loading and conversion instructions.

For any initial register contents, executing the three instructions succeeds,
advances the program counter by three, and writes the row register. When the
total thread count is at most `2^32`, its unsigned value is

$$
i = \mathrm{threadsPerBlock}\cdot\mathrm{blockIndex}+\mathrm{threadIndex},
\qquad 0\le i < \mathrm{blockCount}\cdot\mathrm{threadsPerBlock}.
$$

This bound proves that the computed row does not wrap. It is a property of
execution through the register updates, independent of the initial contents.
The block preserves every 64-bit register and the global-memory snapshot.
The instruction semantics and general theorem contain no layer dimensions or
chosen launch counts.

The concrete layer launch has 12 blocks and 256 threads per block, hence 3072
rows. Its three-instruction program corresponds to the row-selection fragment
in `kernels/relu_layer.ptx`. Source correspondence is checked by comparing the
rendered formal instructions and configured launch against the PTX file; the
comparison is an executable check, rather than a proved PTX parser.

`rowSelection_correct` gives an exact state equation: executing the placed block
returns `some (rowSelected t initial)`. The public `rowSelected` function performs
the two identity writes and the word-level multiply-add write, advancing the PC
with each write. Its unchanged fields express preservation of the 64-bit
registers, memory, predicates and active status, and it updates only the three
selected 32-bit registers. No separate row-result structure is needed.

Execution requires an active thread, a matching context identity, and `CodeAt`
for the three instructions at the initial PC. It succeeds even for a launch
whose mathematical row does not fit in 32 bits; the arithmetic is modular.
The separate `rowSelected_row` lemma uses the launch-size bound to identify
the unsigned row word with the mathematical owned row. `Thread.rowIndex`
packages that natural row with its tensor range proof. Kernel composition uses
the state equation for execution and preservation, and the row lemma for
address selection.

The concrete launch and layout are defined together in `Layer.lean`; its output
count agrees with the shared layer output size.

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
its multiply-add immediate. It starts at fragment-relative program counter
zero, after the kernel's pointer-loading and conversion instructions.

For any initial register contents, executing the three instructions succeeds,
leaves the program counter at three, and writes the row register. When the
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

One state-form theorem accepts arbitrary initial scratch registers and
predicates, an active thread at PC zero, and arbitrary arithmetic operation
parameters. It preserves 64-bit registers, memory, predicates, and active status
while computing the same owned row. Its named `RowResult` contract records
these facts; `Thread.rowIndex` packages the natural row with its range proof.
The configured output count agrees with the shared layer output size.

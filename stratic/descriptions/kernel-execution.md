# Whole-kernel execution

An active thread starts at the kernel entry with four initialized pointer
parameters. Their generic addresses convert successfully to the supplied
global buffer addresses through the environment's partial conversion function.
Those buffer addresses identify the supplied tensor words. Output
separation is not required for one thread: its only store follows all reads.
Scratch registers and predicates may initially contain arbitrary values.

The kernel loads and converts the bases, selects the thread's output row,
computes the row pointers, and executes the initialized dot loop followed by
bias addition, maximum with positive zero, store, and return. Launch dimensions
and input width are parameters. The launch's row count fits in an unsigned
32-bit register; the weight stride is derived from the input width.

For each valid thread, execution terminates with its output word equal to the
ordered `rowResult` and every other memory location unchanged. Register and PC bookkeeping remains internal to the block proofs. A required
execution context supplies the thread identity, environment and arithmetic.
`Returns` means some finite successful run reaches a state whose returned flag
is set; the public theorem does not expose instruction fuel. The
instruction count is the setup length plus `2 + 8 * inputWidth + 5`.
The selected 1024-input kernel has 31 static instructions and executes 8215
instructions per thread.

The four block theorems compose on the actual concatenated instruction list.
All block theorems execute directly at their positions in the kernel under
instruction-placement conditions, including the base-loading prefix. The loop and suffix likewise execute in the enclosing kernel. The loop
instruction list is constructed with its actual entry offset, so its branch
already targets the loop body in that program. Block boundaries do
not reset the machine's PC or change registers or memory.

This is an execution theorem for each thread in the abstract word-memory model,
using supplied FMA and addition operations on binary32 bit patterns. It assumes
valid word accesses and the input-value contract. A concrete FloatLib specialization supplies binary32 nearest-even addition and
FMA and identifies the result with FloatLib reference arithmetic. Correspondence
with PTX arithmetic rules, CUDA memory refinement, concurrent GPU execution,
and hardware correspondence remain separate obligations. For the selected
layer and a finite row, the existing numerical bound compares the stored word
directly with the upstream real-valued Sequential forward coordinate on decoded
weights, bias and input.

Verification uses three groups: kernel execution, numerical semantics, and the
Sequential bridge. The kernel group exercises individual instructions, block
placement, failed accesses, address wraparound, aliasing, and memory frames.
One complete source comparison checks the instruction list and branch label,
the ordered pointer parameters, and the required and documented launch sizes.
These executable source checks are comparisons for this kernel, rather than a
verified PTX parser. The repository's single verification command runs all
three groups and prints their theorem axiom audits.

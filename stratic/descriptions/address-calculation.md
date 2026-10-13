# Address calculation

After row selection, five instructions form the pointers to `W[i,0]`, `b[i]`,
and `y[i]`. The input width is a parameter. FP32 elements occupy four bytes,
so the weight-row stride is derived as `inputWidth * 4`. The concrete width
1024 gives stride 4096; it is not an independent layout parameter.

The machine has separate private 32-bit and 64-bit register banks.
`mul.wide.u32` reads two unsigned 32-bit operands and writes their full 64-bit
product. `add.u64` adds two 64-bit operands modulo `2^64`. Operands are read
from the incoming state before the destination is written. Each instruction
advances the program counter and preserves the other registers. These rules
follow NVIDIA's [integer multiplication](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)
and [integer addition](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-add)
descriptions.

The fragment starts with an active thread, an unsigned row register, and converted
global base-address registers for weights, bias, and output. `CodeAt` requires
its five instructions to occur at the current PC in the enclosing program. It executes two wide
multiplies and three additions through instruction fetches and register
updates. For arbitrary initial scratch registers, its program counter
advances by the block length, the row, base registers, input pointer, and global-memory snapshot are
preserved, and its computed pointers
satisfy

$$
W_{\mathrm{read}}=W_{\mathrm{base}}+i\,(\mathrm{inputWidth}\cdot4),
\qquad b_{\mathrm{read}}=b_{\mathrm{base}}+i\cdot4,
\qquad y_{\mathrm{write}}=y_{\mathrm{base}}+i\cdot4.
$$

The main execution statement gives modular pointer values, including when
pointer addition wraps. The layout requires a positive input width and a row
stride fitting a 32-bit immediate.

The layer specializes the width to 1024 and the row count to the configured
launch's 3072 rows. The five formal instructions are included in the complete
kernel source comparison. The correspondence checker
is not a proved parser. Base-address initialization and conversion, actual
memory contents, loads and stores, allocation, alignment, and disjointness
are outside this address-arithmetic result.

The word-level exit contract also preserves predicates and active status.
Its execution theorem accepts a required execution context; this block does
not use the context's arithmetic.
Whole-kernel composition uses the modular word pointer equations directly.

The public `addressesComputed width initial` state specifies both offsets and
the three resulting pointers, with PC advance and every unchanged field.
`addressCalculation_exec` states that execution equals this explicit state.
The offsets and additions remain word arithmetic, including wraparound; kernel
composition separately uses the fitting launch and stride to identify the
selected tensor addresses.

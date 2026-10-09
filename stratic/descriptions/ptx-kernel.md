# Scalar FP32 kernel

The kernel computes the 1024-input, 3072-output dense ReLU layer with one thread
per output coordinate. It uses PTX 8.0, target `sm_80`, and 64-bit addressing.
The four kernel arguments are generic pointers to global buffers, in the order
`W`, `x`, `b`, `y`.

The launch has grid dimensions `(12, 1, 1)` and block dimensions `(256, 1, 1)`.
Thread `tid.x` in block `ctaid.x` owns row `i = 256 * ctaid.x + tid.x`.
These 3072 threads cover the output coordinates exactly once. The `.reqntid`
directive fixes the block dimensions; the grid dimensions are a launch
precondition. The kernel has no extra-row guard.

The buffers contain contiguous FP32 words with four-byte alignment. Weights use
row-major layout. All four buffers are disjoint, their address ranges do not
wrap, and their sizes are:

| Buffer | Shape | Bytes | Address of an element |
|---|---|---:|---|
| `W` | `[3072, 1024]` | 12582912 | `W_base + 4 * (1024 * i + j)` |
| `x` | `[1024]` | 4096 | `x_base + 4 * j` |
| `b` | `[3072]` | 12288 | `b_base + 4 * i` |
| `y` | `[3072]` | 12288 | `y_base + 4 * i` |

Each base plus its buffer size is less than `2^64`. The three input buffers are
initialized before launch and remain unchanged until the kernel completes.
The output is read after completion.

Each thread initializes its accumulator to positive zero and reads columns in
the order `j = 0, ..., 1023`. Each iteration loads `W[i,j]` and `x[j]` and updates
the accumulator with `fma.rn.f32`. After the loop, `add.rn.f32` adds `b[i]`,
`max.f32` selects the maximum with positive zero, and `st.global.f32` writes
`y[i]`.

The FMA rounds the combined product and sum once to binary32, using nearest-even
rounding. The bias addition also uses nearest-even rounding. No instruction has
the `.ftz` modifier. The final maximum maps negative values and either signed
zero to positive zero. A NaN preactivation yields positive zero with this
`max.f32` form; positive infinity is retained. These are the selected PTX
arithmetic rules, as described in NVIDIA's archived
[FMA](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#floating-point-instructions-fma),
[addition](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#floating-point-instructions-add), and
[maximum](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#floating-point-instructions-max)
sections.

The register groups hold thread and row indices (`%thread`, `%block`, `%row`),
the loop counter (`%j`), global buffer addresses and byte offsets, the loaded
weight and input (`%w`, `%x`), the accumulator (`%acc`), the bias (`%bias`),
the result (`%result`), and the loop predicate (`%more`). The loop advances both
read pointers by four bytes after each FMA. Its final pointer increments are
not dereferenced. Threads use private registers and communicate only through
their own output stores.

The instruction subset contains the following exact forms. Counts refer to
static instructions in the PTX source; the loop body executes 1024 times per
thread.

| Exact form | Count | Role |
|---|---:|---|
| `ld.param.u64` | 4 | Read pointer arguments |
| `cvta.to.global.u64` | 4 | Convert generic pointers to global addresses |
| `mov.u32` | 3 | Read thread/block IDs and initialize the counter |
| `mad.lo.u32` | 1 | Compute the output row |
| `mul.wide.u32` | 2 | Compute 64-bit byte offsets |
| `add.u64` | 5 | Form addresses and advance pointers |
| `mov.f32` | 1 | Initialize positive zero |
| `ld.global.f32` | 3 | Load weight, input, and bias |
| `fma.rn.f32` | 1 | Accumulate a product |
| `add.u32` | 1 | Increment the column counter |
| `setp.lt.u32` | 1 | Test the loop bound |
| `bra` | 1 | Repeat while `%more` is true |
| `add.rn.f32` | 1 | Add bias |
| `max.f32` | 1 | Apply the selected ReLU rule |
| `st.global.f32` | 1 | Store the output |
| `ret` | 1 | Finish the thread program |

The source has 31 static instructions and 16 exact forms. The execution model
also requires per-thread program counters, predicate-guarded execution, reads
of `%tid.x` and `%ctaid.x`, parameter memory, global memory, and the declared
launch dimensions. NVIDIA's [PTX 8.0 specification](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html)
defines these instruction forms and computing rules.

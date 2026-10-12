# Ordered dot-product loop

The loop initializes `%j` to zero and `%acc` to positive zero. For each column it
loads the weight and input words, performs one FMA update, advances both pointers
by four bytes, increments the counter, compares it with the input width, and
branches back while another column remains. It exits before the bias load.

After `k` iterations, the counter is `k`, the accumulator is the ordered prefix
of `k` FMA updates, and the read pointers select column `k`. The memory map is
unchanged. Other registers remain available for the bias/ReLU/store suffix.
The input width is positive and fits the 32-bit counter; the layer specializes
it to 1024. The final one-past-end pointers are never dereferenced by the loop.

The proof follows instruction fetches and execution, including the predicate
comparison and taken or untaken branch. FMA is a supplied operation on FP32 bit
patterns. No concrete rounded-arithmetic correctness, real-valued equality,
bias/ReLU/store execution, whole-kernel composition, or device-memory refinement
is claimed by this loop result.

The machine stores Boolean predicates separately from 32-bit register words.
A global load reads the word at the supplied address; a missing word has no
modeled successor. FMA reads all three old operands before updating its
destination and requires an explicit arithmetic implementation. Unsigned
`setp.lt.u32` writes the less-than test to `%more`; the guarded branch either
sets the PC to the loop entry or advances to the next instruction. These rules
follow NVIDIA's [comparison](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)
and [branch](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#control-flow-instructions-bra)
descriptions within the selected word-memory/arithmetic abstractions.

Two initialization instructions precede eight instructions per iteration.
The fragment-relative loop entry is PC 2 and its exit is PC 10, so a width `n`
executes `2 + 8 * n` instructions. These counts describe the instruction layout,
not extra shape parameters. Positive input width and the existing row-stride
bound ensure the counter never wraps in the proof.

The executable checker compares the ten static instructions and loop-label
position with PTX lines 43–55. It checks taken/untaken branches, unsigned
comparison, missing operands and FMA aliasing, small ordered traces and a full
1024-column run using explicitly artificial arithmetic. This is a tested source
correspondence check, not a verified PTX parser or GPU execution.

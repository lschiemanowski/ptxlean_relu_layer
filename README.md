# ptxlean_relu_layer

A minimal Lean formalization of a PTX implementation of `ReLU(Wx + b)`, with
1024 input coordinates and 3072 output coordinates.

The specification uses upstream TorchLean's `nn.Sequential`, with a linear layer
followed by ReLU. [Specification.lean](ReluLayer/Specification.lean) defines the
architecture, explicit weight and bias state, and the upstream forward program.
[RealForward.lean](ReluLayer/RealForward.lean) interprets that program over real
numbers, proves evaluation succeeds, and proves the coordinate formula for
arbitrary weights, bias, and input.

[relu_layer.ptx](kernels/relu_layer.ptx) is a scalar FP32 kernel with one thread
per output row. It requires 12 blocks of 256 threads and four disjoint contiguous
buffers, passed as `W`, `x`, `b`, `y`. It accumulates 1024 fused multiply-adds in
column order, adds bias, and takes the maximum with positive zero. Its launch,
memory layout, arithmetic, and instruction forms are described in
[Stratic](stratic/descriptions/ptx-kernel.md).

[Machine.lean](ReluLayer/PTX/Machine.lean) defines private 32-bit registers,
a program counter, and execution of moves and low multiply-adds.
[RowSelection.lean](ReluLayer/PTX/RowSelection.lean) defines launch indexing and
proves the three-instruction row-selection fragment computes an in-range row
for arbitrary initial registers. The theorem takes positive launch dimensions
as parameters; `layerLaunch` selects the concrete 12-by-256 configuration.
Memory operations, the loop, and floating-point correspondence remain unproved.

With [elan](https://github.com/leanprover/elan) installed, build with:

```sh
LEAN_NUM_THREADS=1 lake build
```

`lean-toolchain` pins Lean 4.34.0; `lakefile.toml` and `lake-manifest.json` pin
TorchLean and its transitive dependencies. Lake downloads dependencies on the
first build. Project descriptions and source links are in `stratic/`.

With the CUDA toolkit installed, check PTX assembly with:

```sh
ptxas -arch=sm_80 -v kernels/relu_layer.ptx -o /tmp/relu-layer-sm80.cubin
```

This checks assembly for the declared target. It does not execute the kernel.

Check row-selection edge cases, theorem axioms, and correspondence between the
formal instructions/configuration and the PTX source with:

```sh
lake env lean --run checks/RowSelection.lean
```

The grid check reads the documented launch precondition. The block-size check
reads `.reqntid`; neither runs a GPU launch.

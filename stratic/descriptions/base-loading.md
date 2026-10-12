# Base-address loading

Eight instructions load the four `.param .u64` kernel arguments and convert
their generic pointers to global addresses. The named parameters are `W_ptr`,
`x_ptr`, `b_ptr`, and `y_ptr`; the destination registers are `%W_base`,
`%x_read`, `%b_base`, and `%y_base`.

Parameter storage is a read-only map from declared symbol names to optional
64-bit words. `ld.param.u64` reads the named slot and writes its word into a
64-bit register. This models direct loads from declared, naturally aligned
64-bit parameter slots. Host argument serialization and byte-level parameter
layout are outside this representation. The rule follows NVIDIA's
[load instruction](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
description.

The address-space configuration supplies a bounded global window and a list
of excluded windows for other state spaces. A generic pointer is admitted
when it lies within the global window and outside the excluded windows.
`cvta.to.global.u64` converts it by subtracting the global-window base.
The conversion reads the incoming register value, including when source and
destination are the same register. The window bounds prevent subtraction or
representation wraparound for admitted pointers. This is the selected window
model described by NVIDIA's [generic addressing](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#generic-addressing)
and [address conversion](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta)
sections. The window configuration is an explicit environment assumption;
its correspondence to a CUDA device is not proved.

The execution rules return no modeled successor for a missing parameter slot
or a pointer outside the admitted global region. This restricts the model to
valid uses of these instructions. NVIDIA leaves invalid address conversions
undefined; the absence of a modeled successor does not specify a GPU trap.

The block theorem assumes all four named slots contain the supplied generic
pointers and that each pointer is admitted by the address-space configuration.
Starting with an active thread at fragment-relative PC zero, all eight
instructions succeeds. The final program counter is eight, and every target
register contains the converted address of its corresponding parameter.
Every 32-bit register, every other 64-bit register, and the global-memory word
map are preserved. The parameter and address-space environment is read-only.

All kernel fragments use the same register-name type and machine state.
Their theorems remain separate. The eight formal instructions and the four
64-bit parameter declarations are compared with the PTX source using an
executable source check. Tensor allocation and contents, later data loads and
stores, complete-kernel composition, and GPU execution remain outside this
block theorem.

The exit contract also preserves the predicate bank and active status.
The block accepts arbitrary arithmetic operation parameters because none of
its instructions uses floating-point arithmetic.

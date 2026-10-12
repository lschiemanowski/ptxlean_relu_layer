# Concrete binary32 arithmetic

The kernel's arithmetic operates on 32-bit IEEE binary32 encodings. A concrete
FloatLib adapter supplies addition and fused multiply-add with nearest-even
rounding and support for subnormal inputs and results. FMA rounds the exact
product-plus-addend once; it is not separate rounded multiplication and addition.

Converting between kernel words and FloatLib values preserves every bit pattern,
including signed zeros, infinities, and NaN payloads. The adapter's executable
addition and FMA agree with FloatLib's reference binary32 operations. The
whole-kernel theorem specializes to these concrete operations while retaining
its input-memory assumptions. Single-thread execution does not require output
separation.

PTX's single-precision arithmetic does not prescribe a unique NaN encoding.
The adapter chooses FloatLib's NaN policy; equality of intermediate NaN payloads
with arbitrary GPU implementations is not claimed. The selected final maximum
with positive zero maps every NaN to positive zero.

Arithmetic refinement establishes agreement with the FloatLib specification.
The finite-row bound connects the stored result to the actual real-valued
TorchLean sequential forward output on decoded inputs. Correspondence with the
documented PTX rules and GPU hardware behavior remain separate claims.

Finite preactivations decode exactly through the selected maximum to real ReLU.
The output remains finite, including at signed zeros and subnormals. ReLU does
not increase an absolute-error bound between the decoded preactivation and any
real comparison value.

For finite inputs and finite executed intermediates, the ordered FMA loop and
bias addition have an accumulated absolute-error bound relative to exact real
arithmetic on the represented inputs. The bound sums local half-ULP budgets;
ReLU preserves it on the stored output.

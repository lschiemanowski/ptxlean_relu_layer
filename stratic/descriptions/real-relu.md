# Real decoding and ReLU

Finite binary32 words denote exact real values through FloatLib's decoding.
Both signed zeros denote the real number zero. Positive and negative subnormal
values are included; NaNs and infinities are excluded by the finite-input
condition.

The kernel's maximum with positive zero preserves finiteness and decodes to
real ReLU exactly:

$$
\operatorname{toReal}(\operatorname{maxPositiveZero}(w))
=\max(\operatorname{toReal}(w),0).
$$

This operation introduces no additional rounding. Its positive result is the
original word, and its zero result is positive zero. Thus an absolute-error
bound on a finite preactivation remains valid after ReLU:

$$
|\operatorname{toReal}(\operatorname{maxPositiveZero}(w))-\max(z,0)|
\le |\operatorname{toReal}(w)-z|.
$$

The real decoding follows FloatLib's total convention of returning zero for
exceptional encodings. The bridge theorems require finiteness; this convention
does not assign a mathematical real value to NaN or infinity. Bit-level
exceptional behavior of the kernel remains described separately.

This bridge relates the already-computed finite preactivation to real ReLU.
The accumulated-error theorem uses this bridge after bounding the ordered FMA
loop and bias addition. The sequential bridge identifies its real target with
the actual TorchLean forward output on decoded weights, bias and input.

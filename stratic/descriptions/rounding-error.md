# Accumulated rounding error

The numerical bound compares the kernel's ordered binary32 computation with
exact real arithmetic on the values represented by the same input words.
It introduces no input-encoding error. The input width and output row count
remain parameters.

For a selected row, weights, input words, bias, every executed accumulator
prefix, and the result of bias addition are finite. These conditions exclude
NaNs and overflow to infinity while allowing signed zeros and subnormals.
They are explicit hypotheses, not consequences of finite inputs alone.

Each FMA contributes at most half an ULP at its exact product-plus-addend,
using the actual rounded accumulator entering that step. The bias addition
contributes the same kind of local bound. The budget is the sum of these local
half-ULP bounds, including the gradual-underflow grid near zero.

If `a_j` is the decoded rounded accumulator, and `s_j` is the exact sum of the
first `j` represented products, then:

$$
|a_k-s_k|\le\sum_{j<k}\frac12\operatorname{ulp}(W_{ij}x_j+a_j).
$$

After bias addition and ReLU, the decoded row result differs from the exact
dense-ReLU formula by at most that FMA budget plus the bias-addition bound.
The ReLU step adds no error term. The same budget therefore bounds the decoded
word stored by the existing concrete kernel execution theorem.

The public budget is a finite sum over the input-column indices plus the
bias-addition term. Prefix sums used in its induction proof are internal.
The budget follows the executed accumulator values. It is not an input-only
relative-error estimate. No overflow-free input range, NaN/infinity behavior,
PTX arithmetic adequacy, or GPU hardware correspondence is established by this
finite-value numerical bound.

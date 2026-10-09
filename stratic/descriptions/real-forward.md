# Real-valued forward evaluation

The sequential layer is interpreted over real numbers using TorchLean's pure
specification backend. Tensor references are ordinary real-valued tensors, and
each operation uses its upstream mathematical specification.

For every weight tensor `W : Tensor ℝ [3072, 1024]`, bias tensor
`b : Tensor ℝ [3072]`, and input `x : Tensor ℝ [1024]`, evaluation succeeds
and produces a tensor `y : Tensor ℝ [3072]`. The supplied weights and bias
are the model's parameter state.

The linear layer's internal singleton batch dimension, transpose, and bias
broadcast preserve the vector affine map `Wx + b`.

Every output coordinate satisfies

$$
  y_i=\max\left(\sum_{j=0}^{1023} W_{ij}x_j+b_i,\;0\right),
  \qquad 0\leq i<3072.
$$

This is an exact equality for the real-valued sequential specification.

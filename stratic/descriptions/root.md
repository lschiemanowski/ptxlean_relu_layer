# A minimal dense ReLU layer

This project formalizes in Lean a PTX implementation of the forward map

$$
  x \longmapsto \operatorname{ReLU}(Wx+b),
  \qquad x\in\mathbb{R}^{1024},\quad
  W\in\mathbb{R}^{3072\times1024},\quad b\in\mathbb{R}^{3072}.
$$

Each output coordinate has the mathematical specification

$$
  y_i=\max\left(\sum_{j=0}^{1023} W_{ij}x_j+b_i,\;0\right),
  \qquad 0\leq i<3072.
$$

The specification is authored using upstream TorchLean's `nn.Sequential`, with
a linear layer from shape `[1024]` to `[3072]` followed by ReLU. Its public
builder syntax is `nn.Sequential![nn.linear 1024 3072, nn.relu]`. The linear
layer includes the bias; it is the affine map in the equation above.

The weights and bias are arbitrary explicit parameter state, and the input is
an arbitrary vector. The formalization relates the output stored by the PTX
kernel to this sequential specification interpreted over real numbers.

The project covers one forward layer for one input vector. Its PTX formalization
contains the instruction forms, machine state, and execution rules needed by
the layer's kernel.

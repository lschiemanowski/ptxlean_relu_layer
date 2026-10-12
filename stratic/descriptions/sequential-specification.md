# Sequential specification

The shared `inputSize` and `outputSize` define the concrete dimensions, 1024
and 3072, for both the specification and PTX layout/launch.

The layer's architecture is an upstream TorchLean `nn.Sequential [1024] [3072]`
containing `nn.linear 1024 3072` followed by `nn.relu`. The linear layer owns a
weight tensor of shape `[3072, 1024]` and a bias tensor of shape `[3072]`. ReLU
has no parameter state.

The model's state layout is therefore `[[3072, 1024], [3072]]`, in weight-then-bias
order. An explicit parameter state contains the supplied weights and bias.
These tensors can have any values of the selected scalar type; they are not
restricted to the builder's initialization.

The evaluation-mode forward program takes the weights, bias, and input in that
order, and returns a tensor of shape `[3072]`. It uses the sequential model's
upstream forward program. Tensor operations are interpreted by the supplied
TorchLean operation backend.

The Lean package pins TorchLean and its transitive dependencies and contains no
dependency on the PTXLean package.

import NN.API.Macros
import NN.API.Seeded

namespace ReluLayer

open TorchLean

/-- A linear layer with bias, followed by elementwise ReLU. -/
def architecture : nn.Builder (nn.Sequential [1024] [3072]) :=
  nn.Sequential![
    nn.linear 1024 3072,
    nn.relu
  ]

/-- Build the architecture; forward evaluation receives its parameters separately. -/
def model : nn.Sequential [1024] [3072] :=
  nn.build 0 architecture

/-- The linear layer owns weights and bias, in that order; ReLU adds no state. -/
theorem state_shapes : nn.stateShapes model = [[3072, 1024], [3072]] := by
  rfl

/-- Supply arbitrary weights and bias instead of using initialized model state. -/
def parameters {α : Type} [Storage α]
    (W : Tensor α [3072, 1024]) (b : Tensor α [3072]) :
    nn.State α (nn.stateShapes model) :=
  (nn.State.empty.push W).push b

/-- The upstream sequential forward program takes W, b, and x and returns y. -/
def forwardProgram {α : Type} [Storage α] [Context α] :
    Runtime.Autograd.Model.Program α [[3072, 1024], [3072], [1024]] [3072] :=
  model.forward .eval

end ReluLayer

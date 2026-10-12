import ReluLayer.Dimensions
import NN.API.Macros
import NN.API.Seeded

namespace ReluLayer

open TorchLean

/-- A linear layer with bias, followed by elementwise ReLU. -/
def architecture : nn.Builder (nn.Sequential [inputSize] [outputSize]) :=
  nn.Sequential![
    nn.linear inputSize outputSize,
    nn.relu
  ]

/-- Build the architecture; forward evaluation receives its parameters separately. -/
def model : nn.Sequential [inputSize] [outputSize] :=
  nn.build 0 architecture

/-- The linear layer owns weights and bias, in that order; ReLU adds no state. -/
theorem state_shapes : nn.stateShapes model = [[outputSize, inputSize], [outputSize]] := by
  rfl

/-- Supply arbitrary weights and bias instead of using initialized model state. -/
def parameters {α : Type} [Storage α]
    (W : Tensor α [outputSize, inputSize]) (b : Tensor α [outputSize]) :
    nn.State α (nn.stateShapes model) :=
  (nn.State.empty.push W).push b

/-- The upstream sequential forward program takes W, b, and x and returns y. -/
def forwardProgram {α : Type} [Storage α] [Context α] :
    Runtime.Autograd.Model.Program α [[outputSize, inputSize], [outputSize], [inputSize]] [outputSize] :=
  model.forward .eval

end ReluLayer

# activation layers, no trainable params
# Dwight Mayer, July 14th, 2026

require_relative "activations"
require_relative "../core/matrix"
require_relative "../neural_network/layers"
require_relative "layers"

class Sigmoid < Layer
  def forward(input)
    @output = input.sigmoid
    return @output
  end
  def backward(output_gradient, learning_rate=nil)
    derivative = @output.sigmoid_derivative
    output_gradient.hadamard_multiply(derivative)
  end
end

class ReLU < Layer
  def forward(x)
    @input = x
    @output = x.relu
    return x.relu
  end
  def backward(output_gradient, learning_rate=nil)
    output_gradient.hadamard_multiply(@input.relu_derivative)
  end
end

class Tanh < Layer
  def forward(x)
    @input = x
    @output = x.tanh
    return x.tanh
  end
  def backward(output_gradient, learning_rate=nil)
    output_gradient.hadamard_multiply(@input.tanh_derivative)
  end
end

class ELU < Layer
  def initialize(alpha:1.0)
    @alpha = alpha
  end
  def forward(x)
    @input = x
    @output = x.elu
  end
  def backward(output_gradient, learning_rate=nil)
    output_gradient.hadamard_multiply(@output.elu_derivative)
  end
end

class Softmax < Layer
  
  def forward(x)
    @input = x
    @output = x.softmax
  end  
  # Holding off on softmax backward until Jacobian Matrix becomes a good use of time
  def backward(output_gradient, learning_rate=nil)
  end

end










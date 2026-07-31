# LAYER superclass :) 
# Dwight Mayer, July 13th 2026

require_relative "../core/tensor"
require_relative "../core/matrix"
require_relative "../neural_network/losses"
require_relative "../neural_network/layers"



require_relative "losses"

require_relative "../neural_network/activations"
#require_relative "../neural_network/activation_layers"
require_relative "../neural_network/neural_network"


class Layer
  attr_reader :params

  def initialize
    @training = true
    @params = {}
    @training_state = {}
    @gradients = {}
  end

  def train!
    @training = true
  end
  def eval!
    @training = false
  end
  def training?
    return @training
  end

  def parameters
    return []
  end
  def forward(input)
    raise NotImplementedError, "Subclass must implement forward"
  end
  def backward(output_gradient, learning_rate)
    raise NotImplementedError, "Subclass must implement backward"
  end

end

class Linear < Layer
  def initialize(input_size, output_size, weights, bias)
    @weights = weights
    @bias = bias
  end
  def parameters
    return [@weights, @bias]
  end
  def forward(input)
    @input = input
    return input.dot_product(@weights).add(@bias)
  end
  def compute_gradients(output_gradient)
    weights_gradient = @input.transpose.dot_product(output_gradient)
    bias_gradient = output_gradient.reduce_keepdims([0]) { |vals| vals.sum }


    input_gradient = output_gradient.dot_product(@weights.transpose)
    grads = {}
    grads["weights_grad"] = weights_gradient
    grads["bias_grad"] = bias_gradient
    grads["input_grad"] = input_gradient

    return grads
  end

  def apply_gradients(grads, learning_rate)
    @weights = @weights.subtract(grads["weights_grad"].scalar_multiply(learning_rate))
    @bias = @bias.subtract(grads["bias_grad"].scalar_multiply(learning_rate))
  end

  def backward(output_gradient, learning_rate)
    grads = self.compute_gradients(output_gradient)
    self.apply_gradients(grads, learning_rate)
    return grads["input_grad"]
  end
end

class Dense < Linear
  def initialize(input_size, output_size, initializer: :he)
  
    weights = case initializer
    	      when :he then Tensor.init_he(input_size, output_size)
              when :xavier then Tensor.init_xavier(input_size, output_size)
              when :random then Tensor.init_rand(input_size, output_size)
              else raise ArgumentError, 'unknown initializer'
	      end

    bias = Tensor.zeros([1, output_size])
    
    super(input_size, output_size, weights, bias)
  end
end


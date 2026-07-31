
# utility layers
require_relative "layers"
require_relative "../core/tensor"


class Dropout < Layer
  # many ways to do dropout; masking, scaling, etc.
  def initialize(rate)
    super()
    unless rate >= 0.0 && rate < 1.0
      raise ArgumentError, "Dropout rate must be in [0, 1)"
    end
    @keep_probability = 1.0 - rate
    @drop_probability = rate
    @mask = nil

  end

  def forward(input)
    return input unless self.training?
    # create mask, apply mask

    @mask = input.map {rand < @keep_probability ? 1.0 : 0.0}
    input.hadamard_multiply(@mask).scalar_divide(@keep_probability)
  end

  def backward(output_gradient, learning_rate=nil)
    return output_gradient unless training?
    return output_gradient.hadamard_multiply(@mask).scalar_divide(@keep_probability)
  end
end


class Flatten < Layer
  def initialize
    super()
  end

  def forward(input)
    @input_shape = input.shape
    batch_size = input.shape[0]
    feature_size = input.shape[1..-1].reduce(1,:*)
    input.reshape([batch_size, feature_size])

  end

  def backward(output_gradient, learning_rate)
    output_gradient.reshape(@input_shape)
  end

end




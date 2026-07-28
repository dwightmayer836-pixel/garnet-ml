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

    # @mask = Matrix.new(Matrix.create_zeroes(input.rows, input.cols)).map do |_|
    #  rand < @keep_probability ? 1.0 : 0.0
    #    end
    input.hadamard_multiply(@mask).scalar_divide(@keep_probability)
  end

  def backward(output_gradient, learning_rate=nil)
    return output_gradient unless training?
    return output_gradient.hadamard_multiply(@mask).scalar_divide(@keep_probability)
  end
end



class BatchNormalization < Layer
  def initialize(num_features, epsilon:1e-7, momentum:0.9)
    super()
    @epsilon = epsilon
    @momentum = momentum

    @params[:gamma] = Array.new(num_features, 1.0)
    @params[:beta] = Array.new(num_features, 0.0)

    @running_mean = Array.new(num_features, 0.0)
    @running_var = Array.new(num_features, 1.0)

    @training_state[:cache] = nil
  end


  def forward(input, training:true)

    ndims = input.shape.length
    feature_axis = 1
    reduce_axes = (0...ndims).to_a - [feature_axis]


    if self.training?
      batch_mean = input.reduce_keepdims(reduce_axes) {|vals| Matrix.vector_mean(vals)}
      batch_var = input.reduce_keepdims(reduce_axes) {|vals| Matrix.vector_variance(vals)}

      mean_flat = batch_mean.flatten
      mean_var = batch_var.flatten

      @running_mean = @running_mean.each_with_index.map {|r, i| @momentum*r+(1-@momentum)*mean_flat[i]}
      @running_var = @running_var.each_with_index.map {|r,i| @momentum*r+(1-@momentum)*mean_var[i]}
      mean, var = batch_mean,batch_var
    else

      mean = wrap_broadcastable(@running_mean, input, ndims, feature_axis)
      var = wrap_broadcastable(@running_var, input, ndims, feature_axis)
    end      
    


    st_dev = var.map {|v| Math.sqrt(v+@epsilon)}

    gamma = wrap_broadcastable(@params[:gamma], input, ndims, feature_axis)
    beta = wrap_broadcastable(@params[:beta], input, ndims, feature_axis)

    centered = input.combine(mean) {|x, m| x-m}
    normalized = centered.combine(st_dev) {|c,s| c/s}

    output = normalized.combine(gamma) {|n,g| n*g}
    output = output.combine(beta) {|o, b| o + b}

    if self.training?
      @training_state[:cache] = {
	normalized:normalized,
	centered:centered,
	st_dev:st_dev,
	reduce_axes:reduce_axes,
        feature_axis:feature_axis
}
    end

    return output

  end


  def backward(output_gradient, learning_rate)
    cache = @training_state[:cache]
    normalized = cache[:normalized]
    centered = cache[:centered]
    st_dev = cache[:st_dev]
    reduce_axes = cache[:reduce_axes]
    feature_axis = cache[:feature_axis]

    batch_size = reduce_axes.map {|ax| output_gradient.shape[ax] }.reduce(1, :*)
    ndims = output_gradient.shape.length

    gamma_grad = output_gradient.hadamard_multiply(normalized).reduce_keepdims(reduce_axes) {|vals| vals.sum}
    beta_grad = output_gradient.reduce_keepdims(reduce_axes) {|vals| vals.sum}
    @gradients = {gamma: gamma_grad.flatten, beta: beta_grad.flatten}

    gamma_b = wrap_broadcastable(@params[:gamma], output_gradient, ndims, feature_axis)

    normalized_gradient = output_gradient.combine(gamma_b) {|d, g| d * g}

    inv_std_cubed = st_dev.map {|s| -0.5 * (s**-3)}
    var_gradient = normalized_gradient.hadamard_multiply(centered)
	.combine(inv_std_cubed, axis:0) {|v, scale| v * scale}
        .reduce_keepdims(reduce_axes) {|vals| vals.sum} 
    neg_inv_std = st_dev.map {|s| -1.0/s}
    direct = normalized_gradient.combine(neg_inv_std) { |d, s| d * s }
				.reduce_keepdims(reduce_axes) {|vals| vals.sum}

    centered_sums = centered.reduce_keepdims(reduce_axes) { |vals| vals.sum }

    indirect = var_gradient.combine(centered_sums) {|dv, cs| dv * -2.0 * cs / batch_size}
    # mean_gradient = direct.each_with_index.map {|d, c| d + indirect[c]}
    mean_gradient = direct.add(indirect)
    
    term1 = normalized_gradient.combine(st_dev) {|d,s| d/s}
    term2 = centered.combine(var_gradient) {|dv, c| dv * 2.0 * c / batch_size}
    term3_per_col = mean_gradient.map {|m| m/batch_size}
    input_gradient = term1.add(term2).combine(term3_per_col) {|v,m| v+m}

    gamma_grad_b = wrap_broadcastable(@gradients[:gamma], output_gradient, ndims, feature_axis)
    beta_grad_b = wrap_broadcastable(@gradients[:beta], output_gradient, ndims, feature_axis)

    gamma_flat_current = wrap_broadcastable(@params[:gamma], output_gradient, ndims, feature_axis)
    beta_flat_current = wrap_broadcastable(@params[:beta], output_gradient, ndims, feature_axis)
    
    @params[:gamma] = gamma_flat_current.subtract(gamma_grad_b.scalar_multiply(learning_rate)).flatten
    @params[:beta] = beta_flat_current.subtract(beta_grad_b.scalar_multiply(learning_rate)).flatten

    return input_gradient
  end

  def wrap_broadcastable(flat_array, input, ndims, feature_axis)
    if input.is_a?(Matrix)
      Matrix.new([flat_array])
    else 
      shape = Array.new(ndims, 1)
      shape[feature_axis] = flat_array.length
      Tensor.new(flat_array, shape)
    end

  end

end


class MaxPool2D < Layer
  def initialize(pool_size, stride:nil)
    super()
    @pool_h = @pool_w = pool_size
    @stride = stride || pool_size
  end

  def forward(input)
    @input_shape = input.shape
    batch, channels, h, w = input.shape    
    out_h = (h-@pool_h) / @stride + 1
    out_w = (w-@pool_w) / @stride + 1
    result = Tensor.new(Array.new(batch*channels*out_h*out_w), [batch, channels, out_h, out_w])
    @max_locations = {}

    (0...batch).each do |b|
      (0...channels).each do |c|
	(0...out_h).each do |oh|
	  (0...out_w).each do |ow|
	    best_val = -Float::INFINITY
	    best_pos = nil
	    (0...@pool_h).each do |ph|
	      (0...@pool_w).each do |pw|
		r, col = oh*@stride+ph, ow*@stride+pw
		val = input.get(b, c, r, col)
		if val > best_val
		  best_val = val 
                  best_pos = [r, col]
		end
	      end
	    end
	    result.set(b, c, oh, ow, best_val)
	    @max_locations[[b,c,oh,ow]] = best_pos
	  end
        end
      end
    end
    return result
  end

  def backward(output_gradient, learning_rate)
    input_gradient = Tensor.new(Array.new(@input_shape.reduce(1, :*), 0.0), @input_shape)
    @max_locations.each do |(b, c, oh, ow), (r, col)|
      current = input_gradient.get(b, c, r, col)
      grad = output_gradient.get(b, c, oh, ow)
      input_gradient.set(b, c, r, col, (current+grad))
    end
    return input_gradient

  end  
end

class Conv2D < Layer
  def initialize(in_channels, out_channels, kernel_size:, stride: 1, padding:0)
    super()
    @in_channels = in_channels 
    @out_channels = out_channels
    @kernel_h = @kernel_w = kernel_size    
    @stride = stride
    @padding = padding


    fan_in = in_channels * @kernel_h * @kernel_w

    @params[:filters] = Tensor.new(
    Array.new(out_channels * fan_in) { rand(-1.0..1.0) * Math.sqrt(2.0 / fan_in)},
    [out_channels, fan_in])

    @params[:bias] = Array.new(out_channels, 0.0)
  end

  def forward(input)

    @input_shape = input.shape
    batch, _, h, w = input.shape
    padded = @padding > 0 ? input.pad(2, @padding, @padding).pad(3, @padding, @padding) : input
    @cols = padded.im2col(@kernel_h, @kernel_w, @stride)

    @padded_h, @padded_w = padded.shape[2], padded.shape[3]


    out_h = (h + 2*@padding - @kernel_h) / @stride + 1
    out_w = (w+ 2*@padding - @kernel_w) / @stride + 1
    raw = @cols.dot_product(@params[:filters].transpose)
    biased = raw.combine(@params[:bias]) {|v,b| v+b}
    biased.reshape([batch, out_h, out_w, @out_channels]).transpose([0,3,1,2])

  end


  def backward(output_gradient, learning_rate)
  
    batch, out_channels, out_h, out_w = output_gradient.shape
    grad_2d = output_gradient.transpose([0,2,3,1]).reshape([batch*out_h*out_w, out_channels])
    filters_grad = @cols.transpose.dot_product(grad_2d).transpose
    bias_grad = grad_2d.reduce([0]) {|vals| vals.sum}
    
    cols_grad = grad_2d.dot_product(@params[:filters])
    # in_channels_times_kernel = @params[:filters].shape[1]
    input_grad_padded = Tensor.col2im(cols_grad, batch, @in_channels, @padded_h, @padded_w, @kernel_h, @kernel_w, @stride)
    input_grad = if @padding > 0
    input_grad_padded.unpad(2, @padding, @padding).unpad(3, @padding, @padding)
  else
    input_grad_padded
  end
    
    @params[:filters] = @params[:filters].subtract(filters_grad.scalar_multiply(learning_rate))
    @params[:bias] = @params[:bias].each_with_index.map {|b,i| b - learning_rate * bias_grad.get(i)}
    return input_grad    

  end

end

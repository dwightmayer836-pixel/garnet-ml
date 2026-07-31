require_relative "layers"
require_relative "../core/tensor"



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






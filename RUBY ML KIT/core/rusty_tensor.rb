# Dwight Mayer, July 28th, 2026
# Tensor module with Rust-accelerated operations

# needs to be an include
require_relative "../backend/tensor_backend/tensor_backend"



module RustOperations

  def dot_product(other)
    self_source = contiguous? ? self : materialize
    other_source = other.contiguous? ? other : other.materialize

    unless self_source.shape[1] == other_source.shape[0]
      raise ArgumentError, "bad dimensions in rust matmul"
    end   

    result = RustyTensor.matrix_multiply(self_source.data, other_source.data, self_source.shape, other_source.shape)
    Tensor.new(result,[self_source.shape[0], other_source.shape[1]])       
  end

  def im2col(kernel_h, kernel_w, stride)
    source = contiguous? ? self : materialize
    batch, channels, h, w = source.shape
    out_h = (h - kernel_h) / stride + 1
    out_w = (w - kernel_w) / stride + 1

    result = RustyTensor.im2col(source.data, source.shape, kernel_h, kernel_w, stride)
    Tensor.new(result, [batch * out_h * out_w, channels * kernel_h * kernel_w])
    # rust implementation
  end

  def col2im_rust(batch, channels, h, w, kernel_h, kernel_w, stride)
    cols_source = contiguous? ? self : materialize
    result = RustyTensor.col2im(cols_source.data, batch, channels, h, w, kernel_h, kernel_w, stride)
    Tensor.new(result, [batch, channels, h, w])

  end

  def add_rust(other)
    # need to pad other / self tensor

    # other = other.to_tensor if other.is_a?(Matrix)
    # other = wrap_as_tensor(other) unless other.is_a?(Tensor)

    result_shape = self.class.broadcast_shape(@shape, other.shape)
    self_expanded = broadcast_to(result_shape)
    other_expanded = other.broadcast_to(result_shape)

    result = RustyTensor.elementwise_add(self_expanded.data, other_expanded.data)
    Tensor.new(result, result_shape)
    
  end

  def scalar_multiply(scalar)
    result = RustyTensor.scalar_multiply(self.data, scalar)
    Tensor.new(result, self.shape)
  end

  def scalar_divide(scalar)
    raise ZeroDivisionError unless scalar != 0

    result = RustyTensor.scalar_divide(self.data, scalar)
    Tensor.new(result, self.shape)
  end

  def hadamard_multiply(other)
    #other = wrap_as_tensor(other) unless other.is_a?(Tensor)
    
    other = other.to_tensor if other.is_a?(Matrix)
    other = wrap_as_tensor(other) unless other.is_a?(Tensor)

    result_shape = self.class.broadcast_shape(@shape, other.shape)
    self_expanded = broadcast_to(result_shape)
    other_expanded = other.broadcast_to(result_shape)        
    
   
    result = RustyTensor.elementwise_multiply(self_expanded.data, other_expanded.data)
    Tensor.new(result, self.shape)
  end

  def sigmoid_rust
    source = contiguous? ? self : materialize
    result = RustyTensor.sigmoid(source.data)
    Tensor.new(result, source.shape)
  end

  def relu_rust
    source = contiguous? ? self : materialize
    result = RustyTensor.relu(source.data)
    Tensor.new(result, source.shape)
  end

  def tanh_rust
    source = contiguous? ? self : materialize
    result = RustyTensor.tanh(source.data)
    Tensor.new(result, source.shape)
  end

  def sigmoid_derivative_rust
    source = contiguous? ? self : materialize
    result = RustyTensor.tanh_derivative(source.data)
    Tensor.new(result, source.shape)

  end

  def relu_derivative_rust
    source = contiguous? ? self : materialize
    result = RustyTensor.relu_derivative(source.data)
    Tensor.new(result, source.shape)
  end

  def tanh_derivative_rust
    source = contiguous? ? self : materialize
    result = RustyTensor.tanh_derivative(source.data)
    Tensor.new(result, source.shape)

  end

  def elu_rust(alpha:1)
    source = contiguous? ? self : materialize
    result = RustyTensor.elu(source.data,alpha)
    Tensor.new(result, source.shape)
  end

  def elu_derivative_rust(alpha:1)
    source = contiguous? ? self : materialize
    result = RustyTensor.elu_derivative(source.data,alpha)
    Tensor.new(result, source.shape)
  end






end

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

  def add(other)
    # need to pad other / self tensor

  end


end

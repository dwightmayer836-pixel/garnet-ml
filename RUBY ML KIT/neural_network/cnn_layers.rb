# CNN-layers separate file, July 30th, 2026
# Dwiggggght Mayyyyerrrrr


require_relative "../core/tensor"
require_relative "layers"


class MaxPool2D < Layer
  def initialize(pool_size, stride:nil)
    super()
    @pool_h = @pool_w = pool_size
    @stride = stride || pool_size
  end

  # New Rusty implementation
  def forward(input)
    @input_shape = input.shape
    source = input.contiguous? ? input : input.materialize
    batch,channels,h,w = source.shape
    out_h = (h-@pool_h) / @stride + 1
    out_w = (w-@pool_w) / @stride + 1

    result_data, winners = RustyTensor.max_pool2d_forward(
                                                source.data,
                                                source.shape,
                                                @pool_h,
                                                @pool_w,
                                                @stride)


    @winners = winners
    Tensor.new(result_data, [batch,channels,out_h,out_w])

  end

  def backward(output_gradient, learning_rate)
    source = output_gradient.contiguous? ? output_gradient : output_gradient.materialize    
    input_length = @input_shape.reduce(1, :*)
    result = RustyTensor.max_pool2d_backward(source.data, @winners, input_length)
    Tensor.new(result, @input_shape)

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
    # puts "output_gradient.shape = #{output_gradient.shape.inspect}"
    batch, out_channels, out_h, out_w = output_gradient.shape
    grad_2d = output_gradient.transpose([0,2,3,1]).reshape([batch*out_h*out_w, out_channels])
    filters_grad = @cols.transpose.dot_product(grad_2d).transpose
    bias_grad = grad_2d.reduce([0]) {|vals| vals.sum}

    cols_grad = grad_2d.dot_product(@params[:filters])

    input_grad_padded = cols_grad.col2im_rust(batch,@in_channels,@padded_h,@padded_w,@kernel_h,@kernel_w,@stride)

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





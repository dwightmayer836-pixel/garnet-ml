# The Tensor Class
# Dwight Mayer, July 15th, 2026

require_relative "../neural_network/activations"
require_relative "rusty_tensor"


class Tensor
  attr_reader :shape, :strides, :data, :ndims
  
  include RustOperations
  include Activations

  def initialize(data, shape, strides:nil)
    expected_size = shape.reduce(1) {|acc, dim| acc * dim}
    unless expected_size == data.length
      raise ArgumentError, "tensor data length doesn't match shape"
    end
  
    @data = data
    @shape = shape
    @ndims = shape.length
    @strides = strides || self.class.compute_strides(shape)

  end
   
  def copy
    Tensor.new(@data.dup, @shape.dup)
  end
  
  def pick_by_row(indices)
    (0...@shape[0]).map { |r| self.get(r, indices.get(r,0))}
  end
  
  def self.compute_strides(shape)
    strides = Array.new(shape.length, 1)
    (shape.length-2).downto(0) do |i|
      strides[i] = strides[i+1] * shape[i + 1]
    end
    return strides
  end

  def validate_indices(indices)
    unless indices.length == @ndims
      raise ArgumentError, "invalid index length #{indices.length} vs #{@ndims} n_dimensions"
    end

    indices.each_with_index do |idx, dim|
      unless idx.between?(0, @shape[dim]-1)
        raise IndexError, "Index #{idx} out of bounds compared to #{@shape[dim]-1}"
      end
    end
  end 

  def flat_index(indices)
    # self.validate_indices(indices)
    indices.each_with_index.sum {|idx, dim| idx * @strides[dim]}
  end

  def get(*indices)
    return @data[flat_index(indices)]
  end

  def set(*indices, value)
    @data[flat_index(indices)] = value
  end
  
  def self.each_index(shape)
    total = shape.reduce(1) {|acc,dim| acc*dim}
    strides = self.compute_strides(shape)
    
    (0...total).each do |flat_idx|
      indices = []
      remaining = flat_idx
      shape.each_with_index do |_, dim|
        indices.push((remaining / strides[dim]))
        remaining %= strides[dim]
      end
      yield(indices, flat_idx)
    end
  end

  def reshape(new_shape)
    original_size = @shape.reduce(1) {|acc,dim| acc*dim}
    new_size = new_shape.reduce(1) {|acc,dim| acc*dim}

    unless original_size == new_size
      raise ArgumentError, "bad sizes in Tensor reshape"
    end
    source_data = contiguous? ? @data : materialize.data


    return Tensor.new(source_data, new_shape)
    
  end

  def transpose(axes=nil)

    axes ||= (0...@ndims).to_a.reverse

    unless axes.sort == (0...@ndims).to_a
      raise ArgumentError, "axes #{axes.inspect} is not a valid permutation of 0..#{@ndims - 1}"
    end

    new_shape = axes.map {|old_axis| @shape[old_axis]}
    new_strides = axes.map {|old_axis| @strides[old_axis]}
    return Tensor.new(@data, new_shape, strides:new_strides)
  end

  def contiguous?
    return @strides == self.class.compute_strides(@shape)
  end

  def materialize
    new_data = Array.new(@data.length)
    self.class.each_index(@shape) do |indices, flat_idx|
      new_data[flat_idx] = self.get(*indices)
    end
    return Tensor.new(new_data, @shape)
  end

  def self.broadcast_shape(shape_a, shape_b)
    ndims = [shape_a.length, shape_b.length].max
    padded_a = Array.new(ndims-shape_a.length, 1) + shape_a
    padded_b = Array.new(ndims-shape_b.length, 1) + shape_b

    result = padded_a.zip(padded_b).map do |dim_a, dim_b|
      if dim_a == dim_b
        dim_a
      elsif dim_a == 1
        dim_b
      elsif dim_b == 1
        dim_a
      else
        raise ArgumentError, "shapes #{shape_a.inspect} and #{shape_b.inspect} are not broadcastable"
      end
     end
    return result
  end

  def combine(other)
    # pass...
    # current, combine assumes the other input is always a tensor. Not always true. 
    other = other.to_tensor if other.is_a?(Matrix)

    other = wrap_as_tensor(other) unless other.is_a?(Tensor)

    result_shape = self.class.broadcast_shape(@shape, other.shape)
    padded_self_shape = Tensor.pad_shape(@shape, result_shape.length)
    padded_other_shape = Tensor.pad_shape(other.shape, result_shape.length)
    
    new_data = Array.new(result_shape.reduce(1) {|a,d| a * d} ) 
    
    self.class.each_index(result_shape) do |out_indices, flat_idx|
      self_indices = Tensor.map_broadcast_indices(out_indices, padded_self_shape)
      other_indices = Tensor.map_broadcast_indices(out_indices, padded_other_shape) 
      
      a_val = self.get(*Tensor.trim_leading(self_indices, @ndims))
      b_val = other.get(*Tensor.trim_leading(other_indices, other.ndims))
      new_data[flat_idx] = yield(a_val, b_val)
    end
    return Tensor.new(new_data, result_shape)
  end

  def self.pad_shape(shape, target_length)
    return Array.new(target_length-shape.length,1) + shape
  end
  def self.map_broadcast_indices(out_indices, padded_shape)
    out_indices.each_with_index.map {|idx, dim| padded_shape[dim] == 1 ? 0 : idx}  
  end

  def self.trim_leading(indices, keep_last_n)
    indices.last(keep_last_n)
  end
  
  def reduce(axes_to_reduce)
    kept_axes = (0...@ndims).to_a - axes_to_reduce
    output_shape = kept_axes.map {|d| @shape[d]}
    output_shape = [1] if output_shape.empty?

    buckets = Hash.new{|h, k| h[k] = []}

    self.class.each_index(@shape) do |indices, flat_idx|
      kept_indices = kept_axes.map {|d| indices[d]}
      buckets[kept_indices] << self.get(*indices)
    end 

    new_data = Array.new(output_shape.reduce(1) {|a,d| a*d})
    result = Tensor.new(Array.new(new_data.length, 0), output_shape)
    
    buckets.each do |kept_indices, values|
      result.set(*kept_indices, yield(values))
    end
    return result
  end

  def map
    source = contiguous? ? self : materialize
    new_data = source.data.map { |val| yield(val) }

    return Tensor.new(new_data, @shape)
  end

  def sum(axes)
    reduce(axes) {|values| values.sum}
  end

  def mean(axes)
    reduce(axes) {|values| values.sum.to_f / values.length}
  end

  def max(axes)
    reduce(axes) {|values| values.max}
  end

  def min(axes)
    reduce(axes) {|values| values.min}
  end

  def add(other)
    self.combine(other) { |a, b| a + b }
  end

  def subtract(other)
    self.combine(other) {|a,b| a-b}
  end

  def scalar_multiply(scalar)
    self.map {|val| val * scalar}
  end

  def scalar_divide(scalar)
    raise ZeroDivisionError unless scalar =! 0

    self.map{|val| val / scalar}
  end

  def reduce_keepdims(axes)
    output_shape = @shape.each_with_index.map {|dim,i| axes.include?(i) ? 1:dim}
    buckets = Hash.new { |h,k| h[k] = [] }
    
    self.class.each_index(@shape) do |indices, flat_idx|
      key = indices.each_with_index.map {|idx, d| axes.include?(d) ? 0:idx}
      buckets[key] << self.get(*indices)
    end

    new_data = Array.new(output_shape.reduce(1) {|a,d| a*d})

    result = Tensor.new(Array.new(new_data.length, 0), output_shape)
    buckets.each {|key, values| result.set(*key, yield(values))}
    result

  end

  def hadamard_multiply(other)
    self.combine(other) {|a, b| a * b}
  end 


  def scalar_divide(scalar)
    raise ZeroDivisionError if scalar==0.0
    self.map {|val| val/scalar}
  end

  def wrap_as_tensor(other)
    if other.is_a?(Numeric)
      Tensor.new([other], [1])
    elsif other.is_a?(Array)
      Tensor.new(other, [other.length])
    else
      raise ArgumentError, "Cannot combine tensor with #{other.class}"
    end

  end

  def dot_product_old(other)
    other = other.to_tensor unless other.is_a?(Tensor)

    rows, inner = @shape
    cols = other.shape[1]

    a = self.reshape([rows, inner, 1])
    b = other.reshape([1, inner, cols])

    product = a.combine(b) {|x,y| x*y}
    product.reduce([1]) {|vals| vals.sum}    

  end


  def pad(axis, before, after, value:0)
    new_shape = @shape.dup
    new_shape[axis] += (before + after)

    result = Tensor.new(Array.new(new_shape.reduce(1,:*), value), new_shape)
    # incomplete...

    self.class.each_index(@shape) do |indices, flat_idx|
      target_indices = indices.dup
      target_indices[axis] += before
      result.set(*target_indices, self.get(*indices))
    end

    result  

  end

  def unpad(axis, before, after)
    new_size = @shape[axis] - before - after
    ranges = Array.new(@ndims)
    ranges[axis] = (before...(before + new_size))
    slice(*ranges)
  end



  def slice(*ranges)
    new_shape = ranges.map {|r| r.is_a?(Range) ? r.size : @shape[ranges.index(r)]}
    resolved = ranges.each_with_index.map {|r, d| r || (0...@shape[d])}
    out_shape = resolved.map(&:size)
    new_data = Array.new(out_shape.reduce(1, :*))
    
    self.class.each_index(out_shape) do |out_indices, flat_idx|
      source_indices = out_indices.each_with_index.map {|idx, d| resolved[d].first + idx}
      new_data[flat_idx] = self.get(*source_indices)
    end
    return Tensor.new(new_data, out_shape)

  end

  def im2col_old(kernel_h, kernel_w, stride)
    # int, int, int(s)

    batch, channels, h, w = @shape
    out_h = (h-kernel_h) / stride + 1
    out_w = (w-kernel_w) / stride + 1
    num_windows = batch * out_h * out_w
    window_size = channels * kernel_h * kernel_w
    new_data = Array.new(num_windows * window_size)

    row = 0 
    (0...batch).each do |b|
      (0...out_h).each do |oh|
        (0...out_w).each do |ow|
          col = 0
          (0...channels).each do |c|
            (0...kernel_h).each do |kh|
              (0...kernel_w).each do |kw|
		val = self.get(b, c, (oh * stride + kh), (ow * stride + kw))
		new_data[row * window_size + col] = val
		col += 1
	      end
	    end
	  end
	  row += 1
        end
      end
    end

    return Tensor.new(new_data, [num_windows, window_size])

  end
  
  # This is old and deprecated... 
  def self.col2im(cols, batch, channels, h, w, kernel_h, kernel_w, stride)

    out_h = (h-kernel_h) / stride+1
    out_w = (w-kernel_w) / stride+1
    result = Tensor.new(Array.new(batch*channels*h*w, 0.0), [batch, channels, h, w])
    row = 0

    (0...batch).each do |b|
      (0...out_h).each do |oh|
        (0...out_w).each do |ow|
          col = 0
          (0...channels).each do |c|
            (0...kernel_h).each do |kh|
              (0...kernel_w).each do |kw|
                r = oh*stride+kh
                cc = ow*stride+kw
                current = result.get(b, c, r, cc)
                result.set(b,c,r,cc,current+cols.get(row,col))
                col += 1
              end
            end
          end
          row +=1
        end
      end
    end
    result
  end

  def equals?(other)
    
    # There is a case where a 2D matrix of a certain IS a tensor. 
    return false unless other.is_a?(Tensor)
    return false unless @shape == other.shape

    self.class.each_index(@shape) do |indices, _flat_idx|
      return false if self.get(*indices) != other.get(*indices)
    end

    return true


  end

  def self.init_he(rows, cols)
    limit = Math.sqrt(6.0 / rows)
    Tensor.new(Array.new(rows * cols) { rand(-limit...limit) }, [rows, cols])
  end

  def self.init_xavier(rows, cols)
    limit = Math.sqrt(6.0 / (rows + cols))
    Tensor.new(Array.new(rows * cols) { rand(-limit...limit) }, [rows, cols])
  end

  def self.init_rand(rows, cols)
    Tensor.new(Array.new(rows * cols) { rand(-1.0..1.0) }, [rows, cols])
  end

  def self.zeros(shape)
    Tensor.new(Array.new(shape.reduce(1, :*), 0.0), shape)
  end

  def subtract(other)
    self.combine(other) {|a, b| a-b}
  end

end






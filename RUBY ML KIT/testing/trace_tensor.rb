require_relative "../neural_network/losses"
require_relative "../neural_network/layers"
require_relative "../core/tensor"
require_relative "../neural_network/activation_layers"

t = Tensor.new((0..11).to_a, [2, 2, 3])
puts t.strides.inspect
# => [6, 3, 1]

# --- get traces ---
puts t.get(0, 0, 0)   # => 0
puts t.get(0, 1, 2)   # => 5
puts t.get(1, 0, 0)   # => 6
puts t.get(1, 1, 2)   # => 11

# --- set trace ---
t.set(1, 0, 2, 99)
puts t.get(1, 0, 2)   # => 99

# --- validation trace ---
begin
  t.get(2, 0, 0)
rescue IndexError => e
  puts e.message
  # => "index 2 out of bounds for dimension 0 (size 2)"
end

# --- each_index trace ---
Tensor.each_index([2, 2, 3]) do |indices, flat_idx|
  puts "flat_idx=#{flat_idx} -> #{indices.inspect}"
end

t = Tensor.new((0..11).to_a, [2, 2, 3])

flat = t.reshape([4, 3])
puts flat.shape.inspect     # => [4, 3]
puts flat.strides.inspect   # => [3, 1]
puts flat.get(2, 1)         # => 7

back = flat.reshape([2, 2, 3])
puts back.get(1, 0, 1)      # => 7  (same value, original indexing scheme)

# Mismatched size should raise
begin
  t.reshape([5, 5])
rescue ArgumentError => e
  puts e.message
  # => "cannot reshape tensor of size 12 into shape [5, 5] (size 25)"
end

t = Tensor.new((0..11).to_a, [2, 2, 3])

transposed = t.transpose([0, 2, 1])
puts transposed.shape.inspect     # => [2, 3, 2]
puts transposed.strides.inspect   # => [6, 1, 3]
puts transposed.get(0, 2, 1)      # => 5, matches t.get(0, 1, 2)

# Invalid permutation should raise
begin
  t.transpose([0, 1])
rescue ArgumentError => e
  puts e.message
  # => "axes [0, 1] is not a valid permutation of 0..2"
end

t = Tensor.new((0..11).to_a, [2, 2, 3])
transposed = t.transpose([0, 2, 1])

puts transposed.contiguous?   # => false (strides are [6, 1, 3], not the fresh [6, 2, 1] for shape [2,3,2])

fixed = transposed.materialize
puts fixed.contiguous?        # => true
puts fixed.strides.inspect    # => [6, 2, 1] — freshly computed for shape [2, 3, 2]
puts fixed.get(0, 2, 1)       # => 5, same logical value as before
puts fixed.data.equal?(transposed.data)  # => false — genuinely separate array now
Tensor.broadcast_shape([8, 1, 64, 64], [3, 1, 64])

# --- broadcast_shape tests ---

# Standard case from the trace: [8,1,64,64] vs [3,1,64]
result = Tensor.broadcast_shape([8, 1, 64, 64], [3, 1, 64])
puts result.inspect
# expected => [8, 3, 64, 64]

# Same-rank, all compatible
result2 = Tensor.broadcast_shape([2, 3], [2, 1])
puts result2.inspect
# expected => [2, 3]

# Incompatible shapes should raise
begin
  Tensor.broadcast_shape([3, 4], [5, 4])
rescue ArgumentError => e
  puts "raised correctly: #{e.message}"
end

# --- combine tests ---

# Case 1: matrix + bias vector, matches the hand-trace from last message
a = Tensor.new((1..6).to_a, [2, 3])   # [[1,2,3],[4,5,6]]
b = Tensor.new([10, 20, 30], [3])     # broadcast across rows

sum = a.combine(b) { |x, y| x + y }
puts sum.shape.inspect        # expected => [2, 3]
puts sum.get(0, 0)            # expected => 1 + 10 = 11
puts sum.get(1, 2)            # expected => 6 + 30 = 36

# Case 2: same-shape tensors, no broadcasting needed
c = Tensor.new([1, 1, 1, 1], [2, 2])
d = Tensor.new([5, 6, 7, 8], [2, 2])
prod = c.combine(d) { |x, y| x * y }
puts prod.get(0, 1)   # expected => 1 * 6 = 6
puts prod.get(1, 1)   # expected => 1 * 8 = 8

# Case 3: scalar-like tensor [1,1] broadcasting against a bigger tensor
e = Tensor.new((1..12).to_a, [2, 2, 3])
scalar_like = Tensor.new([100], [1, 1, 1])
shifted = e.combine(scalar_like) { |x, y| x + y }
puts shifted.shape.inspect   # expected => [2, 2, 3]
puts shifted.get(1, 1, 2)    # expected => 12 + 100 = 112

# Case 4: incompatible combine should raise via broadcast_shape internally
begin
  f = Tensor.new([1, 2, 3, 4, 5], [5])
  g = Tensor.new([1, 2, 3, 4], [4])
  f.combine(g) { |x, y| x + y }
rescue ArgumentError => e
  puts "raised correctly: #{e.message}"
end

t = Tensor.new((0..5).to_a, [2, 3])
doubled = t.map { |v| v * 2 }

puts doubled.shape.inspect   # => [2, 3]
puts doubled.get(0, 0)       # => 0
puts doubled.get(1, 2)       # => 10

relu = ReLU.new
t = Tensor.new([-2, -1, 0, 1, 2, 3], [2, 3])
result = relu.forward(t)
puts result.data.inspect   # expect [0, 0, 0, 1, 2, 3]



=begin
bn = BatchNormalization.new(num_features=784)
bn.forward(Matrix.new([[]]))
bn.forward(Tensor.new([], []))
=end


d = Dropout.new(0.5)

m = Matrix.new([[1,2,3],[4,5,6]])
out_m = d.forward(m)
puts out_m.class   # => Matrix

t = Tensor.new((1..6).to_a, [2,3])
out_t = d.forward(t)
puts out_t.class   # => Tensor

t = Tensor.new([1,2,3,4], [2,2])
t.hadamard_multiply(2)          # does this work, or raise?
t.hadamard_multiply([10, 20])

a = Tensor.new([1,2,3,4,5,6], [2,3])    # [[1,2,3],[4,5,6]]
b = Tensor.new([7,8,9,10,11,12], [3,2]) # [[7,8],[9,10],[11,12]]

result = a.dot_product(b)
puts result.shape.inspect   # => [2, 2]
puts result.get(0,0)        # => 1*7 + 2*9 + 3*11 = 58
puts result.get(1,1)        # => 4*8 + 5*10 + 6*12 = 154

t = Tensor.new((1..16).to_a, [1, 1, 4, 4])
cols = t.im2col(2, 2, 1)

puts cols.shape.inspect     # => [9, 4]
puts cols.get(0, 0)         # => 1
puts cols.get(0, 3)         # => 6
puts cols.get(3, 0)         # => 5
puts cols.get(8, 3)         # => 16

t = Tensor.new((1..8).to_a, [2, 4])   # [[1,2,3,4],[5,6,7,8]]

padded = t.pad(1, 1, 1)               # pad axis 1 (columns), 1 before, 1 after
puts padded.shape.inspect             # => [2, 6]
puts padded.get(0, 0)                 # => 0 (padding)
puts padded.get(0, 1)                 # => 1 (original data, shifted right by 1)
restored = padded.unpad(1, 1, 1)
puts restored.shape.inspect           # => [2, 4]
puts restored.get(0, 0)               # => 1
puts restored.get(1, 3)               # => 8
puts restored.equals?(t)              # => true (Matrix#equals?-style check if Tensor has one, else compare .data)
t = Tensor.new((1..8).to_a, [2, 4])
padded = t.pad(1, 1, 1)
puts padded.shape.inspect   # sanity check pad worked: expect [2, 6]

# Manually build the ranges unpad *should* be building, to compare:
ranges = Array.new(2)
ranges[1] = (1...(1+4))
puts ranges.inspect   # expect [nil, 1...5]

result = padded.slice(*ranges)
puts result.shape.inspect  # expect [2, 4] if slice itself is fine



# Matrix path (should behave exactly as before)
m = Matrix.new([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
y_true_m = Matrix.new([[1], [2]])
ce = CrossEntropy.new
puts ce.forward(m, y_true_m)   # same value as before this change

# Tensor path (the new case)
t = Tensor.new([0.1, 0.7, 0.2, 0.3, 0.3, 0.4], [2, 3])
y_true_t = Matrix.new([[1], [2]])
puts ce.forward(t, y_true_t)   # should match the Matrix result numerically

=begin
class GradientChecker
  EPSILON = 1e-4
  TOLERANCE = 1e-2

  def self.check_layer(layer, input)
    layer.params.each_key.map { |name| [name, check_param(layer, input, name)] }.to_h
  end

  def self.check_param(layer, input, param_name)
    param = layer.params[param_name]
    param.is_a?(Tensor) ? check_tensor_param(layer, input, param_name) : check_array_param(layer, input, param_name)
  end

  def self.check_tensor_param(layer, input, param_name)
    original_param = layer.params[param_name].copy
    small_lr = 1e-6

    # one real backward pass -> analytical gradient for the WHOLE param at once
    output = layer.forward(input)
    d_output = output.map { |v| 2 * v }
    layer.backward(d_output, small_lr)
    updated_param = layer.params[param_name]
    analytical_grad = original_param.combine(updated_param) { |o, u| (o - u) / small_lr }

    layer.params[param_name] = original_param.copy   # restore before numerical checks

    max_diff = 0.0
    worst = nil

    Tensor.each_index(original_param.shape) do |indices, _|
      current = layer.params[param_name]
      original = current.get(*indices)

      current.set(*indices, original + EPSILON)
      loss_plus = layer.forward(input).map { |v| v**2 }.data.sum

      current.set(*indices, original - EPSILON)
      loss_minus = layer.forward(input).map { |v| v**2 }.data.sum

      current.set(*indices, original)

      numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON)
      analytical = analytical_grad.get(*indices)

      diff = (numerical_grad - analytical).abs
      if diff > max_diff
        max_diff = diff
        worst = [indices, numerical_grad, analytical]
      end
    end

    { max_diff: max_diff, passed: max_diff < TOLERANCE, worst: worst }
  end

  def self.check_array_param(layer, input, param_name)
    original_param = layer.params[param_name].dup
    small_lr = 1e-6

    output = layer.forward(input)
    d_output = output.map { |v| 2 * v }
    layer.backward(d_output, small_lr)
    updated_param = layer.params[param_name]
    analytical_grad = original_param.each_with_index.map { |o, i| (o - updated_param[i]) / small_lr }

    layer.params[param_name] = original_param.dup

    max_diff = 0.0
    worst = nil

    (0...original_param.length).each do |i|
      current = layer.params[param_name]
      original = current[i]

      current[i] = original + EPSILON
      loss_plus = layer.forward(input).map { |v| v**2 }.data.sum

      current[i] = original - EPSILON
      loss_minus = layer.forward(input).map { |v| v**2 }.data.sum

      current[i] = original

      numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON)
      analytical = analytical_grad[i]

      diff = (numerical_grad - analytical).abs
      if diff > max_diff
        max_diff = diff
        worst = [i, numerical_grad, analytical]
      end
    end

    { max_diff: max_diff, passed: max_diff < TOLERANCE, worst: worst }
  end
end

=end

class GradientChecker
  EPSILON = 1e-4
  TOLERANCE = 1e-2

  def self.check_layer(layer, input)
    original_snapshot = layer.params.transform_values do |p|
      p.is_a?(Tensor) ? p.copy : p.dup
    end

    results = layer.params.each_key.map do |name|
      [name, check_param(layer, input, name, original_snapshot)]
    end.to_h

    restore_all!(layer, original_snapshot)
    results
  end

  def self.restore_all!(layer, snapshot)
    snapshot.each do |name, val|
      layer.params[name] = val.is_a?(Tensor) ? val.copy : val.dup
    end
  end

  def self.check_param(layer, input, param_name, snapshot)
    restore_all!(layer, snapshot)   # clean slate before computing analytical grad

    small_lr = 1e-6
    before = snapshot[param_name]

    output = layer.forward(input)
    d_output = output.map { |v| 2 * v }
    layer.backward(d_output, small_lr)
    after = layer.params[param_name]

    analytical_grad = if before.is_a?(Tensor)
      before.combine(after) { |o, u| (o - u) / small_lr }
    else
      before.each_with_index.map { |o, i| (o - after[i]) / small_lr }
    end

    restore_all!(layer, snapshot)   # clean slate again before numerical loop

    max_diff = 0.0
    worst = nil

    if before.is_a?(Tensor)
      Tensor.each_index(before.shape) do |indices, _|
        current = layer.params[param_name]
        original = current.get(*indices)

        current.set(*indices, original + EPSILON)
        loss_plus = layer.forward(input).map { |v| v**2 }.data.sum
        current.set(*indices, original - EPSILON)
        loss_minus = layer.forward(input).map { |v| v**2 }.data.sum
        current.set(*indices, original)

        numerical = (loss_plus - loss_minus) / (2 * EPSILON)
        analytical = analytical_grad.get(*indices)
        diff = (numerical - analytical).abs
        if diff > max_diff
          max_diff = diff
          worst = [indices, numerical, analytical]
        end
      end
    else
      (0...before.length).each do |i|
        current = layer.params[param_name]
        original = current[i]

        current[i] = original + EPSILON
        loss_plus = layer.forward(input).map { |v| v**2 }.data.sum
        current[i] = original - EPSILON
        loss_minus = layer.forward(input).map { |v| v**2 }.data.sum
        current[i] = original

        numerical = (loss_plus - loss_minus) / (2 * EPSILON)
        analytical = analytical_grad[i]
        diff = (numerical - analytical).abs
        if diff > max_diff
          max_diff = diff
          worst = [i, numerical, analytical]
        end
      end
    end

    restore_all!(layer, snapshot)
    { max_diff: max_diff, passed: max_diff < TOLERANCE, worst: worst }
  end
end









conv = Conv2D.new(1, 2, kernel_size: 4, stride: 1, padding: 0)   # kernel == input size
input = Tensor.new(Array.new(1*1*4*4) { rand(-1.0..1.0) }, [1, 1, 4, 4])
results = GradientChecker.check_layer(conv, input)
puts results.inspect



t = Tensor.new((1..12).to_a, [2, 2, 3])
transposed = t.transpose([0,2,1])
puts transposed.contiguous?          # expect false
fixed = transposed.reshape([2, 6])    # should now correctly materialize first
puts fixed.data.inspect               # should reflect transposed's LOGICAL values, not raw old @data


conv = Conv2D.new(1, 3, kernel_size: 4, stride: 1, padding: 0)  # kernel == input size -> 1 window, 3 filters
input = Tensor.new(Array.new(1*1*4*4) { rand(-1.0..1.0) }, [1, 1, 4, 4])
puts GradientChecker.check_layer(conv, input).inspect


conv = Conv2D.new(1, 1, kernel_size: 2, stride: 1, padding: 0)  # 1 filter, many overlapping windows
input = Tensor.new(Array.new(1*1*4*4) { rand(-1.0..1.0) }, [1, 1, 4, 4])
puts GradientChecker.check_layer(conv, input).inspect

t = Tensor.new((1..12).to_a, [3, 4])
transposed = t.transpose   # non-contiguous
puts transposed.contiguous?   # expect false


identity_ish = Tensor.new([1,0,0, 0,1,0, 0,0,1], [3, 3])   # 9 elements, shape [3,3]
result = transposed.dot_product(identity_ish)
puts result.shape.inspect   # => [4, 3]
#identity_ish = Tensor.new([1,0,0,0,1,0,0,0,1,0,0,0], [4,3])  # arbitrary compatible shape
#result = transposed.dot_product(identity_ish)
#puts result.shape.inspect   # expect [4,3]

# cross-check against materializing first
materialized_result = transposed.materialize.dot_product(identity_ish)
puts result.data == materialized_result.data   # should be true if dot_product handles non-contiguous input correctly

a = Tensor.new([1,2,3,4,5,6], [2,3])      # [[1,2,3],[4,5,6]]
b = Tensor.new([7,8,9,10,11,12], [3,2])   # [[7,8],[9,10],[11,12]]
result = a.dot_product(b)
puts result.get(0,0)   # expect 58
puts result.get(0,1)   # expect 64
puts result.get(1,0)   # expect 139
puts result.get(1,1)   # expect 154



puts "end of file"

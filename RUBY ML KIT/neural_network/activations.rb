# Activation functions

# July 13th, 2026, Dwight Mayer

module Activations

  def sigmoid
    return self.map {|val| 1.0 / (1.0 + Math.exp(-val))}
  end
  
  def sigmoid_derivative
    s = self.sigmoid
    return s.hadamard_multiply(s.map {|val| 1-val})
  end
  
  def relu
    return self.map {|x| [0,x].max}    
  end

  def relu_derivative
    s = self.relu
    return s.map{|x| x > 0 ? 1 : 0}
  end

  def elu(alpha:1)
    return self.map {|x| x > 0 ? x: alpha * (Math.exp(x) -1)}
  end

  def elu_derivative(alpha:1)
    e=self.elu
    return e.map{|x| x > 0 ? 1.0 : alpha * Math.exp(x)}    
  end

  def selu(scale:1.0507, alpha:1.6733)
    # scaled ELU    
    return self.map {|value| value > 0 ? (scale*value) : scale*alpha*(Math.exp(value) - 1)}
  end

  def softmax

    row_max = self.reduce_keepdims([1]) {|vals| vals.max}
    shifted = self.combine(row_max) {|x,m| x-m}
    exp_vals = shifted.map {|v| Math.exp(v)}
    row_sum = exp_vals.reduce_keepdims([1]) {|vals| vals.sum}
    exp_vals.combine(row_sum) {|v,s| v/s}


  end

  def softmax_derivative
    s = self.softmax
    # This is NOT implemented yet. 
    # Full Jacobian Matrix necessary

  end

  # should add gelu, swish, mish, ...
  def tanh
    self.map {|x| Math.tanh(x)}
  end

  def tanh_derivative
    t = self.tanh
    t.map {|x| 1-x**2}
  end
end


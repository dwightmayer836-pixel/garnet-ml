# Loss functions

# July 14, 2026
# Dwight Mayer

class Loss
  def forward(y_pred, y_true)
    raise NotImplementedError
  end

  def backward(y_pred, y_true)
    raise NotImplementedError
  end
end

class MeanSquaredError < Loss
  def forward(y_pred, y_true)
    diff = y_pred.subtract(y_true)
    diff.hadamard_multiply(diff).sum / y_pred.rows
  end

  def backward(y_pred, y_true)
    diff = y_pred.subtract(y_true)
    diff.scalar_multiply(2.0 / y_pred.rows)
  end
end

class CrossEntropy < Loss
  def forward(y_pred, y_true)
    probs = y_pred.softmax
    correct_probs = probs.pick_by_row(y_true)
    losses = correct_probs.map { |p| -Math.log([p, 1e-7].max) }
    losses.sum / y_pred.shape[0].to_f
  end
 
  def backward(y_pred, y_true)
    probs = y_pred.softmax
    rows, cols = y_pred.shape
    one_hot = if y_pred.is_a?(Matrix)
      Matrix.build(rows, cols) { |r, c| y_true.get(r, 0) == c ? 1 : 0 }
    else
      Tensor.new(
        (0...rows).flat_map { |r| (0...cols).map { |c| y_true.get(r, 0) == c ? 1 : 0 } },
        [rows, cols]
      )
    end

    probs.subtract(one_hot).scalar_divide(rows)
  end

end




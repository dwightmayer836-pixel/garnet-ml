# Dwight Mayer, custom matrix class, July 2nd, 2026

require_relative "vector_operations"
require_relative "linear_algebra"
require_relative "statistics"
require_relative "../neural_network/activations"

require_relative "tensor"

class Matrix

  extend VectorOperations
  include LinearAlgebra
  include Statistics
  include Activations

  attr_reader :rows, :cols
  attr_accessor :data
  def initialize(data)
    
    @data = data
    @rows = data.length
    @cols = data[0].length
  
  end
  
  def get(row_idx, col_idx)
    self.validate_row_index(row_idx)
    self.validate_column_index(col_idx) 
    return @data[row_idx][col_idx]
  end  

  def set(row_idx, col_idx, new_value)
    self.validate_row_index(row_idx)
    self.validate_column_index(col_idx)    
    @data[row_idx][col_idx] = new_value
  end

  def to_s
    # UNORIGINAL string representation method
    formatted_matrix = self.map {|val| Matrix.format_cell(val)}
    rounded_data = formatted_matrix.data
    widest_col = rounded_data.flatten.map(&:length).max
    lines = rounded_data.map do |row|
      row.map { |v| v.rjust(widest_col) }.join("  ")
    end
    return lines.join("\n")
  end

  def self.format_cell(value)
    value.is_a?(Float) ? format("%.4f", value) : value.to_s
  end

  def get_row(row_idx)
    self.validate_row_index(row_idx)
    return @data[row_idx].dup
  end

  def set_row!(row_idx, new_row)
    # need a method to set row
    unless new_row.length == @cols
      raise ArgumentError, 'bad num columns in row setting'
    end
    self.validate_row_index(row_idx)
    for i in 0...new_row.length
      self.set(row_idx, i, new_row[i])
    end
  end

  def get_col(col_idx)
    self.validate_column_index(col_idx)
    column = []
    for row_idx in 0...@rows
      column.push(@data[row_idx][col_idx])
    end
    return column
  end

  def set_col!(col_idx, new_column)
    # need method to set column
    unless new_column.length == @rows
      raise ArgumentError, 'bad num rows in column setting'
    end
    
    self.validate_column_index(col_idx)
    for i in 0...new_column.length
      self.set(i, col_idx, new_column[i])
    end

  end

  def validate_row_index(row_idx)
    raise IndexError, "row index out of bounds" unless row_idx.between?(0, @rows-1)
  end

  def validate_column_index(col_idx)
    raise IndexError, "column index out of bounds" unless col_idx.between?(0, @cols-1)
  end

  def self.create_zeroes(row_len, col_len)
    return Array.new(row_len) { Array.new(col_len, 0) }
  end

  def shape
    return [@rows, @cols]
  end

  def same_shape?(other)
    return (@cols == other.cols && @rows == other.rows)
  end

  def matching_dimensions?(other)
    return (@cols == other.rows)
  end
 
  def square?
    return @rows == @cols
  end

  def validate_square
    unless self.square?
      raise ArgumentError, "matrix is not square"
    end
  end

  def add(other)
  #  other = self.broadcast(other)
    self.combine(other) {|a, b| a + b}
  end

  def subtract(other)
   #  other = self.broadcast(other)
    self.combine(other) {|a, b| a - b}
  end

  def transpose
    Matrix.build(@cols, @rows) {|row_idx, col_idx| self.get(col_idx, row_idx)}
  end


  def dot_product(other)
    
    if other.is_a?(Array)
      # do matrix on array multiplication
      #Matrix.validate_length(other, @cols)
      raise ArgumentError, "inner dimensions must match" unless @cols == other.rows


      result = Array.new(@rows)
      for row_index in 0...@rows
        result[row_index] = Matrix.vector_dot_product(self.get_row(row_index), other)
      end
      return Matrix.create_column_vector(result)

    elsif other.is_a?(Matrix)
      # do mat on mat operation MUST have good dimensions
      #Matrix.validate_length(other, @cols)
      return Matrix.build(@rows, other.cols) {|row_idx, col_idx| Matrix.vector_dot_product(
	self.get_row(row_idx), other.get_col(col_idx))}

    else
      raise ArgumentError, "bad shape for dot product"
    end

  end

  def matrix_multiply(other)
    unless self.matching_dimensions?(other)
      raise ArgumentError, "Bad dimensions in matrix multiplication"
    end
    
    Matrix.build(@rows, other.cols) {|r_idx, c_idx| Matrix.vector_dot_product(
      self.get_row(r_idx), other.get_col(c_idx))}
 
  end
  
  def scalar_multiply(scalar) 
    self.map {|val| val * scalar}
  end

  def scalar_divide(scalar)
    raise ZeroDivisionError if scalar == 0
    self.map {|val| val / scalar}
  end
  
  def copy
    return Matrix.new(@data.map(&:dup))
  end

  def self.identity(n)
    # create N x N identity matrix
    Matrix.build(n, n) {|row_idx, col_idx| row_idx == col_idx ? 1 : 0}

  end

  def equals?(other)
  return false unless self.same_shape?(other)
    for row_idx in 0...@rows
      for col_idx in 0...@cols
        if other.get(row_idx, col_idx) != self.get(row_idx, col_idx)
          return false
        end
      end
    end
    return true
  end
 
  def hadamard_multiply(other)
    self.combine(other) {|a, b| a * b}
  end

  def hadamard_divide(other)
    self.combine(other) {|a, b| a / b}
  end

  def flatten
    return @data.flatten
  end

  def map
    # this applies a function to every cell in the dataset
    new_grid = Matrix.create_zeroes(@rows, @cols)
    for row_idx in 0...@rows
      for col_idx in 0...@cols
        new_grid[row_idx][col_idx] = yield(self.get(row_idx, col_idx))
      end
    end
    return Matrix.new(new_grid)
  end

  def combine(other, axis: nil)
    other = self.broadcast(other, axis:axis)
    new_grid = Matrix.create_zeroes(@rows, @cols)    
    for row_idx in 0...@rows
      for col_idx in 0...@cols
        new_grid[row_idx][col_idx] = yield(self.get(row_idx, col_idx), other.get(row_idx, col_idx))
      end
    end
    return Matrix.new(new_grid) 
  end

  def self.build(rows, cols)
    # builds new matrix according to optional function / idx calls...
    new_grid = Matrix.create_zeroes(rows, cols)
    for row_idx in 0...rows
      for col_idx in 0...cols
        new_grid[row_idx][col_idx] = yield(row_idx, col_idx)
      end
    end
    return Matrix.new(new_grid)
  end

  def swap_rows!(row1_idx, row2_idx)
  # in-place mutation of the Matrix object!
    @data[row1_idx], @data[row2_idx] = @data[row2_idx], @data[row1_idx]
  end

  def scale_row!(row_idx, scalar) 
    # scales each element in row 
    self.validate_row_index(row_idx)
    for idx in 0... (@cols)
      val = self.get(row_idx, idx) * scalar
      self.set(row_idx, idx, val)
    end
  end

  def add_scaled_row!(target_row, source_row, scalar)
    # add a scaled version ofsource row onto target row
    self.validate_row_index(target_row)
    self.validate_row_index(source_row)

    for col_idx in 0...@cols
      scaled_value = self.get(source_row, col_idx) * scalar
      new_value = self.get(target_row, col_idx) + scaled_value
      self.set(target_row, col_idx, new_value)
    end
  end  

  def hstack(other)
    # horizontall glues together two matrices
    unless @rows == other.rows
      raise ArgumentError, "Bad shape in matrix H stack"
    end
    new_grid = []
    for row_idx in 0...@rows
      left_row = self.get_row(row_idx)
      right_row = other.get_row(row_idx)
      new_grid.push(left_row + right_row)
    end
    return Matrix.new(new_grid)
  end

  def vstack(other)
    # vertically glues together two matrices
    unless @cols == other.cols
      raise ArgumentError, "Bad shape in matrix V stack"
    end

    new_grid = []
    for row_idx in 0...@rows
      new_grid.push(self.get_row(row_idx))
    end
    
    for row_idx in 0...other.rows
      new_grid.push(other.get_row(row_idx))
    end
    return Matrix.new(new_grid)
  end


  def slice(row_start:0, row_end:@rows-1, col_start:0, col_end:@cols-1)
    # gets submatrices from the original matrix
   
    # input validation 
    self.validate_row_index(row_start)
    self.validate_row_index(row_end)
    self.validate_column_index(col_start)
    self.validate_column_index(col_end)

    sliced_grid = Matrix.create_zeroes((row_end-row_start+1),(col_end-col_start+1))
    
    for row_idx in row_start..row_end
      for col_idx in col_start..col_end
        sliced_grid[row_idx-row_start][col_idx-col_start] = self.get(row_idx, col_idx)
      end
    end
    return Matrix.new(sliced_grid)
  end 
  def self.init_rand(rows, cols)
    Matrix.build(rows, cols) {|rows, cols| rand(-1.0.. 1.0)}
  end


  def self.vector_sum(values)
    return values.sum
  end

 
  
  def self.validate_length(other, expected_length)
    # other = Array, expected_length = INT
    unless other.length == expected_length
      raise ArgumentError, "expected length: #{expected_length} received: #{other.length}"
    end

  end

  def broadcast(other, axis:nil)

    if other.is_a?(Matrix)
      return other if self.same_shape?(other)
    
      if other.rows==1 && other.cols == @cols
        return Matrix.build(@rows, @cols) {|row, col| other.get(0, col)}
      elsif other.cols==1 && other.rows==@rows
        return Matrix.build(@rows,@cols) {|row, col| other.get(row,0)} 
      else
        raise ArgumentError, 'bad shape'
      end


    elsif other.is_a?(Numeric)
      scalar_mat = Matrix.build(@rows, @cols) {|row, col| other}
      return scalar_mat
    elsif other.is_a?(Array)
      if self.square?
        if axis == 0
	  #Matrix.validate_length(other, @cols)
          return Matrix.build(@rows, @cols) {|row, col| other[col]}
        elsif axis == 1
	  #Matrix.validate_length(other, @rows)
          return Matrix.build(@rows, @cols) {|row, col| other[row]}
        else raise ArgumentError, 'nil axis on square broadcast'
      end      
      elsif other.length==@cols
         # return matrix where every row is OTHER
        return Matrix.build(@rows, @cols) {|row, col| other[col]}
      elsif other.length==@rows
	# return matrix where every column is OTHER   
        return Matrix.build(@rows, @cols) {|row, col| other[row]} 
      end   
    else raise ArgumentError, "bad shape"
    end
  end
=begin
  def self.mean_squared_error(y_true, y_pred)
    errors = y_true.subtract(y_pred)
    errors = errors.hadamard_multiply(errors)
    return Matrix.vector_mean(errors.flatten)
  end  
=end
  def reshape(*args)

    new_rows, new_cols = args.length ==1 ? args[0]:args


    unless @rows * @cols == new_rows * new_cols
      raise ArgumentError, "bad dimensions in reshape operation"
    end
    
    index = 0
    reshaped = Matrix.new(Matrix.create_zeroes(new_rows, new_cols))
    
    for row in 0...@rows
      for col in 0...@cols
        new_row = index / new_cols
        new_col = index % new_cols
        reshaped.set(new_row, new_col, self.get(row, col))
        index += 1
      end
    end
    return reshaped
  end
  # BEST FOR ELU family functions
  def self.init_he(rows, cols)
    limit = Math.sqrt(6.0 / rows)
    return Matrix.build(rows, cols) { rand(-limit...limit) }
  end
  # BEST FOR TANH / SIGMOID
  def self.init_xavier(rows, cols)
    limit = Math.sqrt(6.0 / (rows + cols))
    return Matrix.build(rows, cols) { rand(-limit...limit) }
    
  end    
  
  def pick_by_row(indices)
   (0...@rows).map { |r| self.get(r, indices.get(r,0)) }
  end    
 
  def self.one_hot(label, num_classes)
    vector = Array.new(num_classes, 0.0)
    vector[label] = 1.0 
    Matrix.new([vector]).transpose
  end

  def reduce_keepdims(axes)

    out_rows = axes.include?(0) ? 1:@rows 
    out_cols = axes.include?(1) ? 1:@cols
    grid = Matrix.create_zeroes(out_rows, out_cols)
    
    (0...out_rows).each do |out_r|
      (0...out_cols).each do |out_c|
        row_range = axes.include?(0) ? (0...@rows):[out_r]
        col_range = axes.include?(1) ? (0...@cols):[out_c]
	values = row_range.flat_map {|r| col_range.map { |c| self.get(r,c)}}
        grid[out_r][out_c] = yield(values)
      end
    end
    Matrix.new(grid)        
  end

  def to_tensor
    Tensor.new(self.flatten, [@rows, @cols])
  end  






end

require "csv"


# This is a data source specifically for CSVs representing the MNIST dataset
class CSVDataSource
  def initialize(path, batch_size, as_image:false)
    @path = path
    @batch_size = batch_size
    @as_image = as_image
  end

  def each_batch
    batch_rows = []

    CSV.foreach(@path, headers:true) do |row|
      batch_rows << row
      if batch_rows.length == @batch_size
        yield to_matrices(batch_rows)
        batch_rows = []
      end
    end
    yield to_matrices(batch_rows) unless batch_rows.empty?
  end
  
  def to_matrices(batch_rows)
    # creates a batch x 784 mat of pixels, batch x 1 mat of labels
    pixel_data = batch_rows.map {|row| row.fields[1..].map(&:to_f)}
    label_data = batch_rows.map {|row| [row.fields[0].to_i]}

    input = Matrix.new(pixel_data)
    labels = Matrix.new(label_data)

    if @as_image
      input = input.to_tensor.reshape([batch_rows.length, 1, 28, 28])
    end


    return [input, labels]


  end
end

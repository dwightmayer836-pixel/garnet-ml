# Can this network learn the XOR function?
# Dwight Mayer, July 14th, 2026

require_relative "../core/matrix"
require_relative "../neural_network/losses"
require_relative "../neural_network/layers"

require_relative "../neural_network/activations"
require_relative "../neural_network/activation_layers"
require_relative "../neural_network/neural_network"

require_relative "../data/csv_data_source"

require "csv"




=begin

csv_path_train = File.join(File.dirname(__FILE__), "..", "data", "mnist_train.csv")
csv_path_test = File.join(File.dirname(__FILE__), "..", "data", "mnist_test.csv")


network_old  = NeuralNetwork.new([
Dense.new(784, 256),
ReLU.new, 
Dense.new(256, 128),
ReLU.new,
Dense.new(128, 10)],
loss:CrossEntropy.new)


network = NeuralNetwork.new([
Dense.new(784, 128),
ReLU.new,
Dense.new(128, 64),
ReLU.new,
Dense.new(64, 10)],
loss:CrossEntropy.new)

train_source = CSVDataSource.new(csv_path_train, 128)
test_source  = CSVDataSource.new(csv_path_test, 128)

=end

#network.train_alt(train_source, epochs: 10, learning_rate: 0.03, verbose: true)
def evaluate_accuracy(network, data_source)
  correct = 0
  total = 0

  data_source.each_batch do |batch_input, batch_target|
    predictions = network.predict(batch_input)   # raw logits, shape [batch, 10]

    (0...predictions.rows).each do |r|
      row_values = predictions.get_row(r)
      predicted_class = row_values.each_with_index.max.last   # argmax
      actual_class = batch_target.get(r, 0)

      correct += 1 if predicted_class == actual_class
      total += 1
    end
  end

  correct.to_f / total
end

csv_path_train = File.join(File.dirname(__FILE__), "..", "data", "mnist_train_small.csv")
csv_path_test = File.join(File.dirname(__FILE__), "..", "data", "mnist_test_small.csv")

train_source = CSVDataSource.new(csv_path_train, 32, as_image:true)
test_source  = CSVDataSource.new(csv_path_test, 32, as_image:true)



network_alt = NeuralNetwork.new([

Conv2D.new(1, 8, kernel_size: 3, padding: 1),
  ReLU.new,
  MaxPool2D.new(2),
  Conv2D.new(8, 16, kernel_size: 3, padding: 1),
  ReLU.new,
  MaxPool2D.new(2),
  Flatten.new,                 # bridges Tensor -> Matrix here
  Dense.new(16 * 7 * 7, 64),   # 28 -> 14 -> 7 after two 2x2 pools
  ReLU.new,
  Dense.new(64, 32),
  ReLU.new,
  Dense.new(32, 10)])

network_alt.train_alt(train_source, epochs: 8, learning_rate: 0.01, verbose: true)
accuracy = evaluate_accuracy(network, test_source)
puts "test accuracy: #{(accuracy * 100).round(2)}%"







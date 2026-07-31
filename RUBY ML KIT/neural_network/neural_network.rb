# legitimate neural network :)
# July 14th, 2026

# Dwight Mayer

class NeuralNetwork
  def initialize(layers, loss:CrossEntropy.new)
    @layers = layers
    @loss = loss
    puts "Neural Network Initialized"
  end  
  
  def train!
    @layers.each {|layer| layer.train!}
  end

  def eval!
    @layers.each {|layer| layer.eval!}
  end

  def parameters
    @layers.flat_map {|layer| layer.parameters} 
  end  


  def predict(input)
    eval!
    forward(input)
  end
  
  def forward(input)
    @layers.reduce(input) {|output, layer| layer.forward(output)}
  end

  def train_step(input, target, learning_rate)
    train!
    prediction = forward(input)

    loss_value = @loss.forward(prediction, target)
    gradient = @loss.backward(prediction, target)

    @layers.reverse.each {|layer| gradient=layer.backward(gradient, learning_rate)}
    return loss_value
  end

  def train(input, target, epochs, learning_rate, tolerance:1e-6, verbose:false, batch_size:nil)
    # Runs a full training loop
    epochs.times do |epoch|

      loss = if batch_size
        train_epoch_batched(input, target, batch_size, learning_rate)
      else
        train_step(input, target, learning_rate)
      end 

      puts "epoch #{epoch}: loss = #{loss}" if verbose && epoch % 100 == 0
      return loss if loss < tolerance
    end
    nil
  end

  def train_epoch_batched(input, target, batch_size, learning_rate)
    total_loss = 0.0
    num_batches = 0

    (0...input.rows).step(batch_size) do |start|
      end_idx = [start + batch_size - 1, input.rows - 1].min
      batch_input = input.slice(row_start:start,row_end:end_idx)
      batch_target = target.slice(row_start:start,row_end:end_idx)
      total_loss += train_step(batch_input, batch_target, learning_rate)
      num_batches += 1
    end

    return (total_loss / num_batches)

  end

  def train_alt(data_source, epochs:10000, learning_rate:0.01,tolerance:1e-6, verbose:true)
    epochs.times do |epoch|
      total_loss = 0.0
      num_batches = 0

      data_source.each_batch do |batch_input, batch_target|
        total_loss += train_step(batch_input, batch_target, learning_rate)
        num_batches += 1
	if num_batches % 3 ==0
          puts "batch #{num_batches} processed"
        end  
      end

      avg_loss = total_loss / num_batches
      puts "epoch #{epoch}: loss = #{avg_loss}" if verbose && epoch % 1 == 0
      return avg_loss if avg_loss < tolerance
    end
    nil
  end

end

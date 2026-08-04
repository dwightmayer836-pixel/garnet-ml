
// this is the first Rust file I've ever touched
// Dwight Mayer, July 28th, 2026

// aiming to recreate the following primitives
// dot product
// im2col
// col2im

use magnus::{
    function,
    prelude::*,
    Error,
};

pub fn matrix_multiply(
    a: Vec<f64>,
    b: Vec<f64>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
) -> Vec<f64>{

    let rows = a_shape[0];
    let inner = a_shape[1];
    let cols = b_shape[1];

    let mut result = vec![0.0; rows*cols];
    
    for i in 0..rows{
        for k in 0..inner{
	    let a_val = a[i*inner+k];

	    for j in 0..cols{
		result[i*cols+j] +=
		   a_val * b[k*cols+j];
	    }
        }    
    }    
    result
}

pub fn im2col(
	data:Vec<f64>,
	shape:Vec<i64>,
	kernel_h:i64,
	kernel_w:i64,
	stride:i64,
) -> Vec<f64> {

	let (batch,channels,h,w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = (h-kernel_h) / stride + 1;
	let out_w = (w-kernel_w) / stride + 1;
	let num_windows = (batch * out_h * out_w) as usize;
	let window_size = (channels * kernel_h * kernel_w) as usize;
	let mut result = vec![0.0; num_windows*window_size];

	let mut row = 0;

	for b in 0..batch{
	    for oh in 0..out_h{
		for ow in 0..out_w{
		    let mut col = 0;
		    for c in 0..channels{
			for kh in 0..kernel_h{
			    for kw in 0..kernel_w{
				let r = oh*stride + kh;
				let cc = ow*stride + kw;
				let idx = (b * channels * h * w + c * h * w + r * w + cc) as usize;

				result[row*window_size+col] = data[idx];
				col += 1;

					}
				}				
			}
			row += 1
		}
	    }
	}
	result
}

pub fn col2im(
    cols: Vec<f64>,
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
) -> Vec<f64> {
    let out_h = (h - kernel_h) / stride + 1;
    let out_w = (w - kernel_w) / stride + 1;
    let window_size = channels * kernel_h * kernel_w;

    let mut result = vec![0.0; batch * channels * h * w];
    let mut row = 0;

    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut col = 0;
                for c in 0..channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let r = oh * stride + kh;
                            let cc = ow * stride + kw;
                            let idx = b * channels * h * w + c * h * w + r * w + cc;

                            result[idx] += cols[row * window_size + col];
                            col += 1;
                        }
                    }
                }
                row += 1;
            }
        }
    }
    result
}


pub fn elementwise_add(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn elementwise_multiply(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

pub fn scalar_multiply(data: Vec<f64>, scalar:f64) -> Vec<f64>{
    data.iter().map(|v| v * scalar).collect()
}

pub fn scalar_divide(data: Vec<f64>, scalar:f64) -> Vec<f64> {
    data.iter().map(|v| v / scalar).collect()
}

pub fn sigmoid(data: Vec<f64>) -> Vec<f64> {
    data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

pub fn sigmoid_derivative(data: Vec<f64>) -> Vec<f64> {
    data.into_iter()
	.map(|v|  {
	let s = 1.0 / (1.0 + (-v).exp());
	s * (1.0 - s)
	})
	.collect()
}

pub fn relu(data: Vec<f64>) -> Vec<f64> {
    data.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
}

pub fn relu_derivative(data: Vec<f64>) -> Vec<f64> {
    data.into_iter()
	.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
	.collect()
		

}

pub fn tanh(data: Vec<f64>) -> Vec<f64> {
    data.iter().map(|&v| v.tanh()).collect()
}

pub fn tanh_derivative(data: Vec<f64>) -> Vec <f64> {
    data.into_iter().map ( |v| { 
	let t = v.tanh();
	1.0 - t * t
	})
	.collect()
}

pub fn elu(data:Vec<f64>, alpha:f64) -> Vec<f64> {
    // logic
    data.into_iter()
	.map ( |v| if v > 0.0 { v } else {alpha * (v.exp() - 1.0)})
	.collect()
}

pub fn elu_derivative(data:Vec<f64>, alpha:f64) -> Vec<f64> {
    // logic
    data.into_iter()
	.map (|v| if v > 0.0 { 1.0 } else {alpha * v.exp()})
	.collect() 
}

pub fn max_pool2d_forward(
    data:Vec<f64>, 
    shape:Vec<i64>, 
    pool_h:i64,
    pool_w:i64, 
    stride:i64
) -> (Vec<f64>, Vec<i64>) {
    // logic...
    let batch = shape[0] as usize;
    let channels = shape[1] as usize;
    let h = shape[2] as usize;
    let w = shape[3] as usize;

    let pool_h = pool_h as usize;
    let pool_w = pool_w as usize;
    let stride = stride as usize;

    let out_h = (h - pool_h) / stride + 1;
    let out_w = (w - pool_w) / stride + 1;

    let mut result = vec![0.0; batch*channels*out_h*out_w]; 
    let mut winners = vec![0i64; batch*channels*out_h*out_w];

    let mut out_idx = 0;
    for b in 0..batch {
	for c in 0..channels {
	    for oh in 0..out_h {
		for ow in 0..out_w {
		    let mut best_val = f64::NEG_INFINITY;
		    let mut best_idx = 0usize;

		    for ph in 0..pool_h {
			for pw in 0..pool_w {
			    let r = oh*stride+ph;
			    let col = ow*stride+pw;
			    let idx = b*channels*h*w+c*h*w+r*w+col;

			    if data[idx] > best_val {
				best_val = data[idx];
				best_idx = idx;
		    }
		}
	    }
	    result[out_idx] = best_val;
	    winners[out_idx] =  best_idx as i64;
	    out_idx += 1;
	    }
        }
    }
}
	(result, winners)	
}

pub fn max_pool2d_backward(output_gradient:Vec<f64>,winners:Vec<i64>,input_size:i64) -> Vec<f64> {
    let input_size = input_size as usize;
    let mut input_gradient = vec![0.0; input_size];

    for (grad, &winner_idx) in output_gradient.iter().zip(winners.iter()) {
	input_gradient[winner_idx as usize] += grad;
    }
    input_gradient
}






#[magnus::init]
fn init() -> Result<(), Error> {
    let module = magnus::define_module("RustyTensor")?;

    module.define_singleton_method(
        "matrix_multiply",
        function!(matrix_multiply, 4),
    )?;

    module.define_singleton_method(
        "im2col",
        function!(im2col, 5),
    )?;

    module.define_singleton_method(
	"col2im",
	function!(col2im, 8)
    )?;

    module.define_singleton_method(
	"elementwise_add",
	function!(elementwise_add, 2)
    )?;

    module.define_singleton_method(
	"scalar_multiply",
	function!(scalar_multiply, 2)
    )?;

    module.define_singleton_method(
	"scalar_divide",
	function!(scalar_divide, 2)
    )?;

    module.define_singleton_method(
	"elementwise_multiply",
	function!(elementwise_multiply, 2)
    )?;

    module.define_singleton_method(
	"sigmoid",
	function!(sigmoid, 1)
    )?;

    module.define_singleton_method(
	"relu",
	function!(relu, 1)
    )?;
    
    module.define_singleton_method(
	"relu_derivative", 
	function!(relu_derivative, 1)
    )?;

    module.define_singleton_method(
	"sigmoid_derivative",
	function!(sigmoid_derivative, 1)
    )?;

    module.define_singleton_method(
	"tanh_derivative",
	function!(tanh_derivative, 1)
    )?;

    module.define_singleton_method(
	"elu",
	function!(elu, 2)
    )?;

    module.define_singleton_method(
	"elu_derivative",
	function!(elu_derivative, 2)
    )?;

    module.define_singleton_method(
	"max_pool2d_forward",
	function!(max_pool2d_forward, 5)
    )?;

    module.define_singleton_method(
	"max_pool2d_backward", 
	function!(max_pool2d_backward, 3)
    )?;

    Ok(())

}







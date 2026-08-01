
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

pub fn relu(data: Vec<f64>) -> Vec<f64> {
    data.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
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

    Ok(())
}







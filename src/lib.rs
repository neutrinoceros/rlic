use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // TODO: parametrize dtype (support f32 too)

    fn advance(vx: f64, vy: f64, x: usize, y: usize, fx: f64, fy: f64, w: usize, h: usize) {}

    fn convolve<'py>(
        u: ArrayView2<'py, f64>,
        v: ArrayView2<'py, f64>,
        kernel: ArrayView1<'py, f64>,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
    ) {
        let ny = u.shape()[0];
        let nx = u.shape()[1];
        let kernellen = kernel.shape()[0];

        for i in 0..ny {
            for j in 0..nx {
                let x = j;
                let y = i;
                let fx = 0.5;
                let fy = 0.5;
                let mut k = kernellen / 2;

                output[[i, j]] += kernel[[k]] * input[[y, x]];

                while k < kernellen - 1 {
                    let ui = u[[y, x]];
                    let vi = v[[y, x]];
                    advance(ui, vi, x, y, fx, fy, nx, ny);
                    k += 1;
                    output[[i, j]] += kernel[[k]] * input[[y, x]];
                }

                let x = j;
                let y = i;
                let fx = 0.5;
                let fy = 0.5;
                k = kernellen / 2;

                while k > 0 {
                    let ui = u[[y, x]];
                    let vi = v[[y, x]];
                    advance(-ui, -vi, x, y, fx, fy, nx, ny);
                    k -= 1;
                    output[[i, j]] += kernel[[k]] * input[[y, x]];
                }
            }
        }
        output.fill(1.0);
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_loop")]
    fn convolve_loop_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        u: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        kernel: PyReadonlyArray1<'py, f64>,
        iterations: i64,
    ) -> Bound<'py, PyArray2<f64>> {
        let u = u.as_array();
        let v = v.as_array();
        let kernel = kernel.as_array();
        let image = image.as_array();
        let mut input =
            Array2::from_shape_vec(image.raw_dim(), image.iter().cloned().collect()).unwrap();
        let mut output = Array2::<f64>::zeros(image.raw_dim());

        let mut it_count = 0;
        while it_count < iterations {
            convolve(u, v, kernel, &input, &mut output);
            it_count += 1;
            if it_count < iterations {
                input.assign(&output);
                output.fill(0.0);
            }
        }

        output.to_pyarray_bound(py)
    }

    Ok(())
}

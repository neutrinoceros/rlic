use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // TODO: parametrize dtype (support f32 too)
    fn convolve<'py>(
        u: ArrayView2<'py, f64>,
        v: ArrayView2<'py, f64>,
        kernel: ArrayView1<'py, f64>,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
    ) {
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

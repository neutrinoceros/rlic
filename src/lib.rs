use std::array::from_ref;

use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{
    Array2, ArrayBase, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut2, ArrayViewMutD,
    OwnedRepr, ViewRepr,
};
use numpy::ToPyArray;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from lick-core!".to_string()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray_bound(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py<'py>(a: f64, x: &Bound<'py, PyArrayDyn<f64>>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    // ... lic ...
    // TODO: parametrize dtype (support f32 too)
    fn convolve<'py>(
        u: ArrayView2<'py, f64>,
        v: ArrayView2<'py, f64>,
        kernel: ArrayView1<'py, f64>,
        input: &Array2<f64>,
        output: &Array2<f64>,
    ) {
        //output.fill(1.0);
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve")]
    fn convolve_py<'py>(
        py: Python<'py>,
        u: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        kernel: PyReadonlyArray1<'py, f64>,
        texture: PyReadonlyArray2<'py, f64>,
        iterations: i64,
    ) -> Bound<'py, PyArray2<f64>> {
        let u = u.as_array();
        let v = v.as_array();
        let kernel = kernel.as_array();
        let texture = texture.as_array();
        let mut input =
            Array2::from_shape_vec(texture.raw_dim(), texture.iter().cloned().collect()).unwrap();
        let mut output = Array2::<f64>::zeros(texture.raw_dim());

        let mut it_count = 0;
        while it_count < iterations {
            convolve(u, v, kernel, &input, &output);
            it_count += 1;
            if it_count < iterations {
                input.assign(&output);
                output.fill(0.0);
            }
        }

        output.to_pyarray_bound(py)
    }

    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    Ok(())
}

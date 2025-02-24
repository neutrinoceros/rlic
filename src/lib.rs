use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::cmp::{max, min};

enum UVMode {
    Velocity,
    Polarization,
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn advance(
        vx: f64,
        vy: f64,
        x: &mut i64,
        y: &mut i64,
        fx: &mut f64,
        fy: &mut f64,
        w: i64,
        h: i64,
    ) {
        let mut zeros: i64 = 0;

        // Think of tx (ty) as the time it takes to reach the next pixel along x (y).
        let tx: f64;
        let ty: f64;
        if vx > 0.0 {
            tx = (1.0 - *fx) / vx;
        } else if vx < 0.0 {
            tx = -*fx / vx;
        } else {
            zeros += 1;
            tx = 1e100;
        }
        if vy > 0.0 {
            ty = (1.0 - *fy) / vy;
        } else if vy < 0.0 {
            ty = -*fy / vy;
        } else {
            zeros += 1;
            ty = 1e100;
        }

        if zeros == 2 {
            return;
        }

        if tx < ty {
            // We reached the next pixel along x first.
            if vx >= 0.0 {
                *x += 1;
                *fx = 0.0;
            } else {
                *x -= 1;
                *fx = 1.0;
            }
            *fy += tx * vy;
        } else {
            // We reached the next pixel along y first.
            if vy >= 0.0 {
                *y += 1;
                *fy = 0.0;
            } else {
                *y -= 1;
                *fy = 1.0;
            }
            *fx += ty * vx;
        }
        *x = max(0, min(w - 1, *x));
        *y = max(0, min(h - 1, *y));
    }

    fn as_array_index(x: i64, nx: usize) -> usize {
        if x >= 0 {
            x as usize
        } else {
            ((nx as i64) + x) as usize
        }
    }

    fn convolve<'py>(
        u: ArrayView2<'py, f64>,
        v: ArrayView2<'py, f64>,
        kernel: ArrayView1<'py, f64>,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        uv_mode: &UVMode,
    ) {
        let ny = u.shape()[0];
        let nx = u.shape()[1];
        let kernellen = kernel.len();

        let w = nx as i64;
        let h = ny as i64;

        for i in 0..ny {
            for j in 0..nx {
                let mut x: i64 = j.try_into().unwrap();
                let mut y: i64 = i.try_into().unwrap();
                let mut fx = 0.5;
                let mut fy = 0.5;
                let mut k = kernellen / 2;
                let mut last_ui = 0.0;
                let mut last_vi = 0.0;

                output[[i, j]] +=
                    kernel[[k]] * input[[as_array_index(y, ny), as_array_index(x, nx)]];

                while k < kernellen - 1 {
                    let mut ui = u[[as_array_index(y, ny), as_array_index(x, nx)]];
                    let mut vi = v[[as_array_index(y, ny), as_array_index(x, nx)]];
                    match uv_mode {
                        UVMode::Polarization => {
                            if (ui * last_ui + vi * last_vi) < 0.0 {
                                ui = -ui;
                                vi = -vi;
                            }
                            last_ui = ui;
                            last_vi = vi;
                        }
                        UVMode::Velocity => {}
                    };
                    advance(ui, vi, &mut x, &mut y, &mut fx, &mut fy, w, h);
                    k += 1;
                    output[[i, j]] +=
                        kernel[[k]] * input[[as_array_index(y, ny), as_array_index(x, nx)]];
                }

                x = j.try_into().unwrap();
                y = i.try_into().unwrap();
                fx = 0.5;
                fy = 0.5;
                k = kernellen / 2;
                last_ui = 0.0;
                last_vi = 0.0;

                while k > 0 {
                    let mut ui = u[[as_array_index(y, ny), as_array_index(x, nx)]];
                    let mut vi = v[[as_array_index(y, ny), as_array_index(x, nx)]];
                    match uv_mode {
                        UVMode::Polarization => {
                            if (ui * last_ui + vi * last_vi) < 0.0 {
                                ui = -ui;
                                vi = -vi;
                            }
                            last_ui = ui;
                            last_vi = vi;
                        }
                        UVMode::Velocity => {}
                    };
                    advance(-ui, -vi, &mut x, &mut y, &mut fx, &mut fy, w, h);
                    k -= 1;
                    output[[i, j]] +=
                        kernel[[k]] * input[[as_array_index(y, ny), as_array_index(x, nx)]];
                }
            }
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_iteratively")]
    fn convolve_interatively_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        u: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        kernel: PyReadonlyArray1<'py, f64>,
        iterations: i64,
        uv_mode: String,
    ) -> Bound<'py, PyArray2<f64>> {
        let u = u.as_array();
        let v = v.as_array();
        let kernel = kernel.as_array();
        let image = image.as_array();
        let mut input =
            Array2::from_shape_vec(image.raw_dim(), image.iter().cloned().collect()).unwrap();
        let mut output = Array2::<f64>::zeros(image.raw_dim());

        let uv_mode_enum: UVMode;
        if uv_mode == "polarization" {
            uv_mode_enum = UVMode::Polarization;
        } else if uv_mode == "velocity" {
            uv_mode_enum = UVMode::Velocity;
        } else {
            panic!("unknown uv_mode")
        }

        let mut it_count = 0;
        while it_count < iterations {
            convolve(u, v, kernel, &input, &mut output, &uv_mode_enum);
            it_count += 1;
            if it_count < iterations {
                input.assign(&output);
                output.fill(0.0);
            }
        }

        output.to_pyarray(py)
    }

    Ok(())
}

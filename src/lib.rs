use num_traits::identities::{One, Zero};
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::cmp::{max, min, PartialOrd};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

enum UVMode {
    Velocity,
    Polarization,
}

struct ImageDimensions {
    nx: usize,
    ny: usize,
    width: i64,
    height: i64,
}

struct PixelCoordinates {
    x: i64,
    y: i64,
}
struct PixelFraction<T> {
    x: T,
    y: T,
}

fn as_array_index(x: i64, nx: usize) -> usize {
    if x >= 0 {
        x as usize
    } else {
        ((nx as i64) + x) as usize
    }
}
impl PixelCoordinates {
    fn x_idx(&self, d: &ImageDimensions) -> usize {
        as_array_index(self.x, d.nx)
    }
    fn y_idx(&self, d: &ImageDimensions) -> usize {
        as_array_index(self.y, d.ny)
    }
}

struct UVPoint<T> {
    u: T,
    v: T,
}

struct PixelSelector {}
impl PixelSelector {
    fn get<T: Copy>(
        &self,
        arr: &Array2<T>,
        coords: &PixelCoordinates,
        dims: &ImageDimensions,
    ) -> T {
        arr[[coords.y_idx(dims), coords.x_idx(dims)]]
    }
    fn get_v<T: Copy>(
        &self,
        arr: &ArrayView2<T>,
        coords: &PixelCoordinates,
        dims: &ImageDimensions,
    ) -> T {
        arr[[coords.y_idx(dims), coords.x_idx(dims)]]
    }
}

trait ParticularValues: From<f32> + Zero + One {
    fn ceil(&self) -> Self;
}
impl ParticularValues for f32 {
    fn ceil(&self) -> Self {
        1e10f32
    }
}
impl ParticularValues for f64 {
    fn ceil(&self) -> Self {
        1e100f64
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn time_to_next_pixel<
        T: PartialOrd + Neg<Output = T> + Div<Output = T> + Sub<Output = T> + ParticularValues,
    >(
        velocity: T,
        current_frac: T,
    ) -> T {
        if velocity > 0.0.into() {
            let one: T = 1.0.into();
            (one - current_frac) / velocity
        } else if velocity < 0.0.into() {
            -(current_frac / velocity)
        } else {
            velocity.ceil()
        }
    }

    fn update_state<
        T: Copy + PartialOrd + Mul<Output = T> + AddAssign<<T as Mul>::Output> + ParticularValues,
    >(
        velocity_parallel: &T,
        velocity_orhtogonal: &T,
        coord_parallel: &mut i64,
        frac_parallel: &mut T,
        frac_orthogonal: &mut T,
        time_parallel: &T,
    ) {
        if *velocity_parallel >= 0.0.into() {
            *coord_parallel += 1;
            *frac_parallel = 0.0.into();
        } else {
            *coord_parallel -= 1;
            *frac_parallel = 1.0.into();
        }
        *frac_orthogonal += *time_parallel * *velocity_orhtogonal;
    }

    fn advance<
        T: Copy
            + PartialOrd
            + Neg<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + AddAssign<<T as Mul>::Output>
            + ParticularValues,
    >(
        uv: &UVPoint<T>,
        coords: &mut PixelCoordinates,
        pix_frac: &mut PixelFraction<T>,
        dims: &ImageDimensions,
    ) {
        if uv.u == 0.0.into() && uv.v == 0.0.into() {
            return;
        }

        let tx = time_to_next_pixel(uv.u, pix_frac.x);
        let ty = time_to_next_pixel(uv.v, pix_frac.y);

        if tx < ty {
            // We reached the next pixel along x first.
            update_state(
                &uv.u,
                &uv.v,
                &mut coords.x,
                &mut pix_frac.x,
                &mut pix_frac.y,
                &tx,
            );
        } else {
            // We reached the next pixel along y first.
            update_state(
                &uv.v,
                &uv.u,
                &mut coords.y,
                &mut pix_frac.y,
                &mut pix_frac.x,
                &ty,
            );
        }
        coords.x = max(0, min(dims.width - 1, coords.x));
        coords.y = max(0, min(dims.height - 1, coords.y));
    }

    fn convolve<
        'py,
        T: Copy
            + PartialOrd
            + Add<Output = T>
            + Sub<Output = T>
            + Neg<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + AddAssign<<T as Mul>::Output>
            + ParticularValues,
    >(
        u: ArrayView2<'py, T>,
        v: ArrayView2<'py, T>,
        kernel: ArrayView1<'py, T>,
        input: &Array2<T>,
        output: &mut Array2<T>,
        uv_mode: &UVMode,
    ) {
        let dims = ImageDimensions {
            nx: u.shape()[1],
            ny: u.shape()[0],
            width: u.shape()[1] as i64,
            height: u.shape()[0] as i64,
        };
        let kernellen = kernel.len();
        let ps = PixelSelector {};

        for i in 0..dims.ny {
            for j in 0..dims.nx {
                let mut coords = PixelCoordinates {
                    x: j.try_into().unwrap(),
                    y: i.try_into().unwrap(),
                };
                let mut pix_frac = PixelFraction {
                    x: 0.5.into(),
                    y: 0.5.into(),
                };
                let mut k = kernellen / 2;
                let mut last_p: UVPoint<T> = UVPoint {
                    u: 0.0.into(),
                    v: 0.0.into(),
                };

                output[[i, j]] += kernel[[k]] * ps.get(input, &coords, &dims);

                while k < kernellen - 1 {
                    let mut p = UVPoint {
                        u: ps.get_v(&u, &coords, &dims),
                        v: ps.get_v(&v, &coords, &dims),
                    };
                    match uv_mode {
                        UVMode::Polarization => {
                            if (p.u * last_p.u + p.v * last_p.v) < 0.0.into() {
                                p.u = -p.u;
                                p.v = -p.v;
                            }
                            last_p.u = p.u;
                            last_p.v = p.u;
                        }
                        UVMode::Velocity => {}
                    };
                    advance(&p, &mut coords, &mut pix_frac, &dims);
                    k += 1;
                    output[[i, j]] += kernel[[k]] * ps.get(input, &coords, &dims);
                }

                coords.x = j.try_into().unwrap();
                coords.y = i.try_into().unwrap();
                pix_frac.x = 0.5.into();
                pix_frac.y = 0.5.into();
                k = kernellen / 2;
                last_p.u = 0.0.into();
                last_p.v = 0.0.into();

                while k > 0 {
                    let mut p = UVPoint {
                        u: ps.get_v(&u, &coords, &dims),
                        v: ps.get_v(&v, &coords, &dims),
                    };
                    match uv_mode {
                        UVMode::Polarization => {
                            if (p.u * last_p.u + p.v * last_p.v) < 0.0.into() {
                                p.u = -p.u;
                                p.v = -p.v;
                            }
                            last_p.u = p.u;
                            last_p.v = p.u;
                        }
                        UVMode::Velocity => {}
                    };
                    let mp = UVPoint { u: -p.u, v: -p.v };

                    advance(&mp, &mut coords, &mut pix_frac, &dims);
                    k -= 1;
                    output[[i, j]] += kernel[[k]] * ps.get(input, &coords, &dims);
                }
            }
        }
    }
    fn convolve_interatively_impl<
        'py,
        T: numpy::Element
            + Clone
            + Copy
            + PartialOrd
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + AddAssign<<T as Mul>::Output>
            + ParticularValues,
    >(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, T>,
        u: PyReadonlyArray2<'py, T>,
        v: PyReadonlyArray2<'py, T>,
        kernel: PyReadonlyArray1<'py, T>,
        iterations: i64,
        uv_mode: String,
    ) -> Bound<'py, PyArray2<T>> {
        let u = u.as_array();
        let v = v.as_array();
        let kernel = kernel.as_array();
        let image = image.as_array();
        let mut input =
            Array2::from_shape_vec(image.raw_dim(), image.iter().cloned().collect()).unwrap();
        let mut output = Array2::<T>::zeros(image.raw_dim());

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
                output.fill(0.0.into());
            }
        }

        output.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_f32")]
    fn convolve_interatively_f32_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        u: PyReadonlyArray2<'py, f32>,
        v: PyReadonlyArray2<'py, f32>,
        kernel: PyReadonlyArray1<'py, f32>,
        iterations: i64,
        uv_mode: String,
    ) -> Bound<'py, PyArray2<f32>> {
        convolve_interatively_impl(py, image, u, v, kernel, iterations, uv_mode)
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_f64")]
    fn convolve_interatively_f64_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        u: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        kernel: PyReadonlyArray1<'py, f64>,
        iterations: i64,
        uv_mode: String,
    ) -> Bound<'py, PyArray2<f64>> {
        convolve_interatively_impl(py, image, u, v, kernel, iterations, uv_mode)
    }
    Ok(())
}

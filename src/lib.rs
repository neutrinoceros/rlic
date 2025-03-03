use either::Either;
use num_traits::{abs, signum, Float, Signed};
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::cmp::{max, min};
use std::ops::{AddAssign, Mul, Neg};

#[derive(Clone)]
enum UVMode {
    Velocity,
    Polarization,
}

struct UVField<'a, T> {
    u: ArrayView2<'a, T>,
    v: ArrayView2<'a, T>,
    mode: UVMode,
}

struct ImageDimensions {
    nx: usize,
    ny: usize,
    width: i64,
    height: i64,
}

struct PixelFraction<T> {
    x: T,
    y: T,
}

fn wrap_array_index(x: i64, nx: usize) -> usize {
    if x >= 0 {
        x as usize
    } else {
        ((nx as i64) + x) as usize
    }
}

#[derive(Clone)]
struct PixelCoordinates {
    x: i64,
    y: i64,
}
impl PixelCoordinates {
    fn x_idx(&self, d: &ImageDimensions) -> usize {
        wrap_array_index(self.x, d.nx)
    }
    fn y_idx(&self, d: &ImageDimensions) -> usize {
        wrap_array_index(self.y, d.ny)
    }
}

#[cfg(test)]
mod test_pixel_coordinates {
    use crate::{ImageDimensions, PixelCoordinates};

    #[test]
    fn coords_as_indices() {
        let dims = ImageDimensions {
            nx: 128,
            ny: 128,
            width: 128,
            height: 128,
        };
        let pc = PixelCoordinates { x: 5, y: -10 };
        assert_eq!(pc.x_idx(&dims), 5);
        assert_eq!(pc.y_idx(&dims), 128 - 10);
    }
}

struct UVPoint<T: Copy> {
    u: T,
    v: T,
}
impl<T: Copy> Clone for UVPoint<T> {
    fn clone(&self) -> Self {
        UVPoint {
            u: self.u,
            v: self.v,
        }
    }
}
impl<T: Neg<Output = T> + Copy> Neg for UVPoint<T> {
    type Output = UVPoint<T>;

    fn neg(self) -> Self::Output {
        UVPoint {
            u: -self.u,
            v: -self.v,
        }
    }
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

#[cfg(test)]
mod test_pixel_selector {
    use numpy::ndarray::array;

    use crate::{ImageDimensions, PixelCoordinates, PixelSelector};
    #[test]
    fn from_array() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let dims = ImageDimensions {
            nx: 4,
            ny: 4,
            width: 4,
            height: 4,
        };
        let coords = PixelCoordinates { x: 1, y: 1 };
        let ps = PixelSelector {};
        let res = ps.get(&arr, &coords, &dims);
        assert_eq!(res, 4.0);
    }
    #[test]
    fn from_view() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let view = arr.view();
        let dims = ImageDimensions {
            nx: 4,
            ny: 4,
            width: 4,
            height: 4,
        };
        let coords = PixelCoordinates { x: 1, y: 1 };
        let ps = PixelSelector {};
        let res = ps.get_v(&view, &coords, &dims);
        assert_eq!(res, 4.0);
    }
}

trait AtLeastF32: Float + From<f32> + Signed + AddAssign<<Self as Mul>::Output> {}
impl AtLeastF32 for f32 {}
impl AtLeastF32 for f64 {}

fn time_to_next_pixel<T: AtLeastF32>(velocity: T, current_frac: T) -> T {
    // this is the branchless version of
    // if velocity > 0.0 {
    //     (1.0 - current_frac) / velocity
    // } else {
    //     -(current_frac / velocity)
    // }
    let one: T = 1.0.into();
    let two: T = 2.0.into();
    let d1 = current_frac;
    let remaining_frac = d1 + (one + signum(velocity)) * (one - two * d1) / two;
    abs(remaining_frac / velocity)
}

#[cfg(test)]
mod test_time_to_next_pixel {
    use super::time_to_next_pixel;
    use std::assert_eq;
    #[test]
    fn positive_vel() {
        let res = time_to_next_pixel(1.0, 0.0);
        assert_eq!(res, 1.0);
    }
    #[test]
    fn negative_vel() {
        let res = time_to_next_pixel(-1.0, 1.0);
        assert_eq!(res, 1.0);
    }
    #[test]
    fn infinite_time_f32() {
        let res = time_to_next_pixel(0.0f32, 0.5f32);
        assert_eq!(res, f32::INFINITY);
    }
    #[test]
    fn infinite_time_f64() {
        let res = time_to_next_pixel(0.0, 0.5);
        assert_eq!(res, f64::INFINITY);
    }
}

#[inline(always)]
fn update_state<T: AtLeastF32>(
    velocity_parallel: &T,
    velocity_orthogonal: &T,
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
    *frac_orthogonal += *time_parallel * *velocity_orthogonal;
}

#[inline(always)]
fn advance<T: AtLeastF32>(
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

enum Direction {
    Forward,
    Backward,
}

#[inline(always)]
fn convole_single_pixel<T: AtLeastF32>(
    pixel_value: &mut T,
    starting_point: &PixelCoordinates,
    uvfield: &UVField<T>,
    kernel: &ArrayView1<T>,
    input: &Array2<T>,
    dims: &ImageDimensions,
    direction: &Direction,
) {
    let mut coords: PixelCoordinates = starting_point.clone();
    let mut pix_frac = PixelFraction {
        x: 0.5.into(),
        y: 0.5.into(),
    };

    let mut last_p: UVPoint<T> = UVPoint {
        u: 0.0.into(),
        v: 0.0.into(),
    };
    let ps = PixelSelector {};

    let kmid = kernel.len() / 2;
    let range = match direction {
        Direction::Forward => Either::Right((kmid + 1)..kernel.len()),
        Direction::Backward => Either::Left((0..kmid).rev()),
    };

    for k in range {
        let mut p = UVPoint {
            u: ps.get_v(&uvfield.u, &coords, dims),
            v: ps.get_v(&uvfield.v, &coords, dims),
        };
        match uvfield.mode {
            UVMode::Polarization => {
                if (p.u * last_p.u + p.v * last_p.v) < 0.0.into() {
                    p = -p;
                }
                last_p = p.clone();
            }
            UVMode::Velocity => {}
        };
        let mp = match direction {
            Direction::Forward => p.clone(),
            Direction::Backward => -p,
        };
        advance(&mp, &mut coords, &mut pix_frac, dims);
        *pixel_value += kernel[[k]] * ps.get(input, &coords, dims);
    }
}

fn convolve<'py, T: AtLeastF32>(
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
    let uvfield = UVField {
        u,
        v,
        mode: uv_mode.clone(),
    };
    let kmid = kernel.len() / 2;

    for i in 0..dims.ny {
        for j in 0..dims.nx {
            let pixel_value = &mut output[[i, j]];
            *pixel_value += kernel[[kmid]] * input[[i, j]];
            let starting_point = PixelCoordinates {
                x: j.try_into().unwrap(),
                y: i.try_into().unwrap(),
            };
            convole_single_pixel(
                pixel_value,
                &starting_point,
                &uvfield,
                &kernel,
                input,
                &dims,
                &Direction::Forward,
            );

            convole_single_pixel(
                pixel_value,
                &starting_point,
                &uvfield,
                &kernel,
                input,
                &dims,
                &Direction::Backward,
            );
        }
    }
}

fn convolve_iteratively_impl<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    texture: PyReadonlyArray2<'py, T>,
    u: PyReadonlyArray2<'py, T>,
    v: PyReadonlyArray2<'py, T>,
    kernel: PyReadonlyArray1<'py, T>,
    iterations: i64,
    uv_mode: String,
) -> Bound<'py, PyArray2<T>> {
    let u = u.as_array();
    let v = v.as_array();
    let kernel = kernel.as_array();
    let texture = texture.as_array();
    let mut input =
        Array2::from_shape_vec(texture.raw_dim(), texture.iter().cloned().collect()).unwrap();
    let mut output = Array2::<T>::zeros(texture.raw_dim());

    let uv_mode = match uv_mode.as_str() {
        "polarization" => UVMode::Polarization,
        "velocity" => UVMode::Velocity,
        _ => panic!("unknown uv_mode"),
    };

    let mut it_count = 0;
    while it_count < iterations {
        convolve(u, v, kernel, &input, &mut output, &uv_mode);
        it_count += 1;
        if it_count < iterations {
            input.assign(&output);
            output.fill(0.0.into());
        }
    }

    output.to_pyarray(py)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "convolve_f32")]
    fn convolve_f32_py<'py>(
        py: Python<'py>,
        texture: PyReadonlyArray2<'py, f32>,
        u: PyReadonlyArray2<'py, f32>,
        v: PyReadonlyArray2<'py, f32>,
        kernel: PyReadonlyArray1<'py, f32>,
        iterations: i64,
        uv_mode: String,
    ) -> Bound<'py, PyArray2<f32>> {
        convolve_iteratively_impl(py, texture, u, v, kernel, iterations, uv_mode)
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_f64")]
    fn convolve_f64_py<'py>(
        py: Python<'py>,
        texture: PyReadonlyArray2<'py, f64>,
        u: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        kernel: PyReadonlyArray1<'py, f64>,
        iterations: i64,
        uv_mode: String,
    ) -> Bound<'py, PyArray2<f64>> {
        convolve_iteratively_impl(py, texture, u, v, kernel, iterations, uv_mode)
    }
    Ok(())
}

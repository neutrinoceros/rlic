use crate::boundaries::{Boundary, BoundaryPair, BoundarySet};
use either::Either;
use num_traits::{abs, signum, Float, Signed};
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::ops::{AddAssign, Mul, Neg};

#[derive(Clone)]
enum UVMode {
    Velocity,
    Polarization,
}
impl UVMode {
    fn new(uv_mode: String) -> UVMode {
        match uv_mode.as_str() {
            "polarization" => UVMode::Polarization,
            "velocity" => UVMode::Velocity,
            _ => panic!("unknown uv_mode"),
        }
    }
}

struct UVField<'a, T> {
    u: ArrayView2<'a, T>,
    v: ArrayView2<'a, T>,
    mode: UVMode,
}

struct PixelFraction<T> {
    x: T,
    y: T,
}

#[derive(Clone)]
struct ImageDimensions {
    x: usize,
    y: usize,
}

#[derive(Clone)]
struct PixelCoordinates {
    x: usize,
    y: usize,
    dimensions: ImageDimensions,
}
impl PixelCoordinates {
    fn apply_one_dir(c: &mut usize, image_size: usize, boundaries: &BoundaryPair) {
        if *c == usize::MAX {
            *c = match boundaries.left {
                Boundary::Closed => 0,
                Boundary::Periodic => image_size - 1,
            };
        } else if *c == image_size {
            *c = match boundaries.right {
                Boundary::Closed => image_size - 1,
                Boundary::Periodic => 0,
            };
        }
    }
    fn apply(&mut self, boundaries: &BoundarySet) {
        PixelCoordinates::apply_one_dir(&mut self.x, self.dimensions.x, &boundaries.x);
        PixelCoordinates::apply_one_dir(&mut self.y, self.dimensions.y, &boundaries.y);
    }
}
mod boundaries {
    pub enum Boundary {
        Closed,
        Periodic,
    }
    impl Boundary {
        fn new(boundary: String) -> Boundary {
            match boundary.as_str() {
                "closed" => Boundary::Closed,
                "periodic" => Boundary::Periodic,
                _ => panic!("unknown boundary"),
            }
        }
    }

    pub struct BoundaryPair {
        pub left: Boundary,
        pub right: Boundary,
    }
    impl BoundaryPair {
        fn new(pair: (String, String)) -> BoundaryPair {
            BoundaryPair {
                left: Boundary::new(pair.0),
                right: Boundary::new(pair.1),
            }
        }
    }

    pub struct BoundarySet {
        pub x: BoundaryPair,
        pub y: BoundaryPair,
    }
    impl BoundarySet {
        pub fn new(set: ((String, String), (String, String))) -> BoundarySet {
            BoundarySet {
                x: BoundaryPair::new(set.0),
                y: BoundaryPair::new(set.1),
            }
        }
    }
}

#[derive(Clone)]
struct UVPoint<T: Copy> {
    u: T,
    v: T,
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

fn select_pixel<T: Copy>(arr: &ArrayView2<T>, coords: &PixelCoordinates) -> T {
    arr[[coords.y, coords.x]]
}

#[cfg(test)]
mod test_pixel_select {
    use numpy::ndarray::array;

    use crate::{select_pixel, ImageDimensions, PixelCoordinates};
    #[test]
    fn selection() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let coords = PixelCoordinates {
            x: 1,
            y: 1,
            dimensions: ImageDimensions { x: 4, y: 4 },
        };
        let res = select_pixel(&arr.view(), &coords);
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
    let mtwo = -two;
    let d1 = current_frac;
    let remaining_frac = (one + signum(velocity)).mul_add((mtwo.mul_add(d1, one)) / two, d1);
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
    coord_parallel: &mut usize,
    frac_parallel: &mut T,
    frac_orthogonal: &mut T,
    time_parallel: &T,
) {
    if *velocity_parallel >= 0.0.into() {
        *coord_parallel += 1;
        *frac_parallel = 0.0.into();
    } else {
        *coord_parallel = coord_parallel.wrapping_sub(1);
        *frac_parallel = 1.0.into();
    }
    *frac_orthogonal = (*time_parallel).mul_add(*velocity_orthogonal, *frac_orthogonal);
}

#[inline(always)]
fn advance<T: AtLeastF32>(
    uv: &UVPoint<T>,
    coords: &mut PixelCoordinates,
    pix_frac: &mut PixelFraction<T>,
    boundaries: &BoundarySet,
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
    // All boundary conditions must be applicable on each step.
    // This is done to allow for complex cases like shearing boxes.
    coords.apply(boundaries);
}

#[cfg(test)]
mod test_advance {
    use crate::{
        advance, Boundary, BoundaryPair, BoundarySet, ImageDimensions, PixelCoordinates,
        PixelFraction, UVPoint,
    };

    #[test]
    fn zero_vel() {
        let uv = UVPoint { u: 0.0, v: 0.0 };
        let mut coords = PixelCoordinates {
            x: 5,
            y: 5,
            dimensions: ImageDimensions { x: 10, y: 10 },
        };
        let mut pix_frac = PixelFraction { x: 0.5, y: 0.5 };
        let boundaries = BoundarySet {
            x: BoundaryPair {
                left: Boundary::Closed,
                right: Boundary::Closed,
            },
            y: BoundaryPair {
                left: Boundary::Closed,
                right: Boundary::Closed,
            },
        };
        advance(&uv, &mut coords, &mut pix_frac, &boundaries);
        assert_eq!(coords.x, 5);
        assert_eq!(coords.y, 5);
        assert_eq!(pix_frac.x, 0.5);
        assert_eq!(pix_frac.y, 0.5);
    }
}

enum Direction {
    Forward,
    Backward,
}

#[inline(always)]
fn convole_single_pixel<T: AtLeastF32>(
    pixel_value: &mut T,
    starting_point: &PixelCoordinates,
    uv: &UVField<T>,
    kernel: &ArrayView1<T>,
    input: &ArrayView2<T>,
    boundaries: &BoundarySet,
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

    let kmid = kernel.len() / 2;
    let range = match direction {
        Direction::Forward => Either::Right((kmid + 1)..kernel.len()),
        Direction::Backward => Either::Left((0..kmid).rev()),
    };

    for k in range {
        let mut p = UVPoint {
            u: select_pixel(&uv.u, &coords),
            v: select_pixel(&uv.v, &coords),
        };
        if p.u.is_nan() || p.v.is_nan() {
            break;
        }
        match uv.mode {
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
        advance(&mp, &mut coords, &mut pix_frac, boundaries);
        *pixel_value = kernel[[k]].mul_add(select_pixel(input, &coords), *pixel_value);
    }
}

fn convolve<'py, T: AtLeastF32>(
    uv: &UVField<'py, T>,
    kernel: ArrayView1<'py, T>,
    boundaries: &BoundarySet,
    input: ArrayView2<T>,
    output: &mut Array2<T>,
) {
    let dims = ImageDimensions {
        x: uv.u.shape()[1],
        y: uv.u.shape()[0],
    };
    let kmid = kernel.len() / 2;

    for i in 0..dims.y {
        for j in 0..dims.x {
            let pixel_value = &mut output[[i, j]];
            *pixel_value = kernel[[kmid]].mul_add(input[[i, j]], *pixel_value);
            let starting_point = PixelCoordinates {
                x: j,
                y: i,
                dimensions: dims.clone(),
            };
            convole_single_pixel(
                pixel_value,
                &starting_point,
                uv,
                &kernel,
                &input,
                boundaries,
                &Direction::Forward,
            );

            convole_single_pixel(
                pixel_value,
                &starting_point,
                uv,
                &kernel,
                &input,
                boundaries,
                &Direction::Backward,
            );
        }
    }
}

fn convolve_once<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    texture: PyReadonlyArray2<'py, T>,
    uv: (PyReadonlyArray2<'py, T>, PyReadonlyArray2<'py, T>, String),
    kernel: PyReadonlyArray1<'py, T>,
    boundaries: BoundarySet,
) -> Bound<'py, PyArray2<T>> {
    let uv = UVField {
        u: uv.0.as_array(),
        v: uv.1.as_array(),
        mode: UVMode::new(uv.2),
    };
    let kernel = kernel.as_array();
    let texture = texture.as_array();
    let mut output = Array2::<T>::zeros(texture.raw_dim());

    convolve(&uv, kernel, &boundaries, texture, &mut output);

    output.to_pyarray(py)
}

fn convolve_iteratively<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    texture: PyReadonlyArray2<'py, T>,
    uv: (PyReadonlyArray2<'py, T>, PyReadonlyArray2<'py, T>, String),
    kernel: PyReadonlyArray1<'py, T>,
    boundaries: BoundarySet,
    iterations: i64,
) -> Bound<'py, PyArray2<T>> {
    let uv = UVField {
        u: uv.0.as_array(),
        v: uv.1.as_array(),
        mode: UVMode::new(uv.2),
    };
    let kernel = kernel.as_array();
    let texture = texture.as_array();
    let mut input =
        Array2::from_shape_vec(texture.raw_dim(), texture.iter().cloned().collect()).unwrap();
    let mut output = Array2::<T>::zeros(texture.raw_dim());

    let mut it_count = 0;
    while it_count < iterations {
        convolve(&uv, kernel, &boundaries, input.view(), &mut output);
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
#[pymodule(gil_used = false)]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "convolve_f32")]
    fn convolve_f32_py<'py>(
        py: Python<'py>,
        texture: PyReadonlyArray2<'py, f32>,
        uv: (
            PyReadonlyArray2<'py, f32>,
            PyReadonlyArray2<'py, f32>,
            String,
        ),
        kernel: PyReadonlyArray1<'py, f32>,
        boundaries: ((String, String), (String, String)),
        iterations: i64,
    ) -> Bound<'py, PyArray2<f32>> {
        let boundaries = BoundarySet::new(boundaries);
        match iterations {
            1 => convolve_once(py, texture, uv, kernel, boundaries),
            _ => convolve_iteratively(py, texture, uv, kernel, boundaries, iterations),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_f64")]
    fn convolve_f64_py<'py>(
        py: Python<'py>,
        texture: PyReadonlyArray2<'py, f64>,
        uv: (
            PyReadonlyArray2<'py, f64>,
            PyReadonlyArray2<'py, f64>,
            String,
        ),
        kernel: PyReadonlyArray1<'py, f64>,
        boundaries: ((String, String), (String, String)),
        iterations: i64,
    ) -> Bound<'py, PyArray2<f64>> {
        let boundaries = BoundarySet::new(boundaries);
        match iterations {
            1 => convolve_once(py, texture, uv, kernel, boundaries),
            _ => convolve_iteratively(py, texture, uv, kernel, boundaries, iterations),
        }
    }
    Ok(())
}

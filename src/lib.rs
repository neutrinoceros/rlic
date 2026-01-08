use crate::boundaries::BoundarySet;
use either::Either;
use interpn::one_dim::linear::LinearHoldLast1D;
use interpn::one_dim::Interp1D;
use interpn::RectilinearGrid1D;
use num_traits::{Float, NumCast, Signed};
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray2, ToPyArray};
use pyo3::types::PyModuleMethods;
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, PyResult, Python};
use std::collections::VecDeque;
use std::ops::{AddAssign, Mul, Neg};

#[cfg(feature = "branchless")]
use num_traits::{abs, signum};

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

#[derive(Clone, Copy, PartialEq)]
struct ArrayDimensions {
    // can be used to represent the size of an image or view
    x: usize,
    y: usize,
}

impl ArrayDimensions {
    fn last_pixel_index(&self) -> PixelIndex {
        PixelIndex {
            i: (self.y - 1),
            j: (self.x - 1),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct PixelIndex {
    i: usize, // y
    j: usize, // x
}

mod boundaries {
    use crate::{ArrayDimensions, PixelIndex};

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
        pub image_size: usize,
    }
    impl BoundaryPair {
        fn new(pair: (String, String), image_size: usize) -> BoundaryPair {
            BoundaryPair {
                left: Boundary::new(pair.0),
                right: Boundary::new(pair.1),
                image_size,
            }
        }
        fn apply_one_dir(&self, c: &mut usize) {
            if *c == usize::MAX {
                *c = match self.left {
                    Boundary::Closed => 0,
                    Boundary::Periodic => self.image_size - 1,
                };
            } else if *c == self.image_size {
                *c = match self.right {
                    Boundary::Closed => self.image_size - 1,
                    Boundary::Periodic => 0,
                };
            }
        }
    }

    pub struct BoundarySet {
        pub x: BoundaryPair,
        pub y: BoundaryPair,
    }
    impl BoundarySet {
        pub fn new(
            set: ((String, String), (String, String)),
            dims: ArrayDimensions,
        ) -> BoundarySet {
            BoundarySet {
                x: BoundaryPair::new(set.0, dims.x),
                y: BoundaryPair::new(set.1, dims.y),
            }
        }
        pub fn apply(&self, pc: &mut PixelIndex) {
            self.x.apply_one_dir(&mut pc.j);
            self.y.apply_one_dir(&mut pc.i);
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

fn select_pixel<T: Copy>(arr: &ArrayView2<T>, idx: PixelIndex) -> T {
    arr[[idx.i, idx.j]]
}

#[cfg(test)]
mod test_pixel_select {
    use numpy::ndarray::array;

    use crate::{select_pixel, PixelIndex};
    #[test]
    fn selection() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let idx = PixelIndex { i: 1, j: 1 };
        let res = select_pixel(&arr.view(), idx);
        assert_eq!(res, 4.0);
    }
}

trait AtLeastF32: Float + From<f32> + Signed + AddAssign<<Self as Mul>::Output> {}
impl AtLeastF32 for f32 {}
impl AtLeastF32 for f64 {}

fn time_to_next_pixel<T: AtLeastF32>(velocity: T, current_frac: T) -> T {
    #[cfg(not(feature = "branchless"))]
    if velocity > 0.0.into() {
        let one: T = 1.0.into();
        (one - current_frac) / velocity
    } else if velocity < 0.0.into() {
        -(current_frac / velocity)
    } else {
        f32::INFINITY.into()
    }

    #[cfg(feature = "branchless")]
    {
        let one: T = 1.0.into();
        let half: T = 0.5.into();
        let d1 = current_frac;

        #[cfg(not(feature = "fma"))]
        let remaining_frac = (one + signum(velocity)) * (half - d1) + d1;
        #[cfg(feature = "fma")]
        let remaining_frac = (one + signum(velocity)).mul_add(half - d1, d1);
        abs(remaining_frac / velocity)
    }
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

    #[cfg(not(feature = "fma"))]
    {
        *frac_orthogonal += *time_parallel * *velocity_orthogonal;
    }
    #[cfg(feature = "fma")]
    {
        *frac_orthogonal = (*time_parallel).mul_add(*velocity_orthogonal, *frac_orthogonal);
    }
}

#[inline(always)]
fn advance<T: AtLeastF32>(
    uv: &UVPoint<T>,
    idx: &mut PixelIndex,
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
            &mut idx.j,
            &mut pix_frac.x,
            &mut pix_frac.y,
            &tx,
        );
    } else {
        // We reached the next pixel along y first.
        update_state(
            &uv.v,
            &uv.u,
            &mut idx.i,
            &mut pix_frac.y,
            &mut pix_frac.x,
            &ty,
        );
    }
    // All boundary conditions must be applicable on each step.
    // This is done to allow for complex cases like shearing boxes.
    boundaries.apply(idx);
}

#[cfg(test)]
mod test_advance {
    use crate::{advance, ArrayDimensions, BoundarySet, PixelFraction, PixelIndex, UVPoint};

    #[test]
    fn zero_vel() {
        let uv = UVPoint { u: 0.0, v: 0.0 };
        let mut idx = PixelIndex { i: 5, j: 5 };
        let mut pix_frac = PixelFraction { x: 0.5, y: 0.5 };
        let boundaries = BoundarySet::new(
            (
                (String::from("closed"), String::from("closed")),
                (String::from("closed"), String::from("closed")),
            ),
            ArrayDimensions { x: 10, y: 10 },
        );
        advance(&uv, &mut idx, &mut pix_frac, &boundaries);
        assert_eq!(idx.j, 5);
        assert_eq!(idx.i, 5);
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
    starting_point: PixelIndex,
    uv: &UVField<T>,
    kernel: &ArrayView1<T>,
    input: &ArrayView2<T>,
    boundaries: &BoundarySet,
    direction: &Direction,
) {
    let mut idx: PixelIndex = starting_point;
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
            u: select_pixel(&uv.u, idx),
            v: select_pixel(&uv.v, idx),
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
        advance(&mp, &mut idx, &mut pix_frac, boundaries);
        #[cfg(not(feature = "fma"))]
        {
            *pixel_value += kernel[[k]] * select_pixel(input, idx);
        }
        #[cfg(feature = "fma")]
        {
            *pixel_value = kernel[[k]].mul_add(select_pixel(input, idx), *pixel_value);
        }
    }
}

fn convolve<'py, T: AtLeastF32>(
    uv: &UVField<'py, T>,
    kernel: ArrayView1<'py, T>,
    boundaries: &BoundarySet,
    input: ArrayView2<T>,
    output: &mut Array2<T>,
) {
    let kmid = kernel.len() / 2;

    for i in 0..boundaries.y.image_size {
        for j in 0..boundaries.x.image_size {
            let pixel_value = &mut output[[i, j]];
            #[cfg(not(feature = "fma"))]
            {
                *pixel_value += kernel[[kmid]] * input[[i, j]];
            }
            #[cfg(feature = "fma")]
            {
                *pixel_value = kernel[[kmid]].mul_add(input[[i, j]], *pixel_value);
            }
            let starting_point = PixelIndex { i, j };
            convole_single_pixel(
                pixel_value,
                starting_point,
                uv,
                &kernel,
                &input,
                boundaries,
                &Direction::Forward,
            );

            convole_single_pixel(
                pixel_value,
                starting_point,
                uv,
                &kernel,
                &input,
                boundaries,
                &Direction::Backward,
            );
        }
    }
}

fn convolve_iteratively<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    texture: PyReadonlyArray2<'py, T>,
    uv: (PyReadonlyArray2<'py, T>, PyReadonlyArray2<'py, T>, String),
    kernel: PyReadonlyArray1<'py, T>,
    boundaries: ((String, String), (String, String)),
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

    let dims = ArrayDimensions {
        x: uv.u.shape()[1],
        y: uv.u.shape()[0],
    };
    let boundaries = BoundarySet::new(boundaries, dims);
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

struct Histogram<T> {
    bins: Array1<usize>,
    vmin: T,
    vmax: T,
}

impl<T: AtLeastF32 + NumCast> Histogram<T> {
    fn bin_width(&self) -> T {
        (self.vmax - self.vmin) / <T as NumCast>::from(self.bins.len()).unwrap()
    }

    fn edges(&self) -> Array1<T> {
        Array1::<T>::linspace(self.vmin, self.vmax, self.bins.len() + 1)
    }

    fn centers(&self) -> Array1<T> {
        let nbins = self.bins.len();
        let mut centers = Array1::<T>::zeros(nbins);
        let half: T = 0.5.into();
        let offset = half * self.bin_width();
        let edges = self.edges();
        for i in 0..nbins {
            centers[i] = edges[i] + offset;
        }
        centers
    }

    fn cdf(&self) -> Array1<usize> {
        // convert histogram to normalized cumulative distribution function
        let mut cdf = self.bins.clone();
        for i in 1..cdf.len() {
            cdf[i] += cdf[i - 1];
        }
        cdf
    }

    fn cdf_as_normalized(&self) -> Array1<T> {
        let cdf_us = self.cdf();
        let nbins = self.bins.len();
        let mut cdf: Array1<T> = cdf_us.mapv(|elem| <T as NumCast>::from(elem).unwrap());
        let tot = cdf[nbins - 1];
        for i in 0..nbins {
            cdf[i] = cdf[i] / tot;
        }
        cdf
    }
}

fn compute_subhistogram<T: AtLeastF32>(
    arr: ArrayView1<T>,
    vmin: T,
    vmax: T,
    nbins: usize,
) -> Histogram<T> {
    let vspan = vmax - vmin;
    let bin_width = vspan / <T as NumCast>::from(nbins).unwrap();

    // padding one extra bin on the right allows for a branchless optimization:
    // pixels that contain exactly vmax are counted in the extra bin
    let mut hvals = Array1::<usize>::zeros(nbins + 1);
    for i in 0..arr.len() {
        let v = arr[i];
        let f = ((v - vmin) / bin_width).floor();
        let idx = <usize as NumCast>::from(f).unwrap();
        hvals[idx] += 1;
    }

    // move data from the last bin to the previous one
    hvals[nbins - 1] += hvals[nbins];
    let hvals = hvals.slice(s![..-1]).to_owned();
    assert_eq!(hvals.len(), nbins);

    Histogram {
        bins: hvals,
        vmin,
        vmax,
    }
}

fn reduce_histogram<T: Copy>(subhists: &VecDeque<Histogram<T>>) -> Histogram<T> {
    // it is assumed that all input histograms have the exact same range and nbins
    let h0 = subhists.front().unwrap();
    let nbins = h0.bins.len();
    let mut hvals = Array1::<usize>::zeros(nbins);
    for h in subhists {
        for i in 0..nbins {
            hvals[i] += h.bins[i];
        }
    }
    Histogram {
        bins: hvals,
        vmin: h0.vmin,
        vmax: h0.vmax,
    }
}

fn compute_histogram<T: AtLeastF32 + numpy::Element + NumCast>(
    image: ArrayView2<T>,
    nbins: usize,
) -> Histogram<T> {
    let dims = ArrayDimensions {
        x: image.shape()[1],
        y: image.shape()[0],
    };
    let mut vmin: T = T::infinity();
    let mut vmax: T = -T::infinity();

    for i in 0..dims.y {
        for j in 0..dims.x {
            let v = image[[i, j]];
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }

    let mut subhistograms = VecDeque::with_capacity(dims.y + 1);
    for i in 0..dims.y {
        let arr = image.index_axis(Axis(0), i);
        subhistograms.push_back(compute_subhistogram(arr, vmin, vmax, nbins));
    }
    reduce_histogram(&subhistograms)
}

#[cfg(test)]
mod test_histogram {
    use crate::Histogram;
    use numpy::ndarray::Array1;

    #[test]
    fn test_ones() {
        let nbins = 8usize;
        let vmin = 0.0;
        let vmax = 8.0;
        let hist = Histogram {
            bins: Array1::<usize>::ones(nbins),
            vmin,
            vmax,
        };
        assert_eq!(hist.bin_width(), 1.0);

        let edges = hist.edges();
        let edges = edges.as_slice().unwrap();
        assert_eq!(edges.len(), nbins + 1);
        assert_eq!(*edges.first().unwrap(), vmin);
        assert_eq!(*edges.last().unwrap(), vmax);
        assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let centers = hist.centers();
        let centers = centers.as_slice().unwrap();
        assert_eq!(centers.len(), nbins);
        assert_eq!(centers, vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);

        let cdf = hist.cdf();
        let cdf = cdf.as_slice().unwrap();
        assert_eq!(cdf, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let cdf = hist.cdf_as_normalized();
        let cdf = cdf.as_slice().unwrap();
        assert_eq!(cdf, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]);
    }
}

fn adjust_intensity<T: AtLeastF32 + numpy::Element + NumCast>(
    image: ArrayView2<T>,
    hist: Histogram<T>,
    out: &mut Array2<T>,
) {
    let bin_centers = hist.centers();
    let cdf = hist.cdf_as_normalized();
    let grid =
        RectilinearGrid1D::new(bin_centers.as_slice().unwrap(), cdf.as_slice().unwrap()).unwrap();
    let interpolator = LinearHoldLast1D::new(grid);

    let locs = image.as_slice().unwrap();
    match interpolator.eval(locs, out.as_slice_mut().unwrap()) {
        Ok(_) => (),
        Err(_) => panic!("interpolation failed"),
    }
}
fn adjust_pixel_intensity<T: AtLeastF32 + numpy::Element + NumCast>(
    pixel: T,
    hist: &Histogram<T>,
    out: &mut T,
) {
    let bin_centers = hist.centers();
    let cdf = hist.cdf_as_normalized();
    let grid =
        RectilinearGrid1D::new(bin_centers.as_slice().unwrap(), cdf.as_slice().unwrap()).unwrap();
    let interpolator = LinearHoldLast1D::new(grid);

    *out = match interpolator.eval_one(pixel) {
        Ok(res) => res,
        Err(_) => panic!("interpolation failed"),
    };
}

fn equalize_histogram<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, T>,
    nbins: usize,
) -> Bound<'py, PyArray2<T>> {
    let image = image.as_array();
    let hist = compute_histogram(image, nbins);
    let mut out = Array2::<T>::zeros(image.raw_dim());
    adjust_intensity(image, hist, &mut out);
    out.to_pyarray(py)
}

fn get_tile_view<'a, T: numpy::Element>(
    image: &'a ArrayView2<'a, T>,
    center_pixel: PixelIndex,
    tile_shape_max: ArrayDimensions,
) -> ArrayView2<'a, T> {
    // since tile_shape_max only contains odd numbers,
    // the result of an integer division by 2 always corresponds
    // to the maximum number of pixels on one side of the central one,
    // *excluding* the latter.
    let half_tile_shape_max = ArrayDimensions {
        x: tile_shape_max.x / 2,
        y: tile_shape_max.y / 2,
    };
    let dims = ArrayDimensions {
        x: image.shape()[1],
        y: image.shape()[0],
    };
    let tile_left_idx = PixelIndex {
        i: center_pixel.i.saturating_sub(half_tile_shape_max.y),
        j: center_pixel.j.saturating_sub(half_tile_shape_max.x),
    };
    let tile_right_idx = PixelIndex {
        i: if center_pixel.i + half_tile_shape_max.y > dims.y - 1 {
            dims.y - 1
        } else {
            center_pixel.i + half_tile_shape_max.y
        },
        j: if center_pixel.j + half_tile_shape_max.x > dims.x - 1 {
            dims.x - 1
        } else {
            center_pixel.j + half_tile_shape_max.x
        },
    };
    image.slice(s![
        tile_left_idx.i..tile_right_idx.i + 1,
        tile_left_idx.j..tile_right_idx.j + 1,
    ])
}

fn equalize_histogram_sliding_tile<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, T>,
    nbins: usize,
    tile_shape_max: (usize, usize),
) -> Bound<'py, PyArray2<T>> {
    let image = image.as_array();
    let dims = ArrayDimensions {
        x: image.shape()[1],
        y: image.shape()[0],
    };

    let tile_shape_max = ArrayDimensions {
        x: tile_shape_max.1,
        y: tile_shape_max.0,
    };
    let mut row_vmin: T;
    let mut row_vmax: T;
    let last_pixel = dims.last_pixel_index();
    let mut center_pixel = PixelIndex { i: 0, j: 0 };
    let mut out = Array2::<T>::zeros(image.raw_dim());
    while center_pixel.j <= last_pixel.j || center_pixel.i <= last_pixel.i {
        center_pixel.i = 0;
        for i in 0..dims.y {
            let tile = get_tile_view(&image, center_pixel, tile_shape_max);
            let tile_dims = ArrayDimensions {
                x: tile.shape()[1],
                y: tile.shape()[0],
            };

            // TODO: separate and reuse this loop
            // TODO: rewrite as to work on any flatten array ?
            let mut vmin = T::infinity();
            let mut vmax = -T::infinity();
            for i in 0..tile_dims.y {
                for j in 0..tile_dims.x {
                    let v = tile[[i, j]];
                    vmin = vmin.min(v);
                    vmax = vmax.max(v);
                }
            }
            let mut subhists = VecDeque::with_capacity(tile_dims.y + 1);
            for k in 0..tile_dims.y {
                let row = tile.index_axis(Axis(0), k);
                subhists.push_back(compute_subhistogram(row, vmin, vmax, nbins));
            }
            let row = image.row(i);
            row_vmin = T::infinity();
            row_vmax = -T::infinity();
            for v in row {
                row_vmin = row_vmin.min(*v);
                row_vmax = row_vmax.max(*v);
            }

            if (row_vmin < vmin) || (row_vmax > vmax) {
                subhists.truncate(0);
                // refresh extrema
                vmin = vmin.min(row_vmin);
                vmax = vmax.max(row_vmax);

                for it in 0..tile_dims.y {
                    let row = tile.index_axis(Axis(0), it);
                    subhists.push_back(compute_subhistogram(row, vmin, vmax, nbins));
                }
            }
            let hist = reduce_histogram(&subhists);

            let in_pix = image[[center_pixel.i, center_pixel.j]];
            let out_pix = &mut out[[center_pixel.i, center_pixel.j]];
            adjust_pixel_intensity(in_pix, &hist, out_pix);
            center_pixel.i += 1;
        }
        assert_eq!(center_pixel.i, last_pixel.i + 1);
        center_pixel.j += 1;
    }
    assert_eq!(center_pixel.j, last_pixel.j + 1);

    out.to_pyarray(py)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(gil_used = false)]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfunction]
    fn convolve_f32<'py>(
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
        convolve_iteratively(py, texture, uv, kernel, boundaries, iterations)
    }
    m.add_function(wrap_pyfunction!(convolve_f32, m)?)?;

    #[pyfunction]
    fn convolve_f64<'py>(
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
        convolve_iteratively(py, texture, uv, kernel, boundaries, iterations)
    }
    m.add_function(wrap_pyfunction!(convolve_f64, m)?)?;

    #[pyfunction]
    fn equalize_histogram_f32<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        nbins: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram(py, image, nbins)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_f32, m)?)?;

    #[pyfunction]
    fn equalize_histogram_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram(py, image, nbins)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_f64, m)?)?;

    #[pyfunction]
    fn equalize_histogram_sliding_tile_f32<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        nbins: usize,
        tile_shape_max: (usize, usize),
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram_sliding_tile(py, image, nbins, tile_shape_max)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_sliding_tile_f32, m)?)?;

    #[pyfunction]
    fn equalize_histogram_sliding_tile_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
        tile_shape_max: (usize, usize),
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram_sliding_tile(py, image, nbins, tile_shape_max)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_sliding_tile_f64, m)?)?;

    Ok(())
}

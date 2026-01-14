use crate::boundaries::BoundarySet;
use either::Either;
use interpn::one_dim::linear::LinearHoldLast1D;
use interpn::one_dim::Interp1D;
use interpn::RegularGrid1D;
use num_traits::{Float, NumCast, Signed};
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{s, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, Dimension};
use numpy::{PyArray2, ToPyArray};
use pyo3::types::PyModuleMethods;
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, PyResult, Python};
use std::collections::VecDeque;
use std::ops::{AddAssign, Mul, Neg, Sub};

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

trait AtLeastF32: Float + From<f32> + Signed + AddAssign<<Self as Mul>::Output> + std::fmt::Display + std::fmt::Debug {}
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

#[derive(Clone, Copy, PartialEq)]
struct Range<T> {
    lo: T,
    hi: T,
}

impl<T: Sub<Output = T> + Copy> Range<T> {
    fn span(&self) -> T {
        self.hi - self.lo
    }
}

struct Histogram<T> {
    bins: Array1<usize>,
    range: Range<T>,
}

impl<T: AtLeastF32 + NumCast> Histogram<T> {
    fn bin_width(&self) -> T {
        self.range.span() / <T as NumCast>::from(self.bins.len()).unwrap()
    }

    fn first_bin_center(&self) -> T {
        self.range.lo + self.bin_width() / 2.0.into()
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
    range: Range<T>,
    nbins: usize,
) -> Histogram<T> {
    let bin_width = range.span() / <T as NumCast>::from(nbins).unwrap();

    // padding one extra bin on the right allows for a branchless optimization:
    // pixels that contain exactly vmax are counted in the extra bin
    let mut bins = Array1::<usize>::zeros(nbins + 1);
    for v in arr.iter() {
        let f = ((*v - range.lo) / bin_width).floor();
        let idx = <usize as NumCast>::from(f).unwrap();
        bins[idx] += 1;
    }

    // move data from the last bin to the previous one
    bins[nbins - 1] += bins[nbins];
    let bins = bins.slice(s![..-1]).to_owned();
    assert_eq!(bins.len(), nbins);

    Histogram { bins, range }
}

fn reduce_histogram<T: Copy>(subhists: &VecDeque<Histogram<T>>) -> Histogram<T> {
    // it is assumed that all input histograms have the exact same range and nbins
    let h0 = subhists.front().unwrap();
    let nbins = h0.bins.len();
    let mut bins = Array1::<usize>::zeros(nbins);
    for h in subhists {
        for i in 0..nbins {
            bins[i] += h.bins[i];
        }
    }
    Histogram {
        bins,
        range: h0.range,
    }
}

fn compute_histogram<T: AtLeastF32 + numpy::Element + NumCast>(
    image: ArrayView2<T>,
    nbins: usize,
) -> Histogram<T> {
    let range = get_value_range(image);
    let mut subhistograms = VecDeque::with_capacity(image.shape()[0] + 1);
    for row in image.axis_iter(Axis(0)) {
        subhistograms.push_back(compute_subhistogram(row, range, nbins));
    }
    reduce_histogram(&subhistograms)
}

#[cfg(test)]
mod test_histogram {
    use crate::{Histogram, Range};
    use numpy::ndarray::Array1;

    #[test]
    fn test_ones() {
        let nbins = 8usize;
        let hist = Histogram {
            bins: Array1::<usize>::ones(nbins),
            range: Range { lo: 0.0, hi: 8.0 },
        };
        assert_eq!(hist.bin_width(), 1.0);
        assert_eq!(hist.first_bin_center(), 0.5);

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
    let cdf = hist.cdf_as_normalized();
    let grid = RegularGrid1D::new(
        hist.first_bin_center(),
        hist.bin_width(),
        cdf.as_slice().unwrap(),
    )
    .unwrap();
    let interpolator = LinearHoldLast1D::new(grid);

    let locs = image.as_slice().unwrap();
    match interpolator.eval(locs, out.as_slice_mut().unwrap()) {
        Ok(_) => (),
        Err(_) => panic!("interpolation failed"),
    }
}
fn adjust_intensity_single_pixel<T: AtLeastF32 + numpy::Element + NumCast>(
    interpolator: &LinearHoldLast1D<RegularGrid1D<'_, T>>,
    pixel: T,
) -> T {
    match interpolator.eval_one(pixel) {
        Ok(res) => res,
        Err(_) => panic!("interpolation failed"),
    }
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

#[derive(Copy, Clone, PartialEq)]
struct ViewRange {
    x: Range<usize>,
    y: Range<usize>,
}

fn get_tile_range(
    dims: ArrayDimensions,
    center_pixel: PixelIndex,
    tile_shape: ArrayDimensions,
) -> ViewRange {
    // since tile_shape only contains odd numbers,
    // the result of an integer division by 2 always corresponds
    // to the maximum number of pixels on one side of the central one,
    // *excluding* the latter.
    let half_tile_shape = ArrayDimensions {
        x: tile_shape.x / 2,
        y: tile_shape.y / 2,
    };
    ViewRange {
        x: Range {
            lo: center_pixel.j.saturating_sub(half_tile_shape.x),
            hi: if center_pixel.j + half_tile_shape.x > dims.x - 1 {
                dims.x - 1
            } else {
                center_pixel.j + half_tile_shape.x
            },
        },
        y: Range {
            lo: center_pixel.i.saturating_sub(half_tile_shape.y),
            hi: if center_pixel.i + half_tile_shape.y > dims.y - 1 {
                dims.y - 1
            } else {
                center_pixel.i + half_tile_shape.y
            },
        },
    }
}

fn get_tile_view<'a, T: numpy::Element>(
    image: &'a ArrayView2<'a, T>,
    range: ViewRange,
) -> ArrayView2<'a, T> {
    image.slice(s![range.y.lo..range.y.hi + 1, range.x.lo..range.x.hi + 1,])
}

fn get_value_range<A: AtLeastF32 + numpy::Element, D>(arr: ArrayView<A, D>) -> Range<A>
where
    D: Dimension,
{
    let mut lo = A::infinity();
    let mut hi = -A::infinity();
    for v in arr.iter() {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    Range { lo, hi }
}

fn equalize_histogram_sliding_tile<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, T>,
    nbins: usize,
    tile_shape: (usize, usize),
) -> Bound<'py, PyArray2<T>> {
    let image = image.as_array();
    let dims = ArrayDimensions {
        x: image.shape()[1],
        y: image.shape()[0],
    };

    let tile_shape = ArrayDimensions {
        x: tile_shape.1,
        y: tile_shape.0,
    };

    let mut center_pixel = PixelIndex { i: 0, j: 0 };
    let mut out = Array2::<T>::zeros(image.raw_dim());
    let mut subhists = VecDeque::with_capacity(tile_shape.y + 1);
    let mut hist = Histogram {
        bins: (Array1::<usize>::zeros(nbins)),
        range: Range {
            lo: 0.0.into(),
            hi: 1.0.into(),
        },
    };
    let mut cdf = hist.cdf_as_normalized();
    let mut cdf_grid = RegularGrid1D::new(
        hist.first_bin_center(),
        hist.bin_width(),
        cdf.as_slice().unwrap(),
    )
    .unwrap();
    let mut cdf_interpolator = LinearHoldLast1D::new(cdf_grid);
    let mut subhists_need_reinit = true;
    let mut hist_reduction_needed = true;
    let mut previous_tile_range = ViewRange {
        x: Range { lo: 0, hi: 0 },
        y: Range { lo: 0, hi: 0 },
    };

    for j in 0..dims.x {
        center_pixel.j = j;
        let mut tile: ArrayView2<T> = image.slice(s![.., ..]);
        let mut tile_dims: ArrayDimensions = dims;
        let mut vrange: Range<T> = Range {
            lo: 0.0.into(),
            hi: 0.0.into(),
        };
        let mut row_vrange: Range<T> = vrange;

        for i in 0..dims.y {
            center_pixel.i = i;
            let tile_range = get_tile_range(dims, center_pixel, tile_shape);
            if tile_range != previous_tile_range {
                tile = get_tile_view(&image, tile_range);
                tile_dims = ArrayDimensions {
                    x: tile.shape()[1],
                    y: tile.shape()[0],
                };
                if tile_range.y.lo != previous_tile_range.y.lo {
                    subhists.pop_front();
                }
                vrange = get_value_range(tile);
                row_vrange = vrange;

                previous_tile_range = tile_range;
                subhists_need_reinit = true;
            }

            if row_vrange.lo < vrange.lo {
                vrange.lo = row_vrange.lo;
                subhists_need_reinit = true;
            }
            if row_vrange.hi > vrange.hi {
                vrange.hi = row_vrange.hi;
                subhists_need_reinit = true;
            }

            if subhists_need_reinit {
                subhists.truncate(0);
                for row in tile.axis_iter(Axis(0)) {
                    subhists.push_back(compute_subhistogram(row, vrange, nbins));
                }
                hist_reduction_needed = true;
                subhists_need_reinit = false;
            }
            if subhists.len() < tile_dims.y {
                let row = image.row(i);
                row_vrange = get_value_range(row);
                subhists.push_back(compute_subhistogram(row, vrange, nbins));
                hist_reduction_needed = true;
            }
            assert_eq!(subhists.len(), tile_dims.y);
            if hist_reduction_needed {
                hist = reduce_histogram(&subhists);
                cdf = hist.cdf_as_normalized();
                cdf_grid = RegularGrid1D::new(
                    hist.first_bin_center(),
                    hist.bin_width(),
                    cdf.as_slice().unwrap(),
                )
                .unwrap();
                cdf_interpolator = LinearHoldLast1D::new(cdf_grid);
                hist_reduction_needed = false;
            }

            let in_pix = image[[center_pixel.i, center_pixel.j]];
            let out_pix = &mut out[[center_pixel.i, center_pixel.j]];
            *out_pix = adjust_intensity_single_pixel(&cdf_interpolator, in_pix);
        }
    }

    out.to_pyarray(py)
}

struct ECRQuadruple<T> {
    top_left: T,
    top_right: T,
    bottom_left: T,
    bottom_right: T,
}

struct InterpolatorInputs<T: Float + numpy::Element> {
    start: T,
    step: T,
    vals: Array1<T>,
}

impl<T: Float + numpy::Element> InterpolatorInputs<T> {
    fn as_interpolator(&self) -> LinearHoldLast1D<RegularGrid1D<'_, T>> {
        let grid =
            RegularGrid1D::new(self.start, self.step, self.vals.as_slice().unwrap()).unwrap();
        LinearHoldLast1D::new(grid)
    }
}
struct EffectiveContextualRegion<'a, T: Float + numpy::Element> {
    top_left: &'a InterpolatorInputs<T>,
    top_right: &'a InterpolatorInputs<T>,
    bottom_left: &'a InterpolatorInputs<T>,
    bottom_right: &'a InterpolatorInputs<T>,
}

impl<T: AtLeastF32 + NumCast + numpy::Element> EffectiveContextualRegion<'_, T> {
    fn get_normalized_intensities(&self, intensity: T) -> ECRQuadruple<T> {
        ECRQuadruple {
            top_left: adjust_intensity_single_pixel(&self.top_left.as_interpolator(), intensity),
            top_right: adjust_intensity_single_pixel(&self.top_right.as_interpolator(), intensity),
            bottom_left: adjust_intensity_single_pixel(
                &self.bottom_left.as_interpolator(),
                intensity,
            ),
            bottom_right: adjust_intensity_single_pixel(
                &self.bottom_right.as_interpolator(),
                intensity,
            ),
        }
    }
}

struct Stencil<T> {
    x: Array1<T>,
    y: Array1<T>,
}
struct TargetTile<'a, T: Float + numpy::Element> {
    ecr: EffectiveContextualRegion<'a, T>,
    stencil: &'a Stencil<T>,
}
fn equalize_histogram_tile_interpolation<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    pimage: PyReadonlyArray2<'py, T>,
    nbins: usize,
    tile_shape: (usize, usize),
) -> Bound<'py, PyArray2<T>> {
    let pimage = pimage.as_array();
    let pdims = ArrayDimensions {
        x: pimage.shape()[1],
        y: pimage.shape()[0],
    };
    let tile_shape = ArrayDimensions {
        x: tile_shape.1,
        y: tile_shape.0,
    };
    // assumptions:
    // - the full p(added)image is an integer multiple of tile sizes in every direction
    // - there's always *exactly* one entirely ghost tile in each direction
    // - it follows that every internal tile
    //   has a top, left, bottom and right neighbor
    // - external tiles do not need connectivity data (they are not to be iterated on)

    // ... define internal tiles ...
    // any "internal" may be only *partially* internal to the non-padded image,
    // but it'll always have a non-zero intersection with the unpadded domain

    let sample_mosaic_shape = ArrayDimensions {
        x: pdims.x / tile_shape.x,
        y: pdims.y / tile_shape.y,
    };

    let mut sample_interpolators = vec![];

    for i in (0..pdims.y).step_by(tile_shape.y) {
        let mut row_interpolators = vec![];
        for j in (0..pdims.x).step_by(tile_shape.x) {
            let vrange = ViewRange {
                x: Range {
                    lo: j,
                    hi: j + tile_shape.x - 1,
                },
                y: Range {
                    lo: i,
                    hi: i + tile_shape.y - 1,
                },
            };
            let tile_view = get_tile_view(&pimage, vrange);
            let hist = compute_histogram(tile_view, nbins);
            let cdf = hist.cdf_as_normalized();
            let ii = InterpolatorInputs {
                start: hist.first_bin_center(),
                step: hist.bin_width(),
                vals: cdf,
            };
            row_interpolators.push(ii);
        }
        sample_interpolators.push(row_interpolators);
    }
    assert_eq!(sample_interpolators.len(), sample_mosaic_shape.y);
    for si in sample_interpolators.iter().take(sample_mosaic_shape.y) {
        assert_eq!(si.len(), sample_mosaic_shape.x);
    }
    let half_tile_offset_x = tile_shape.x / 2;
    let half_tile_offset_y = tile_shape.y / 2;
    let mut tiles: Vec<Vec<TargetTile<T>>> = vec![];

    // x offset, in pixel width, between the top left tile center and the center
    // of the first pixel in the target tile. Since all tile sizes are even,
    // it follows that this is always 0.5
    let xoff: T = 0.5.into();
    let mut alpha = Array1::<T>::zeros(tile_shape.x);
    let tsx = <T as NumCast>::from(tile_shape.x).unwrap();
    for j in 0..tile_shape.x {
        let fj = <T as NumCast>::from(j).unwrap();
        alpha[j] = xoff + fj / tsx;
    }

    // y offset, in pixel height, between the top left tile center and the center
    // of the first pixel in the target tile.
    let yoff: T = 0.5.into();
    let mut beta = Array1::<T>::zeros(tile_shape.y);
    let tsy = <T as NumCast>::from(tile_shape.y).unwrap();
    for i in 0..tile_shape.y {
        let fi = <T as NumCast>::from(i).unwrap();
        beta[i] = yoff + fi / tsy;
    }

    let stencil = Stencil { x: alpha, y: beta };

    for imos in 0..sample_mosaic_shape.y - 1 {
        let mut row = vec![];
        for jmos in 0..sample_mosaic_shape.x - 1 {
            row.push(TargetTile {
                ecr: EffectiveContextualRegion {
                    top_left: &sample_interpolators[imos][jmos],
                    top_right: &sample_interpolators[imos][jmos + 1],
                    bottom_left: &sample_interpolators[imos + 1][jmos],
                    bottom_right: &sample_interpolators[imos + 1][jmos + 1],
                },
                stencil: &stencil,
            })
        }
        tiles.push(row);
    }
    let one: T = 1.0.into();

    let mut out = Array2::<T>::zeros(pimage.raw_dim());
    for (itiles, row) in tiles.iter().enumerate().take(sample_mosaic_shape.y - 1) {
        let ioff = half_tile_offset_y + itiles * tile_shape.y;
        for (jtiles, tile) in row.iter().enumerate().take(sample_mosaic_shape.x - 1) {
            let joff = half_tile_offset_x + jtiles * tile_shape.x;
            for i in 0..tile_shape.y {
                for j in 0..tile_shape.x {
                    let v = pimage[[ioff + i, joff + j]];
                    let q = tile.ecr.get_normalized_intensities(v);
                    let a = tile.stencil.x[j];
                    let b = tile.stencil.y[i];
                    // my convention for a VS b is completely different from Fizer 1987
                    // maybe I should align them.
                    out[[ioff + i, joff + j]] = b
                        * (a * q.bottom_right + (one - a) * q.bottom_left)
                        + (one - b) * (a * q.top_right + (one - a) * q.top_left);
                }
            }
        }
    }
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
        tile_shape: (usize, usize),
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram_sliding_tile(py, image, nbins, tile_shape)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_sliding_tile_f32, m)?)?;

    #[pyfunction]
    fn equalize_histogram_sliding_tile_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
        tile_shape: (usize, usize),
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram_sliding_tile(py, image, nbins, tile_shape)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_sliding_tile_f64, m)?)?;

    #[pyfunction]
    fn equalize_histogram_tile_interpolation_f32<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        nbins: usize,
        tile_shape: (usize, usize),
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram_tile_interpolation(py, image, nbins, tile_shape)
    }
    m.add_function(wrap_pyfunction!(
        equalize_histogram_tile_interpolation_f32,
        m
    )?)?;

    #[pyfunction]
    fn equalize_histogram_tile_interpolation_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
        tile_shape: (usize, usize),
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram_tile_interpolation(py, image, nbins, tile_shape)
    }
    m.add_function(wrap_pyfunction!(
        equalize_histogram_tile_interpolation_f64,
        m
    )?)?;

    Ok(())
}

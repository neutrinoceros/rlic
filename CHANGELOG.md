# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.4.0 - 2025-07-10

- DOC: consistently indicate argument types (positional-only and keyword-only)
  in `rlic.convolve`'s docstring
- DOC: fix invalid example code and re-generate output image
- DOC: document memory usage
- PERF: turn rust panics into fatal errors, improving performance and
  reducing binary size by 10% each
- PERF: leverage Fused Multiply-Add (FMA) instructions on supporting architectures
  (including Apple Silicon and AMD processors). Except about 2 to 4% gain
- PERF: avoid back-and-forth conversions between usize and i64 representations
  for image coordinates. Expect about 2% gain
- DEP: drop support for NumPy 1.21.x, require 1.22.0 or newer
- WHL: upgrade `manylinux` image from `2014` to `2_28`, following NumPy

## 0.3.4 - 2025-04-09

- BLD: declare compatibility with free-threaded CPython
- DOC: fix an incomplete sentence in `rlic.convolve`'s docstring
- TYP: add array shape information to type hints
- BLD: bump MSRV (1.64.0 -> 1.65.0)
- ENH: report all exceptions instead of just the first one when validating inputs,
  on all supported Python versions (using the backport package `exceptiongroup`
  on 3.9 and 3.10)

## 0.3.3 - 2025-03-08

- BUG: fix a typo in a user visible error message
- ENH: report all exceptions instead of just the first one when validating inputs, on
  Python 3.11 and newer
- BUG: always run all validators before returning

## 0.3.2 - 2025-03-07

- DOC: prefer using a convex kernel shape in examples
- ENH: terminate streamline integration when a NaN vector component is encountered
- ENH: explicitly forbid non-finite values in `kernel`

## 0.3.1 - 2025-03-04

- DOC: add example scripts and visualizations to `README.md`
- ENH: allow negative values in kernels. It's still an unusual use case,
  but there is no strong argument for forbidding it.
- ENH: Previously, only kernels with at least 3 elements and at most as long as
  the shortest dimension of the image were allowed. These restrictions are now
  lifted.
- DOC: improve `rlic.convolve`'s docstring. Add detail to the description of the
  `texture` and `kernel` arguments

## 0.3.0 - 2025-03-02

- PERF: disable build parallelism to the benefit of runtime performance
  (expect about 5 to 10% gain)
- PERF: avoid conditional branching in pixel traversal time evaluation
  (expect about 30 to 40% gain)
- TST: test for regressions against reference implementation

## 0.2.2 - 2025-03-01

- BUG: fix a defect in polarization mode

## 0.2.1 - 2025-02-27

- BUG: fix incorrect implementation of infinite-time edge cases
- TST: improve Rust and Python test coverage
- TST: test against future versions of direct rust dependencies
- BLD/TST: fix incorrect Minimal Supported Rust Version (MSRV) (bumped from 1.63
  to 1.64) and check it in CI

## 0.2.0 - 2025-02-25

- WHL: build and publish `abi3` wheels for all platforms + architecture combos
  supported by NumPy as of 2.2.3
- API: rename the first argument to `rlic.convolve` from `image` to `texture`,
  and make it positional-only.

## 0.1.2 - 2025-02-24

- ENH: add support for single precision (`float32`) input data

## 0.1.1 - 2025-02-23

- DEP: add support for numpy 1.x
- REL: ensure source distribution is testable and enable auto-publishing

## 0.1.0 - 2025-02-23

This is the first release. It's missing wheels and documentation.
Only source distribution is published on PyPI.

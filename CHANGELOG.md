# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

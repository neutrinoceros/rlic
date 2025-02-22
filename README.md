# rLIC
[![PyPI](https://img.shields.io/pypi/v/rlic.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/rlic/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

***Line Integral Convolution for Python, written in Rust***

`rLIC` (pronounced 'relic') is a minimal implementation of the [Line Integral Convolution](https://en.wikipedia.org/wiki/Line_integral_convolution) algorithm for in-memory `numpy` arrays, written in Rust.


## Installation
```
python -m pip install rLIC
```

## Example usage

`rLIC` consists in a single Python function, `rlic.convolve`.

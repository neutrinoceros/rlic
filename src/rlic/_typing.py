__all__ = [
    "D1",
    "D2",
    "F",
    "FArray1D",
    "FArray2D",
    "Pair",
    "PairSpec",
    "UVMode",
    "UnsetType",
    "UNSET",
]
from enum import Enum, auto
from typing import Literal, TypeAlias, TypeVar

import numpy as np
from numpy import float32 as f32
from numpy import float64 as f64

T = TypeVar("T")
Pair: TypeAlias = tuple[T, T]
PairSpec: TypeAlias = T | Pair[T]

UVMode = Literal["velocity", "polarization"]


class UnsetType(Enum):
    UNSET = auto()


UNSET = UnsetType.UNSET

F = TypeVar("F", f32, f64)
D1 = tuple[int]
D2 = tuple[int, int]
FArray1D = np.ndarray[D1, np.dtype[F]]
FArray2D = np.ndarray[D2, np.dtype[F]]

__all__ = [
    "Pair",
    "PairSpec",
    "UVMode",
]

from typing import Literal, TypeAlias, TypeVar

T = TypeVar("T")
Pair: TypeAlias = tuple[T, T]
PairSpec: TypeAlias = T | Pair[T]

UVMode = Literal["velocity", "polarization"]

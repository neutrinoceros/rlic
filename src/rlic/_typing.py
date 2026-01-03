__all__ = [
    "Boundary",
    "Pair",
    "PairSpec",
    "UVMode",
]

from typing import Literal, TypeAlias, TypedDict, TypeVar

T = TypeVar("T")
Pair: TypeAlias = tuple[T, T]
PairSpec: TypeAlias = T | Pair[T]

Boundary = Literal["closed", "periodic"]


class BoundarySpec(TypedDict):
    x: PairSpec[Boundary]
    y: PairSpec[Boundary]


UVMode = Literal["velocity", "polarization"]

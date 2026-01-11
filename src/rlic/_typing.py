__all__ = [
    "Pair",
    "PairSpec",
    "UVMode",
    "UnsetType",
    "UNSET",
]
from enum import Enum, auto
from typing import Literal, TypeAlias, TypeVar

T = TypeVar("T")
Pair: TypeAlias = tuple[T, T]
PairSpec: TypeAlias = T | Pair[T]

UVMode = Literal["velocity", "polarization"]


class UnsetType(Enum):
    UNSET = auto()


UNSET = UnsetType.UNSET

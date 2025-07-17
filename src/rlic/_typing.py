__all__ = [
    "Boundary",
    "UVMode",
]

from typing import Literal, TypeAlias, TypedDict

Boundary = Literal["closed", "periodic"]
BoundaryPair = tuple[Boundary, Boundary]
AnyBoundary: TypeAlias = Boundary | BoundaryPair


class BoundaryDict(TypedDict):
    x: AnyBoundary
    y: AnyBoundary


UVMode = Literal["velocity", "polarization"]

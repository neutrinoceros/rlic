from typing import NewType, TypeAlias, TypedDict

Bound = NewType("Bound", str)
BoundPair = tuple[Bound, Bound]
AnyBound: TypeAlias = Bound | BoundPair


class AbstractBounds(TypedDict):
    x: AnyBound
    y: AnyBound


class ConcreteBounds(TypedDict):
    x: BoundPair
    y: BoundPair


def as_pair(b: AnyBound) -> BoundPair:
    match b:
        case (str(), str()):
            return (b[0], b[1])
        case _:
            return (b, b)


def expand_bounds(bounds: Bound | AbstractBounds | ConcreteBounds) -> ConcreteBounds:
    """Expand the requirements string or dictionary into a concrete bounds dictionary."""
    match bounds:
        case str():
            return {
                "x": as_pair(bounds),
                "y": as_pair(bounds),
            }
        case {"x": str() | (str(), str()), "y": str() | (str(), str())} if (
            len(bounds) == 2
        ):
            return {
                "x": as_pair(bounds["x"]),
                "y": as_pair(bounds["y"]),
            }
        case _:
            raise TypeError(f"Invalid boundary specification {bounds}")

__all__ = [
    "Strategy",
]

import sys
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from rlic._typing import Pair, PairSpec

if sys.version_info >= (3, 11):
    from typing import NotRequired, assert_never  # pyright: ignore[reportUnreachable]
else:
    from typing_extensions import (  # pyright: ignore[reportUnreachable]
        NotRequired,
        assert_never,
    )

SUPPORTED_KINDS = frozenset({"sliding-tile"})
StrategyKind: TypeAlias = Literal["sliding-tile"]


SlidingTileSpec = TypedDict(
    "SlidingTileSpec",
    {
        "kind": Literal["sliding-tile"],
        "tile-size": NotRequired[PairSpec[int]],
        "tile-size-max": NotRequired[PairSpec[int]],
    },
)


class TileSizeSpec(TypedDict):
    tile_size: PairSpec[int]


class TileSizeMaxSpec(TypedDict):
    tile_size_max: PairSpec[int]


def as_pair(s: PairSpec[int], /) -> Pair[int]:
    match s:
        case int():
            return (s, s)
        case (int(s1), int(s2)):
            return (s1, s2)
        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)  # type: ignore


@dataclass(frozen=True, slots=True, kw_only=True)
class Strategy:
    kind: StrategyKind
    tile_size: Pair[int] | None = None
    tile_size_max: Pair[int] | None = None

    @staticmethod
    def from_spec(spec: SlidingTileSpec, /) -> "Strategy":
        match kind := spec.get("kind"):
            case None:  # pyright: ignore[reportUnnecessaryComparison]
                raise TypeError("strategy dict is missing a 'kind' key.")  # pyright: ignore[reportUnreachable]
            case "sliding-tile":
                pass
            case _:  # pyright: ignore[reportUnnecessaryComparison]
                raise ValueError(  # pyright: ignore[reportUnreachable]
                    f"Unknown strategy kind {kind!r}. Expected one of {sorted(SUPPORTED_KINDS)}"
                )

        kwarg: TileSizeSpec | TileSizeMaxSpec

        match (spec.get("tile-size"), spec.get("tile-size-max")):
            case (None, None):
                raise TypeError(
                    "Neither 'tile-size' nor 'tile-size-max' keys were found. "
                    "Either are allowed, but exactly one is expected."
                )
            case (((int() | (int(), int())) as ts), None):
                kwarg = {"tile_size": as_pair(ts)}  # type: ignore[arg-type]
            case (None, ((int() | (int(), int())) as ts)):
                kwarg = {"tile_size_max": as_pair(ts)}
            case (ts, None):  # pyright: ignore[reportUnnecessaryComparison]
                raise TypeError(  # pyright: ignore[reportUnreachable]
                    "Incorrect type associated with key 'tile-size'. "
                    f"Received {ts} with type {type(ts)}. "
                    "Expected a single int, or a pair thereof."
                )
            case (None, ts):  # pyright: ignore[reportUnnecessaryComparison]
                raise TypeError(  # pyright: ignore[reportUnreachable]
                    "Incorrect type associated with key 'tile-size-max'. "
                    f"Received {ts} with type {type(ts)}. "
                    "Expected a single int, or a pair thereof."
                )
            case _:
                raise TypeError(
                    "Both 'tile-size' and 'tile-size-max' keys were provided. "
                    "Only one of them can be specified at a time."
                )

        return Strategy(
            kind=kind,
            **kwarg,  # pyright: ignore[reportArgumentType]
        )

# pyright: reportUnreachable=false, reportUnnecessaryComparison=false
__all__ = [
    "SlidingTile",
    "Strategy",
    "StrategySpec",
    "SUPPORTED_AHE_KINDS",
    "TileInterpolation",
]

import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, TypedDict, TypeVar, final

from rlic._typing import UNSET, Pair, PairSpec, UnsetType

if sys.version_info >= (3, 11):
    from typing import NotRequired, Self, assert_never
else:
    from exceptiongroup import ExceptionGroup
    from typing_extensions import NotRequired, Self, assert_never

SUPPORTED_AHE_KINDS = frozenset({"sliding-tile", "tile-interpolation"})
StrategyKind: TypeAlias = Literal["sliding-tile", "tile-interpolation"]


SlidingTileSpec = TypedDict(
    "SlidingTileSpec", {"kind": Literal["sliding-tile"], "tile-size-max": PairSpec[int]}
)
TileInterpolationSpec = TypedDict(
    "TileInterpolationSpec",
    {
        "kind": Literal["tile-interpolation"],
        "tile-into": NotRequired[PairSpec[int]],
        "tile-size-max": NotRequired[PairSpec[int]],
    },
)

StrategySpec: TypeAlias = SlidingTileSpec | TileInterpolationSpec


def as_pair(s: PairSpec[int], /) -> Pair[int]:
    match s:
        case int():
            return (s, s)
        case (int(s1), int(s2)):
            return (s1, s2)
        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[arg-type]


def report(exceptions: list[Exception]) -> None:
    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise ExceptionGroup(
            "Found multiple issues with adaptive strategy specifications",
            exceptions,
        )


MSG_EVEN = "Maximum {axis} tile size {size} is even. Only odd values are allowed."
MSG_TOO_LOW = "Maximum {axis} tile size {size} is too low. The minimum allowed positive value is 3"
MSG_TOO_LOW_NEG = "Maximum {axis} tile size {size} is too low. The minimum allowed negative value is -2"


def collect_exceptions_tile_size_max(tile_size_max: Pair[int]) -> list[Exception]:
    exceptions: list[Exception] = []
    for axis, size in zip(("x", "y"), tile_size_max, strict=True):
        if size < -2:
            exceptions.append(ValueError(MSG_TOO_LOW_NEG.format(axis=axis, size=size)))
        if size < 0:
            continue
        if size < 3:
            exceptions.append(ValueError(MSG_TOO_LOW.format(axis=axis, size=size)))
        if not size % 2:
            exceptions.append(ValueError(MSG_EVEN.format(axis=axis, size=size)))
    return exceptions


MSG_INTO_NEG = (
    "Cannot produce {into} tiles on axis {axis}. The minimum meaningful value is 1"
)


def collect_exceptions_tile_into(tile_into: Pair[int]) -> list[Exception]:
    exceptions: list[Exception] = []
    for axis, into in zip(("x", "y"), tile_into, strict=True):
        if into < 1:
            exceptions.append(ValueError(MSG_INTO_NEG.format(axis=axis, into=into)))

    return exceptions


def resolve_tile_shape(
    tile_shape: Pair[int], image_shape: Pair[int], *, require_odd: bool
) -> Pair[int]:
    assert all(s > 0 for s in image_shape)
    base_shape = tile_shape
    ret_shape_mut = list(base_shape)
    for i in range(2):
        s = image_shape[i]
        if base_shape[i] == -1:
            ret_shape_mut[i] = s
        elif base_shape[i] == -2:
            ret_shape_mut[i] = 2 * s
        if require_odd:
            ret_shape_mut[i] |= 1  # add 1 if the value is even
    ret_shape = (ret_shape_mut[0], ret_shape_mut[1])
    assert all(s > 0 for s in ret_shape)
    if require_odd:
        assert all(s % 2 for s in ret_shape)
    return ret_shape


def minimal_divisor_size(size: int, into: int) -> int:
    d, r = divmod(size, into)
    # 1 if there's a non-zero remainder, 0 otherwise
    round_up = int(bool(r))
    return max(1, d + round_up)


V = TypeVar("V")


class Strategy(Protocol):
    @classmethod
    def from_spec(cls, spec: Mapping[str, V], /) -> Self: ...
    def resolve_tile_shape(self, image_shape: Pair[int]) -> Pair[int]: ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SlidingTile:
    tile_shape_max: Pair[int]

    @classmethod
    def from_spec(cls, spec: Mapping[str, V], /) -> Self:
        if spec.get("kind") != "sliding-tile":
            raise AssertionError

        tsp: Pair[int] | None = None
        match spec.get("tile-size-max", UNSET):
            case (int() | (int(), int())) as ts:
                tsp = as_pair(ts)  # type: ignore[arg-type]
            case UnsetType():
                raise TypeError(
                    "Sliding tile specification is missing a 'tile-size-max' key. "
                    "Expected a single int, or a pair thereof."
                )
            case _ as invalid:
                raise TypeError(
                    "Incorrect type associated with key 'tile-size-max'. "
                    f"Received {invalid} with type {type(invalid)}. "
                    "Expected a single int, or a pair thereof."
                )

        exceptions = collect_exceptions_tile_size_max(tsp)
        report(exceptions)
        return cls(tile_shape_max=tsp)

    def resolve_tile_shape(self, image_shape: Pair[int]) -> Pair[int]:
        return resolve_tile_shape(self.tile_shape_max, image_shape, require_odd=True)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TileInterpolation:
    tile_into: Pair[int] | None = None
    tile_shape_max: Pair[int] | None = None

    @classmethod
    def from_spec(cls, spec: Mapping[str, V], /) -> Self:
        if spec.get("kind") != "tile-interpolation":
            raise AssertionError

        exceptions: list[Exception] = []

        tsp: Pair[int] | None = None
        tip: Pair[int] | None = None
        match (spec.get("tile-into", UNSET), spec.get("tile-size-max", UNSET)):
            case (UnsetType(), UnsetType()):
                exceptions.append(
                    TypeError(
                        "Neither 'tile-into' nor 'tile-size-max' keys were found. "
                        "Either are allowed, but exactly one is expected."
                    )
                )
            case (((int() | (int(), int())) as ti), UnsetType()):
                tip = as_pair(ti)  # type: ignore[arg-type]
            case (UnsetType(), ((int() | (int(), int())) as ts)):
                tsp = as_pair(ts)  # type: ignore[arg-type]
            case (ts, UnsetType()):
                exceptions.append(
                    TypeError(
                        "Incorrect type associated with key 'tile-into'. "
                        f"Received {ts} with type {type(ts)}. "
                        "Expected a single int, or a pair thereof."
                    )
                )
            case (UnsetType(), ts):
                exceptions.append(
                    TypeError(
                        "Incorrect type associated with key 'tile-size-max'. "
                        f"Received {ts} with type {type(ts)}. "
                        "Expected a single int, or a pair thereof."
                    )
                )
            case _:
                exceptions.append(
                    TypeError(
                        "Both 'tile-into' and 'tile-size-max' keys were provided. "
                        "Only one of them can be specified at a time."
                    )
                )

        if tip is not None:
            exceptions.extend(collect_exceptions_tile_into(tip))

        if tsp is not None:
            exceptions.extend(collect_exceptions_tile_size_max(tsp))

        report(exceptions)
        return cls(tile_into=tip, tile_shape_max=tsp)

    def resolve_tile_shape(self, image_shape: Pair[int]) -> Pair[int]:
        assert all(s > 0 for s in image_shape)
        assert Counter([self.tile_into, self.tile_shape_max])[None] == 1

        if self.tile_into is not None:
            tsm = (
                minimal_divisor_size(image_shape[0], self.tile_into[0]),
                minimal_divisor_size(image_shape[1], self.tile_into[1]),
            )
        elif self.tile_shape_max is not None:
            tsm = resolve_tile_shape(
                self.tile_shape_max, image_shape, require_odd=False
            )
        else:
            raise AssertionError

        return tsm

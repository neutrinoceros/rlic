# pyright: reportUnreachable=false, reportUnnecessaryComparison=false
__all__ = [
    "Strategy",
]

import sys
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict, cast

from rlic._typing import Pair, PairSpec

if sys.version_info >= (3, 11):
    from typing import NotRequired, assert_never
else:
    from exceptiongroup import ExceptionGroup
    from typing_extensions import NotRequired, assert_never

if sys.version_info >= (3, 13):
    from copy import replace as copy_replace
else:
    from dataclasses import replace as copy_replace

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


class TileShapeSpec(TypedDict):
    tile_shape: PairSpec[int]


class TileShapeMaxSpec(TypedDict):
    tile_shape_max: PairSpec[int]


def as_pair(s: PairSpec[int], /) -> Pair[int]:
    match s:
        case int():
            return (s, s)
        case (int(s1), int(s2)):
            return (s1, s2)
        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[arg-type]


MSG_EVEN = "{prefix}{axis} tile size {size} is even. Only odd values are allowed."
MSG_TOO_LOW = (
    "{prefix}{axis} tile size {size} is too low. The minimum allowed value is 3"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Strategy:
    kind: StrategyKind
    tile_shape: Pair[int] | None = None
    tile_shape_max: Pair[int] | None = None

    @classmethod
    def from_spec(cls, spec: SlidingTileSpec, /) -> "Strategy":
        exceptions: list[Exception] = []

        match kind := spec.get("kind"):
            case None:
                exceptions.append(TypeError("strategy dict is missing a 'kind' key."))
            case "sliding-tile":
                pass
            case _:
                exceptions.append(
                    ValueError(
                        f"Unknown strategy kind {kind!r}. Expected one of {sorted(SUPPORTED_KINDS)}"
                    )
                )

        kwarg: TileShapeSpec | TileShapeMaxSpec | None = None

        match (spec.get("tile-size"), spec.get("tile-size-max")):
            case (None, None):
                exceptions.append(
                    TypeError(
                        "Neither 'tile-size' nor 'tile-size-max' keys were found. "
                        "Either are allowed, but exactly one is expected."
                    )
                )
            case (((int() | (int(), int())) as ts), None):
                kwarg = {"tile_shape": as_pair(ts)}  # type: ignore[arg-type, assignment]
            case (None, ((int() | (int(), int())) as ts)):
                kwarg = {"tile_shape_max": as_pair(ts)}
            case (ts, None):
                exceptions.append(
                    TypeError(
                        "Incorrect type associated with key 'tile-size'. "
                        f"Received {ts} with type {type(ts)}. "
                        "Expected a single int, or a pair thereof."
                    )
                )
            case (None, ts):
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
                        "Both 'tile-size' and 'tile-size-max' keys were provided. "
                        "Only one of them can be specified at a time."
                    )
                )

        if kwarg is None:
            cls.report(exceptions)
            raise AssertionError

        key = next(iter(kwarg.keys()))
        pair = cast("Pair[int]", next(iter(kwarg.values())))
        if key == "tile_shape_max":
            prefix = "Maximum "
        else:
            prefix = ""
        for size, axis in zip(pair, ("x", "y"), strict=True):
            if size < 0:
                continue
            if size < 3:
                exceptions.append(
                    ValueError(MSG_TOO_LOW.format(prefix=prefix, axis=axis, size=size))
                )
            if not size % 2:
                exceptions.append(
                    ValueError(MSG_EVEN.format(prefix=prefix, axis=axis, size=size))
                )

        cls.report(exceptions)

        return Strategy(
            kind=kind,
            **kwarg,  # pyright: ignore[reportArgumentType]
        )

    @staticmethod
    def report(exceptions: list[Exception]) -> None:
        if len(exceptions) == 1:
            raise exceptions[0]
        elif exceptions:
            raise ExceptionGroup(
                "Found multiple issues with adaptive strategy specifications",
                exceptions,
            )

    def resolve_tile_shape(self, containing_shape: Pair[int], /) -> "Strategy":
        if self.tile_shape_max is None:
            raise AssertionError
        assert all(s > 0 for s in containing_shape)
        base_shape = self.tile_shape_max
        ret_shape_mut = list(base_shape)
        for i in range(2):
            if not (s := containing_shape[i]) % 2:
                s += 1
            if base_shape[i] < 0:
                ret_shape_mut[i] = s
        ret_shape = (ret_shape_mut[0], ret_shape_mut[1])
        assert all(s > 0 for s in ret_shape)
        assert all(s % 2 for s in ret_shape)
        return copy_replace(self, tile_shape_max=ret_shape)

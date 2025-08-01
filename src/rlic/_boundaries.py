from __future__ import annotations

__all__ = [
    "COMBO_ALLOWED_BOUNDS",
    "COMBO_DISALLOWED_BOUNDS",
    "SUPPORTED_BOUNDS",
    "BoundarySet",
]

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlic._typing import AnyBoundary, Boundary, BoundaryDict, BoundaryPair
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup  # pyright: ignore[reportUnreachable]


# boundaries that can be combined with another value on the opposite side
COMBO_ALLOWED_BOUNDS = frozenset({"closed"})
# boundaries that require the exact same value be used on the opposite side
COMBO_DISALLOWED_BOUNDS = frozenset({"periodic"})

SUPPORTED_BOUNDS = frozenset(COMBO_ALLOWED_BOUNDS | COMBO_DISALLOWED_BOUNDS)


def as_pair(b: AnyBoundary, /) -> BoundaryPair:
    match b:
        case str():
            return (b, b)
        case (str(), str()):
            return (b[0], b[1])
        case _:  # pragma: no cover # pyright: ignore[reportUnnecessaryComparison]
            raise RuntimeError  # pyright: ignore[reportUnreachable]


@dataclass(frozen=True, slots=True, kw_only=True)
class BoundarySet:
    x: BoundaryPair
    y: BoundaryPair

    @staticmethod
    def from_user_input(bounds: Boundary | BoundaryDict, /) -> BoundarySet | None:
        # this function is responsible for validating the keys in a dict input, but
        # lets through any str as keys. It should output a very predictable structure.
        match bounds:
            case str():
                return BoundarySet(x=as_pair(bounds), y=as_pair(bounds))
            case {"x": str() | (str(), str()), "y": str() | (str(), str())} if (
                len(bounds) == 2
            ):
                return BoundarySet(x=as_pair(bounds["x"]), y=as_pair(bounds["y"]))
            case _:
                # signal an invalid input
                return None

    def collect_exceptions(self) -> list[Exception]:
        # this function is responsible for invalidating unsupported keys (or combos)
        msg_unknown = "Unknown {side} {ax} boundary {name!r}"
        msg_invalid_combo = (
            "{side} {ax} boundary {name!r} cannot be combined with "
            "a different boundary ({other!r})"
        )
        exceptions: list[Exception] = []

        for axis, (left, right) in [("x", self.x), ("y", self.y)]:
            if {left, right}.issubset(COMBO_ALLOWED_BOUNDS):
                continue

            if {left, right}.issubset(SUPPORTED_BOUNDS):
                # intentionally only report on disallowed combos
                # if both bounds are known (avoid over-reporting for simple typos)
                if left == right:
                    continue
                if left in COMBO_DISALLOWED_BOUNDS:
                    msg = msg_invalid_combo.format(
                        side="left", name=left, other=right, ax=axis
                    )
                    exceptions.append(ValueError(msg))
                if right in COMBO_DISALLOWED_BOUNDS:
                    msg = msg_invalid_combo.format(
                        side="right", name=right, other=left, ax=axis
                    )
                    exceptions.append(ValueError(msg))
            else:
                if left not in SUPPORTED_BOUNDS:
                    msg = msg_unknown.format(side="left", name=left, ax=axis)
                    exceptions.append(ValueError(msg))
                if right not in SUPPORTED_BOUNDS:
                    msg = msg_unknown.format(side="right", name=right, ax=axis)
                    exceptions.append(ValueError(msg))

        return exceptions

    def validate(self) -> None:
        exceptions = self.collect_exceptions()
        if len(exceptions) == 1:
            raise exceptions[0]
        elif exceptions:
            raise ExceptionGroup(
                "Found multiple issues with boundary specifications", exceptions
            )

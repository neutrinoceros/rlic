from dataclasses import replace
from itertools import permutations, product

import pytest
from pytest import RaisesExc, RaisesGroup

from rlic._boundaries import (
    COMBO_ALLOWED_BOUNDS,
    COMBO_DISALLOWED_BOUNDS,
    SUPPORTED_BOUNDS,
    BoundarySet,
)


@pytest.mark.parametrize(
    "bounds_input, expected_output",
    [
        pytest.param("a", BoundarySet(x=("a", "a"), y=("a", "a")), id="expand-all"),
        pytest.param(
            {"x": "a", "y": "b"},
            BoundarySet(x=("a", "a"), y=("b", "b")),
            id="expand-keys",
        ),
        pytest.param(
            {"x": ("a", "b"), "y": ("c", "w")},
            BoundarySet(x=("a", "b"), y=("c", "w")),
            id="already-expanded",
        ),
        pytest.param(
            {"x": ["a", "b"], "y": ["c", "w"]},
            BoundarySet(x=("a", "b"), y=("c", "w")),
            id="lists-to-tuples",
        ),
    ],
)
def test_from_user_input(bounds_input, expected_output):
    result = BoundarySet.from_user_input(bounds_input)
    assert result == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        123,
        None,
        ["a", "b"],  # List is not a valid input type
        {"x": "a"},  # Missing 'y' key
        {"y": "b"},  # Missing 'x' key
        {"x": "a", "y": "b", "z": "c"},  # extra key not allowed
    ],
)
def test_from_user_input_invalid_type(invalid_input):
    assert BoundarySet.from_user_input(invalid_input) is None


@pytest.mark.parametrize(
    "bound_input",
    [
        pytest.param(BoundarySet(x=(b1, b1), y=(b2, b2)), id=f"doubles_x={b1}_y={b2}")
        for (b1, b2) in product(SUPPORTED_BOUNDS, SUPPORTED_BOUNDS)
    ],
)
def test_validate_simple_bounds(bound_input):
    assert not bound_input.collect_exceptions()
    assert bound_input.validate() is None


@pytest.mark.parametrize(
    "bound_input",
    [
        pytest.param(BoundarySet(x=(b1, b1), y=(b2, b2)), id=f"doubles_x={b1}_y={b2}")
        for (b1, b2) in product(COMBO_ALLOWED_BOUNDS, COMBO_ALLOWED_BOUNDS)
    ],
)
def test_validate_bounds_combos(bound_input):
    assert not bound_input.collect_exceptions()
    assert bound_input.validate() is None


@pytest.mark.parametrize(
    "bound_input",
    [
        pytest.param(BoundarySet(x=(b1, b2), y=(b1, b1)), id=f"{b1}+{b2}")
        for (b1, b2) in product(COMBO_DISALLOWED_BOUNDS, SUPPORTED_BOUNDS)
        if b1 != b2
    ],
)
def test_invalid_combos(bound_input):
    with pytest.raises(ValueError, match=r"^left x boundary '\w+' cannot be combined"):
        bound_input.validate()

    reversed_x = replace(bound_input, x=reversed(bound_input.x))
    with pytest.raises(ValueError, match=r"^right x boundary '\w+' cannot be combined"):
        reversed_x.validate()

    swapped = BoundarySet(x=bound_input.y, y=bound_input.x)
    with pytest.raises(ValueError, match=r"^left y boundary '\w+' cannot be combined"):
        swapped.validate()

    swapped_reversed_y = replace(swapped, y=reversed(swapped.y))
    with pytest.raises(ValueError, match=r"^right y boundary '\w+' cannot be combined"):
        swapped_reversed_y.validate()


@pytest.mark.parametrize(
    "bound_input",
    [
        pytest.param(
            BoundarySet(x=(b1, b2), y=(b3, b4)),
            id=f"x=({b1}, {b2}), y=({b3}, {b4})",
        )
        for (b1, b2, b3, b4) in set(
            permutations(("periodic", "periodic", "periodic", "unknown"))
        )
    ],
)
def test_unknown_bounds(bound_input):
    with pytest.raises(
        ValueError, match=r"^Unknown (left|right) (x|y) boundary 'unknown'$"
    ):
        bound_input.validate()


def test_multiple_errors():
    with RaisesGroup(
        RaisesExc(ValueError, match=r"^Unknown left x boundary 'unknown'$"),
        RaisesExc(ValueError, match=r"^left y boundary 'periodic' cannot be combined"),
        match="Found multiple issues with boundary specifications",
    ):
        BoundarySet(x=("unknown", "periodic"), y=("periodic", "closed")).validate()

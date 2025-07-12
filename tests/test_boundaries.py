import pytest

from rlic._boundaries import expand_bounds


@pytest.mark.parametrize(
    "bounds_input, expected_output",
    [
        pytest.param("a", {"x": ("a", "a"), "y": ("a", "a")}, id="expand-all"),
        pytest.param(
            {"x": "a", "y": "b"},
            {"x": ("a", "a"), "y": ("b", "b")},
            id="expand-keys",
        ),
        pytest.param(
            {"x": ("a", "b"), "y": ("c", "w")},
            {"x": ("a", "b"), "y": ("c", "w")},
            id="already-expanded",
        ),
        pytest.param(
            {"x": ["a", "b"], "y": ["c", "w"]},
            {"x": ("a", "b"), "y": ("c", "w")},
            id="lists-to-tuples",
        ),
    ],
)
def test_expand_bounds(bounds_input, expected_output):
    """Test the expand_bounds function with various inputs."""
    result = expand_bounds(bounds_input)
    assert result == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        123,  # Invalid type, should raise TypeError
        None,  # None is not a valid input
        ["a", "b"],  # List is not a valid input type
        {"x": "a"},  # Missing 'y' key
        {"y": "b"},  # Missing 'x' key
        {"x": "a", "y": "b", "wdir": "c"},  # extra key not allowed
    ],
)
def test_expand_bounds_invalid_type(invalid_input):
    """Test that expand_bounds raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        expand_bounds(invalid_input)

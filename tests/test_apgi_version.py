"""
Tests for apgi_version.py - semantic versioning module.
"""

from apgi_version import __version__, get_version


def test_version_constant_exists():
    """Test that __version__ constant is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_version_format():
    """Test that version follows semantic versioning format."""
    assert __version__ is not None
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_get_version():
    """Test get_version() returns the version string."""
    assert get_version() == __version__


def test_get_version_returns_string():
    """Test get_version() returns a string."""
    assert isinstance(get_version(), str)


def test_version_not_empty():
    """Test that version is not empty."""
    assert __version__ != ""
    assert get_version() != ""


def test_version_matches_expected_format():
    """Test that version matches semantic versioning format exactly."""
    import re

    # Should match MAJOR.MINOR.PATCH format
    pattern = r"^\d+\.\d+\.\d+$"
    assert re.match(pattern, __version__)


def test_version_major_is_int():
    """Test that major version is a valid integer."""
    major = __version__.split(".")[0]
    assert int(major) >= 0


def test_version_minor_is_int():
    """Test that minor version is a valid integer."""
    minor = __version__.split(".")[1]
    assert int(minor) >= 0


def test_version_patch_is_int():
    """Test that patch version is a valid integer."""
    patch = __version__.split(".")[2]
    assert int(patch) >= 0


def test_get_version_multiple_calls():
    """Test that get_version() returns consistent results."""
    v1 = get_version()
    v2 = get_version()
    v3 = get_version()
    assert v1 == v2 == v3 == __version__

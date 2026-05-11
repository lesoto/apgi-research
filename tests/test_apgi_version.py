"""
Tests for apgi_version.py - semantic versioning module.
"""

from utils.apgi_version import get_version


def test_version_constant_exists():
    """Test that get_version() function works."""
    version = get_version()
    assert version is not None  # nosec: B101 - Test assertion
    assert isinstance(version, str)  # nosec: B101 - Test assertion


def test_version_format():
    """Test that version follows semantic versioning format."""
    version = get_version()
    assert version is not None  # nosec: B101 - Test assertion
    parts = version.split(".")
    assert len(parts) >= 3  # At least major.minor.patch  # nosec: B101 - Test assertion
    assert all(  # nosec: B101 - Test assertion
        part.replace(".", "").replace("-", "").isdigit() or part == "" for part in parts
    )


def test_get_version():
    """Test get_version() returns a version string."""
    version = get_version()
    assert version is not None  # nosec: B101 - Test assertion
    assert isinstance(version, str)  # nosec: B101 - Test assertion


def test_get_version_returns_string():
    """Test get_version() returns a string."""
    assert isinstance(get_version(), str)  # nosec: B101 - Test assertion


def test_version_not_empty():
    """Test that version is not empty."""
    version = get_version()
    assert version != ""  # nosec: B101 - Test assertion
    assert get_version() != ""  # nosec: B101 - Test assertion


def test_version_matches_expected_format():
    """Test that version matches semantic versioning format exactly."""
    import re

    # Should match MAJOR.MINOR.PATCH format
    pattern = r"^\d+\.\d+\.\d+$"
    version = get_version()
    assert re.match(pattern, version)  # nosec: B101 - Test assertion


def test_version_major_is_int():
    """Test that major version is a valid integer."""
    version = get_version()
    major = version.split(".")[0]
    assert int(major) >= 0  # nosec: B101 - Test assertion


def test_version_minor_is_int():
    """Test that minor version is a valid integer."""
    version = get_version()
    minor = version.split(".")[1]
    assert int(minor) >= 0  # nosec: B101 - Test assertion


def test_version_patch_is_int():
    """Test that patch version is a valid integer."""
    version = get_version()
    parts = version.split(".")
    if len(parts) >= 3:
        patch = parts[2]
        assert int(patch) >= 0  # nosec: B101 - Test assertion


def test_get_version_multiple_calls():
    """Test that get_version() returns consistent results."""
    v1 = get_version()
    v2 = get_version()
    v3 = get_version()
    assert v1 == v2 == v3  # nosec: B101 - Test assertion

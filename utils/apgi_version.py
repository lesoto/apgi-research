"""
Semantic versioning for APGI.
Uses importlib.metadata for single source of truth from pyproject.toml.
"""

try:
    from importlib.metadata import version

    __version__ = version("autoresearch")
except ImportError:
    # Fallback for older Python versions
    try:
        import pkg_resources

        __version__ = pkg_resources.get_distribution("autoresearch").version
    except Exception:
        # Ultimate fallback if package is not installed
        __version__ = "0.1.0"


def get_version() -> str:
    """Get version from importlib.metadata (single source of truth)."""
    return __version__

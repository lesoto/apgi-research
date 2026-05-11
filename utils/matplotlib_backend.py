"""
Matplotlib backend management utilities.

Provides context managers and utilities to handle matplotlib backend conflicts
between GUI and non-GUI contexts.
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator, Optional

import matplotlib


def get_safe_backend() -> str:
    """Get appropriate matplotlib backend based on context."""
    # Check if we're in a GUI context
    in_gui_context = (
        os.environ.get("DISPLAY") is not None  # Unix-like systems
        or os.environ.get("WAYLAND_DISPLAY") is not None  # Wayland
        or sys.platform == "win32"  # Windows
        or (
            sys.platform == "darwin" and os.environ.get("SSH_CONNECTION") is None
        )  # macOS not over SSH
    )

    if in_gui_context and not is_headless_environment():
        return "TkAgg"  # GUI backend
    else:
        return "Agg"  # Non-interactive backend


def is_headless_environment() -> bool:
    """Check if we're running in a headless environment."""
    return (
        os.environ.get("DISPLAY") is None
        and os.environ.get("WAYLAND_DISPLAY") is None
        and os.environ.get("SSH_CONNECTION") is not None
        or os.environ.get("CI") == "true"
        or os.environ.get("PYTEST_CURRENT_TEST") is not None
    )


@contextmanager
def matplotlib_backend(backend: Optional[str] = None) -> Generator[str, None, None]:
    """Context manager for temporarily setting matplotlib backend.

    Args:
        backend: Backend to use. If None, uses get_safe_backend().
    """
    if backend is None:
        backend = get_safe_backend()

    # Store current backend
    original_backend = matplotlib.get_backend()

    try:
        # Set new backend
        matplotlib.use(backend, force=True)
        yield backend
    finally:
        # Restore original backend
        matplotlib.use(original_backend, force=True)


@contextmanager
def non_interactive_backend() -> Generator[None, None, None]:
    """Context manager for using non-interactive backend (Agg)."""
    with matplotlib_backend("Agg"):
        yield


@contextmanager
def gui_backend() -> Generator[None, None, None]:
    """Context manager for using GUI backend (TkAgg)."""
    with matplotlib_backend("TkAgg"):
        yield

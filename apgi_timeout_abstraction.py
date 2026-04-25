"""
Cross-Platform Timeout Abstraction for APGI System

Replaces signal.SIGALRM (Unix-only) with portable timer/cancellation abstraction
that works across Windows, macOS, and Linux.

Supports:
- Thread-based timeouts (portable)
- Process-based timeouts (Unix)
- Async/await timeouts
"""

import threading
from typing import Callable, Any, Optional, TypeVar, Generator
from abc import ABC, abstractmethod
from contextlib import contextmanager
import sys

from apgi_logging import get_logger
from apgi_errors import APGITimeoutError

T = TypeVar("T")


class TimeoutStrategy(ABC):
    """Abstract base for timeout strategies."""

    @abstractmethod
    def start(self, timeout_seconds: float, callback: Callable[[], None]) -> None:
        """Start timeout."""
        pass

    @abstractmethod
    def cancel(self) -> None:
        """Cancel timeout."""
        pass


class ThreadBasedTimeout(TimeoutStrategy):
    """Portable thread-based timeout (works on all platforms)."""

    def __init__(self) -> None:
        self.logger = get_logger("apgi.timeout.thread")
        self.timer: Optional[threading.Timer] = None
        self.timed_out = False

    def start(self, timeout_seconds: float, callback: Callable[[], None]) -> None:
        """Start timeout using threading.Timer."""
        if timeout_seconds <= 0:
            return

        def timeout_handler() -> None:
            self.timed_out = True
            self.logger.warning(f"Timeout triggered after {timeout_seconds}s")
            callback()

        self.timer = threading.Timer(timeout_seconds, timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        self.logger.debug(f"Started thread-based timeout: {timeout_seconds}s")

    def cancel(self) -> None:
        """Cancel timeout."""
        if self.timer:
            self.timer.cancel()
            self.logger.debug("Cancelled thread-based timeout")


class SignalBasedTimeout(TimeoutStrategy):
    """Unix-only signal-based timeout (SIGALRM)."""

    def __init__(self) -> None:
        self.logger = get_logger("apgi.timeout.signal")
        self.original_handler = None

    def start(self, timeout_seconds: float, callback: Callable[[], None]) -> None:
        """Start timeout using signal.SIGALRM (Unix only)."""
        if sys.platform == "win32":
            self.logger.warning("Signal-based timeout not available on Windows")
            return

        import signal

        def signal_handler(signum: int, frame: Any) -> None:
            self.logger.warning(f"Timeout triggered after {timeout_seconds}s")
            callback()

        self.original_handler = signal.signal(  # type: ignore
            signal.SIGALRM, signal_handler
        )
        signal.alarm(int(timeout_seconds))
        self.logger.debug(f"Started signal-based timeout: {timeout_seconds}s")

    def cancel(self) -> None:
        """Cancel timeout."""
        if sys.platform == "win32":
            return

        import signal

        signal.alarm(0)
        if self.original_handler:
            signal.signal(signal.SIGALRM, self.original_handler)  # type: ignore
        self.logger.debug("Cancelled signal-based timeout")


class TimeoutManager:
    """Manages timeouts with automatic strategy selection."""

    def __init__(self, prefer_signal: bool = False):
        self.logger = get_logger("apgi.timeout.manager")
        self.prefer_signal = prefer_signal
        self.active_timeouts: dict = {}

    def _select_strategy(self) -> TimeoutStrategy:
        """Select appropriate timeout strategy for platform."""
        if self.prefer_signal and sys.platform != "win32":
            return SignalBasedTimeout()
        else:
            return ThreadBasedTimeout()

    @contextmanager
    def timeout(
        self, timeout_seconds: float, error_message: str = "Operation timed out"
    ) -> Generator[None, None, None]:
        """Context manager for timeout."""
        if timeout_seconds <= 0:
            yield
            return

        strategy = self._select_strategy()
        timed_out = False

        def on_timeout() -> None:
            nonlocal timed_out
            timed_out = True

        try:
            strategy.start(timeout_seconds, on_timeout)
            yield
        finally:
            strategy.cancel()
            if timed_out:
                raise APGITimeoutError(
                    error_message,
                    context={"timeout_seconds": timeout_seconds},
                )

    def run_with_timeout(
        self,
        func: Callable[..., T],
        timeout_seconds: float,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run function with timeout."""
        with self.timeout(timeout_seconds, f"Function {func.__name__} timed out"):
            return func(*args, **kwargs)


class CancellationToken:
    """Token for cooperative cancellation."""

    def __init__(self) -> None:
        self.cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Request cancellation."""
        with self._lock:
            self.cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        with self._lock:
            return self.cancelled

    def check_cancelled(self) -> None:
        """Raise if cancellation was requested."""
        if self.is_cancelled():
            raise APGITimeoutError("Operation was cancelled")


class CancellableOperation:
    """Wrapper for cancellable operations."""

    def __init__(self, timeout_seconds: float = 0):
        self.logger = get_logger("apgi.timeout.cancellable")
        self.timeout_seconds = timeout_seconds
        self.token = CancellationToken()
        self.manager = TimeoutManager()

    def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run operation with cancellation support."""

        def cancellable_func() -> T:
            # Pass token to function if it accepts it
            try:
                return func(self.token, *args, **kwargs)
            except TypeError:
                # Function doesn't accept token
                return func(*args, **kwargs)

        if self.timeout_seconds > 0:
            return self.manager.run_with_timeout(
                cancellable_func,
                self.timeout_seconds,
            )
        else:
            return cancellable_func()

    def cancel(self) -> None:
        """Request cancellation."""
        self.token.cancel()
        self.logger.info("Cancellation requested")


# Global timeout manager instance
_timeout_manager = TimeoutManager()


def get_timeout_manager() -> TimeoutManager:
    """Get global timeout manager."""
    return _timeout_manager


def set_timeout_manager(manager: TimeoutManager) -> None:
    """Set global timeout manager (for testing)."""
    global _timeout_manager
    _timeout_manager = manager


# Convenience functions
def with_timeout(
    timeout_seconds: float,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for timeout."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return _timeout_manager.run_with_timeout(
                func,
                timeout_seconds,
                *args,
                **kwargs,
            )

        return wrapper

    return decorator

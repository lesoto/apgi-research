"""
Comprehensive tests for apgi_timeout_abstraction.py module.
Aiming for 100% code coverage.
"""

import sys
import threading
import time
from unittest.mock import Mock, patch

import pytest

from apgi_timeout_abstraction import (
    APGITimeoutError,
    CancellableOperation,
    CancellationToken,
    SignalBasedTimeout,
    ThreadBasedTimeout,
    TimeoutManager,
    TimeoutStrategy,
    get_timeout_manager,
    set_timeout_manager,
    with_timeout,
)


class TestTimeoutStrategyABC:
    """Test TimeoutStrategy abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that TimeoutStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TimeoutStrategy()  # type: ignore


class TestThreadBasedTimeout:
    """Test ThreadBasedTimeout class."""

    def test_init(self):
        """Test initialization."""
        timeout = ThreadBasedTimeout()
        assert timeout.timer is None
        assert timeout.timed_out is False

    def test_start_timeout(self):
        """Test starting timeout."""
        timeout = ThreadBasedTimeout()
        callback = Mock()

        timeout.start(0.1, callback)

        assert timeout.timer is not None
        assert isinstance(timeout.timer, threading.Timer)
        assert timeout.timer.daemon is True

        # Wait for timeout
        time.sleep(0.15)
        callback.assert_called_once()
        assert timeout.timed_out is True

    def test_start_zero_timeout(self):
        """Test starting with zero timeout."""
        timeout = ThreadBasedTimeout()
        callback = Mock()

        timeout.start(0, callback)

        # Should return immediately without creating timer
        assert timeout.timer is None
        callback.assert_not_called()

    def test_cancel_timeout(self):
        """Test cancelling timeout."""
        timeout = ThreadBasedTimeout()
        callback = Mock()

        timeout.start(1.0, callback)
        timeout.cancel()

        # Wait and verify callback not called
        time.sleep(0.1)
        callback.assert_not_called()

    def test_cancel_no_timer(self):
        """Test cancelling when no timer exists."""
        timeout = ThreadBasedTimeout()

        # Should not raise error
        timeout.cancel()


class TestSignalBasedTimeout:
    """Test SignalBasedTimeout class."""

    def test_init(self):
        """Test initialization."""
        timeout = SignalBasedTimeout()
        assert timeout.original_handler is None

    @pytest.mark.skipif(
        sys.platform == "win32", reason="SIGALRM not available on Windows"
    )
    def test_start_timeout_unix(self):
        """Test starting timeout on Unix."""
        timeout = SignalBasedTimeout()
        callback = Mock()

        with patch("signal.signal") as mock_signal:
            with patch("signal.alarm") as mock_alarm:
                timeout.start(1, callback)

                mock_signal.assert_called_once()
                mock_alarm.assert_called_once_with(1)

    def test_start_timeout_windows(self):
        """Test starting timeout on Windows (should log warning)."""
        with patch("sys.platform", "win32"):
            with patch("apgi_timeout_abstraction.get_logger") as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger

                timeout = SignalBasedTimeout()
                callback = Mock()

                timeout.start(1, callback)

                mock_logger.warning.assert_called_once()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="SIGALRM not available on Windows"
    )
    def test_cancel_timeout_unix(self):
        """Test cancelling timeout on Unix."""
        timeout = SignalBasedTimeout()

        with patch("signal.alarm") as mock_alarm:
            with patch("signal.signal") as mock_signal:
                timeout.original_handler = Mock()  # type: ignore
                timeout.cancel()

                mock_alarm.assert_called_once_with(0)
                mock_signal.assert_called_once()

    def test_cancel_timeout_windows(self):
        """Test cancelling timeout on Windows."""
        timeout = SignalBasedTimeout()

        # Should return without error
        timeout.cancel()


class TestTimeoutManager:
    """Test TimeoutManager class."""

    def test_init_default(self):
        """Test default initialization."""
        manager = TimeoutManager()
        assert manager.prefer_signal is False
        assert manager.active_timeouts == {}

    def test_init_with_signal_preference(self):
        """Test initialization with signal preference."""
        manager = TimeoutManager(prefer_signal=True)
        assert manager.prefer_signal is True

    def test_select_strategy_thread_based(self):
        """Test selecting thread-based strategy."""
        manager = TimeoutManager()
        strategy = manager._select_strategy()

        assert isinstance(strategy, ThreadBasedTimeout)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="SIGALRM not available on Windows"
    )
    def test_select_strategy_signal_based(self):
        """Test selecting signal-based strategy."""
        manager = TimeoutManager(prefer_signal=True)
        strategy = manager._select_strategy()

        assert isinstance(strategy, SignalBasedTimeout)

    def test_select_strategy_windows_fallback(self):
        """Test fallback to thread-based on Windows."""
        with patch("sys.platform", "win32"):
            manager = TimeoutManager(prefer_signal=True)
            strategy = manager._select_strategy()

            # Should fall back to thread-based on Windows
            assert isinstance(strategy, ThreadBasedTimeout)

    def test_timeout_context_manager_success(self):
        """Test timeout context manager with successful completion."""
        manager = TimeoutManager()

        with manager.timeout(1.0, "Operation timed out"):
            # Complete immediately
            pass

    def test_timeout_context_manager_zero_timeout(self):
        """Test timeout context manager with zero timeout."""
        manager = TimeoutManager()

        with manager.timeout(0, "Operation timed out"):
            pass

    def test_timeout_context_manager_actual_timeout(self):
        """Test timeout context manager that actually times out."""
        manager = TimeoutManager()

        with pytest.raises(APGITimeoutError) as exc_info:
            with manager.timeout(0.01, "Operation timed out"):
                time.sleep(0.1)  # Sleep longer than timeout

        assert "Operation timed out" in str(exc_info.value)

    def test_run_with_timeout_success(self):
        """Test running function with timeout successfully."""
        manager = TimeoutManager()

        def test_func(x, y=10):
            return x + y

        result = manager.run_with_timeout(test_func, 1.0, 5, y=5)
        assert result == 10

    def test_run_with_timeout_failure(self):
        """Test running function that times out."""
        manager = TimeoutManager()

        def slow_func():
            time.sleep(0.1)
            return "completed"

        with pytest.raises(APGITimeoutError):
            manager.run_with_timeout(slow_func, 0.01)


class TestCancellationToken:
    """Test CancellationToken class."""

    def test_init(self):
        """Test initialization."""
        token = CancellationToken()
        assert token.cancelled is False

    def test_cancel(self):
        """Test cancellation."""
        token = CancellationToken()
        token.cancel()

        assert token.cancelled is True

    def test_is_cancelled(self):
        """Test checking cancellation status."""
        token = CancellationToken()

        assert token.is_cancelled() is False
        token.cancel()
        assert token.is_cancelled() is True

    def test_check_cancelled_no_exception(self):
        """Test check when not cancelled."""
        token = CancellationToken()

        # Should not raise
        token.check_cancelled()

    def test_check_cancelled_raises(self):
        """Test check when cancelled."""
        token = CancellationToken()
        token.cancel()

        with pytest.raises(APGITimeoutError) as exc_info:
            token.check_cancelled()

        assert "Operation was cancelled" in str(exc_info.value)


class TestCancellableOperation:
    """Test CancellableOperation class."""

    def test_init_with_timeout(self):
        """Test initialization with timeout."""
        op = CancellableOperation(timeout_seconds=5.0)
        assert op.timeout_seconds == 5.0
        assert isinstance(op.token, CancellationToken)
        assert isinstance(op.manager, TimeoutManager)

    def test_init_without_timeout(self):
        """Test initialization without timeout."""
        op = CancellableOperation()
        assert op.timeout_seconds == 0

    def test_run_with_token(self):
        """Test running function that accepts token."""
        op = CancellableOperation()

        def func(token, x):
            return x * 2

        result = op.run(func, 5)
        assert result == 10

    def test_run_without_token(self):
        """Test running function that doesn't accept token."""
        op = CancellableOperation()

        def func(x):
            return x * 2

        result = op.run(func, 5)
        assert result == 10

    def test_run_with_timeout(self):
        """Test running with timeout."""
        op = CancellableOperation(timeout_seconds=1.0)

        def func(token):
            return "success"

        result = op.run(func)
        assert result == "success"

    def test_run_with_timeout_exceeded(self):
        """Test running with timeout that is exceeded."""
        op = CancellableOperation(timeout_seconds=0.01)

        def slow_func(token):
            time.sleep(0.1)
            return "completed"

        with pytest.raises(APGITimeoutError):
            op.run(slow_func)

    def test_cancel(self):
        """Test cancelling operation."""
        op = CancellableOperation()
        op.cancel()

        assert op.token.cancelled is True


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_timeout_manager(self):
        """Test getting global timeout manager."""
        manager = get_timeout_manager()
        assert isinstance(manager, TimeoutManager)

    def test_set_timeout_manager(self):
        """Test setting global timeout manager."""
        new_manager = TimeoutManager()
        set_timeout_manager(new_manager)

        assert get_timeout_manager() is new_manager

    def test_with_timeout_decorator(self):
        """Test with_timeout decorator."""

        @with_timeout(1.0)
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_with_timeout_decorator_failure(self):
        """Test with_timeout decorator with timeout."""

        @with_timeout(0.01)
        def slow_func():
            time.sleep(0.1)
            return "completed"

        with pytest.raises(APGITimeoutError):
            slow_func()

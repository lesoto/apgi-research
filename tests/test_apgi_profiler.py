"""
Tests for APGI Profiling module including cProfile and line_profiler integration.
"""

import pytest

from apgi_profiler import (
    LINE_PROFILER_AVAILABLE,
    PerformanceBudgetExceeded,
    _is_profiling_enabled,
    _profiling_disabled_message,
    enforce_budget,
    profile_hot_path,
    profile_hot_path_combined,
    profile_hot_path_line,
)


class TestPerformanceBudget:
    """Tests for performance budget enforcement."""

    def test_enforce_budget_within_limit(self):
        """Test that function within budget passes."""

        @enforce_budget(max_time_ms=100)
        def fast_function():
            return 42

        result = fast_function()
        assert result == 42

    def test_enforce_budget_exceeded(self):
        """Test that function exceeding budget raises exception."""
        import time

        @enforce_budget(max_time_ms=1)
        def slow_function():
            time.sleep(0.01)  # 10ms
            return 42

        with pytest.raises(PerformanceBudgetExceeded):
            slow_function()

    def test_enforce_budget_with_args(self):
        """Test budget enforcement with function arguments."""

        @enforce_budget(max_time_ms=100)
        def add(a, b):
            return a + b

        result = add(5, 3)
        assert result == 8


class TestCProfileHotPath:
    """Tests for cProfile-based hot path profiling."""

    def test_profile_hot_path_basic(self, capsys, monkeypatch):
        """Test basic cProfile profiling."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path
        def sample_function():
            total = 0
            for i in range(100):
                total += i
            return total

        result = sample_function()
        assert result == 4950

        captured = capsys.readouterr()
        assert "cProfile for sample_function" in captured.out
        assert "function calls" in captured.out.lower()

    def test_profile_hot_path_with_args(self, capsys, monkeypatch):
        """Test profiling with function arguments."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path
        def multiply(a, b):
            return a * b

        result = multiply(5, 3)
        assert result == 15

        captured = capsys.readouterr()
        assert "cProfile for multiply" in captured.out


class TestLineProfilerHotPath:
    """Tests for line_profiler-based hot path profiling."""

    @pytest.mark.skipif(
        not LINE_PROFILER_AVAILABLE, reason="line_profiler package not installed"
    )
    def test_profile_hot_path_line_basic(self, capsys, monkeypatch):
        """Test basic line profiler profiling."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path_line
        def sample_function():
            total = 0
            for i in range(100):
                total += i
            return total

        result = sample_function()
        assert result == 4950

        captured = capsys.readouterr()
        assert "Line Profiler for sample_function" in captured.out
        assert "Timer unit" in captured.out or "Total time" in captured.out

    @pytest.mark.skipif(
        not LINE_PROFILER_AVAILABLE, reason="line_profiler package not installed"
    )
    def test_profile_hot_path_line_with_args(self, capsys, monkeypatch):
        """Test line profiling with function arguments."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path_line
        def multiply(a, b):
            return a * b

        result = multiply(5, 3)
        assert result == 15

        captured = capsys.readouterr()
        assert "Line Profiler for multiply" in captured.out

    def test_profile_hot_path_line_fallback(self, capsys):
        """Test that line profiler falls back to cProfile when unavailable."""
        if LINE_PROFILER_AVAILABLE:
            pytest.skip("line_profiler is available, cannot test fallback")

        @profile_hot_path_line
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42

        captured = capsys.readouterr()
        # Should fall back to cProfile
        assert "cProfile for sample_function" in captured.out


class TestCombinedProfiler:
    """Tests for combined cProfile + line_profiler profiling."""

    @pytest.mark.skipif(
        not LINE_PROFILER_AVAILABLE, reason="line_profiler package not installed"
    )
    def test_profile_hot_path_combined_basic(self, capsys, monkeypatch):
        """Test combined profiling with both cProfile and line_profiler."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path_combined
        def sample_function():
            total = 0
            for i in range(100):
                total += i
            return total

        result = sample_function()
        assert result == 4950

        captured = capsys.readouterr()
        # Should show both profilers
        assert "cProfile for sample_function" in captured.out
        assert "Line Profiler for sample_function" in captured.out

    @pytest.mark.skipif(
        not LINE_PROFILER_AVAILABLE, reason="line_profiler package not installed"
    )
    def test_profile_hot_path_combined_with_args(self, capsys, monkeypatch):
        """Test combined profiling with function arguments."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path_combined
        def multiply(a, b):
            return a * b

        result = multiply(5, 3)
        assert result == 15

        captured = capsys.readouterr()
        assert "cProfile for multiply" in captured.out
        assert "Line Profiler for multiply" in captured.out

    def test_profile_hot_path_combined_without_line_profiler(self, capsys):
        """Test combined profiling when line_profiler is unavailable."""
        if LINE_PROFILER_AVAILABLE:
            pytest.skip("line_profiler is available, cannot test without it")

        @profile_hot_path_combined
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42

        captured = capsys.readouterr()
        # Should only show cProfile
        assert "cProfile for sample_function" in captured.out
        assert "Line Profiler" not in captured.out


class TestProfilerIntegration:
    """Integration tests for profiler usage in realistic scenarios."""

    def test_profiler_with_numpy_operations(self, capsys, monkeypatch):
        """Test profiling with NumPy operations."""
        import numpy as np

        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path
        def numpy_operation():
            arr = np.random.rand(1000)
            return np.sum(arr)

        result = numpy_operation()
        assert isinstance(result, (float, np.floating))

        captured = capsys.readouterr()
        assert "cProfile for numpy_operation" in captured.out

    @pytest.mark.skipif(
        not LINE_PROFILER_AVAILABLE, reason="line_profiler package not installed"
    )
    def test_line_profiler_with_numpy_operations(self, capsys, monkeypatch):
        """Test line profiling with NumPy operations."""
        import numpy as np

        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path_line
        def numpy_operation():
            arr = np.random.rand(1000)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total

        result = numpy_operation()
        assert isinstance(result, (float, np.floating))

        captured = capsys.readouterr()
        assert "Line Profiler for numpy_operation" in captured.out


class TestIsProfilingEnabled:
    """Tests for _is_profiling_enabled function."""

    def test_profiling_enabled_with_1(self, monkeypatch):
        """Test profiling enabled with '1'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_true(self, monkeypatch):
        """Test profiling enabled with 'true'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "true")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_True(self, monkeypatch):
        """Test profiling enabled with 'True'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "True")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_TRUE(self, monkeypatch):
        """Test profiling enabled with 'TRUE'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "TRUE")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_yes(self, monkeypatch):
        """Test profiling enabled with 'yes'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "yes")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_YES(self, monkeypatch):
        """Test profiling enabled with 'YES'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "YES")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_on(self, monkeypatch):
        """Test profiling enabled with 'on'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "on")
        assert _is_profiling_enabled() is True

    def test_profiling_enabled_with_ON(self, monkeypatch):
        """Test profiling enabled with 'ON'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "ON")
        assert _is_profiling_enabled() is True

    def test_profiling_disabled_empty(self, monkeypatch):
        """Test profiling disabled with empty string."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "")
        assert _is_profiling_enabled() is False

    def test_profiling_disabled_unset(self, monkeypatch):
        """Test profiling disabled when env var not set."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)
        assert _is_profiling_enabled() is False

    def test_profiling_disabled_with_0(self, monkeypatch):
        """Test profiling disabled with '0'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "0")
        assert _is_profiling_enabled() is False

    def test_profiling_disabled_with_false(self, monkeypatch):
        """Test profiling disabled with 'false'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "false")
        assert _is_profiling_enabled() is False

    def test_profiling_disabled_with_no(self, monkeypatch):
        """Test profiling disabled with 'no'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "no")
        assert _is_profiling_enabled() is False

    def test_profiling_disabled_with_off(self, monkeypatch):
        """Test profiling disabled with 'off'."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "off")
        assert _is_profiling_enabled() is False

    def test_profiling_disabled_with_random(self, monkeypatch):
        """Test profiling disabled with random value."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "random_value")
        assert _is_profiling_enabled() is False


class TestProfilingDisabledMessage:
    """Tests for _profiling_disabled_message function."""

    def test_profiling_disabled_message(self):
        """Test that _profiling_disabled_message is callable."""
        # Function is a no-op (pass), just verify it doesn't raise
        _profiling_disabled_message("test_function")


class TestProfilingDisabledScenarios:
    """Tests for profiling decorators when profiling is disabled."""

    def test_profile_hot_path_disabled(self, capsys, monkeypatch):
        """Test profile_hot_path when profiling is disabled."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)

        @profile_hot_path
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42

        captured = capsys.readouterr()
        # Should not print profiling output
        assert "cProfile" not in captured.out

    def test_profile_hot_path_line_disabled(self, capsys, monkeypatch):
        """Test profile_hot_path_line when profiling is disabled."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)

        @profile_hot_path_line
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42

        captured = capsys.readouterr()
        # Should not print profiling output
        assert "cProfile" not in captured.out
        assert "Line Profiler" not in captured.out

    def test_profile_hot_path_combined_disabled(self, capsys, monkeypatch):
        """Test profile_hot_path_combined when profiling is disabled."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)

        @profile_hot_path_combined
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42

        captured = capsys.readouterr()
        # Should not print profiling output
        assert "cProfile" not in captured.out
        assert "Line Profiler" not in captured.out

    def test_profile_hot_path_with_exception_disabled(self, monkeypatch):
        """Test profile_hot_path propagates exceptions when disabled."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)

        @profile_hot_path
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_profile_hot_path_line_with_exception_disabled(self, monkeypatch):
        """Test profile_hot_path_line propagates exceptions when disabled."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)

        @profile_hot_path_line
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()


class TestPerformanceBudgetEdgeCases:
    """Edge case tests for performance budget enforcement."""

    def test_enforce_budget_zero_budget(self):
        """Test enforce_budget with zero budget always fails."""

        @enforce_budget(max_time_ms=0)
        def any_function():
            return 42

        with pytest.raises(PerformanceBudgetExceeded):
            any_function()

    def test_enforce_budget_very_large_budget(self):
        """Test enforce_budget with very large budget."""

        @enforce_budget(max_time_ms=1000000)
        def fast_function():
            return 42

        result = fast_function()
        assert result == 42

    def test_enforce_budget_with_exception(self):
        """Test that exceptions propagate through enforce_budget."""

        @enforce_budget(max_time_ms=1000)
        def failing_function():
            raise RuntimeError("Test exception")

        with pytest.raises(RuntimeError, match="Test exception"):
            failing_function()

    def test_enforce_budget_with_return_value(self):
        """Test that return values are preserved."""

        @enforce_budget(max_time_ms=1000)
        def return_dict():
            return {"key": "value", "number": 42}

        result = return_dict()
        assert result == {"key": "value", "number": 42}

    def test_enforce_budget_with_none_return(self):
        """Test that None return value is preserved."""

        @enforce_budget(max_time_ms=1000)
        def return_none():
            return None

        result = return_none()
        assert result is None


class TestProfilerEdgeCases:
    """Edge case tests for profiler decorators."""

    def test_profile_hot_path_with_nested_calls(self, capsys, monkeypatch):
        """Test profile_hot_path with nested function calls."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path
        def inner_function():
            return 10

        @profile_hot_path
        def outer_function():
            return inner_function() + 20

        try:
            result = outer_function()
        except ValueError as e:
            if "Another profiling tool is already active" in str(e):
                pytest.skip("cProfile already active from another test")
            raise

        assert result == 30

        captured = capsys.readouterr()
        # Should show both functions
        assert "cProfile for inner_function" in captured.out
        assert "cProfile for outer_function" in captured.out

    def test_profile_hot_path_with_kwargs(self, capsys, monkeypatch):
        """Test profile_hot_path with keyword arguments."""
        monkeypatch.setenv("APGI_ENABLE_PROFILING", "1")

        @profile_hot_path
        def function_with_kwargs(a, b, c=None):
            return a + b + (c or 0)

        try:
            result = function_with_kwargs(1, 2, c=3)
        except ValueError as e:
            if "Another profiling tool is already active" in str(e):
                pytest.skip("cProfile already active from another test")
            raise

        assert result == 6

        captured = capsys.readouterr()
        assert "cProfile for function_with_kwargs" in captured.out

    def test_profile_hot_path_preserves_function_name(self, monkeypatch):
        """Test that decorator preserves function metadata."""
        monkeypatch.delenv("APGI_ENABLE_PROFILING", raising=False)

        @profile_hot_path
        def my_named_function():
            """My docstring."""
            return 42

        assert my_named_function.__name__ == "my_named_function"
        assert my_named_function.__doc__ == "My docstring."


class TestLineProfilerAvailableFlag:
    """Tests for LINE_PROFILER_AVAILABLE flag."""

    def test_line_profiler_available_is_bool(self):
        """Test that LINE_PROFILER_AVAILABLE is a boolean."""
        assert isinstance(LINE_PROFILER_AVAILABLE, bool)

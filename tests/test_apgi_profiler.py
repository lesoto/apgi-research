"""
Tests for APGI Profiling module including cProfile and line_profiler integration.
"""

import pytest
from apgi_profiler import (
    enforce_budget,
    profile_hot_path,
    profile_hot_path_line,
    profile_hot_path_combined,
    PerformanceBudgetExceeded,
    LINE_PROFILER_AVAILABLE,
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

    def test_profile_hot_path_basic(self, capsys):
        """Test basic cProfile profiling."""

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

    def test_profile_hot_path_with_args(self, capsys):
        """Test profiling with function arguments."""

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
    def test_profile_hot_path_line_basic(self, capsys):
        """Test basic line profiler profiling."""

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
    def test_profile_hot_path_line_with_args(self, capsys):
        """Test line profiling with function arguments."""

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
    def test_profile_hot_path_combined_basic(self, capsys):
        """Test combined profiling with both cProfile and line_profiler."""

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
    def test_profile_hot_path_combined_with_args(self, capsys):
        """Test combined profiling with function arguments."""

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

    def test_profiler_with_numpy_operations(self, capsys):
        """Test profiling with NumPy operations."""
        import numpy as np

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
    def test_line_profiler_with_numpy_operations(self, capsys):
        """Test line profiling with NumPy operations."""
        import numpy as np

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

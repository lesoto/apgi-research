"""
APGI Profiling and Performance Budget module.

Provides both function-level (cProfile) and line-level (line_profiler) profiling
for comprehensive hot path optimization.

All profiling decorators are gated behind APGI_ENABLE_PROFILING environment variable
to prevent performance overhead and noise in production runs.
"""

import cProfile
import io
import os
import pstats
import time
from functools import wraps
from typing import Any, Callable, TypeVar

try:
    from line_profiler import LineProfiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False


F = TypeVar("F", bound=Callable[..., Any])


def _is_profiling_enabled() -> bool:
    """Check if profiling is enabled via environment variable."""
    return os.environ.get("APGI_ENABLE_PROFILING", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _profiling_disabled_message(func_name: str) -> None:
    """Print message when profiling is called but disabled."""
    pass  # Silent by default to avoid production noise


class PerformanceBudgetExceeded(Exception):
    pass


def enforce_budget(max_time_ms: float) -> Callable:
    """
    Decorator to enforce a strict performance budget (max execution time).
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000

            if elapsed_ms > max_time_ms:
                raise PerformanceBudgetExceeded(
                    f"Performance budget exceeded for {func.__name__}: "
                    f"took {elapsed_ms:.2f}ms (budget: {max_time_ms:.2f}ms)"
                )
            return result

        return wrapper

    return decorator


def profile_hot_path(func: F) -> F:
    """
    Decorator to profile hot paths using cProfile (function-level profiling).

    Only activates when APGI_ENABLE_PROFILING environment variable is set.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _is_profiling_enabled():
            return func(*args, **kwargs)

        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(10)  # Print top 10 bottlenecks

        print(f"--- cProfile for {func.__name__} ---")
        print(s.getvalue())
        return result

    return wrapper  # type: ignore[return-value]


def profile_hot_path_line(func: F) -> F:
    """
    Decorator to profile hot paths using line_profiler (line-level profiling).

    This provides line-by-line execution time analysis for detailed optimization.
    Requires the 'line_profiler' package to be installed.
    Only activates when APGI_ENABLE_PROFILING environment variable is set.

    Usage:
        @profile_hot_path_line
        def my_function():
            # code to profile line-by-line
            pass
    """

    if not LINE_PROFILER_AVAILABLE:
        # Fallback to cProfile if line_profiler is not available
        return profile_hot_path(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _is_profiling_enabled():
            return func(*args, **kwargs)

        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)

        s = io.StringIO()
        lp.print_stats(stream=s)

        print(f"--- Line Profiler for {func.__name__} ---")
        print(s.getvalue())
        return result

    return wrapper  # type: ignore[return-value]


def profile_hot_path_combined(func: F) -> F:
    """
    Decorator to profile hot paths using both cProfile and line_profiler.

    Provides comprehensive profiling with both function-level and line-level analysis.
    Requires the 'line_profiler' package to be installed.
    Only activates when APGI_ENABLE_PROFILING environment variable is set.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _is_profiling_enabled():
            return func(*args, **kwargs)

        # Run cProfile first
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(10)

        print(f"--- cProfile for {func.__name__} ---")
        print(s.getvalue())

        # Run line profiler if available
        if LINE_PROFILER_AVAILABLE:
            lp = LineProfiler()
            lp_wrapper = lp(func)
            result = lp_wrapper(*args, **kwargs)

            s_line = io.StringIO()
            lp.print_stats(stream=s_line)

            print(f"--- Line Profiler for {func.__name__} ---")
            print(s_line.getvalue())

        return result

    return wrapper  # type: ignore[return-value]

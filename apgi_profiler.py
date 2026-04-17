"""
APGI Profiling and Performance Budget module.

Provides both function-level (cProfile) and line-level (line_profiler) profiling
for comprehensive hot path optimization.
"""

import cProfile
import pstats
import io
import time
from functools import wraps

try:
    from line_profiler import LineProfiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False


class PerformanceBudgetExceeded(Exception):
    pass


def enforce_budget(max_time_ms: float):
    """
    Decorator to enforce a strict performance budget (max execution time).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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


def profile_hot_path(func):
    """
    Decorator to profile hot paths using cProfile (function-level profiling).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
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

    return wrapper


def profile_hot_path_line(func):
    """
    Decorator to profile hot paths using line_profiler (line-level profiling).

    This provides line-by-line execution time analysis for detailed optimization.
    Requires the 'line_profiler' package to be installed.

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
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)

        s = io.StringIO()
        lp.print_stats(stream=s)

        print(f"--- Line Profiler for {func.__name__} ---")
        print(s.getvalue())
        return result

    return wrapper


def profile_hot_path_combined(func):
    """
    Decorator to profile hot paths using both cProfile and line_profiler.

    Provides comprehensive profiling with both function-level and line-level analysis.
    Requires the 'line_profiler' package to be installed.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
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

    return wrapper

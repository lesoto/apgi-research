"""
APGI Profiling and Performance Budget module.
"""

import cProfile
import pstats
import io
import time
from functools import wraps


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
    Decorator to profile hot paths using cProfile.
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

        print(f"--- Profiling for {func.__name__} ---")
        print(s.getvalue())
        return result

    return wrapper

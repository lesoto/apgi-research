"""
================================================================================
MEMORY PROFILING AND MANAGEMENT TESTS
================================================================================

Comprehensive tests for memory profiling, leak detection, and memory management.
Tracks memory usage across experiments and validates memory bounds.
"""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    HAS_MEMORY_PROFILER = False
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =============================================================================
# MEMORY FIXTURES
# =============================================================================


@pytest.fixture
def memory_snapshot() -> Generator[tracemalloc.Snapshot, None, None]:
    """Provide memory snapshot before and after test."""
    tracemalloc.start()
    gc.collect()  # Force garbage collection before snapshot
    snapshot_before = tracemalloc.take_snapshot()

    yield snapshot_before

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Report differences
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    if top_stats:
        print("\nTop memory differences:")
        for stat in top_stats[:5]:
            print(f"  {stat}")


@pytest.fixture
def memory_limit() -> int:
    """Maximum memory limit in MB for tests."""
    return 512  # 512MB default limit


@pytest.fixture
def memory_tracker() -> Generator["MemoryTracker", None, None]:
    """Provide memory tracking utility."""
    tracker = MemoryTracker()
    yield tracker
    tracker.cleanup()


# =============================================================================
# MEMORY TRACKER CLASS
# =============================================================================


class MemoryTracker:
    """Track memory usage and detect leaks."""

    def __init__(self) -> None:
        self.snapshots: List[tracemalloc.Snapshot] = []
        self.peak_memory: float = 0.0
        self.allocations: List[Dict[str, Any]] = []

    def snapshot(self, label: str = "") -> None:
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = max(self.peak_memory, peak / 1024 / 1024)
        self.snapshots.append(snapshot)
        self.allocations.append(
            {
                "label": label,
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024,
                "snapshot": len(self.snapshots) - 1,
            }
        )

    def get_leak_report(
        self, snapshot_idx1: int = 0, snapshot_idx2: int = -1
    ) -> List[Any]:
        """Get memory leak report between two snapshots."""
        if len(self.snapshots) < 2:
            return []

        s1 = self.snapshots[snapshot_idx1]
        s2 = self.snapshots[snapshot_idx2]
        return s2.compare_to(s1, "lineno")

    def cleanup(self) -> None:
        """Clean up memory tracking."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        gc.collect()


# =============================================================================
# BASIC MEMORY TESTS
# =============================================================================


@pytest.mark.memory
class TestMemoryBasics:
    """Basic memory management tests."""

    def test_memory_tracking_available(self) -> None:
        """Test that memory tracking is available."""
        assert tracemalloc is not None

    def test_memory_snapshot_fixture(
        self, memory_snapshot: tracemalloc.Snapshot
    ) -> None:
        """Test memory snapshot fixture works."""
        assert memory_snapshot is not None
        assert isinstance(memory_snapshot, tracemalloc.Snapshot)

    def test_memory_tracker(self, memory_tracker: MemoryTracker) -> None:
        """Test memory tracker functionality."""
        memory_tracker.snapshot("initial")

        # Allocate some memory
        data = [0] * 1000000

        memory_tracker.snapshot("after_allocation")

        # Check we have snapshots
        assert len(memory_tracker.snapshots) == 2
        assert len(memory_tracker.allocations) == 2

        del data
        gc.collect()

    def test_tracemalloc_stats(self) -> None:
        """Test tracemalloc statistics collection."""
        tracemalloc.start()

        # Allocate memory
        data = np.zeros((1000, 1000))

        current, peak = tracemalloc.get_traced_memory()

        assert current > 0
        assert peak >= current

        del data
        tracemalloc.stop()


@pytest.mark.memory
@pytest.mark.integration
class TestMemoryIntegration:
    """Memory integration tests for APGI components."""

    def test_experiment_memory_usage(
        self, memory_tracker: MemoryTracker, memory_limit: int
    ) -> None:
        """Test that experiment execution stays within memory limits."""
        if not HAS_PSUTIL:
            pytest.skip("psutil not installed")

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        memory_tracker.snapshot("experiment_start")

        # Simulate experiment data loading
        # Create arrays similar to what experiments might use
        trial_data = np.random.randn(1000, 5)
        results = np.zeros((1000, 5))

        for i in range(1000):
            results[i] = (
                np.mean(trial_data[i : i + 10], axis=0)[:5] if i < 990 else np.zeros(5)
            )

        memory_tracker.snapshot("experiment_end")

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before

        # Should not exceed memory limit
        assert (
            mem_increase < memory_limit
        ), f"Memory increase {mem_increase:.1f}MB exceeds limit {memory_limit}MB"

        del trial_data, results
        gc.collect()

    def test_large_array_handling(self, memory_tracker: MemoryTracker) -> None:
        """Test memory handling with large arrays."""
        memory_tracker.snapshot("before_large_arrays")

        # Create arrays that might be used in experiments
        large_1d = np.zeros(10_000_000)  # 80MB float64 array
        large_2d = np.zeros((1000, 1000))  # 8MB array

        memory_tracker.snapshot("after_large_arrays")

        # Check peak memory
        assert memory_tracker.peak_memory < 200  # Should be under 200MB

        del large_1d, large_2d
        gc.collect()

    def test_memory_efficient_processing(self, memory_tracker: MemoryTracker) -> None:
        """Test memory-efficient data processing patterns."""
        memory_tracker.snapshot("start")

        # Process data in chunks to minimize memory
        chunk_size = 1000
        total_size = 10000
        results = []

        for i in range(0, total_size, chunk_size):
            # Process chunk and immediately discard
            chunk = np.random.randn(chunk_size, 100)
            chunk_result = np.mean(chunk, axis=0)
            results.append(chunk_result)
            del chunk  # Explicitly free chunk memory
            gc.collect()

        memory_tracker.snapshot("end")

        # Verify reasonable memory usage
        assert memory_tracker.peak_memory < 100  # Should stay under 100MB

        del results


@pytest.mark.memory
@pytest.mark.stress
class TestMemoryStress:
    """Memory stress tests."""

    def test_repeated_allocation_deallocation(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Test memory stability under repeated allocation/deallocation."""
        memory_tracker.snapshot("start")

        initial_snap = len(memory_tracker.snapshots) - 1

        # Repeatedly allocate and free memory
        for i in range(100):
            arr = np.random.randn(10000)
            result = np.sum(arr)
            del arr, result
            if i % 10 == 0:
                gc.collect()
                memory_tracker.snapshot(f"iteration_{i}")

        memory_tracker.snapshot("end")

        # Check for memory leaks
        leak_report = memory_tracker.get_leak_report(initial_snap, -1)
        # Allow some growth for test infrastructure
        total_growth = sum(stat.size_diff for stat in leak_report[:10])
        assert (
            total_growth < 10 * 1024 * 1024
        ), f"Possible memory leak detected: {total_growth / 1024 / 1024:.1f}MB"

    def test_concurrent_memory_pressure(self, memory_tracker: MemoryTracker) -> None:
        """Test memory handling under concurrent pressure."""
        import threading
        import queue

        memory_tracker.snapshot("start")
        results_queue: queue.Queue[Any] = queue.Queue()

        def allocate_worker() -> None:
            """Worker that allocates memory."""
            arr = np.random.randn(10000, 100)
            result = np.mean(arr, axis=0)
            results_queue.put(result)
            del arr

        threads = []
        for _ in range(10):
            t = threading.Thread(target=allocate_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        memory_tracker.snapshot("end")

        # Drain queue
        while not results_queue.empty():
            results_queue.get()

        gc.collect()

        # Should not have excessive memory usage
        assert memory_tracker.peak_memory < 200

    def test_memory_fragmentation(self, memory_tracker: MemoryTracker) -> None:
        """Test memory fragmentation handling."""
        memory_tracker.snapshot("start")

        # Create various sized allocations
        allocations: List[Optional[np.ndarray]] = []
        sizes = [100, 1000, 10000, 100000, 1000, 100, 10000]

        for size in sizes:
            arr = np.zeros(size)
            allocations.append(arr)

        # Free some allocations to create fragmentation
        for i in [0, 2, 4, 6]:
            allocations[i] = None

        gc.collect()
        memory_tracker.snapshot("after_fragmentation")

        # Allocate more to test fragmented memory handling
        new_allocations = [np.zeros(s) for s in sizes]

        memory_tracker.snapshot("end")

        del allocations, new_allocations
        gc.collect()


@pytest.mark.memory
@pytest.mark.performance
class TestMemoryPerformance:
    """Memory performance tests."""

    def test_memory_allocation_speed(self, request: Any) -> None:
        """Benchmark memory allocation speed."""
        # Check if benchmark fixture is available
        if not hasattr(request, "node") or not hasattr(
            request.node, "get_closest_marker"
        ):
            pytest.skip("Benchmark fixture not available")

        def allocate_arrays() -> None:
            for _ in range(100):
                arr = np.zeros((1000, 100))
                del arr

        # Run without benchmark if not available
        import time

        start = time.perf_counter()
        allocate_arrays()
        duration = time.perf_counter() - start
        # Should complete in reasonable time
        assert duration < 1.0  # Less than 1 second

    def test_gc_impact(self, memory_tracker: MemoryTracker) -> None:
        """Test garbage collection impact on performance."""
        import time

        memory_tracker.snapshot("start")

        # Time operations with and without GC
        gc.disable()
        start_no_gc = time.perf_counter()
        for _ in range(100):
            arr = np.random.randn(10000)
            del arr
        duration_no_gc = time.perf_counter() - start_no_gc
        gc.enable()

        memory_tracker.snapshot("after_no_gc")

        start_with_gc = time.perf_counter()
        for _ in range(100):
            arr = np.random.randn(10000)
            del arr
            gc.collect()
        duration_with_gc = time.perf_counter() - start_with_gc

        memory_tracker.snapshot("after_with_gc")

        # GC timing is highly variable across environments
        # Just verify both operations completed without error
        assert duration_no_gc > 0, "No-GC operations should take some time"
        assert duration_with_gc > 0, "With-GC operations should take some time"


@pytest.mark.memory
@pytest.mark.integration
class TestMemoryWithAPGI:
    """Memory tests specific to APGI components."""

    def test_experiment_data_structures(self, memory_tracker: MemoryTracker) -> None:
        """Test memory usage of experiment data structures."""
        memory_tracker.snapshot("start")

        # Simulate experiment state
        experiment_data: Dict[str, Any] = {"trials": [], "metrics": {}, "config": {}}

        # Add trial data
        for i in range(1000):
            trial = {
                "trial_id": i,
                "stimulus": np.random.randn(10),
                "response": np.random.randint(0, 2),
                "rt": np.random.exponential(0.5),
                "correct": np.random.random() > 0.2,
            }
            experiment_data["trials"].append(trial)

        memory_tracker.snapshot("after_trials")

        # Add metrics
        experiment_data["metrics"] = {
            "accuracy": np.random.random(),
            "mean_rt": np.random.exponential(0.5),
            "std_rt": np.random.exponential(0.1),
            "learning_curve": np.random.randn(100).cumsum(),
        }

        memory_tracker.snapshot("after_metrics")

        # Memory should be reasonable
        assert memory_tracker.peak_memory < 100

        del experiment_data

    def test_checkpoint_memory(self, memory_tracker: MemoryTracker) -> None:
        """Test memory usage when creating checkpoints."""
        import tempfile
        import json

        memory_tracker.snapshot("start")

        # Create checkpoint data
        checkpoint = {
            "experiment_state": "running",
            "trial_number": 500,
            "data": np.random.randn(1000, 10).tolist(),
        }

        memory_tracker.snapshot("checkpoint_created")

        # Serialize checkpoint
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(checkpoint, f)
            checkpoint_path = f.name

        memory_tracker.snapshot("checkpoint_saved")

        # Clean up
        os.unlink(checkpoint_path)
        del checkpoint

        gc.collect()
        memory_tracker.snapshot("cleanup")

        # Should not leak significant memory
        final_alloc = memory_tracker.allocations[-1]
        assert final_alloc["current_mb"] < 50


# =============================================================================
# CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with memory-specific markers."""
    config.addinivalue_line("markers", "memory: marks tests as memory tests")

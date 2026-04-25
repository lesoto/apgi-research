"""
================================================================================
APGI BENCHMARKS: Microbenchmarks and End-to-End Throughput Tests
================================================================================

This module provides comprehensive benchmarks for:
- process_trial / process_trials microbenchmarks
- End-to-end throughput benchmarks per experiment family
- Batch processing performance comparison
- APGI integration overhead measurement

Usage:
    pytest tests/test_apgi_benchmarks.py -v
    pytest tests/test_apgi_benchmarks.py -m benchmark --benchmark-only

Output:
    Performance metrics with throughput (trials/sec) and latency (ms/trial)
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# APGI imports
from apgi_integration import APGIIntegration, APGIParameters
from standard_apgi_runner import StandardAPGIRunner
from experiment_apgi_integration import ExportedAPGIParams

# Import experiment runners for end-to-end benchmarks
EXPERIMENT_RUNNERS: Dict[str, Tuple[str, str]] = {
    "attention": ("run_attentional_blink", "AttentionalBlinkRunner"),
    "inhibition": ("run_go_no_go", "GoNoGoRunner"),
    "interference": ("run_stroop_effect", "StroopRunner"),
    "masking": ("run_masking", "MaskingRunner"),
    "learning": ("run_artificial_grammar_learning", "AGLRunner"),
    "memory": ("run_sternberg_memory", "SternbergRunner"),
    "navigation": ("run_virtual_navigation", "VirtualNavigationRunner"),
    "perception": ("run_change_blindness", "ChangeBlindnessRunner"),
    "timing": ("run_time_estimation", "TimeEstimationRunner"),
    "visual_search": ("run_visual_search", "VisualSearchRunner"),
}


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    trials_count: int
    total_time_ms: float
    trials_per_second: float
    ms_per_trial: float
    memory_delta_mb: float

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.trials_per_second:,.0f} trials/sec, "
            f"{self.ms_per_trial:.3f} ms/trial, "
            f"{self.total_time_ms:.1f}ms total ({self.trials_count} trials)"
        )


@pytest.fixture
def benchmark_runner() -> Callable[..., BenchmarkResult]:
    """Factory fixture for creating benchmark runners."""

    def _run_benchmark(
        name: str,
        func: Callable[[], Any],
        trials_count: int,
        setup: Optional[Callable[[], None]] = None,
    ) -> BenchmarkResult:
        """Run a benchmark and collect metrics."""
        # Setup phase
        if setup:
            setup()

        # Force GC before measurement
        gc.collect()
        gc.collect()

        # Memory baseline
        baseline_mem = _get_memory_usage()

        # Time measurement
        start = time.perf_counter()
        func()
        end = time.perf_counter()

        total_time_ms = (end - start) * 1000
        memory_delta = _get_memory_usage() - baseline_mem

        return BenchmarkResult(
            name=name,
            trials_count=trials_count,
            total_time_ms=total_time_ms,
            trials_per_second=trials_count / (total_time_ms / 1000),
            ms_per_trial=total_time_ms / trials_count,
            memory_delta_mb=memory_delta,
        )

    return _run_benchmark


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# =============================================================================
# MICROBENCHMARKS: process_trial vs process_trials
# =============================================================================


class TestProcessTrialMicrobenchmarks:
    """Microbenchmarks for individual trial processing functions."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_trials", [100, 1000, 10000])
    def test_apgi_integration_process_trial_sequential(
        self, benchmark_runner: Callable[..., BenchmarkResult], num_trials: int
    ) -> None:
        """Benchmark APGIIntegration.process_trial sequential processing."""
        apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
        observed = np.random.uniform(400, 800, num_trials)
        predicted = np.random.uniform(400, 800, num_trials)

        def run() -> None:
            apgi.reset()
            for i in range(num_trials):
                apgi.process_trial(
                    observed=float(observed[i]),
                    predicted=float(predicted[i]),
                    trial_type="neutral",
                )

        result = benchmark_runner(
            f"APGIIntegration.process_trial_seq_{num_trials}",
            run,
            num_trials,
            setup=lambda: apgi.reset(),
        )

        print(f"\n{result}")
        # Assert reasonable performance
        assert result.ms_per_trial < 10, f"Too slow: {result.ms_per_trial:.3f} ms/trial"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_trials", [100, 1000, 10000])
    def test_apgi_integration_process_trials_batch(
        self, benchmark_runner: Callable[..., BenchmarkResult], num_trials: int
    ) -> None:
        """Benchmark APGIIntegration.process_trials batch processing."""
        apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
        observed = np.random.uniform(400, 800, num_trials)
        predicted = np.random.uniform(400, 800, num_trials)

        def run() -> None:
            apgi.reset()
            apgi.process_trials(
                observed=observed,
                predicted=predicted,
                trial_type="neutral",
            )

        result = benchmark_runner(
            f"APGIIntegration.process_trials_batch_{num_trials}",
            run,
            num_trials,
            setup=lambda: apgi.reset(),
        )

        print(f"\n{result}")
        assert result.ms_per_trial < 5, f"Too slow: {result.ms_per_trial:.3f} ms/trial"

    @pytest.mark.benchmark
    def test_batch_vs_sequential_speedup(self) -> None:
        """Compare batch processing speedup vs sequential."""
        num_trials = 5000
        apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
        observed = np.random.uniform(400, 800, num_trials)
        predicted = np.random.uniform(400, 800, num_trials)

        # Sequential timing
        gc.collect()
        apgi.reset()
        start_seq = time.perf_counter()
        for i in range(num_trials):
            apgi.process_trial(
                observed=float(observed[i]),
                predicted=float(predicted[i]),
                trial_type="neutral",
            )
        seq_time_ms = (time.perf_counter() - start_seq) * 1000

        # Batch timing
        gc.collect()
        apgi.reset()
        start_batch = time.perf_counter()
        apgi.process_trials(
            observed=observed,
            predicted=predicted,
            trial_type="neutral",
        )
        batch_time_ms = (time.perf_counter() - start_batch) * 1000

        speedup = seq_time_ms / batch_time_ms
        print(f"\nBatch vs Sequential: {speedup:.2f}x speedup")
        print(
            f"  Sequential: {seq_time_ms:.1f}ms ({num_trials / seq_time_ms * 1000:.0f} trials/sec)"
        )
        print(
            f"  Batch: {batch_time_ms:.1f}ms ({num_trials / batch_time_ms * 1000:.0f} trials/sec)"
        )

        # Batch should be significantly faster
        assert (
            speedup > 1.5
        ), f"Batch processing should be >1.5x faster, got {speedup:.2f}x"


class TestStandardAPGIRunnerMicrobenchmarks:
    """Microbenchmarks for StandardAPGIRunner methods."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_trials", [100, 1000])
    def test_standard_runner_process_trial_full_apgi(
        self, benchmark_runner: Callable[..., BenchmarkResult], num_trials: int
    ) -> None:
        """Benchmark StandardAPGIRunner.process_trial_with_full_apgi."""
        from unittest.mock import MagicMock

        mock_runner = MagicMock()
        mock_runner.experiment_name = "benchmark"

        apgi_params = ExportedAPGIParams(
            experiment_name="benchmark",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        runner = StandardAPGIRunner(
            base_runner=mock_runner,
            experiment_name="benchmark",
            apgi_params=apgi_params,
            enable_hierarchical=True,
            enable_precision_gap=True,
        )

        observed = np.random.uniform(400, 800, num_trials)
        predicted = np.random.uniform(400, 800, num_trials)

        def run() -> None:
            runner.trial_count = 0
            runner.apgi_metrics_history.clear()
            runner.apgi.reset() if runner.apgi else None
            for i in range(num_trials):
                runner.process_trial_with_full_apgi(
                    observed=float(observed[i]),
                    predicted=float(predicted[i]),
                    trial_type="neutral",
                    hierarchical_level=(i % 5) + 1,
                )

        result = benchmark_runner(
            f"StandardAPGIRunner.process_trial_full_{num_trials}",
            run,
            num_trials,
        )

        print(f"\n{result}")
        assert result.ms_per_trial < 15, f"Too slow: {result.ms_per_trial:.3f} ms/trial"


# =============================================================================
# END-TO-END THROUGHPUT BENCHMARKS
# =============================================================================


class TestEndToEndThroughput:
    """End-to-end throughput benchmarks per experiment family."""

    @pytest.mark.benchmark
    def test_experiment_runner_attentional_blink(
        self, benchmark_runner: Callable[..., BenchmarkResult]
    ) -> None:
        """Benchmark attentional blink experiment throughput."""
        try:
            from run_attentional_blink import (
                EnhancedAttentionalBlinkRunner as AttentionalBlinkRunner,
            )
        except ImportError:
            pytest.skip("run_attentional_blink not available")

        runner = AttentionalBlinkRunner()

        def run() -> None:
            runner.reset()
            runner.run_experiment()

        result = benchmark_runner(
            "AttentionalBlink",
            run,
            trials_count=1,
        )

        print(f"\n{result}")
        assert (
            result.ms_per_trial < 150
        ), f"Too slow: {result.ms_per_trial:.1f} ms/trial"

    @pytest.mark.benchmark
    def test_experiment_runner_go_no_go(
        self, benchmark_runner: Callable[..., BenchmarkResult]
    ) -> None:
        """Benchmark Go/No-Go experiment throughput."""
        try:
            from run_go_no_go import EnhancedGoNoGoRunner as GoNoGoRunner
        except ImportError:
            pytest.skip("run_go_no_go not available")

        runner = GoNoGoRunner()

        def run() -> None:
            runner.reset()
            runner.run_experiment()

        result = benchmark_runner(
            "GoNoGo",
            run,
            trials_count=1,
        )

        print(f"\n{result}")
        assert result.ms_per_trial < 100

    @pytest.mark.benchmark
    def test_experiment_runner_stroop(
        self, benchmark_runner: Callable[..., BenchmarkResult]
    ) -> None:
        """Benchmark Stroop effect experiment throughput."""
        try:
            from run_stroop_effect import EnhancedStroopRunner as StroopRunner
        except ImportError:
            pytest.skip("run_stroop_effect not available")

        runner = StroopRunner()

        def run() -> None:
            runner.reset()
            runner.run_experiment()

        result = benchmark_runner(
            "Stroop",
            run,
            trials_count=1,
        )

        print(f"\n{result}")
        assert result.ms_per_trial < 100

    @pytest.mark.benchmark
    def test_experiment_runner_sternberg(
        self, benchmark_runner: Callable[..., BenchmarkResult]
    ) -> None:
        """Benchmark Sternberg memory experiment throughput."""
        try:
            from run_sternberg_memory import EnhancedSternbergRunner as SternbergRunner
        except ImportError:
            pytest.skip("run_sternberg_memory not available")

        runner = SternbergRunner()

        def run() -> None:
            runner.reset()
            runner.run_experiment()

        result = benchmark_runner(
            "Sternberg",
            run,
            trials_count=1,
        )

        print(f"\n{result}")
        assert result.ms_per_trial < 100


# =============================================================================
# COMPARATIVE BENCHMARKS
# =============================================================================


class TestComparativeBenchmarks:
    """Comparative benchmarks showing improvements."""

    @pytest.mark.benchmark
    def test_apgi_overhead_measurement(self) -> None:
        """Measure APGI integration overhead."""
        num_trials = 1000

        # Baseline: simple computation without APGI
        observed = np.random.uniform(400, 800, num_trials)
        predicted = np.random.uniform(400, 800, num_trials)

        gc.collect()
        start_baseline = time.perf_counter()
        for i in range(num_trials):
            _ = abs(observed[i] - predicted[i])  # Simple error computation
        baseline_time_ms = (time.perf_counter() - start_baseline) * 1000

        # With APGI batch processing
        apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
        gc.collect()
        apgi.reset()
        start_apgi = time.perf_counter()
        apgi.process_trials(
            observed=observed, predicted=predicted, trial_type="neutral"
        )
        apgi_time_ms = (time.perf_counter() - start_apgi) * 1000

        # With APGI sequential (legacy)
        apgi.reset()
        gc.collect()
        start_seq = time.perf_counter()
        for i in range(num_trials):
            apgi.process_trial(
                observed=float(observed[i]),
                predicted=float(predicted[i]),
                trial_type="neutral",
            )
        seq_time_ms = (time.perf_counter() - start_seq) * 1000

        overhead_baseline = (apgi_time_ms / baseline_time_ms) - 1
        overhead_vs_seq = (seq_time_ms / apgi_time_ms) - 1

        print("\n=== APGI Overhead Analysis ===")
        print(f"Baseline (no APGI): {baseline_time_ms:.2f}ms")
        print(
            f"APGI batch: {apgi_time_ms:.2f}ms ({overhead_baseline * 100:.1f}% overhead vs baseline)"
        )
        print(f"APGI sequential: {seq_time_ms:.2f}ms")
        print(f"Batch speedup: {overhead_vs_seq * 100:.1f}% faster than sequential")

        # Batch should be within reasonable overhead of baseline (100x allows for CI variability)
        assert apgi_time_ms < baseline_time_ms * 100, "APGI batch overhead too high"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("trial_types", [["neutral"], ["neutral", "survival"]])
    def test_trial_type_performance_variation(self, trial_types: List[str]) -> None:
        """Benchmark performance across different trial types."""
        num_trials = 500
        num_per_type = num_trials // len(trial_types)

        apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)

        results: Dict[str, BenchmarkResult] = {}
        for trial_type in trial_types:
            observed = np.random.uniform(400, 800, num_per_type)
            predicted = np.random.uniform(400, 800, num_per_type)

            gc.collect()
            apgi.reset()
            start = time.perf_counter()
            apgi.process_trials(
                observed=observed,
                predicted=predicted,
                trial_type=trial_type,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            results[trial_type] = BenchmarkResult(
                name=f"trial_type_{trial_type}",
                trials_count=num_per_type,
                total_time_ms=elapsed_ms,
                trials_per_second=num_per_type / (elapsed_ms / 1000),
                ms_per_trial=elapsed_ms / num_per_type,
                memory_delta_mb=0.0,
            )

        print("\n=== Trial Type Performance ===")
        for trial_type, result in results.items():
            print(f"  {trial_type}: {result.ms_per_trial:.3f} ms/trial")


# =============================================================================
# SCALABILITY BENCHMARKS
# =============================================================================


class TestScalabilityBenchmarks:
    """Scalability benchmarks for different load levels."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_trials", [100, 1000, 5000, 10000])
    def test_batch_scalability(self, num_trials: int) -> None:
        """Test how batch processing scales with trial count."""
        apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
        observed = np.random.uniform(400, 800, num_trials)
        predicted = np.random.uniform(400, 800, num_trials)

        gc.collect()
        apgi.reset()
        start = time.perf_counter()
        apgi.process_trials(observed=observed, predicted=predicted)
        elapsed_ms = (time.perf_counter() - start) * 1000

        tps = num_trials / (elapsed_ms / 1000)
        ms_per = elapsed_ms / num_trials

        print(f"\nBatch scalability ({num_trials} trials):")
        print(f"  Throughput: {tps:,.0f} trials/sec")
        print(f"  Latency: {ms_per:.4f} ms/trial")
        print(f"  Total time: {elapsed_ms:.1f}ms")

        # Performance should scale roughly linearly
        assert ms_per < 5, f"Degraded performance at scale: {ms_per:.3f} ms/trial"

    @pytest.mark.benchmark
    def test_memory_scaling(self) -> None:
        """Test memory usage scaling with trial count."""
        trial_counts = [100, 1000, 5000]
        memory_per_trial: List[float] = []

        for num_trials in trial_counts:
            apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
            observed = np.random.uniform(400, 800, num_trials)
            predicted = np.random.uniform(400, 800, num_trials)

            gc.collect()
            gc.collect()
            mem_before = _get_memory_usage()

            apgi.process_trials(observed=observed, predicted=predicted)

            mem_after = _get_memory_usage()
            mem_per_trial = (mem_after - mem_before) / num_trials
            memory_per_trial.append(mem_per_trial)

            print(f"  {num_trials} trials: {mem_per_trial:.4f} MB/trial")

        # Memory per trial should be roughly constant
        max_variation = max(memory_per_trial) - min(memory_per_trial)
        print(f"\nMemory variation: {max_variation:.4f} MB/trial")


# =============================================================================
# SUMMARY BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("num_trials", [100, 1000])
def test_full_benchmark_summary(num_trials: int) -> None:
    """Generate full benchmark summary report."""
    print("\n" + "=" * 70)
    print(f"APGI BENCHMARK SUMMARY ({num_trials} trials)")
    print("=" * 70)
    # Test 1: Sequential processing
    apgi = APGIIntegration(params=APGIParameters(), enable_neuromodulators=True)
    observed = np.random.uniform(400, 800, num_trials)
    predicted = np.random.uniform(400, 800, num_trials)

    gc.collect()
    apgi.reset()
    start = time.perf_counter()
    for i in range(num_trials):
        apgi.process_trial(
            observed=float(observed[i]),
            predicted=float(predicted[i]),
            trial_type="neutral",
        )
    seq_time = (time.perf_counter() - start) * 1000

    # Test 2: Batch processing
    gc.collect()
    apgi.reset()
    start = time.perf_counter()
    apgi.process_trials(observed=observed, predicted=predicted, trial_type="neutral")
    batch_time = (time.perf_counter() - start) * 1000

    # Print summary
    print("\nProcessing Performance:")
    print(
        f"  Sequential: {seq_time:.1f}ms ({num_trials / seq_time * 1000:.0f} trials/sec)"
    )
    print(
        f"  Batch: {batch_time:.1f}ms ({num_trials / batch_time * 1000:.0f} trials/sec)"
    )
    print(f"  Speedup: {seq_time / batch_time:.2f}x")

    # Verify batch is faster
    assert batch_time < seq_time, "Batch should be faster than sequential"
    assert batch_time / num_trials < 5, "Should complete within 5ms per trial"

    print(f"{'=' * 70}")

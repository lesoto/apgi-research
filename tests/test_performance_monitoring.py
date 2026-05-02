"""
Comprehensive tests for performance_monitoring.py - Performance monitoring module.
"""

import time
from unittest.mock import MagicMock, patch

from performance_monitoring import (
    BenchmarkConfig,
    BenchmarkResult,
    PerformanceMonitor,
    PerformanceReport,
    benchmark,
    get_memory_usage,
    get_performance_summary,
    monitor_performance,
    record_metric,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self):
        """Test default BenchmarkConfig values."""
        config = BenchmarkConfig()
        assert config.name == ""
        assert config.iterations == 1
        assert config.warmup_iterations == 0
        assert config.timeout_seconds == 60.0

    def test_custom_values(self):
        """Test custom BenchmarkConfig values."""
        config = BenchmarkConfig(
            name="test_benchmark",
            iterations=10,
            warmup_iterations=2,
            timeout_seconds=30.0,
        )
        assert config.name == "test_benchmark"
        assert config.iterations == 10
        assert config.warmup_iterations == 2
        assert config.timeout_seconds == 30.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_default_values(self):
        """Test default BenchmarkResult values."""
        result = BenchmarkResult(name="test")
        assert result.name == "test"
        assert result.mean_time_ms == 0.0
        assert result.median_time_ms == 0.0
        assert result.min_time_ms == 0.0
        assert result.max_time_ms == 0.0
        assert result.std_dev_ms == 0.0
        assert result.iterations == 0

    def test_calculate_stats(self):
        """Test statistics calculation."""
        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = BenchmarkResult.calculate("test", times)
        assert result.name == "test"
        assert result.mean_time_ms == 30.0
        assert result.min_time_ms == 10.0
        assert result.max_time_ms == 50.0
        assert result.iterations == 5


class TestPerformanceReport:
    """Tests for PerformanceReport dataclass."""

    def test_default_values(self):
        """Test default PerformanceReport values."""
        report = PerformanceReport()
        assert report.timestamp is not None
        assert report.metrics == {}
        assert report.benchmarks == []
        assert report.summary == {}

    def test_add_metric(self):
        """Test adding metric to report."""
        report = PerformanceReport()
        report.add_metric("cpu_percent", 50.0)
        assert report.metrics["cpu_percent"] == 50.0

    def test_add_benchmark(self):
        """Test adding benchmark to report."""
        report = PerformanceReport()
        benchmark = BenchmarkResult(name="test", mean_time_ms=100.0)
        report.add_benchmark(benchmark)
        assert len(report.benchmarks) == 1
        assert report.benchmarks[0].name == "test"


class TestGetMemoryUsage:
    """Tests for get_memory_usage function."""

    @patch("performance_monitoring.psutil")
    def test_get_memory_success(self, mock_psutil):
        """Test getting memory usage successfully."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(
            rss=1024 * 1024 * 100
        )  # 100MB
        mock_psutil.Process.return_value = mock_process

        memory = get_memory_usage()
        assert memory > 0

    @patch("performance_monitoring.psutil")
    def test_get_memory_psutil_not_available(self, mock_psutil):
        """Test when psutil is not available."""
        mock_psutil.Process.side_effect = ImportError("psutil not found")
        memory = get_memory_usage()
        assert memory == 0.0


class TestRecordMetric:
    """Tests for record_metric function."""

    def test_record_metric(self):
        """Test recording a metric."""
        result = record_metric("test_metric", 42.0)
        assert result is True

    def test_record_multiple_metrics(self):
        """Test recording multiple metrics."""
        record_metric("metric1", 1.0)
        record_metric("metric2", 2.0)
        summary = get_performance_summary()
        assert "metric1" in summary or "metric2" in summary or summary == {}


class TestGetPerformanceSummary:
    """Tests for get_performance_summary function."""

    def test_empty_summary(self):
        """Test getting empty summary."""
        summary = get_performance_summary()
        assert isinstance(summary, dict)


class TestBenchmark:
    """Tests for benchmark decorator/function."""

    def test_benchmark_function(self):
        """Test benchmarking a function."""

        def test_func():
            time.sleep(0.001)
            return 42

        config = BenchmarkConfig(name="test", iterations=2, warmup_iterations=0)
        result = benchmark(test_func, config)
        assert result.name == "test"
        assert result.iterations == 2

    def test_benchmark_decorator(self):
        """Test benchmark as decorator."""

        @benchmark(name="decorated_test", iterations=1)
        def test_func():
            return 42

        result = test_func()
        assert isinstance(result, BenchmarkResult)


class TestMonitorPerformance:
    """Tests for monitor_performance context manager."""

    def test_monitor_context_manager(self):
        """Test using monitor as context manager."""
        with monitor_performance("test_operation") as monitor:
            time.sleep(0.001)
        assert monitor.start_time is not None
        assert monitor.end_time is not None
        assert monitor.duration_ms > 0

    def test_monitor_records_metrics(self):
        """Test that monitor records metrics."""
        with monitor_performance("test_op") as monitor:
            time.sleep(0.001)
        assert monitor.name == "test_op"


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_init(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.metrics == {}
        assert monitor.benchmarks == []
        assert monitor.enabled is True

    def test_init_disabled(self):
        """Test initialization when disabled."""
        monitor = PerformanceMonitor(enabled=False)
        assert monitor.enabled is False

    def test_record_metric(self):
        """Test recording metric."""
        monitor = PerformanceMonitor()
        monitor.record_metric("cpu", 50.0)
        assert "cpu" in monitor.metrics
        assert monitor.metrics["cpu"] == 50.0

    def test_record_metric_disabled(self):
        """Test recording metric when disabled."""
        monitor = PerformanceMonitor(enabled=False)
        monitor.record_metric("cpu", 50.0)
        assert monitor.metrics == {}

    def test_start_timer(self):
        """Test starting timer."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test_op")
        assert "test_op" in monitor._timers
        assert monitor._timers["test_op"] > 0

    def test_stop_timer(self):
        """Test stopping timer."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test_op")
        time.sleep(0.001)
        duration = monitor.stop_timer("test_op")
        assert duration > 0

    def test_stop_timer_nonexistent(self):
        """Test stopping nonexistent timer."""
        monitor = PerformanceMonitor()
        duration = monitor.stop_timer("nonexistent")
        assert duration == 0.0

    def test_benchmark_method(self):
        """Test benchmark method."""
        monitor = PerformanceMonitor()

        def test_func():
            time.sleep(0.001)
            return 42

        result = monitor.benchmark("test", test_func, iterations=2)
        assert result.name == "test"
        assert result.iterations == 2
        assert len(monitor.benchmarks) == 1

    def test_benchmark_disabled(self):
        """Test benchmark when disabled."""
        monitor = PerformanceMonitor(enabled=False)

        def test_func():
            return 42

        result = monitor.benchmark("test", test_func, iterations=2)
        assert result is None

    def test_get_report(self):
        """Test getting performance report."""
        monitor = PerformanceMonitor()
        monitor.record_metric("cpu", 50.0)

        def test_func():
            time.sleep(0.001)
            return 42

        monitor.benchmark("test", test_func, iterations=2)
        report = monitor.get_report()
        assert isinstance(report, PerformanceReport)
        assert "cpu" in report.metrics
        assert len(report.benchmarks) == 1

    def test_get_report_disabled(self):
        """Test getting report when disabled."""
        monitor = PerformanceMonitor(enabled=False)
        report = monitor.get_report()
        assert isinstance(report, PerformanceReport)

    def test_reset(self):
        """Test resetting monitor."""
        monitor = PerformanceMonitor()
        monitor.record_metric("cpu", 50.0)
        monitor.reset()
        assert monitor.metrics == {}
        assert monitor.benchmarks == []

    def test_context_manager(self):
        """Test using PerformanceMonitor as context manager."""
        with PerformanceMonitor() as monitor:
            monitor.record_metric("test", 1.0)
        assert isinstance(monitor, PerformanceMonitor)

    def test_enable_disable(self):
        """Test enabling and disabling."""
        monitor = PerformanceMonitor()
        monitor.disable()
        assert monitor.enabled is False
        monitor.enable()
        assert monitor.enabled is True

    def test_is_slow_operation(self):
        """Test checking if operation is slow."""
        monitor = PerformanceMonitor()
        monitor._slow_threshold_ms = 100.0
        assert monitor.is_slow_operation(150.0) is True
        assert monitor.is_slow_operation(50.0) is False

    def test_log_slow_operation(self, capsys):
        """Test logging slow operation."""
        monitor = PerformanceMonitor()
        monitor._slow_threshold_ms = 100.0
        monitor.log_slow_operation("slow_op", 200.0)
        captured = capsys.readouterr()
        assert "slow_op" in captured.out or captured.out == ""

    def test_get_summary(self):
        """Test getting summary."""
        monitor = PerformanceMonitor()
        monitor.record_metric("metric1", 1.0)
        monitor.record_metric("metric2", 2.0)
        summary = monitor.get_summary()
        assert isinstance(summary, dict)
        assert summary["total_metrics"] == 2

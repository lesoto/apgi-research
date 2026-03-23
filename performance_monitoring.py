"""
Performance trending and memory monitoring for APGI experiments.

Provides system resource monitoring, performance metrics tracking,
and optimization convergence detection.
"""

import gc
import psutil
import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""

    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    available_mb: float  # Available memory in MB
    gc_objects: int  # Number of objects tracked by garbage collector
    gc_stats: Dict[str, int]  # GC statistics


@dataclass
class CPUSnapshot:
    """CPU usage snapshot at a point in time."""

    timestamp: float
    percent: float  # CPU usage percentage
    count: int  # Number of CPU cores
    freq_mhz: float  # CPU frequency in MHz
    load_avg: List[float]  # Load averages (1, 5, 15 minutes)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation."""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[MemorySnapshot] = None
    memory_after: Optional[MemorySnapshot] = None
    memory_peak: Optional[MemorySnapshot] = None
    cpu_before: Optional[CPUSnapshot] = None
    cpu_after: Optional[CPUSnapshot] = None
    cpu_peak: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None


class PerformanceMonitor:
    """Monitors system performance and memory usage."""

    def __init__(self, output_dir: str = "performance_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 1.0  # seconds

        # Data storage
        self.memory_history: List[MemorySnapshot] = []
        self.cpu_history: List[CPUSnapshot] = []
        self.operation_metrics: List[PerformanceMetrics] = []

        # Thread safety
        self._lock = threading.Lock()

        # Performance thresholds
        self.memory_threshold_mb = 1024  # 1GB
        self.cpu_threshold_percent = 80.0
        self.memory_leak_threshold_mb = 100  # 100MB growth

        # Callbacks
        self.threshold_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    def add_threshold_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Add callback for threshold violations."""
        self.threshold_callbacks.append(callback)

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background monitoring."""
        if self.is_monitoring:
            return

        self.monitoring_interval = interval
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Take memory snapshot
                memory_snapshot = self._take_memory_snapshot()
                cpu_snapshot = self._take_cpu_snapshot()

                with self._lock:
                    self.memory_history.append(memory_snapshot)
                    self.cpu_history.append(cpu_snapshot)

                    # Keep only recent history (last 1000 snapshots)
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]
                    if len(self.cpu_history) > 1000:
                        self.cpu_history = self.cpu_history[-1000:]

                # Check thresholds
                self._check_thresholds(memory_snapshot, cpu_snapshot)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                print(f"Monitoring error: {e}")

    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # System memory info
        system_memory = psutil.virtual_memory()

        # GC info - handle both dict and tuple formats
        gc.collect()
        gc_objects = len(gc.get_objects())
        gc_stats_raw = gc.get_stats()

        # gc.get_stats() returns list of dicts in Python 3.7+, tuples in older versions
        gc_stats = {}
        for i, stat in enumerate(gc_stats_raw):
            if isinstance(stat, dict):
                # New format: dictionary with keys
                gc_stats[f"generation_{i}"] = stat.get("collected", 0)
            else:
                # Old format: tuple (generation, collections, collected, uncollectable)
                try:
                    gc_stats[f"generation_{i}"] = stat[2] if len(stat) > 2 else 0
                except (IndexError, TypeError):
                    gc_stats[f"generation_{i}"] = 0

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=system_memory.available / 1024 / 1024,
            gc_objects=gc_objects,
            gc_stats=gc_stats,
        )

    def _take_cpu_snapshot(self) -> CPUSnapshot:
        """Take a CPU usage snapshot."""
        cpu_percent = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else []

        return CPUSnapshot(
            timestamp=time.time(),
            percent=cpu_percent,
            count=cpu_count,
            freq_mhz=cpu_freq.current if cpu_freq else 0,
            load_avg=load_avg,
        )

    def _check_thresholds(
        self, memory_snapshot: MemorySnapshot, cpu_snapshot: CPUSnapshot
    ) -> None:
        """Check for threshold violations."""
        violations = []

        # Memory threshold
        if memory_snapshot.rss_mb > self.memory_threshold_mb:
            violations.append(
                {
                    "type": "memory",
                    "value": memory_snapshot.rss_mb,
                    "threshold": self.memory_threshold_mb,
                    "message": f"Memory usage exceeded: {memory_snapshot.rss_mb:.1f}MB > {self.memory_threshold_mb}MB",
                }
            )

        # CPU threshold
        if cpu_snapshot.percent > self.cpu_threshold_percent:
            violations.append(
                {
                    "type": "cpu",
                    "value": cpu_snapshot.percent,
                    "threshold": self.cpu_threshold_percent,
                    "message": f"CPU usage exceeded: {cpu_snapshot.percent:.1f}% > {self.cpu_threshold_percent}%",
                }
            )

        # Check for memory leaks
        if len(self.memory_history) > 10:
            recent_memory = [s.rss_mb for s in self.memory_history[-10:]]
            memory_growth = recent_memory[-1] - recent_memory[0]
            if memory_growth > self.memory_leak_threshold_mb:
                violations.append(
                    {
                        "type": "memory_leak",
                        "value": memory_growth,
                        "threshold": self.memory_leak_threshold_mb,
                        "message": f"Potential memory leak detected: {memory_growth:.1f}MB growth in last {len(recent_memory)} snapshots",
                    }
                )

        # Call callbacks for violations
        for violation in violations:
            for callback in self.threshold_callbacks:
                try:
                    callback("threshold_violation", violation)
                except Exception as e:
                    print(f"Threshold callback error: {e}")

    def start_operation(self, operation_name: str) -> PerformanceMetrics:
        """Start monitoring an operation."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            memory_before=self._take_memory_snapshot(),
            cpu_before=self._take_cpu_snapshot(),
        )

        with self._lock:
            self.operation_metrics.append(metrics)

        return metrics

    def end_operation(
        self,
        metrics: PerformanceMetrics,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """End monitoring an operation."""
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time
        metrics.memory_after = self._take_memory_snapshot()
        metrics.cpu_after = self._take_cpu_snapshot()
        metrics.success = success
        metrics.error = error

        # Calculate peak values
        if self.memory_history:
            operation_memory = [
                s
                for s in self.memory_history
                if s.timestamp >= metrics.start_time and s.timestamp <= metrics.end_time
            ]
            if operation_memory:
                metrics.memory_peak = max(operation_memory, key=lambda x: x.rss_mb)

        if self.cpu_history:
            operation_cpu = [
                s
                for s in self.cpu_history
                if s.timestamp >= metrics.start_time and s.timestamp <= metrics.end_time
            ]
            if operation_cpu:
                metrics.cpu_peak = max(s.percent for s in operation_cpu)

    def get_performance_summary(
        self, operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance summary for operations."""
        with self._lock:
            metrics = [
                m
                for m in self.operation_metrics
                if operation_name is None or m.operation_name == operation_name
            ]

            if not metrics:
                return {}

            successful_metrics = [m for m in metrics if m.success is True]
            failed_metrics = [m for m in metrics if m.success is False]

            summary = {
                "total_operations": len(metrics),
                "successful_operations": len(successful_metrics),
                "failed_operations": len(failed_metrics),
                "success_rate": len(successful_metrics) / len(metrics)
                if metrics
                else 0,
            }

            if successful_metrics:
                durations = [
                    m.duration for m in successful_metrics if m.duration is not None
                ]
                memory_usage = [
                    m.memory_before.rss_mb
                    for m in successful_metrics
                    if m.memory_before is not None
                ]
                cpu_usage = [
                    m.cpu_before.percent
                    for m in successful_metrics
                    if m.cpu_before is not None
                ]

                summary.update(
                    {
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                        "max_memory_mb": max(memory_usage),
                        "avg_cpu_percent": sum(cpu_usage) / len(cpu_usage),
                        "max_cpu_percent": max(cpu_usage),
                    }
                )

            return summary

    def get_memory_trend(self) -> Dict[str, Any]:
        """Get memory usage trend analysis."""
        with self._lock:
            if len(self.memory_history) < 2:
                return {}

            recent_memory = self.memory_history[-100:]  # Last 100 snapshots
            memory_values = [s.rss_mb for s in recent_memory]
            timestamps = [s.timestamp for s in recent_memory]

            # Calculate trend
            if len(memory_values) >= 2:
                x = np.arange(len(memory_values))
                slope, intercept = np.polyfit(x, memory_values, 1)
                trend_slope = slope

                # Determine trend direction
                if abs(trend_slope) < 0.1:
                    trend_direction = "stable"
                elif trend_slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
            else:
                trend_slope = 0
                trend_direction = "stable"

            return {
                "current_mb": memory_values[-1],
                "min_mb": min(memory_values),
                "max_mb": max(memory_values),
                "avg_mb": sum(memory_values) / len(memory_values),
                "trend_slope": trend_slope,
                "trend_direction": trend_direction,
                "samples": len(memory_values),
                "time_span_minutes": (timestamps[-1] - timestamps[0]) / 60
                if len(timestamps) > 1
                else 0,
            }

    def get_cpu_trend(self) -> Dict[str, Any]:
        """Get CPU usage trend analysis."""
        with self._lock:
            if len(self.cpu_history) < 2:
                return {}

            recent_cpu = self.cpu_history[-100:]  # Last 100 snapshots
            cpu_values = [s.percent for s in recent_cpu]
            timestamps = [s.timestamp for s in recent_cpu]

            # Calculate trend
            if len(cpu_values) >= 2:
                x = np.arange(len(cpu_values))
                slope, intercept = np.polyfit(x, cpu_values, 1)
                trend_slope = slope

                # Determine trend direction
                if abs(trend_slope) < 0.5:
                    trend_direction = "stable"
                elif trend_slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
            else:
                trend_slope = 0
                trend_direction = "stable"

            return {
                "current_percent": cpu_values[-1],
                "min_percent": min(cpu_values),
                "max_percent": max(cpu_values),
                "avg_percent": sum(cpu_values) / len(cpu_values),
                "trend_slope": trend_slope,
                "trend_direction": trend_direction,
                "samples": len(cpu_values),
                "time_span_minutes": (timestamps[-1] - timestamps[0]) / 60
                if len(timestamps) > 1
                else 0,
            }

    def detect_performance_regression(
        self, baseline_window: int = 10, current_window: int = 5
    ) -> Dict[str, Any]:
        """Detect performance regression by comparing recent performance to baseline."""
        with self._lock:
            if len(self.operation_metrics) < baseline_window + current_window:
                return {"status": "insufficient_data"}

            # Get baseline and recent performance
            recent_metrics = self.operation_metrics[-current_window:]
            baseline_metrics = self.operation_metrics[
                -(baseline_window + current_window) : -current_window
            ]

            # Compare durations
            recent_durations = [
                m.duration
                for m in recent_metrics
                if m.duration is not None and m.success
            ]
            baseline_durations = [
                m.duration
                for m in baseline_metrics
                if m.duration is not None and m.success
            ]

            if not recent_durations or not baseline_durations:
                return {"status": "insufficient_data"}

            recent_avg = sum(recent_durations) / len(recent_durations)
            baseline_avg = sum(baseline_durations) / len(baseline_durations)

            # Calculate regression
            regression_percent = ((recent_avg - baseline_avg) / baseline_avg) * 100

            # Determine significance
            if abs(regression_percent) < 5:
                significance = "no_change"
            elif regression_percent > 0:
                significance = (
                    "significant_regression"
                    if regression_percent > 20
                    else "moderate_regression"
                )
            else:
                significance = (
                    "significant_improvement"
                    if regression_percent < -20
                    else "moderate_improvement"
                )

            return {
                "status": "detected",
                "regression_percent": regression_percent,
                "significance": significance,
                "recent_avg_duration": recent_avg,
                "baseline_avg_duration": baseline_avg,
                "recent_samples": len(recent_durations),
                "baseline_samples": len(baseline_durations),
            }

    def save_monitoring_data(self) -> Path:
        """Save monitoring data to file."""
        data = {
            "memory_history": [asdict(m) for m in self.memory_history],
            "cpu_history": [asdict(c) for c in self.cpu_history],
            "operation_metrics": [asdict(o) for o in self.operation_metrics],
            "summary": {
                "memory_trend": self.get_memory_trend(),
                "cpu_trend": self.get_cpu_trend(),
                "performance_summary": self.get_performance_summary(),
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.output_dir / f"performance_data_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return file_path

    def generate_performance_plots(self) -> List[Path]:
        """Generate performance plots."""
        plot_files = []

        with self._lock:
            if self.memory_history:
                # Memory usage plot
                fig, ax = plt.subplots(figsize=(10, 6))
                timestamps = [
                    datetime.fromtimestamp(s.timestamp) for s in self.memory_history
                ]
                memory_values = [s.rss_mb for s in self.memory_history]

                ax.plot(timestamps, memory_values, "b-", label="RSS Memory")
                ax.set_xlabel("Time")
                ax.set_ylabel("Memory Usage (MB)")
                ax.set_title("Memory Usage Over Time")
                ax.legend()
                ax.grid(True)

                # Rotate x-axis labels
                plt.xticks(rotation=45)
                plt.tight_layout()

                memory_plot_file = self.output_dir / "memory_usage.png"
                plt.savefig(memory_plot_file)
                plt.close()
                plot_files.append(memory_plot_file)

            if self.cpu_history:
                # CPU usage plot
                fig, ax = plt.subplots(figsize=(10, 6))
                timestamps = [
                    datetime.fromtimestamp(s.timestamp) for s in self.cpu_history
                ]
                cpu_values = [s.percent for s in self.cpu_history]

                ax.plot(timestamps, cpu_values, "r-", label="CPU Usage")
                ax.set_xlabel("Time")
                ax.set_ylabel("CPU Usage (%)")
                ax.set_title("CPU Usage Over Time")
                ax.legend()
                ax.grid(True)

                # Rotate x-axis labels
                plt.xticks(rotation=45)
                plt.tight_layout()

                cpu_plot_file = self.output_dir / "cpu_usage.png"
                plt.savefig(cpu_plot_file)
                plt.close()
                plot_files.append(cpu_plot_file)

        return plot_files


# Convenience functions
def create_performance_monitor(
    output_dir: str = "performance_logs",
) -> PerformanceMonitor:
    """Create a new performance monitor."""
    return PerformanceMonitor(output_dir)


def monitor_function_performance(monitor: PerformanceMonitor, operation_name: str):
    """Decorator to monitor function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = monitor.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                monitor.end_operation(metrics, success=True)
                return result
            except Exception as e:
                monitor.end_operation(metrics, success=False, error=str(e))
                raise

        return wrapper

    return decorator

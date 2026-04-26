"""
Experiment progress tracking for APGI experiments.

Provides comprehensive progress tracking, logging, and monitoring capabilities.
"""

import json
import pickle
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from apgi_integration import APGIIntegration


@dataclass
class TrialResult:
    """Result of a single trial."""

    trial_number: int
    timestamp: float
    response_time: Optional[float] = None
    accuracy: Optional[float] = None
    apgi_metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentProgress:
    """Overall experiment progress."""

    experiment_name: str
    participant_id: str
    start_time: float
    current_trial: int
    total_trials: int
    completed_trials: int
    trials: List[TrialResult]
    apgi_summary: Optional[Dict[str, float]] = None
    status: str = "running"  # running, completed, failed, paused
    end_time: Optional[float] = None
    error_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        return


class ProgressTracker:
    """Tracks experiment progress with persistence and monitoring."""

    def __init__(
        self,
        experiment_name: str,
        participant_id: str,
        total_trials: int,
        output_dir: str = "progress",
        load_existing: bool = False,
    ):
        self.experiment_name = experiment_name
        self.participant_id = participant_id
        self.total_trials = total_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize progress
        self.progress = ExperimentProgress(
            experiment_name=experiment_name,
            participant_id=participant_id,
            start_time=time.time(),
            current_trial=0,
            total_trials=total_trials,
            completed_trials=0,
            trials=[],
            status="running",
        )

        # Progress callbacks
        self.progress_callbacks: List[Callable[[ExperimentProgress], None]] = []
        self.trial_callbacks: List[Callable[[TrialResult], None]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Load existing progress if requested
        if load_existing:
            self.load_progress()

        # Auto-save timer
        self._save_interval = 30  # seconds
        self._last_save = time.time()
        self._auto_save_enabled = True

    def add_progress_callback(
        self, callback: Callable[[ExperimentProgress], None]
    ) -> None:
        """Add a callback to be called when progress updates."""
        self.progress_callbacks.append(callback)

    def add_trial_callback(self, callback: Callable[[TrialResult], None]) -> None:
        """Add a callback to be called when a trial completes."""
        self.trial_callbacks.append(callback)

    def start_trial(self, trial_number: int) -> None:
        """Mark the start of a trial."""
        with self._lock:
            self.progress.current_trial = trial_number
            self._auto_save()

    def complete_trial(self, trial_result: TrialResult) -> None:
        """Record completion of a trial."""
        with self._lock:
            self.progress.trials.append(trial_result)
            self.progress.completed_trials += 1

            # Call trial callbacks
            for callback in self.trial_callbacks:
                try:
                    callback(trial_result)
                except Exception as e:
                    self.log_error(f"Trial callback error: {e}")

            # Call progress callbacks
            for callback in self.progress_callbacks:  # type: ignore[assignment]
                try:
                    callback(self.progress)  # type: ignore[arg-type]
                except Exception as e:
                    self.log_error(f"Progress callback error: {e}")

            self._auto_save()

    def log_error(self, error_message: str) -> None:
        """Log an error message."""
        with self._lock:
            self.progress.error_log.append(
                f"[{datetime.now().isoformat()}] {error_message}"
            )
            self._auto_save()

    def update_apgi_metrics(self, apgi_integration: APGIIntegration) -> None:
        """Update APGI metrics from integration."""
        with self._lock:
            try:
                self.progress.apgi_summary = apgi_integration.finalize()
                self._auto_save()
            except Exception as e:
                self.log_error(f"Failed to update APGI metrics: {e}")

    def get_progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        with self._lock:
            if self.progress.total_trials == 0:
                return 0.0
            return (self.progress.completed_trials / self.progress.total_trials) * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        with self._lock:
            end_time = self.progress.end_time or time.time()
            return end_time - self.progress.start_time

    def get_estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        with self._lock:
            if self.progress.completed_trials == 0:
                return None

            elapsed = self.get_elapsed_time()
            avg_time_per_trial = elapsed / self.progress.completed_trials
            remaining_trials = (
                self.progress.total_trials - self.progress.completed_trials
            )
            return remaining_trials * avg_time_per_trial

    def get_trial_statistics(self) -> Dict[str, Any]:
        """Get statistics about completed trials."""
        with self._lock:
            if not self.progress.trials:
                return {}

            response_times = [
                t.response_time
                for t in self.progress.trials
                if t.response_time is not None
            ]
            accuracies = [
                t.accuracy for t in self.progress.trials if t.accuracy is not None
            ]

            stats: Dict[str, Union[int, float]] = {
                "completed_trials": len(self.progress.trials),
                "error_trials": len(
                    [t for t in self.progress.trials if t.error is not None]
                ),
            }

            if response_times:
                stats.update(
                    {
                        "avg_response_time": float(
                            sum(response_times) / len(response_times)
                        ),
                        "min_response_time": float(min(response_times)),
                        "max_response_time": float(max(response_times)),
                    }
                )

            if accuracies:
                stats.update(
                    {
                        "avg_accuracy": float(sum(accuracies) / len(accuracies)),
                        "min_accuracy": float(min(accuracies)),
                        "max_accuracy": float(max(accuracies)),
                    }
                )

            return stats

    def get_apgi_statistics(self) -> Dict[str, Any]:
        """Get APGI-specific statistics."""
        with self._lock:
            if not self.progress.apgi_summary:
                return {}

            return self.progress.apgi_summary.copy()

    def complete_experiment(self, status: str = "completed") -> None:
        """Mark experiment as completed."""
        with self._lock:
            self.progress.status = status
            self.progress.end_time = time.time()
            self._save_progress()

    def pause_experiment(self) -> None:
        """Pause the experiment."""
        with self._lock:
            self.progress.status = "paused"
            self._save_progress()

    def resume_experiment(self) -> None:
        """Resume a paused experiment."""
        with self._lock:
            if self.progress.status == "paused":
                self.progress.status = "running"
                self._save_progress()

    def _auto_save(self) -> None:
        """Auto-save if enough time has passed."""
        if not self._auto_save_enabled:
            return

        current_time = time.time()
        if current_time - self._last_save > self._save_interval:
            self._save_progress()
            self._last_save = current_time

    def _save_progress(self) -> None:
        """Save progress to file."""
        try:
            progress_file = (
                self.output_dir
                / f"{self.experiment_name}_{self.participant_id}_progress.json"
            )
            with open(progress_file, "w") as f:
                json.dump(asdict(self.progress), f, indent=2, default=str)

            # Also save as pickle for faster loading
            pickle_file = (
                self.output_dir
                / f"{self.experiment_name}_{self.participant_id}_progress.pkl"
            )
            with open(pickle_file, "wb") as f:
                pickle.dump(self.progress, f)

        except Exception as e:
            print(f"Failed to save progress: {e}")

    def load_progress(self) -> bool:
        """Load progress from file."""
        try:
            pickle_file = (
                self.output_dir
                / f"{self.experiment_name}_{self.participant_id}_progress.pkl"
            )
            if pickle_file.exists():
                with open(pickle_file, "rb") as f:
                    self.progress = pickle.load(f)
                return True
        except Exception as e:
            print(f"Failed to load progress: {e}")
        return False

    def export_summary(self) -> Dict[str, Any]:
        """Export a comprehensive summary of the experiment."""
        with self._lock:
            summary = {
                "experiment_info": {
                    "name": self.progress.experiment_name,
                    "participant_id": self.progress.participant_id,
                    "status": self.progress.status,
                    "start_time": datetime.fromtimestamp(
                        self.progress.start_time
                    ).isoformat(),
                    "end_time": (
                        datetime.fromtimestamp(self.progress.end_time).isoformat()
                        if self.progress.end_time
                        else None
                    ),
                    "elapsed_time": self.get_elapsed_time(),
                },
                "progress": {
                    "total_trials": self.progress.total_trials,
                    "completed_trials": self.progress.completed_trials,
                    "current_trial": self.progress.current_trial,
                    "progress_percentage": self.get_progress_percentage(),
                    "estimated_remaining_time": self.get_estimated_remaining_time(),
                },
                "trial_statistics": self.get_trial_statistics(),
                "apgi_statistics": self.get_apgi_statistics(),
                "errors": {
                    "total_errors": len(self.progress.error_log),
                    "error_log": (
                        self.progress.error_log[-10:] if self.progress.error_log else []
                    ),  # Last 10 errors
                },
            }
            return summary

    def save_summary_report(self) -> Path:
        """Save a detailed summary report."""
        summary = self.export_summary()
        report_file = (
            self.output_dir
            / f"{self.experiment_name}_{self.participant_id}_summary.json"
        )

        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Also save progress as pickle for loading
        self._save_progress()

        return report_file


class ProgressMonitor:
    """Monitors multiple experiments and provides aggregated statistics."""

    def __init__(self, progress_dir: str = "progress"):
        self.progress_dir = Path(progress_dir)
        self.active_experiments: Dict[str, ProgressTracker] = {}

    def register_experiment(self, tracker: ProgressTracker) -> None:
        """Register an experiment tracker."""
        key = f"{tracker.experiment_name}_{tracker.participant_id}"
        self.active_experiments[key] = tracker

    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall status of all experiments."""
        status: Dict[str, Any] = {
            "total_experiments": len(self.active_experiments),
            "running_experiments": len(
                [
                    t
                    for t in self.active_experiments.values()
                    if t.progress.status == "running"
                ]
            ),
            "completed_experiments": len(
                [
                    t
                    for t in self.active_experiments.values()
                    if t.progress.status == "completed"
                ]
            ),
            "failed_experiments": len(
                [
                    t
                    for t in self.active_experiments.values()
                    if t.progress.status == "failed"
                ]
            ),
            "experiments": {},
        }

        for key, tracker in self.active_experiments.items():
            status["experiments"][key] = {
                "status": tracker.progress.status,
                "progress_percentage": tracker.get_progress_percentage(),
                "completed_trials": tracker.progress.completed_trials,
                "total_trials": tracker.progress.total_trials,
                "elapsed_time": tracker.get_elapsed_time(),
            }

        return status

    def get_experiment_tracker(
        self, experiment_name: str, participant_id: str
    ) -> Optional[ProgressTracker]:
        """Get a specific experiment tracker."""
        key = f"{experiment_name}_{participant_id}"
        return self.active_experiments.get(key)


# Convenience functions
def create_progress_tracker(
    experiment_name: str,
    participant_id: str,
    total_trials: int,
    output_dir: str = "progress",
) -> ProgressTracker:
    """Create a new progress tracker."""
    return ProgressTracker(experiment_name, participant_id, total_trials, output_dir)


def load_progress_tracker(
    experiment_name: str,
    participant_id: str,
    total_trials: int,
    output_dir: str = "progress",
) -> Optional[ProgressTracker]:
    """Load an existing progress tracker."""
    tracker = ProgressTracker(experiment_name, participant_id, total_trials, output_dir)
    if tracker.load_progress():
        return tracker
    return None


def monitor_experiment_progress(
    experiment_name: str,
    participant_id: str,
    update_callback: Optional[Callable[[ExperimentProgress], None]] = None,
) -> None:
    """Monitor experiment progress with optional callback."""
    tracker = load_progress_tracker(experiment_name, participant_id, 0)
    if tracker and update_callback:
        update_callback(tracker.progress)

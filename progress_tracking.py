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
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from apgi_integration import APGIIntegration


class ProgressStatus(Enum):
    """Status for task progress tracking."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Progress for a single task."""

    task_id: str
    status: ProgressStatus = ProgressStatus.PENDING
    progress_percent: float = 0.0
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set start time when task becomes in progress."""
        if self.status == ProgressStatus.IN_PROGRESS and self.start_time is None:
            self.start_time = time.time()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class Checkpoint:
    """A checkpoint for saving progress state."""

    checkpoint_id: str
    timestamp: float = field(default_factory=time.time)
    task_states: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressReport:
    """Report summarizing progress across tasks."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0
    pending_tasks: int = 0
    overall_progress: float = 0.0

    @classmethod
    def calculate(cls, tasks: List[TaskProgress]) -> "ProgressReport":
        """Calculate a progress report from a list of tasks."""
        if not tasks:
            return cls()

        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == ProgressStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == ProgressStatus.FAILED)
        in_progress = sum(1 for t in tasks if t.status == ProgressStatus.IN_PROGRESS)
        pending = sum(1 for t in tasks if t.status == ProgressStatus.PENDING)

        # Calculate overall progress as average of all task progress
        overall = sum(t.progress_percent for t in tasks) / total if total > 0 else 0.0

        return cls(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            in_progress_tasks=in_progress,
            pending_tasks=pending,
            overall_progress=overall,
        )


# Default checkpoint directory
CHECKPOINT_DIR = Path("checkpoints")


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
        experiment_name: str = "",
        participant_id: str = "",
        total_trials: int = 0,
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

        # Task-based tracking (new API)
        self.tasks: Dict[str, TaskProgress] = {}
        self.checkpoints: List[Checkpoint] = []

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

    def add_task(
        self, task_id: str, message: str = "", progress_percent: float = 0.0
    ) -> TaskProgress:
        """Add a new task to track."""
        with self._lock:
            task = TaskProgress(
                task_id=task_id,
                message=message,
                progress_percent=progress_percent,
            )
            self.tasks[task_id] = task
            return task

    def update_task(
        self,
        task_id: str,
        progress_percent: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[ProgressStatus] = None,
    ) -> Optional[TaskProgress]:
        """Update a task's progress."""
        with self._lock:
            if task_id not in self.tasks:
                # Create new task if it doesn't exist
                self.add_task(task_id, message or "", progress_percent or 0.0)

            task = self.tasks[task_id]

            if progress_percent is not None:
                task.progress_percent = progress_percent
            if message is not None:
                task.message = message
            if status is not None:
                task.status = status
                if status == ProgressStatus.IN_PROGRESS and task.start_time is None:
                    task.start_time = time.time()

            return task

    def complete_task(self, task_id: str, message: str = "") -> Optional[TaskProgress]:
        """Mark a task as completed."""
        with self._lock:
            task = self.update_task(
                task_id,
                status=ProgressStatus.COMPLETED,
                progress_percent=100.0,
                message=message,
            )
            if task:
                task.end_time = time.time()
            return task

    def fail_task(self, task_id: str, message: str = "") -> Optional[TaskProgress]:
        """Mark a task as failed."""
        with self._lock:
            task = self.update_task(
                task_id,
                status=ProgressStatus.FAILED,
                message=message,
            )
            if task:
                task.end_time = time.time()
            return task

    def get_report(self) -> ProgressReport:
        """Get a progress report for all tasks."""
        with self._lock:
            return ProgressReport.calculate(list(self.tasks.values()))

    def create_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """Create a checkpoint from current task states."""
        with self._lock:
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                task_states={
                    tid: {
                        "status": t.status.value,
                        "progress_percent": t.progress_percent,
                        "message": t.message,
                    }
                    for tid, t in self.tasks.items()
                },
            )
            self.checkpoints.append(checkpoint)
            return checkpoint

    def save_checkpoint(self, checkpoint_id: str, path: Optional[Path] = None) -> bool:
        """Save a checkpoint to file."""
        try:
            checkpoint = self.create_checkpoint(checkpoint_id)
            save_path = path or (CHECKPOINT_DIR / f"{checkpoint_id}.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w") as f:
                json.dump(
                    {
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "timestamp": checkpoint.timestamp,
                        "task_states": checkpoint.task_states,
                        "metadata": checkpoint.metadata,
                    },
                    f,
                    indent=2,
                )
            return True
        except Exception:
            return False

    def load_checkpoint(self, path: Path) -> bool:
        """Load a checkpoint from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Restore task states
            for task_id, state in data.get("task_states", {}).items():
                self.tasks[task_id] = TaskProgress(
                    task_id=task_id,
                    status=ProgressStatus(state.get("status", "pending")),
                    progress_percent=state.get("progress_percent", 0.0),
                    message=state.get("message", ""),
                )

            return True
        except Exception:
            return False

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


# Convenience functions for checkpoint management
def create_checkpoint(
    checkpoint_id: str,
    tasks: Optional[List[TaskProgress]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Checkpoint:
    """Create a checkpoint."""
    task_states = {}
    if tasks:
        for task in tasks:
            task_states[task.task_id] = {
                "status": task.status.value,
                "progress_percent": task.progress_percent,
                "message": task.message,
            }

    return Checkpoint(
        checkpoint_id=checkpoint_id,
        task_states=task_states,
        metadata=metadata or {},
    )


def save_checkpoint(checkpoint: Checkpoint, path: Optional[Path] = None) -> bool:
    """Save a checkpoint to file."""
    try:
        save_path = path or (CHECKPOINT_DIR / f"{checkpoint.checkpoint_id}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "timestamp": checkpoint.timestamp,
                    "task_states": checkpoint.task_states,
                    "metadata": checkpoint.metadata,
                },
                f,
                indent=2,
            )
        return True
    except Exception:
        return False


def load_checkpoint(path: Path) -> Optional[Checkpoint]:
    """Load a checkpoint from file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)

        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data.get("timestamp", time.time()),
            task_states=data.get("task_states", {}),
            metadata=data.get("metadata", {}),
        )
    except Exception:
        return None


def resume_from_checkpoint(
    path: Path,
) -> tuple[Optional[Checkpoint], List[TaskProgress]]:
    """Resume from a checkpoint, returning the checkpoint and restored tasks."""
    checkpoint = load_checkpoint(path)
    if not checkpoint:
        return None, []

    tasks = []
    for task_id, state in checkpoint.task_states.items():
        tasks.append(
            TaskProgress(
                task_id=task_id,
                status=ProgressStatus(state.get("status", "pending")),
                progress_percent=state.get("progress_percent", 0.0),
                message=state.get("message", ""),
            )
        )

    return checkpoint, tasks


def get_progress_summary(tasks: List[TaskProgress]) -> Dict[str, Any]:
    """Get a summary of progress for a list of tasks."""
    report = ProgressReport.calculate(tasks)
    return {
        "total": report.total_tasks,
        "completed": report.completed_tasks,
        "failed": report.failed_tasks,
        "in_progress": report.in_progress_tasks,
        "pending": report.pending_tasks,
        "progress_percent": report.overall_progress,
    }


class ProgressManager:
    """Manages multiple progress trackers."""

    def __init__(self) -> None:
        self.trackers: Dict[str, ProgressTracker] = {}

    def get_tracker(self, session_id: str) -> ProgressTracker:
        """Get or create a tracker for a session."""
        if session_id not in self.trackers:
            self.trackers[session_id] = ProgressTracker()
        return self.trackers[session_id]

    def remove_tracker(self, session_id: str) -> None:
        """Remove a tracker."""
        if session_id in self.trackers:
            del self.trackers[session_id]

    def get_all_reports(self) -> Dict[str, ProgressReport]:
        """Get progress reports for all trackers."""
        return {sid: tracker.get_report() for sid, tracker in self.trackers.items()}

    def get_overall_report(self) -> ProgressReport:
        """Get a combined report for all trackers."""
        all_tasks: List[TaskProgress] = []
        for tracker in self.trackers.values():
            all_tasks.extend(tracker.tasks.values())
        return ProgressReport.calculate(all_tasks)


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

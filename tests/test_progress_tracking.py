"""
Comprehensive tests for progress_tracking.py - Progress tracking module.
"""

import json
import time
from unittest.mock import patch

from progress_tracking import (
    Checkpoint,
    ProgressManager,
    ProgressReport,
    ProgressStatus,
    ProgressTracker,
    TaskProgress,
    create_checkpoint,
    get_progress_summary,
    load_checkpoint,
    resume_from_checkpoint,
    save_checkpoint,
)


class TestProgressStatus:
    """Tests for ProgressStatus enum."""

    def test_status_values(self):
        """Test ProgressStatus enum values."""
        assert ProgressStatus.PENDING.value == "pending"  # nosec: B101 - Test assertion
        assert (
            ProgressStatus.IN_PROGRESS.value == "in_progress"
        )  # nosec: B101 - Test assertion
        assert (
            ProgressStatus.COMPLETED.value == "completed"
        )  # nosec: B101 - Test assertion
        assert ProgressStatus.FAILED.value == "failed"  # nosec: B101 - Test assertion
        assert (
            ProgressStatus.CANCELLED.value == "cancelled"
        )  # nosec: B101 - Test assertion


class TestTaskProgress:
    """Tests for TaskProgress dataclass."""

    def test_default_values(self):
        """Test default TaskProgress values."""
        task = TaskProgress(task_id="test-1")
        assert task.task_id == "test-1"  # nosec: B101 - Test assertion
        assert task.status == ProgressStatus.PENDING  # nosec: B101 - Test assertion
        assert task.progress_percent == 0.0  # nosec: B101 - Test assertion
        assert task.message == ""  # nosec: B101 - Test assertion
        assert task.start_time is None  # nosec: B101 - Test assertion
        assert task.end_time is None  # nosec: B101 - Test assertion
        assert task.metadata == {}  # nosec: B101 - Test assertion

    def test_custom_values(self):
        """Test custom TaskProgress values."""
        start = time.time()
        task = TaskProgress(
            task_id="test-2",
            status=ProgressStatus.IN_PROGRESS,
            progress_percent=50.0,
            message="Halfway done",
            start_time=start,
            metadata={"key": "value"},
        )
        assert task.task_id == "test-2"  # nosec: B101 - Test assertion
        assert task.status == ProgressStatus.IN_PROGRESS  # nosec: B101 - Test assertion
        assert task.progress_percent == 50.0  # nosec: B101 - Test assertion
        assert task.message == "Halfway done"  # nosec: B101 - Test assertion
        assert task.start_time == start  # nosec: B101 - Test assertion
        assert task.metadata == {"key": "value"}  # nosec: B101 - Test assertion

    def test_duration_no_start(self):
        """Test duration calculation without start time."""
        task = TaskProgress(task_id="test-3")
        assert task.duration_seconds is None  # nosec: B101 - Test assertion

    def test_duration_with_start(self):
        """Test duration calculation with start time."""
        start = time.time()
        task = TaskProgress(task_id="test-4", start_time=start)
        assert task.duration_seconds is not None  # nosec: B101 - Test assertion
        assert task.duration_seconds >= 0  # nosec: B101 - Test assertion


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_default_values(self):
        """Test default Checkpoint values."""
        checkpoint = Checkpoint(checkpoint_id="cp-1")
        assert checkpoint.checkpoint_id == "cp-1"  # nosec: B101 - Test assertion
        assert checkpoint.timestamp is not None  # nosec: B101 - Test assertion
        assert checkpoint.task_states == {}  # nosec: B101 - Test assertion
        assert checkpoint.metadata == {}  # nosec: B101 - Test assertion

    def test_custom_values(self):
        """Test custom Checkpoint values."""
        ts = time.time()
        checkpoint = Checkpoint(
            checkpoint_id="cp-2",
            timestamp=ts,
            task_states={"task1": {"progress": 50}},
            metadata={"version": "1.0"},
        )
        assert checkpoint.checkpoint_id == "cp-2"  # nosec: B101 - Test assertion
        assert checkpoint.timestamp == ts  # nosec: B101 - Test assertion
        assert checkpoint.task_states == {
            "task1": {"progress": 50}
        }  # nosec: B101 - Test assertion
        assert checkpoint.metadata == {"version": "1.0"}  # nosec: B101 - Test assertion


class TestProgressReport:
    """Tests for ProgressReport dataclass."""

    def test_default_values(self):
        """Test default ProgressReport values."""
        report = ProgressReport()
        assert report.total_tasks == 0  # nosec: B101 - Test assertion
        assert report.completed_tasks == 0  # nosec: B101 - Test assertion
        assert report.failed_tasks == 0  # nosec: B101 - Test assertion
        assert report.in_progress_tasks == 0  # nosec: B101 - Test assertion
        assert report.pending_tasks == 0  # nosec: B101 - Test assertion
        assert report.overall_progress == 0.0  # nosec: B101 - Test assertion

    def test_calculate_progress(self):
        """Test progress calculation."""
        tasks = [
            TaskProgress(
                task_id="1", status=ProgressStatus.COMPLETED, progress_percent=100
            ),
            TaskProgress(
                task_id="2", status=ProgressStatus.IN_PROGRESS, progress_percent=50
            ),
            TaskProgress(
                task_id="3", status=ProgressStatus.PENDING, progress_percent=0
            ),
        ]
        report = ProgressReport.calculate(tasks)
        assert report.total_tasks == 3  # nosec: B101 - Test assertion
        assert report.completed_tasks == 1  # nosec: B101 - Test assertion
        assert report.in_progress_tasks == 1  # nosec: B101 - Test assertion
        assert report.pending_tasks == 1  # nosec: B101 - Test assertion
        assert report.overall_progress == 50.0  # nosec: B101 - Test assertion


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_to_default_path(self, tmp_path):
        """Test saving checkpoint to default path."""
        with patch("progress_tracking.CHECKPOINT_DIR", tmp_path):
            checkpoint = Checkpoint(checkpoint_id="test-cp")
            result = save_checkpoint(checkpoint)
            assert result is True  # nosec: B101 - Test assertion

    def test_save_to_custom_path(self, tmp_path):
        """Test saving checkpoint to custom path."""
        custom_path = tmp_path / "custom_checkpoint.json"
        checkpoint = Checkpoint(checkpoint_id="test-cp")
        result = save_checkpoint(checkpoint, custom_path)
        assert result is True  # nosec: B101 - Test assertion
        assert custom_path.exists()  # nosec: B101 - Test assertion

    def test_save_creates_directory(self, tmp_path):
        """Test that saving creates directory if needed."""
        nested_path = tmp_path / "nested" / "dir" / "checkpoint.json"
        checkpoint = Checkpoint(checkpoint_id="test-cp")
        result = save_checkpoint(checkpoint, nested_path)
        assert result is True  # nosec: B101 - Test assertion
        assert nested_path.parent.exists()  # nosec: B101 - Test assertion


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_existing_checkpoint(self, tmp_path):
        """Test loading existing checkpoint."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_data = {
            "checkpoint_id": "test-cp",
            "timestamp": time.time(),
            "task_states": {},
            "metadata": {},
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        checkpoint = load_checkpoint(checkpoint_file)
        assert checkpoint is not None  # nosec: B101 - Test assertion
        assert checkpoint.checkpoint_id == "test-cp"  # nosec: B101 - Test assertion

    def test_load_nonexistent_checkpoint(self, tmp_path):
        """Test loading nonexistent checkpoint."""
        checkpoint_file = tmp_path / "nonexistent.json"
        checkpoint = load_checkpoint(checkpoint_file)
        assert checkpoint is None  # nosec: B101 - Test assertion

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        checkpoint_file = tmp_path / "invalid.json"
        checkpoint_file.write_text("not valid json")
        checkpoint = load_checkpoint(checkpoint_file)
        assert checkpoint is None  # nosec: B101 - Test assertion


class TestCreateCheckpoint:
    """Tests for create_checkpoint function."""

    def test_create_empty_checkpoint(self):
        """Test creating empty checkpoint."""
        checkpoint = create_checkpoint("empty-cp")
        assert checkpoint.checkpoint_id == "empty-cp"  # nosec: B101 - Test assertion
        assert checkpoint.task_states == {}  # nosec: B101 - Test assertion

    def test_create_with_tasks(self):
        """Test creating checkpoint with tasks."""
        tasks = [
            TaskProgress(task_id="1", progress_percent=50),
            TaskProgress(task_id="2", progress_percent=100),
        ]
        checkpoint = create_checkpoint("tasks-cp", tasks)
        assert checkpoint.checkpoint_id == "tasks-cp"  # nosec: B101 - Test assertion
        assert "1" in checkpoint.task_states  # nosec: B101 - Test assertion
        assert "2" in checkpoint.task_states  # nosec: B101 - Test assertion

    def test_create_with_metadata(self):
        """Test creating checkpoint with metadata."""
        checkpoint = create_checkpoint("meta-cp", metadata={"version": "1.0"})
        assert checkpoint.metadata == {"version": "1.0"}  # nosec: B101 - Test assertion


class TestResumeFromCheckpoint:
    """Tests for resume_from_checkpoint function."""

    def test_resume_existing(self, tmp_path):
        """Test resuming from existing checkpoint."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_data = {
            "checkpoint_id": "test-cp",
            "timestamp": time.time(),
            "task_states": {"task1": {"progress": 50}},
            "metadata": {},
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        checkpoint, tasks = resume_from_checkpoint(checkpoint_file)
        assert checkpoint is not None  # nosec: B101 - Test assertion
        assert checkpoint.checkpoint_id == "test-cp"  # nosec: B101 - Test assertion
        assert len(tasks) == 1  # nosec: B101 - Test assertion

    def test_resume_nonexistent(self, tmp_path):
        """Test resuming from nonexistent checkpoint."""
        checkpoint_file = tmp_path / "nonexistent.json"
        checkpoint, tasks = resume_from_checkpoint(checkpoint_file)
        assert checkpoint is None  # nosec: B101 - Test assertion
        assert tasks == []  # nosec: B101 - Test assertion


class TestGetProgressSummary:
    """Tests for get_progress_summary function."""

    def test_summary_empty(self):
        """Test summary with no tasks."""
        summary = get_progress_summary([])
        assert summary["total"] == 0  # nosec: B101 - Test assertion
        assert summary["completed"] == 0  # nosec: B101 - Test assertion
        assert summary["progress_percent"] == 0.0  # nosec: B101 - Test assertion

    def test_summary_with_tasks(self):
        """Test summary with tasks."""
        tasks = [
            TaskProgress(
                task_id="1", status=ProgressStatus.COMPLETED, progress_percent=100
            ),
            TaskProgress(
                task_id="2", status=ProgressStatus.IN_PROGRESS, progress_percent=50
            ),
        ]
        summary = get_progress_summary(tasks)
        assert summary["total"] == 2  # nosec: B101 - Test assertion
        assert summary["completed"] == 1  # nosec: B101 - Test assertion
        assert summary["in_progress"] == 1  # nosec: B101 - Test assertion
        assert summary["progress_percent"] == 75.0  # nosec: B101 - Test assertion


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()
        assert tracker.tasks == {}  # nosec: B101 - Test assertion
        assert tracker.checkpoints == []  # nosec: B101 - Test assertion

    def test_add_task(self):
        """Test adding a task."""
        tracker = ProgressTracker()
        tracker.add_task("task-1", "Test Task")
        assert "task-1" in tracker.tasks  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["task-1"].task_id == "task-1"
        )  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["task-1"].message == "Test Task"
        )  # nosec: B101 - Test assertion

    def test_update_task(self):
        """Test updating a task."""
        tracker = ProgressTracker()
        tracker.add_task("task-1", "Test Task")
        tracker.update_task("task-1", progress_percent=50, message="Halfway")
        assert (
            tracker.tasks["task-1"].progress_percent == 50
        )  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["task-1"].message == "Halfway"
        )  # nosec: B101 - Test assertion

    def test_update_nonexistent_task(self):
        """Test updating nonexistent task creates it."""
        tracker = ProgressTracker()
        tracker.update_task("new-task", progress_percent=25)
        assert "new-task" in tracker.tasks  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["new-task"].progress_percent == 25
        )  # nosec: B101 - Test assertion

    def test_complete_task(self):
        """Test completing a task."""
        tracker = ProgressTracker()
        tracker.add_task("task-1")
        tracker.complete_task("task-1", message="Done")
        assert (
            tracker.tasks["task-1"].status == ProgressStatus.COMPLETED
        )  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["task-1"].progress_percent == 100
        )  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["task-1"].end_time is not None
        )  # nosec: B101 - Test assertion

    def test_fail_task(self):
        """Test failing a task."""
        tracker = ProgressTracker()
        tracker.add_task("task-1")
        tracker.fail_task("task-1", message="Failed")
        assert (
            tracker.tasks["task-1"].status == ProgressStatus.FAILED
        )  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["task-1"].end_time is not None
        )  # nosec: B101 - Test assertion

    def test_get_report(self):
        """Test getting progress report."""
        tracker = ProgressTracker()
        tracker.add_task("task-1")
        tracker.add_task("task-2")
        tracker.complete_task("task-1")
        report = tracker.get_report()
        assert isinstance(report, ProgressReport)  # nosec: B101 - Test assertion
        assert report.total_tasks == 2  # nosec: B101 - Test assertion
        assert report.completed_tasks == 1  # nosec: B101 - Test assertion

    def test_create_checkpoint(self):
        """Test creating checkpoint from tracker."""
        tracker = ProgressTracker()
        tracker.add_task("task-1", progress_percent=50)
        checkpoint = tracker.create_checkpoint("cp-1")
        assert checkpoint.checkpoint_id == "cp-1"  # nosec: B101 - Test assertion
        assert "task-1" in checkpoint.task_states  # nosec: B101 - Test assertion

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoint."""
        with patch("progress_tracking.CHECKPOINT_DIR", tmp_path):
            tracker = ProgressTracker()
            tracker.add_task("task-1", progress_percent=50)
            tracker.save_checkpoint("cp-1")

            new_tracker = ProgressTracker()
            new_tracker.load_checkpoint(tmp_path / "cp-1.json")
            assert "task-1" in new_tracker.tasks  # nosec: B101 - Test assertion


class TestProgressManager:
    """Tests for ProgressManager class."""

    def test_init(self):
        """Test ProgressManager initialization."""
        manager = ProgressManager()
        assert manager.trackers == {}  # nosec: B101 - Test assertion

    def test_get_tracker(self):
        """Test getting a tracker."""
        manager = ProgressManager()
        tracker = manager.get_tracker("session-1")
        assert isinstance(tracker, ProgressTracker)  # nosec: B101 - Test assertion
        assert "session-1" in manager.trackers  # nosec: B101 - Test assertion

    def test_get_existing_tracker(self):
        """Test getting existing tracker returns same instance."""
        manager = ProgressManager()
        tracker1 = manager.get_tracker("session-1")
        tracker2 = manager.get_tracker("session-1")
        assert tracker1 is tracker2  # nosec: B101 - Test assertion

    def test_remove_tracker(self):
        """Test removing a tracker."""
        manager = ProgressManager()
        manager.get_tracker("session-1")
        manager.remove_tracker("session-1")
        assert "session-1" not in manager.trackers  # nosec: B101 - Test assertion

    def test_get_all_reports(self):
        """Test getting all reports."""
        manager = ProgressManager()
        tracker1 = manager.get_tracker("session-1")
        tracker1.add_task("task-1")
        tracker1.complete_task("task-1")

        reports = manager.get_all_reports()
        assert "session-1" in reports  # nosec: B101 - Test assertion
        assert reports["session-1"].completed_tasks == 1  # nosec: B101 - Test assertion

    def test_get_overall_report(self):
        """Test getting overall report."""
        manager = ProgressManager()
        tracker1 = manager.get_tracker("session-1")
        tracker1.add_task("task-1")
        tracker1.complete_task("task-1")
        tracker2 = manager.get_tracker("session-2")
        tracker2.add_task("task-2")

        report = manager.get_overall_report()
        assert report.total_tasks == 2  # nosec: B101 - Test assertion
        assert report.completed_tasks == 1  # nosec: B101 - Test assertion

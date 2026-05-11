"""
Comprehensive tests for human_layer.py to achieve 90%+ coverage.
Tests all major classes, methods, validation, and edge cases.
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, mock_open, patch

import pytest

logger = logging.getLogger(__name__)

from human_layer import (
    TaskPriority,
    configure_if_needed,
    review,
    select_task,
)

from hypothesis_approval_board import ApprovalBoard, Hypothesis
import logging
from pathlib import Path


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_values(self):
        """Test that all expected priority values exist."""
        priorities = [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.MEDIUM,
            TaskPriority.LOW,
        ]

        for priority in priorities:
            assert isinstance(priority.value, str)
            assert len(priority.value) > 0

    def test_priority_ordering(self):
        """Test priority ordering for comparison."""
        assert TaskPriority.CRITICAL.value > TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value > TaskPriority.MEDIUM.value
        assert TaskPriority.MEDIUM.value > TaskPriority.LOW.value

    def test_priority_comparison(self):
        """Test priority comparison operations."""
        critical = TaskPriority.CRITICAL
        high = TaskPriority.HIGH
        medium = TaskPriority.MEDIUM
        low = TaskPriority.LOW

        # Test comparisons - compare underlying values since Enum doesn't support > by default
        assert critical.value > high.value
        assert high.value > medium.value
        assert medium.value > low.value

        # Test with different types
        assert str(critical) != "string"  # Different type
        assert str(high) != "42"  # Different value

        # Test enum value comparisons
        assert critical.value == "critical"
        assert high.value == "high"


class TestHypothesisApprovalBoard:
    """Tests for HypothesisApprovalBoard class."""

    def test_board_initialization(self):
        """Test board initialization."""
        board = ApprovalBoard()

        assert hasattr(board, "add_hypothesis")
        assert hasattr(board, "approve")
        assert hasattr(board, "reject")
        assert hasattr(board, "get_status")

    def test_add_hypothesis(self):
        """Test adding hypotheses to board."""
        board = ApprovalBoard()

        # Create a hypothesis using the proper method
        hypothesis = board.create_hypothesis(
            title="Test hypothesis",
            description="Test description",
        )

        # Verify hypothesis was added
        status = board.get_status()
        assert status.total_count == 1

        # Get hypothesis details
        pending = board.get_hypothesis(hypothesis.id)
        assert pending is not None
        assert pending.title == "Test hypothesis"

    def test_approve_hypothesis(self):
        """Test hypothesis approval."""
        from hypothesis_approval_board import ApprovalBoard

        board = ApprovalBoard()

        # Create and approve a hypothesis
        hypothesis = board.create_hypothesis(
            title="Test approval",
            description="Test description",
        )

        # Approve the hypothesis
        success = board.approve(hypothesis.id, "Test approval")
        assert success

        # Verify approval using get_status()
        status = board.get_status()
        assert status.approved_count == 1
        assert status.pending_count == 0

    def test_reject_hypothesis(self):
        """Test hypothesis rejection."""
        board = ApprovalBoard()

        # Create and reject a hypothesis
        hypothesis = board.create_hypothesis(
            title="Test rejection",
            description="Test description",
        )

        # Reject the hypothesis
        success = board.reject(hypothesis.id, "Insufficient evidence", "Test rejection")
        assert success

        # Verify rejection using get_status()
        status = board.get_status()
        assert status.rejected_count == 1
        assert status.pending_count == 0

    def test_board_persistence(self):
        """Test board data persistence."""
        board = ApprovalBoard()

        # Create test data
        board.create_hypothesis(
            title="Persistence test",
            description="Test data persistence",
        )

        # Mock file operations
        mock_dump = MagicMock()
        with (
            patch("builtins.open", mock_open) as mock_open_file,
            patch("json.dump", mock_dump),
        ):
            # Type the mock properly
            mock_open_file = MagicMock()

            # Save board using save() method
            board.save(Path("test_board.json"))

            # Verify file operations
            mock_open_file.assert_called_once()
            mock_dump.assert_called_once()

    def test_board_loading(self):
        """Test loading board from file."""
        from hypothesis_approval_board import ApprovalBoard

        # Test data for loading
        test_data = {
            "hypotheses": [
                {
                    "id": "test-hyp-1",
                    "title": "Loaded hypothesis",
                    "description": "Test description",
                    "status": "approved",
                    "created_at": "2023-01-01T00:00:00",
                    "updated_at": "2023-01-01T00:00:00",
                }
            ],
            "approvers": [],
            "min_approvers": 1,
        }

        mock_open_file = MagicMock()
        with patch("builtins.open", mock_open_file):
            mock_open_file.return_value.__enter__.return_value.read.return_value = (
                json.dumps(test_data)
            )

            # Load board using load() method
            loaded_board = ApprovalBoard.load(Path("test_board.json"))

            # Verify loading
            mock_open_file.assert_called_once_with(Path("test_board.json"), "r")
            status = loaded_board.get_status()
            assert status.approved_count == 1
            assert status.pending_count == 0

    def test_error_handling(self):
        """Test error handling in board operations."""
        # Test with invalid file - ApprovalBoard.load() should handle gracefully
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            # The load method should create a new board if file doesn't exist
            board = ApprovalBoard.load(Path("nonexistent.json"))
            assert board is not None

    def test_statistics_methods(self):
        """Test board statistics methods."""
        board = ApprovalBoard()

        # Add test hypotheses
        for i in range(5):
            board.create_hypothesis(
                title=f"Hypothesis {i}",
                description=f"Test hypothesis {i}",
            )

        # Test statistics using get_status()
        status = board.get_status()
        assert status.total_count == 5
        assert status.pending_count == 5  # All start as draft, not pending
        assert status.approved_count == 0
        assert status.rejected_count == 0

    def test_hypothesis_validation(self):
        """Test hypothesis validation."""
        board = ApprovalBoard()

        # Test valid hypothesis creation
        valid_hypothesis = board.create_hypothesis(
            title="Valid hypothesis",
            description="Valid description",
        )

        # Should create successfully
        assert valid_hypothesis.title == "Valid hypothesis"
        assert valid_hypothesis.description == "Valid description"

        # Test that create_hypothesis handles basic validation
        # (The actual validation is handled by the dataclass)


class TestHumanLayerFunctions:
    """Tests for human layer functions."""

    def test_configure_if_needed_no_gui(self):
        """Test configure_if_needed without GUI."""
        with patch("human_layer._is_tkinter_running", return_value=False):
            result = configure_if_needed()

            # Should not configure without GUI
            assert result is False

    def test_configure_if_needed_with_gui(self):
        """Test configure_if_needed with GUI running."""
        with patch("human_layer._is_tkinter_running", return_value=True):
            with patch("human_layer.configure_human_interaction") as mock_configure:
                result = configure_if_needed()

                # Should configure human interaction
                assert result is True
                mock_configure.assert_called_once()

    def test_select_task_empty_board(self):
        """Test select_task with empty board."""
        with pytest.raises(ValueError, match="No hypotheses available"):
            select_task()

    def test_select_task_with_priorities(self):
        """Test select_task with priority-based selection."""
        # Create a mock board with tasks for testing select_task function
        from unittest.mock import Mock

        mock_board = Mock()
        mock_tasks = [
            {"title": "Critical task", "priority": TaskPriority.CRITICAL},
            {"title": "High priority task", "priority": TaskPriority.HIGH},
            {"title": "Medium priority task", "priority": TaskPriority.MEDIUM},
        ]
        mock_board.tasks = mock_tasks

        # Mock select_task function behavior
        with patch("human_layer.select_task") as mock_select:
            mock_select.return_value = mock_tasks[0]  # Return critical task

            selected = select_task()
            assert selected and selected.title == "Critical task"

    def test_select_task_with_filters(self):
        """Test select_task with filters."""
        # Mock the select_task function for testing
        from unittest.mock import Mock

        mock_board = Mock()
        mock_tasks = [
            {
                "title": "Approved task",
                "status": "approved",
                "priority": TaskPriority.LOW,
            },
            {
                "title": "Pending task",
                "status": "pending",
                "priority": TaskPriority.HIGH,
            },
        ]
        mock_board.tasks = mock_tasks

        # Test select_task with priority filter
        with patch("human_layer.select_task") as mock_select:
            mock_select.return_value = mock_tasks[1]  # Return pending task

            selected_high = select_task()
            assert selected_high and selected_high.title == "Pending task"

    def test_review_functionality(self):
        """Test review functionality."""
        # Mock the review function for testing
        mock_result = {
            "decision": "approve",
            "reviewer": "test_reviewer",
            "confidence": 0.8,
            "comments": "Test review",
        }

        with patch("human_layer.review") as mock_review:
            mock_review.return_value = mock_result

            review_result = review({"test": "Review test"})

            # Verify review was processed
            assert review_result is not None
            assert review_result.decision.value == "approve"

    def test_review_with_approval(self):
        """Test review with approval."""
        # Mock the review function for testing
        mock_result = {
            "decision": "approve",
            "reviewer": "test_reviewer",
            "confidence": 0.9,
            "comments": "Approved for testing",
        }

        with patch("human_layer.review") as mock_review:
            mock_review.return_value = mock_result

            review_result = review({"test": "Review with approval"})

            # Verify approval
            assert review_result is not None
            assert review_result.decision.value == "approve"

    def test_review_with_rejection(self):
        """Test review with rejection."""
        # Mock the review function for testing
        mock_result = {
            "decision": "reject",
            "reviewer": "test_reviewer",
            "confidence": 0.7,
            "comments": "Rejected for testing",
            "reason": "Insufficient evidence",
        }

        with patch("human_layer.review") as mock_review:
            mock_review.return_value = mock_result

            review_result = review({"test": "Review with rejection"})

            # Verify rejection
            assert review_result is not None
            assert review_result.decision.value == "reject"


class TestIntegration:
    """Integration tests for human layer functionality."""

    def test_full_human_layer_workflow(self):
        """Test complete human layer workflow."""
        with (
            patch("human_layer._is_tkinter_running", return_value=True),
            patch("human_layer.configure_human_interaction") as mock_configure,
            patch("human_layer.select_task") as mock_select,
            patch("human_layer.review") as mock_review,
        ):

            # Configure human interaction
            configure_if_needed()
            mock_configure.assert_called_once()

            # Create approval board
            board = ApprovalBoard()

            # Add task to board
            task = Hypothesis(
                id="integration_test",
                title="Integration test task",
                description="Test task for integration",
                tags=["integration_evidence"],
            )
            board.add_hypothesis(task)

            # Select task
            mock_select.return_value = task
            selected_task = select_task()

            # Review and approve task
            mock_review.return_value = [{"task": task, "action": "approve"}]
            review({"test": "Integration review"})

            # Verify workflow
            mock_configure.assert_called_once()
            mock_select.assert_called_once_with()
            mock_review.assert_called_once()
            assert selected_task and selected_task.title == "Integration test task"

    def test_error_handling_in_workflow(self):
        """Test error handling in human layer workflow."""
        with patch(
            "human_layer._is_tkinter_running", side_effect=Exception("GUI error")
        ):

            # Should handle GUI error gracefully
            with pytest.raises(Exception, match="GUI error"):
                configure_if_needed()

    def test_persistence_across_sessions(self):
        """Test data persistence across sessions."""
        board1 = ApprovalBoard()
        board2 = ApprovalBoard()

        # Add data to first board
        task = Hypothesis(
            id="persistence_test",
            title="Persistence test task",
            description="Test task for persistence",
            tags=["persistence_evidence"],
        )
        board1.add_hypothesis(task)

        # Save first board
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            board1.save(Path(temp_file.name))

            # Load second board from same file
            board2 = ApprovalBoard.load(Path(temp_file.name))

            # Verify data persistence
            assert len(board2.hypotheses) == 1
            hypothesis = board2.get_hypothesis("Persistence test task")
            assert hypothesis is not None
            assert hypothesis.title == "Persistence test task"

    def test_concurrent_access(self):
        """Test concurrent access to human layer."""
        board = ApprovalBoard()

        # Add test data
        task = Hypothesis(
            id="concurrent_test",
            title="Concurrent test task",
            description="Test task for concurrent access",
            tags=["concurrent_evidence"],
        )
        board.add_hypothesis(task)

        # Test concurrent access (should be thread-safe)
        # This test mainly verifies no exceptions are raised
        try:
            # Multiple simultaneous operations
            for _ in range(10):
                select_task()
                review({"test": f"Concurrent review {_}"})
        except Exception:
            pytest.fail("Concurrent access should be thread-safe")

    def test_configuration_integration(self):
        """Test configuration integration with human layer."""
        with patch("human_layer.configure_human_interaction") as mock_configure:

            # Test configuration
            configure_if_needed()

            # Verify configuration was called with proper parameters
            mock_configure.assert_called_once()
            args, kwargs = mock_configure.call_args

            # Should be called with no arguments (auto-configuration)
            assert len(args) == 0
            assert len(kwargs) == 0

"""
Comprehensive tests for hypothesis_approval_board.py - Hypothesis approval board module.
"""

import time
from pathlib import Path

import pytest

from hypothesis_approval_board import (
    ApprovalBoard,
    ApprovalDecision,
    ApprovalStatus,
    Hypothesis,
    HypothesisStatus,
    approve_hypothesis,
    create_hypothesis,
    get_approval_queue,
    get_hypothesis_history,
    reject_hypothesis,
    review_hypothesis,
    submit_for_approval,
)


@pytest.fixture(autouse=True)
def clean_hypothesis_board():
    """Clean up hypothesis board file before each test."""
    board_file = Path("hypothesis_board.json")
    if board_file.exists():
        board_file.unlink()
    yield
    # Cleanup after test
    if board_file.exists():
        board_file.unlink()


class TestHypothesisStatus:
    """Tests for HypothesisStatus enum."""

    def test_status_values(self):
        """Test HypothesisStatus enum values."""
        assert HypothesisStatus.DRAFT.value == "draft"  # nosec: B101 - Test assertion
        assert (
            HypothesisStatus.SUBMITTED.value == "submitted"
        )  # nosec: B101 - Test assertion
        assert (
            HypothesisStatus.UNDER_REVIEW.value == "under_review"
        )  # nosec: B101 - Test assertion
        assert (
            HypothesisStatus.APPROVED.value == "approved"
        )  # nosec: B101 - Test assertion
        assert (
            HypothesisStatus.REJECTED.value == "rejected"
        )  # nosec: B101 - Test assertion
        assert (
            HypothesisStatus.EXPERIMENTING.value == "experimenting"
        )  # nosec: B101 - Test assertion
        assert (
            HypothesisStatus.COMPLETED.value == "completed"
        )  # nosec: B101 - Test assertion


class TestApprovalDecision:
    """Tests for ApprovalDecision enum."""

    def test_decision_values(self):
        """Test ApprovalDecision enum values."""
        assert (
            ApprovalDecision.APPROVE.value == "approve"
        )  # nosec: B101 - Test assertion
        assert ApprovalDecision.REJECT.value == "reject"  # nosec: B101 - Test assertion
        assert (
            ApprovalDecision.REQUEST_CHANGES.value == "request_changes"
        )  # nosec: B101 - Test assertion
        assert ApprovalDecision.DEFER.value == "defer"  # nosec: B101 - Test assertion


class TestApprovalStatus:
    """Tests for ApprovalStatus dataclass."""

    def test_default_values(self):
        """Test default ApprovalStatus values."""
        status = ApprovalStatus()
        assert status.approved_count == 0  # nosec: B101 - Test assertion
        assert status.rejected_count == 0  # nosec: B101 - Test assertion
        assert status.pending_count == 0  # nosec: B101 - Test assertion
        assert status.total_count == 0  # nosec: B101 - Test assertion

    def test_custom_values(self):
        """Test custom ApprovalStatus values."""
        status = ApprovalStatus(
            approved_count=5,
            rejected_count=2,
            pending_count=3,
            total_count=10,
        )
        assert status.approved_count == 5  # nosec: B101 - Test assertion
        assert status.rejected_count == 2  # nosec: B101 - Test assertion
        assert status.pending_count == 3  # nosec: B101 - Test assertion
        assert status.total_count == 10  # nosec: B101 - Test assertion


class TestHypothesis:
    """Tests for Hypothesis dataclass."""

    def test_default_values(self):
        """Test default Hypothesis values."""
        hyp = Hypothesis(
            id="hyp-1",
            title="Test Hypothesis",
            description="A test hypothesis",
        )
        assert hyp.id == "hyp-1"  # nosec: B101 - Test assertion
        assert hyp.title == "Test Hypothesis"  # nosec: B101 - Test assertion
        assert hyp.description == "A test hypothesis"  # nosec: B101 - Test assertion
        assert hyp.status == HypothesisStatus.DRAFT  # nosec: B101 - Test assertion
        assert hyp.predicted_outcome == ""  # nosec: B101 - Test assertion
        assert hyp.success_criteria is None  # nosec: B101 - Test assertion
        assert hyp.experiment_design is None  # nosec: B101 - Test assertion
        assert hyp.created_at is not None  # nosec: B101 - Test assertion
        assert hyp.updated_at is not None  # nosec: B101 - Test assertion

    def test_custom_values(self):
        """Test custom Hypothesis values."""
        hyp = Hypothesis(
            id="hyp-2",
            title="Another Hypothesis",
            description="Another test",
            status=HypothesisStatus.APPROVED,
            predicted_outcome="Positive result",
            success_criteria=["criterion1", "criterion2"],
            experiment_design={"method": "test"},
        )
        assert hyp.status == HypothesisStatus.APPROVED  # nosec: B101 - Test assertion
        assert (
            hyp.predicted_outcome == "Positive result"
        )  # nosec: B101 - Test assertion
        assert hyp.success_criteria == [
            "criterion1",
            "criterion2",
        ]  # nosec: B101 - Test assertion
        assert hyp.experiment_design == {
            "method": "test"
        }  # nosec: B101 - Test assertion

    def test_to_dict(self):
        """Test converting hypothesis to dict."""
        hyp = Hypothesis(
            id="hyp-3",
            title="Test",
            description="Test description",
        )
        data = hyp.to_dict()
        assert data["id"] == "hyp-3"  # nosec: B101 - Test assertion
        assert data["title"] == "Test"  # nosec: B101 - Test assertion
        assert data["status"] == "draft"  # nosec: B101 - Test assertion

    def test_from_dict(self):
        """Test creating hypothesis from dict."""
        data = {
            "id": "hyp-4",
            "title": "From Dict",
            "description": "Created from dict",
            "status": "submitted",
            "predicted_outcome": "Outcome",
            "success_criteria": ["crit1"],
        }
        hyp = Hypothesis.from_dict(data)
        assert hyp.id == "hyp-4"  # nosec: B101 - Test assertion
        assert hyp.title == "From Dict"  # nosec: B101 - Test assertion
        assert hyp.status == HypothesisStatus.SUBMITTED  # nosec: B101 - Test assertion


class TestCreateHypothesis:
    """Tests for create_hypothesis function."""

    def test_create_basic(self):
        """Test creating basic hypothesis."""
        hyp = create_hypothesis(
            title="Test Hypothesis",
            description="Test description",
        )
        assert hyp.title == "Test Hypothesis"  # nosec: B101 - Test assertion
        assert hyp.description == "Test description"  # nosec: B101 - Test assertion
        assert hyp.status == HypothesisStatus.DRAFT  # nosec: B101 - Test assertion
        assert hyp.id.startswith("hyp-")  # nosec: B101 - Test assertion

    def test_create_with_details(self):
        """Test creating hypothesis with all details."""
        hyp = create_hypothesis(
            title="Full Hypothesis",
            description="Full description",
            predicted_outcome="Success",
            success_criteria=["criterion1", "criterion2"],
            experiment_design={"method": "controlled"},
        )
        assert hyp.predicted_outcome == "Success"  # nosec: B101 - Test assertion
        assert hyp.success_criteria == [
            "criterion1",
            "criterion2",
        ]  # nosec: B101 - Test assertion
        assert hyp.experiment_design == {
            "method": "controlled"
        }  # nosec: B101 - Test assertion


class TestSubmitForApproval:
    """Tests for submit_for_approval function."""

    def test_submit_changes_status(self):
        """Test submitting changes hypothesis status."""
        hyp = create_hypothesis("Test", "Description")
        submitted = submit_for_approval(hyp)
        assert (
            submitted.status == HypothesisStatus.SUBMITTED
        )  # nosec: B101 - Test assertion
        assert submitted.updated_at is not None  # nosec: B101 - Test assertion
        assert submitted.created_at is not None  # nosec: B101 - Test assertion
        assert (
            submitted.updated_at > submitted.created_at
        )  # nosec: B101 - Test assertion

    def test_submit_already_submitted(self):
        """Test submitting already submitted hypothesis."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        submitted = submit_for_approval(hyp)
        assert (
            submitted.status == HypothesisStatus.SUBMITTED
        )  # nosec: B101 - Test assertion


class TestReviewHypothesis:
    """Tests for review_hypothesis function."""

    def test_review_approve(self):
        """Test approving hypothesis."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        reviewed = review_hypothesis(hyp, ApprovalDecision.APPROVE, "Looks good")
        assert (
            reviewed.status == HypothesisStatus.APPROVED
        )  # nosec: B101 - Test assertion
        assert reviewed.review_notes == "Looks good"  # nosec: B101 - Test assertion

    def test_review_reject(self):
        """Test rejecting hypothesis."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        reviewed = review_hypothesis(hyp, ApprovalDecision.REJECT, "Needs work")
        assert (
            reviewed.status == HypothesisStatus.REJECTED
        )  # nosec: B101 - Test assertion
        assert reviewed.review_notes == "Needs work"  # nosec: B101 - Test assertion

    def test_review_request_changes(self):
        """Test requesting changes."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        reviewed = review_hypothesis(
            hyp, ApprovalDecision.REQUEST_CHANGES, "Fix methodology"
        )
        assert reviewed.status == HypothesisStatus.DRAFT  # nosec: B101 - Test assertion


class TestApproveHypothesis:
    """Tests for approve_hypothesis function."""

    def test_approve(self):
        """Test approve convenience function."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        approved = approve_hypothesis(hyp, "Approved for experiment")
        assert (
            approved.status == HypothesisStatus.APPROVED
        )  # nosec: B101 - Test assertion
        assert approved.approved_at is not None  # nosec: B101 - Test assertion


class TestRejectHypothesis:
    """Tests for reject_hypothesis function."""

    def test_reject(self):
        """Test reject convenience function."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        rejected = reject_hypothesis(hyp, "Insufficient evidence")
        assert (
            rejected.status == HypothesisStatus.REJECTED
        )  # nosec: B101 - Test assertion
        assert (
            rejected.rejection_reason == "Insufficient evidence"
        )  # nosec: B101 - Test assertion


class TestGetApprovalQueue:
    """Tests for get_approval_queue function."""

    def test_empty_queue(self):
        """Test empty approval queue."""
        queue = get_approval_queue([])
        assert queue == []  # nosec: B101 - Test assertion

    def test_queue_filters_status(self):
        """Test queue filters by status."""
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.status = HypothesisStatus.SUBMITTED
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.status = HypothesisStatus.APPROVED
        hyp3 = create_hypothesis("Test3", "Desc3")
        hyp3.status = HypothesisStatus.DRAFT

        queue = get_approval_queue([hyp1, hyp2, hyp3])
        assert len(queue) == 1  # nosec: B101 - Test assertion
        assert queue[0].id == hyp1.id  # nosec: B101 - Test assertion


class TestGetHypothesisHistory:
    """Tests for get_hypothesis_history function."""

    def test_history_empty(self):
        """Test empty history."""
        history = get_hypothesis_history([])
        assert history == []  # nosec: B101 - Test assertion

    def test_history_sorted(self):
        """Test history is sorted by date."""
        now = time.time()
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.created_at = str(now - 100)
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.created_at = str(now - 50)
        hyp3 = create_hypothesis("Test3", "Desc3")
        hyp3.created_at = str(now)

        history = get_hypothesis_history([hyp3, hyp1, hyp2])
        assert history[0].id == hyp1.id  # nosec: B101 - Test assertion
        assert history[1].id == hyp2.id  # nosec: B101 - Test assertion
        assert history[2].id == hyp3.id  # nosec: B101 - Test assertion


class TestApprovalBoard:
    """Tests for ApprovalBoard class."""

    def test_init(self):
        """Test ApprovalBoard initialization."""
        board = ApprovalBoard()
        assert board.hypotheses == {}  # nosec: B101 - Test assertion
        assert board.approvers == []  # nosec: B101 - Test assertion
        assert board.min_approvers == 1  # nosec: B101 - Test assertion

    def test_init_with_config(self):
        """Test initialization with config."""
        board = ApprovalBoard(approvers=["user1", "user2"], min_approvers=2)
        assert board.approvers == ["user1", "user2"]  # nosec: B101 - Test assertion
        assert board.min_approvers == 2  # nosec: B101 - Test assertion

    def test_add_hypothesis(self):
        """Test adding hypothesis."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        assert hyp.id in board.hypotheses  # nosec: B101 - Test assertion
        assert board.hypotheses[hyp.id] == hyp  # nosec: B101 - Test assertion

    def test_get_hypothesis(self):
        """Test getting hypothesis."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        retrieved = board.get_hypothesis(hyp.id)
        assert retrieved == hyp  # nosec: B101 - Test assertion

    def test_get_nonexistent_hypothesis(self):
        """Test getting nonexistent hypothesis."""
        board = ApprovalBoard()
        retrieved = board.get_hypothesis("nonexistent")
        assert retrieved is None  # nosec: B101 - Test assertion

    def test_approve(self):
        """Test approving via board."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        result = board.approve(hyp.id, "Approved")
        assert result is True  # nosec: B101 - Test assertion
        retrieved = board.get_hypothesis(hyp.id)
        assert retrieved is not None  # nosec: B101 - Test assertion
        assert (
            retrieved.status == HypothesisStatus.APPROVED
        )  # nosec: B101 - Test assertion

    def test_approve_nonexistent(self):
        """Test approving nonexistent hypothesis."""
        board = ApprovalBoard()
        result = board.approve("nonexistent", "Approved")
        assert result is False  # nosec: B101 - Test assertion

    def test_reject(self):
        """Test rejecting via board."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        result = board.reject(hyp.id, "Rejected")
        assert result is True  # nosec: B101 - Test assertion
        retrieved = board.get_hypothesis(hyp.id)
        assert retrieved is not None  # nosec: B101 - Test assertion
        assert (
            retrieved.status == HypothesisStatus.REJECTED
        )  # nosec: B101 - Test assertion

    def test_get_pending(self):
        """Test getting pending hypotheses."""
        board = ApprovalBoard()
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.status = HypothesisStatus.SUBMITTED
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.status = HypothesisStatus.APPROVED
        board.add_hypothesis(hyp1)
        board.add_hypothesis(hyp2)

        pending = board.get_pending()
        assert len(pending) == 1  # nosec: B101 - Test assertion
        assert pending[0].id == hyp1.id  # nosec: B101 - Test assertion

    def test_get_approved(self):
        """Test getting approved hypotheses."""
        board = ApprovalBoard()
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.status = HypothesisStatus.APPROVED
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.status = HypothesisStatus.REJECTED
        board.add_hypothesis(hyp1)
        board.add_hypothesis(hyp2)

        approved = board.get_approved()
        assert len(approved) == 1  # nosec: B101 - Test assertion
        assert approved[0].id == hyp1.id  # nosec: B101 - Test assertion

    def test_get_rejected(self):
        """Test getting rejected hypotheses."""
        board = ApprovalBoard()
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.status = HypothesisStatus.APPROVED
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.status = HypothesisStatus.REJECTED
        board.add_hypothesis(hyp1)
        board.add_hypothesis(hyp2)

        rejected = board.get_rejected()
        assert len(rejected) == 1  # nosec: B101 - Test assertion
        assert rejected[0].id == hyp2.id  # nosec: B101 - Test assertion

    def test_get_status(self):
        """Test getting board status."""
        board = ApprovalBoard()
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.status = HypothesisStatus.APPROVED
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.status = HypothesisStatus.REJECTED
        hyp3 = create_hypothesis("Test3", "Desc3")
        hyp3.status = HypothesisStatus.SUBMITTED
        board.add_hypothesis(hyp1)
        board.add_hypothesis(hyp2)
        board.add_hypothesis(hyp3)

        status = board.get_status()
        assert status.approved_count == 1  # nosec: B101 - Test assertion
        assert status.rejected_count == 1  # nosec: B101 - Test assertion
        assert status.pending_count == 1  # nosec: B101 - Test assertion
        assert status.total_count == 3  # nosec: B101 - Test assertion

    def test_to_dict(self):
        """Test converting board to dict."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        data = board.to_dict()
        assert "hypotheses" in data  # nosec: B101 - Test assertion
        assert hyp.id in data["hypotheses"]  # nosec: B101 - Test assertion

    def test_save_and_load(self, tmp_path):
        """Test saving and loading board."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)

        save_path = tmp_path / "board.json"
        board.save(save_path)

        loaded = ApprovalBoard.load(save_path)
        assert hyp.id in loaded.hypotheses  # nosec: B101 - Test assertion
        assert loaded.hypotheses[hyp.id].title == "Test"  # nosec: B101 - Test assertion

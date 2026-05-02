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
        assert HypothesisStatus.DRAFT.value == "draft"
        assert HypothesisStatus.SUBMITTED.value == "submitted"
        assert HypothesisStatus.UNDER_REVIEW.value == "under_review"
        assert HypothesisStatus.APPROVED.value == "approved"
        assert HypothesisStatus.REJECTED.value == "rejected"
        assert HypothesisStatus.EXPERIMENTING.value == "experimenting"
        assert HypothesisStatus.COMPLETED.value == "completed"


class TestApprovalDecision:
    """Tests for ApprovalDecision enum."""

    def test_decision_values(self):
        """Test ApprovalDecision enum values."""
        assert ApprovalDecision.APPROVE.value == "approve"
        assert ApprovalDecision.REJECT.value == "reject"
        assert ApprovalDecision.REQUEST_CHANGES.value == "request_changes"
        assert ApprovalDecision.DEFER.value == "defer"


class TestApprovalStatus:
    """Tests for ApprovalStatus dataclass."""

    def test_default_values(self):
        """Test default ApprovalStatus values."""
        status = ApprovalStatus()
        assert status.approved_count == 0
        assert status.rejected_count == 0
        assert status.pending_count == 0
        assert status.total_count == 0

    def test_custom_values(self):
        """Test custom ApprovalStatus values."""
        status = ApprovalStatus(
            approved_count=5,
            rejected_count=2,
            pending_count=3,
            total_count=10,
        )
        assert status.approved_count == 5
        assert status.rejected_count == 2
        assert status.pending_count == 3
        assert status.total_count == 10


class TestHypothesis:
    """Tests for Hypothesis dataclass."""

    def test_default_values(self):
        """Test default Hypothesis values."""
        hyp = Hypothesis(
            id="hyp-1",
            title="Test Hypothesis",
            description="A test hypothesis",
        )
        assert hyp.id == "hyp-1"
        assert hyp.title == "Test Hypothesis"
        assert hyp.description == "A test hypothesis"
        assert hyp.status == HypothesisStatus.DRAFT
        assert hyp.predicted_outcome == ""
        assert hyp.success_criteria is None
        assert hyp.experiment_design is None
        assert hyp.created_at is not None
        assert hyp.updated_at is not None

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
        assert hyp.status == HypothesisStatus.APPROVED
        assert hyp.predicted_outcome == "Positive result"
        assert hyp.success_criteria == ["criterion1", "criterion2"]
        assert hyp.experiment_design == {"method": "test"}

    def test_to_dict(self):
        """Test converting hypothesis to dict."""
        hyp = Hypothesis(
            id="hyp-3",
            title="Test",
            description="Test description",
        )
        data = hyp.to_dict()
        assert data["id"] == "hyp-3"
        assert data["title"] == "Test"
        assert data["status"] == "draft"

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
        assert hyp.id == "hyp-4"
        assert hyp.title == "From Dict"
        assert hyp.status == HypothesisStatus.SUBMITTED


class TestCreateHypothesis:
    """Tests for create_hypothesis function."""

    def test_create_basic(self):
        """Test creating basic hypothesis."""
        hyp = create_hypothesis(
            title="Test Hypothesis",
            description="Test description",
        )
        assert hyp.title == "Test Hypothesis"
        assert hyp.description == "Test description"
        assert hyp.status == HypothesisStatus.DRAFT
        assert hyp.id.startswith("hyp-")

    def test_create_with_details(self):
        """Test creating hypothesis with all details."""
        hyp = create_hypothesis(
            title="Full Hypothesis",
            description="Full description",
            predicted_outcome="Success",
            success_criteria=["criterion1", "criterion2"],
            experiment_design={"method": "controlled"},
        )
        assert hyp.predicted_outcome == "Success"
        assert hyp.success_criteria == ["criterion1", "criterion2"]
        assert hyp.experiment_design == {"method": "controlled"}


class TestSubmitForApproval:
    """Tests for submit_for_approval function."""

    def test_submit_changes_status(self):
        """Test submitting changes hypothesis status."""
        hyp = create_hypothesis("Test", "Description")
        submitted = submit_for_approval(hyp)
        assert submitted.status == HypothesisStatus.SUBMITTED
        assert submitted.updated_at is not None
        assert submitted.created_at is not None
        assert submitted.updated_at > submitted.created_at

    def test_submit_already_submitted(self):
        """Test submitting already submitted hypothesis."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        submitted = submit_for_approval(hyp)
        assert submitted.status == HypothesisStatus.SUBMITTED


class TestReviewHypothesis:
    """Tests for review_hypothesis function."""

    def test_review_approve(self):
        """Test approving hypothesis."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        reviewed = review_hypothesis(hyp, ApprovalDecision.APPROVE, "Looks good")
        assert reviewed.status == HypothesisStatus.APPROVED
        assert reviewed.review_notes == "Looks good"

    def test_review_reject(self):
        """Test rejecting hypothesis."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        reviewed = review_hypothesis(hyp, ApprovalDecision.REJECT, "Needs work")
        assert reviewed.status == HypothesisStatus.REJECTED
        assert reviewed.review_notes == "Needs work"

    def test_review_request_changes(self):
        """Test requesting changes."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        reviewed = review_hypothesis(
            hyp, ApprovalDecision.REQUEST_CHANGES, "Fix methodology"
        )
        assert reviewed.status == HypothesisStatus.DRAFT


class TestApproveHypothesis:
    """Tests for approve_hypothesis function."""

    def test_approve(self):
        """Test approve convenience function."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        approved = approve_hypothesis(hyp, "Approved for experiment")
        assert approved.status == HypothesisStatus.APPROVED
        assert approved.approved_at is not None


class TestRejectHypothesis:
    """Tests for reject_hypothesis function."""

    def test_reject(self):
        """Test reject convenience function."""
        hyp = create_hypothesis("Test", "Description")
        hyp.status = HypothesisStatus.SUBMITTED
        rejected = reject_hypothesis(hyp, "Insufficient evidence")
        assert rejected.status == HypothesisStatus.REJECTED
        assert rejected.rejection_reason == "Insufficient evidence"


class TestGetApprovalQueue:
    """Tests for get_approval_queue function."""

    def test_empty_queue(self):
        """Test empty approval queue."""
        queue = get_approval_queue([])
        assert queue == []

    def test_queue_filters_status(self):
        """Test queue filters by status."""
        hyp1 = create_hypothesis("Test1", "Desc1")
        hyp1.status = HypothesisStatus.SUBMITTED
        hyp2 = create_hypothesis("Test2", "Desc2")
        hyp2.status = HypothesisStatus.APPROVED
        hyp3 = create_hypothesis("Test3", "Desc3")
        hyp3.status = HypothesisStatus.DRAFT

        queue = get_approval_queue([hyp1, hyp2, hyp3])
        assert len(queue) == 1
        assert queue[0].id == hyp1.id


class TestGetHypothesisHistory:
    """Tests for get_hypothesis_history function."""

    def test_history_empty(self):
        """Test empty history."""
        history = get_hypothesis_history([])
        assert history == []

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
        assert history[0].id == hyp1.id
        assert history[1].id == hyp2.id
        assert history[2].id == hyp3.id


class TestApprovalBoard:
    """Tests for ApprovalBoard class."""

    def test_init(self):
        """Test ApprovalBoard initialization."""
        board = ApprovalBoard()
        assert board.hypotheses == {}
        assert board.approvers == []
        assert board.min_approvers == 1

    def test_init_with_config(self):
        """Test initialization with config."""
        board = ApprovalBoard(approvers=["user1", "user2"], min_approvers=2)
        assert board.approvers == ["user1", "user2"]
        assert board.min_approvers == 2

    def test_add_hypothesis(self):
        """Test adding hypothesis."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        assert hyp.id in board.hypotheses
        assert board.hypotheses[hyp.id] == hyp

    def test_get_hypothesis(self):
        """Test getting hypothesis."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        retrieved = board.get_hypothesis(hyp.id)
        assert retrieved == hyp

    def test_get_nonexistent_hypothesis(self):
        """Test getting nonexistent hypothesis."""
        board = ApprovalBoard()
        retrieved = board.get_hypothesis("nonexistent")
        assert retrieved is None

    def test_approve(self):
        """Test approving via board."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        result = board.approve(hyp.id, "Approved")
        assert result is True
        retrieved = board.get_hypothesis(hyp.id)
        assert retrieved is not None
        assert retrieved.status == HypothesisStatus.APPROVED

    def test_approve_nonexistent(self):
        """Test approving nonexistent hypothesis."""
        board = ApprovalBoard()
        result = board.approve("nonexistent", "Approved")
        assert result is False

    def test_reject(self):
        """Test rejecting via board."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        result = board.reject(hyp.id, "Rejected")
        assert result is True
        retrieved = board.get_hypothesis(hyp.id)
        assert retrieved is not None
        assert retrieved.status == HypothesisStatus.REJECTED

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
        assert len(pending) == 1
        assert pending[0].id == hyp1.id

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
        assert len(approved) == 1
        assert approved[0].id == hyp1.id

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
        assert len(rejected) == 1
        assert rejected[0].id == hyp2.id

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
        assert status.approved_count == 1
        assert status.rejected_count == 1
        assert status.pending_count == 1
        assert status.total_count == 3

    def test_to_dict(self):
        """Test converting board to dict."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)
        data = board.to_dict()
        assert "hypotheses" in data
        assert hyp.id in data["hypotheses"]

    def test_save_and_load(self, tmp_path):
        """Test saving and loading board."""
        board = ApprovalBoard()
        hyp = create_hypothesis("Test", "Description")
        board.add_hypothesis(hyp)

        save_path = tmp_path / "board.json"
        board.save(save_path)

        loaded = ApprovalBoard.load(save_path)
        assert hyp.id in loaded.hypotheses
        assert loaded.hypotheses[hyp.id].title == "Test"

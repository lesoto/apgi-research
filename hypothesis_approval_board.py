"""
Hypothesis Approval Board for APGI Experiment Review System.

This module provides a structured way to manage, review, and approve
hypotheses for experiments before execution.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Status of hypothesis in the approval workflow."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPERIMENTING = "experimenting"
    COMPLETED = "completed"


class ApprovalDecision(Enum):
    """Decision options for hypothesis review."""

    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    DEFER = "defer"


@dataclass
class ApprovalStatus:
    """Status summary for the approval board."""

    approved_count: int = 0
    rejected_count: int = 0
    pending_count: int = 0
    total_count: int = 0


@dataclass
class Hypothesis:
    """Represents a scientific hypothesis for experiment validation."""

    id: str
    title: str
    description: str
    predicted_outcome: str = ""
    confidence_score: float = 0.5  # 0.0 to 1.0
    risk_assessment: str = "medium"  # "low", "medium", "high"
    created_at: Optional[str] = None
    success_criteria: Optional[List[str]] = None  # Metrics that define success
    reviewer_comments: str = ""
    status: HypothesisStatus = HypothesisStatus.DRAFT
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    tags: Optional[List[str]] = None  # e.g., ["cognitive", "performance", "safety"]
    experiment_design: Optional[Dict[str, Any]] = None
    review_notes: str = ""
    approved_at: Optional[str] = None
    rejection_reason: str = ""
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "predicted_outcome": self.predicted_outcome,
            "confidence_score": self.confidence_score,
            "success_criteria": self.success_criteria or [],
            "risk_assessment": self.risk_assessment,
            "reviewer_comments": self.reviewer_comments,
            "status": (
                self.status.value
                if isinstance(self.status, HypothesisStatus)
                else self.status
            ),
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": self.reviewed_by,
            "tags": self.tags or [],
            "experiment_design": self.experiment_design or {},
            "review_notes": self.review_notes,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        """Reconstruct Hypothesis from dictionary."""
        # Make a copy to avoid modifying the input
        data_copy = dict(data)

        # Convert status string back to enum
        if "status" in data_copy and isinstance(data_copy["status"], str):
            try:
                data_copy["status"] = HypothesisStatus(data_copy["status"])
            except ValueError:
                data_copy["status"] = HypothesisStatus.DRAFT

        # Ensure confidence_score is float
        if "confidence_score" in data_copy:
            data_copy["confidence_score"] = float(data_copy["confidence_score"])

        # Set default values for optional fields
        if "experiment_design" not in data_copy:
            data_copy["experiment_design"] = {}
        if "review_notes" not in data_copy:
            data_copy["review_notes"] = ""
        if "rejection_reason" not in data_copy:
            data_copy["rejection_reason"] = ""

        return cls(**data_copy)


class ApprovalBoard:
    """Manages the hypothesis approval workflow."""

    def __init__(
        self,
        storage_path: str = "hypothesis_board.json",
        approvers: Optional[List[str]] = None,
        min_approvers: int = 1,
    ):
        self.storage_path = Path(storage_path)
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.approvers: List[str] = approvers or []
        self.min_approvers = min_approvers
        self._load_hypotheses()

    def _load_hypotheses(self) -> None:
        """Load hypotheses from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "hypotheses" in data:
                        # New format with metadata
                        hypotheses_data = data["hypotheses"]
                        if isinstance(hypotheses_data, dict):
                            # Dict format: {"id1": {...}, "id2": {...}}
                            self.hypotheses = {
                                k: Hypothesis.from_dict(v)
                                for k, v in hypotheses_data.items()
                            }
                        elif isinstance(hypotheses_data, list):
                            # List format: [{...}, {...}]
                            self.hypotheses = {
                                h["id"]: Hypothesis.from_dict(h)
                                for h in hypotheses_data
                            }
                        self.approvers = data.get("approvers", [])
                        self.min_approvers = data.get("min_approvers", 1)
                    elif isinstance(data, list):
                        # Legacy format
                        self.hypotheses = {
                            h.id: h for h in [Hypothesis.from_dict(h) for h in data]
                        }
            except Exception as e:
                logger.error(f"Failed to load hypotheses: {e}")

    def _save_hypotheses(self) -> None:
        """Save hypotheses to storage file."""
        try:
            with open(self.storage_path, "w") as f:
                # Convert to dict first to handle nested dataclasses/enums
                serializable_data = {
                    "hypotheses": [h.to_dict() for h in self.hypotheses.values()],
                    "approvers": self.approvers,
                    "min_approvers": self.min_approvers,
                }
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save hypotheses: {e}")

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to the board."""
        self.hypotheses[hypothesis.id] = hypothesis
        self._save_hypotheses()

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)

    def approve(
        self, hypothesis_id: str, comments: str = "", approved_by: str = ""
    ) -> bool:
        """Approve a hypothesis."""
        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False

        now = datetime.now().isoformat()
        hypothesis.status = HypothesisStatus.APPROVED
        hypothesis.review_notes = comments
        hypothesis.reviewed_by = approved_by
        hypothesis.approved_at = now
        hypothesis.reviewed_at = now
        hypothesis.updated_at = now

        self._save_hypotheses()
        logger.info(f"Approved hypothesis {hypothesis_id}")
        return True

    def reject(
        self, hypothesis_id: str, reason: str = "", rejected_by: str = ""
    ) -> bool:
        """Reject a hypothesis."""
        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False

        now = datetime.now().isoformat()
        hypothesis.status = HypothesisStatus.REJECTED
        hypothesis.rejection_reason = reason
        hypothesis.reviewed_by = rejected_by
        hypothesis.reviewed_at = now
        hypothesis.updated_at = now

        self._save_hypotheses()
        logger.info(f"Rejected hypothesis {hypothesis_id}")
        return True

    def get_pending(self) -> List[Hypothesis]:
        """Get all hypotheses pending review (submitted status)."""
        return [
            h
            for h in self.hypotheses.values()
            if h.status == HypothesisStatus.SUBMITTED
        ]

    def get_approved(self) -> List[Hypothesis]:
        """Get all approved hypotheses."""
        return [
            h for h in self.hypotheses.values() if h.status == HypothesisStatus.APPROVED
        ]

    def get_rejected(self) -> List[Hypothesis]:
        """Get all rejected hypotheses."""
        return [
            h for h in self.hypotheses.values() if h.status == HypothesisStatus.REJECTED
        ]

    def get_status(self) -> ApprovalStatus:
        """Get the current approval status summary."""
        approved = len(self.get_approved())
        rejected = len(self.get_rejected())
        pending = len(self.get_pending())
        total = len(self.hypotheses)

        return ApprovalStatus(
            approved_count=approved,
            rejected_count=rejected,
            pending_count=pending,
            total_count=total,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert board to dictionary."""
        return {
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            "approvers": self.approvers,
            "min_approvers": self.min_approvers,
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save the board to a file."""
        save_path = path or self.storage_path
        try:
            with open(save_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save board: {e}")

    @classmethod
    def load(cls, path: Path) -> "ApprovalBoard":
        """Load a board from a file."""
        board = cls(storage_path=str(path))
        return board

    def create_hypothesis(
        self,
        title: str,
        description: str,
        predicted_outcome: str = "",
        confidence_score: float = 0.5,
        success_criteria: Optional[List[str]] = None,
        risk_assessment: str = "medium",
        tags: Optional[List[str]] = None,
        experiment_design: Optional[Dict[str, Any]] = None,
    ) -> Hypothesis:
        """Create a new hypothesis."""
        import uuid

        now = datetime.now().isoformat()
        hypothesis = Hypothesis(
            id=f"hyp-{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            predicted_outcome=predicted_outcome,
            confidence_score=confidence_score,
            success_criteria=success_criteria or [],
            risk_assessment=risk_assessment,
            tags=tags or [],
            status=HypothesisStatus.DRAFT,
            created_at=now,
            updated_at=now,
            experiment_design=experiment_design or {},
        )
        self.hypotheses[hypothesis.id] = hypothesis
        self._save_hypotheses()
        logger.info(f"Created new hypothesis: {title}")
        return hypothesis

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        reviewer_comments: str = "",
        reviewed_by: str = "",
    ) -> bool:
        """Update hypothesis status and review information."""
        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            logger.warning(f"Hypothesis not found: {hypothesis_id}")
            return False

        hypothesis.status = status
        hypothesis.reviewer_comments = reviewer_comments
        hypothesis.reviewed_by = reviewed_by
        hypothesis.reviewed_at = datetime.now().isoformat()
        hypothesis.updated_at = datetime.now().isoformat()

        self._save_hypotheses()
        logger.info(f"Updated hypothesis {hypothesis_id} status to {status.value}")
        return True

    def get_pending_hypotheses(self) -> List[Hypothesis]:
        """Get all hypotheses pending review."""
        return [
            h for h in self.hypotheses.values() if h.status == HypothesisStatus.PENDING
        ]

    def get_approved_hypotheses(self) -> List[Hypothesis]:
        """Get all approved hypotheses."""
        return [
            h for h in self.hypotheses.values() if h.status == HypothesisStatus.APPROVED
        ]

    def search_hypotheses(self, query: str) -> List[Hypothesis]:
        """Search hypotheses by title, description, or tags."""
        query_lower = query.lower()
        results = []
        for h in self.hypotheses.values():
            # Search in title, description, and tags
            searchable_text = (
                f"{h.title} {h.description} {' '.join(h.tags or [])}".lower()
            )
            if query_lower in searchable_text:
                results.append(h)
        return results


# Convenience functions for hypothesis management


def create_hypothesis(
    title: str,
    description: str,
    predicted_outcome: str = "",
    success_criteria: Optional[List[str]] = None,
    experiment_design: Optional[Dict[str, Any]] = None,
) -> Hypothesis:
    """Create a new hypothesis."""
    import uuid

    now = datetime.now().isoformat()
    return Hypothesis(
        id=f"hyp-{uuid.uuid4().hex[:8]}",
        title=title,
        description=description,
        predicted_outcome=predicted_outcome,
        success_criteria=success_criteria or [],
        experiment_design=experiment_design or {},
        created_at=now,
        updated_at=now,
    )


def submit_for_approval(hypothesis: Hypothesis) -> Hypothesis:
    """Submit a hypothesis for approval."""
    hypothesis.status = HypothesisStatus.SUBMITTED
    hypothesis.updated_at = datetime.now().isoformat()
    return hypothesis


def review_hypothesis(
    hypothesis: Hypothesis,
    decision: ApprovalDecision,
    notes: str = "",
) -> Hypothesis:
    """Review a hypothesis with a decision."""
    now = datetime.now().isoformat()
    hypothesis.reviewed_at = now
    hypothesis.updated_at = now
    hypothesis.review_notes = notes

    if decision == ApprovalDecision.APPROVE:
        hypothesis.status = HypothesisStatus.APPROVED
        hypothesis.approved_at = now
    elif decision == ApprovalDecision.REJECT:
        hypothesis.status = HypothesisStatus.REJECTED
        hypothesis.rejection_reason = notes
    elif decision == ApprovalDecision.REQUEST_CHANGES:
        hypothesis.status = HypothesisStatus.DRAFT

    return hypothesis


def approve_hypothesis(hypothesis: Hypothesis, notes: str = "") -> Hypothesis:
    """Convenience function to approve a hypothesis."""
    now = datetime.now().isoformat()
    hypothesis.status = HypothesisStatus.APPROVED
    hypothesis.review_notes = notes
    hypothesis.approved_at = now
    hypothesis.reviewed_at = now
    hypothesis.updated_at = now
    return hypothesis


def reject_hypothesis(hypothesis: Hypothesis, reason: str = "") -> Hypothesis:
    """Convenience function to reject a hypothesis."""
    now = datetime.now().isoformat()
    hypothesis.status = HypothesisStatus.REJECTED
    hypothesis.rejection_reason = reason
    hypothesis.reviewed_at = now
    hypothesis.updated_at = now
    return hypothesis


def get_approval_queue(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Get hypotheses that are submitted for approval."""
    return [h for h in hypotheses if h.status == HypothesisStatus.SUBMITTED]


def get_hypothesis_history(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Get hypotheses sorted by creation date (oldest first)."""
    return sorted(hypotheses, key=lambda h: h.created_at or "")

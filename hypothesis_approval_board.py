"""
Hypothesis Approval Board for APGI Experiment Review System.

This module provides a structured way to manage, review, and approve
hypotheses for experiments before execution.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Status of hypothesis in the approval workflow."""

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class Hypothesis:
    """Represents a scientific hypothesis for experiment validation."""

    id: str
    title: str
    description: str
    predicted_outcome: str
    confidence_score: float  # 0.0 to 1.0
    risk_assessment: str  # "low", "medium", "high"
    created_at: str
    success_criteria: Optional[List[str]] = None  # Metrics that define success
    reviewer_comments: str = ""
    status: HypothesisStatus = HypothesisStatus.PENDING
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    tags: Optional[List[str]] = None  # e.g., ["cognitive", "performance", "safety"]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "predicted_outcome": self.predicted_outcome,
            "confidence_score": self.confidence_score,
            "success_criteria": self.success_criteria,
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
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Hypothesis":
        """Reconstruct Hypothesis from dictionary."""
        # Convert status string back to enum
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = HypothesisStatus(data["status"])
            except ValueError:
                data["status"] = HypothesisStatus.PENDING

        # Ensure confidence_score is float
        if "confidence_score" in data:
            data["confidence_score"] = float(data["confidence_score"])

        return cls(**data)


@dataclass
class ApprovalBoard:
    """Manages the hypothesis approval workflow."""

    def __init__(self, storage_path: str = "hypothesis_board.json"):
        self.storage_path = Path(storage_path)
        self.hypotheses: List[Hypothesis] = self._load_hypotheses()

    def _load_hypotheses(self) -> List[Hypothesis]:
        """Load hypotheses from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    return [Hypothesis.from_dict(h) for h in data]
            except Exception as e:
                logger.error(f"Failed to load hypotheses: {e}")
        return []

    def _save_hypotheses(self) -> None:
        """Save hypotheses to storage file."""
        try:
            with open(self.storage_path, "w") as f:
                # Custom encoder to handle Enums specifically
                def enum_encoder(obj: Any) -> Any:
                    if isinstance(obj, HypothesisStatus):
                        return obj.value
                    return asdict(obj)

                # Convert to dict first to handle nested dataclasses/enums
                serializable_data = []
                for h in self.hypotheses:
                    h_dict = h.to_dict()
                    serializable_data.append(h_dict)

                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save hypotheses: {e}")

    def create_hypothesis(
        self,
        title: str,
        description: str,
        predicted_outcome: str,
        confidence_score: float = 0.5,
        success_criteria: Optional[List[str]] = None,
        risk_assessment: str = "medium",
        tags: Optional[List[str]] = None,
    ) -> Hypothesis:
        """Create a new hypothesis."""
        import uuid

        hypothesis = Hypothesis(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            predicted_outcome=predicted_outcome,
            confidence_score=confidence_score,
            success_criteria=success_criteria or [],
            risk_assessment=risk_assessment,
            tags=tags or [],
            status=HypothesisStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )
        self.hypotheses.append(hypothesis)
        self._save_hypotheses()
        logger.info(f"Created new hypothesis: {title}")
        return hypothesis

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None

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

        self._save_hypotheses()
        logger.info(f"Updated hypothesis {hypothesis_id} status to {status.value}")
        return True

    def get_pending_hypotheses(self) -> List[Hypothesis]:
        """Get all hypotheses pending review."""
        return [h for h in self.hypotheses if h.status == HypothesisStatus.PENDING]

    def get_approved_hypotheses(self) -> List[Hypothesis]:
        """Get all approved hypotheses."""
        return [h for h in self.hypotheses if h.status == HypothesisStatus.APPROVED]

    def search_hypotheses(self, query: str) -> List[Hypothesis]:
        """Search hypotheses by title, description, or tags."""
        query_lower = query.lower()
        results = []
        for h in self.hypotheses:
            # Search in title, description, and tags
            searchable_text = (
                f"{h.title} {h.description} {' '.join(h.tags or [])}".lower()
            )
            if query_lower in searchable_text:
                results.append(h)
        return results

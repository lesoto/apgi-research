"""
APGI Compliance-by-design Module

Provides Data classification, retention TTLs, deletion routines,
Audit trails for parameter changes and experiment runs,
Optional pseudonymization pipeline for participant identifiers.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class RetentionPolicy:
    classification: DataClassification
    ttl_days: int
    deletion_routine: str


RETENTION_POLICIES = {
    DataClassification.PUBLIC: RetentionPolicy(
        DataClassification.PUBLIC, 3650, "archive"
    ),
    DataClassification.INTERNAL: RetentionPolicy(
        DataClassification.INTERNAL, 365, "soft_delete"
    ),
    DataClassification.CONFIDENTIAL: RetentionPolicy(
        DataClassification.CONFIDENTIAL, 90, "secure_erase"
    ),
    DataClassification.RESTRICTED: RetentionPolicy(
        DataClassification.RESTRICTED, 30, "crypto_shred"
    ),
}


class ComplianceManager:
    """Manages system-wide compliance, auditing, and data lifecycle rules."""

    def __init__(self) -> None:
        self.audit_trail: list[dict[str, Any]] = []

    def log_parameter_change(
        self, user: str, param_name: str, old_value: Any, new_value: Any
    ) -> None:
        """Creates an audit trail entry for configuration parameter changes."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "parameter_change",
            "user": user,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
        }
        self.audit_trail.append(entry)
        logger.info(f"Compliance Audit: {json.dumps(entry)}")

    def log_experiment_run(
        self, experiment_id: str, classification: DataClassification
    ) -> None:
        """Creates an audit trail entry for experiment invocations."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "experiment_run",
            "experiment_id": experiment_id,
            "classification": classification.value,
        }
        self.audit_trail.append(entry)
        logger.info(f"Compliance Audit: {json.dumps(entry)}")

    def enforce_retention(self, data_records: list) -> list:
        """Applies TTL and filters out expired records based on data classification."""
        current_time = datetime.now()
        valid_records = []
        for record in data_records:
            record_class = record.get("classification", DataClassification.INTERNAL)
            if isinstance(record_class, str):
                try:
                    record_class = DataClassification(record_class)
                except ValueError:
                    record_class = DataClassification.INTERNAL

            policy = RETENTION_POLICIES.get(record_class)
            if policy is None:
                continue

            created_at_str = record.get("created_at", current_time.isoformat())
            created_at = datetime.fromisoformat(created_at_str)

            if current_time - created_at <= timedelta(days=policy.ttl_days):
                valid_records.append(record)
            else:
                self._execute_deletion(record, policy.deletion_routine)
        return valid_records

    def _execute_deletion(self, record: dict, routine: str) -> None:
        """Simulates deletion routines based on classification."""
        logger.info(
            f"Applying deletion routine '{routine}' to record {record.get('id', 'unknown')}"
        )
        # In a real system, this would call out to database delete, crypto key trash, etc.


def pseudonymize_participant(
    participant_id: str, salt: str = "apgi_default_salt_x9Z"
) -> str:
    """
    Optional pseudonymization pipeline for participant identifiers.
    Hashes the participant ID to protect privacy in exported datasets.
    """
    if not participant_id:
        return ""
    pipeline_input = f"{participant_id}:{salt}".encode("utf-8")
    return hashlib.sha256(pipeline_input).hexdigest()

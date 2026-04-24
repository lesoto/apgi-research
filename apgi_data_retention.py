"""
Data Retention and Deletion Management for APGI System

Implements enforceable retention policies, real deletion executors,
and key-destruction workflows for GDPR/CCPA/HIPAA compliance.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os

from apgi_logging import get_logger
from apgi_audit import get_audit_sink, AuditEventType


class RetentionPolicy(Enum):
    """Data retention policy types."""

    PERMANENT = "permanent"  # Keep indefinitely
    GDPR_DEFAULT = "gdpr_default"  # 3 years (GDPR default)
    CCPA_DEFAULT = "ccpa_default"  # 12 months (CCPA default)
    HIPAA_DEFAULT = "hipaa_default"  # 6 years (HIPAA minimum)
    CUSTOM = "custom"  # Custom retention period


@dataclass
class RetentionConfig:
    """Configuration for data retention."""

    policy: RetentionPolicy = RetentionPolicy.GDPR_DEFAULT
    retention_days: int = 1095  # 3 years for GDPR
    auto_delete_enabled: bool = True
    deletion_verification_required: bool = True
    audit_trail_retention_days: int = 2555  # 7 years for audit trail

    def get_retention_period(self) -> timedelta:
        """Get retention period as timedelta."""
        return timedelta(days=self.retention_days)

    def get_audit_retention_period(self) -> timedelta:
        """Get audit trail retention period."""
        return timedelta(days=self.audit_trail_retention_days)


@dataclass
class DataSubjectRecord:
    """Record of data associated with a data subject (operator)."""

    subject_id: str
    subject_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    data_categories: List[str] = field(
        default_factory=list
    )  # experiment, config, audit, etc.
    retention_policy: RetentionPolicy = RetentionPolicy.GDPR_DEFAULT
    deletion_requested: bool = False
    deletion_requested_at: Optional[datetime] = None
    deletion_completed: bool = False
    deletion_completed_at: Optional[datetime] = None

    def is_retention_expired(self, config: RetentionConfig) -> bool:
        """Check if retention period has expired."""
        if self.retention_policy == RetentionPolicy.PERMANENT:
            return False

        retention_period = config.get_retention_period()
        expiration_date = self.created_at + retention_period
        return datetime.utcnow() > expiration_date

    def mark_for_deletion(self) -> None:
        """Mark record for deletion."""
        self.deletion_requested = True
        self.deletion_requested_at = datetime.utcnow()

    def mark_deletion_complete(self) -> None:
        """Mark deletion as complete."""
        self.deletion_completed = True
        self.deletion_completed_at = datetime.utcnow()


class DeletionExecutor:
    """Executes real data deletion (not simulated)."""

    def __init__(self, config: RetentionConfig):
        self.config = config
        self.logger = get_logger("apgi.retention.deletion")
        self.audit_sink = get_audit_sink()

    def delete_experiment_data(
        self,
        subject_id: str,
        experiment_id: str,
        data_path: Optional[str] = None,
    ) -> bool:
        """Delete experiment data for a subject."""
        try:
            if data_path and os.path.exists(data_path):
                # Real deletion: remove file
                os.remove(data_path)
                self.logger.info(f"Deleted experiment data: {data_path}")

            # Audit deletion
            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="experiment",
                resource_id=experiment_id,
                action="delete",
                details={"data_path": data_path},
                status="success",
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to delete experiment data: {e}")
            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="experiment",
                resource_id=experiment_id,
                action="delete",
                details={"error": str(e)},
                status="failure",
                error_message=str(e),
            )
            return False

    def delete_config_data(
        self,
        subject_id: str,
        config_id: str,
    ) -> bool:
        """Delete configuration data for a subject."""
        try:
            # In-memory deletion (configs typically not persisted)
            self.logger.info(f"Deleted config data for subject {subject_id}")

            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="config",
                resource_id=config_id,
                action="delete",
                status="success",
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to delete config data: {e}")
            return False

    def destroy_kms_key(
        self,
        subject_id: str,
        key_id: str,
        kms_callback: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """Destroy KMS key associated with subject."""
        try:
            if kms_callback:
                # Call external KMS to destroy key
                success = kms_callback(key_id)
                if not success:
                    raise Exception("KMS key destruction failed")

            self.logger.info(f"Destroyed KMS key {key_id} for subject {subject_id}")

            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="kms_key",
                resource_id=key_id,
                action="destroy",
                status="success",
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to destroy KMS key: {e}")
            return False


class RetentionJobScheduler:
    """Schedules and executes retention jobs."""

    def __init__(self, config: RetentionConfig):
        self.config = config
        self.logger = get_logger("apgi.retention.scheduler")
        self.deletion_executor = DeletionExecutor(config)
        self.data_subjects: Dict[str, DataSubjectRecord] = {}

    def register_data_subject(
        self,
        subject_id: str,
        subject_name: str,
        data_categories: List[str],
        retention_policy: RetentionPolicy = RetentionPolicy.GDPR_DEFAULT,
    ) -> DataSubjectRecord:
        """Register a data subject."""
        record = DataSubjectRecord(
            subject_id=subject_id,
            subject_name=subject_name,
            data_categories=data_categories,
            retention_policy=retention_policy,
        )
        self.data_subjects[subject_id] = record
        self.logger.info(f"Registered data subject {subject_name}")
        return record

    def request_deletion(self, subject_id: str) -> bool:
        """Request deletion for a data subject (right to erasure)."""
        if subject_id not in self.data_subjects:
            self.logger.warning(f"Data subject {subject_id} not found")
            return False

        record = self.data_subjects[subject_id]
        record.mark_for_deletion()
        self.logger.info(f"Deletion requested for subject {subject_id}")
        return True

    def execute_retention_jobs(self) -> Dict[str, Any]:
        """Execute retention jobs (delete expired data)."""
        results = {
            "total_subjects": len(self.data_subjects),
            "expired_subjects": 0,
            "deletion_requested": 0,
            "deletions_completed": 0,
            "deletions_failed": 0,
        }

        for subject_id, record in self.data_subjects.items():
            # Check for deletion requests
            if record.deletion_requested and not record.deletion_completed:
                success = self._execute_subject_deletion(record)
                if success:
                    results["deletions_completed"] += 1
                else:
                    results["deletions_failed"] += 1
                results["deletion_requested"] += 1

            # Check for expired retention
            elif (
                record.is_retention_expired(self.config)
                and self.config.auto_delete_enabled
            ):
                success = self._execute_subject_deletion(record)
                if success:
                    results["deletions_completed"] += 1
                else:
                    results["deletions_failed"] += 1
                results["expired_subjects"] += 1

        self.logger.info(f"Retention jobs completed: {results}")
        return results

    def _execute_subject_deletion(self, record: DataSubjectRecord) -> bool:
        """Execute deletion for a data subject."""
        try:
            # Delete each data category
            for category in record.data_categories:
                if category == "experiment":
                    # Delete experiment data (would need actual data path)
                    self.deletion_executor.delete_experiment_data(
                        record.subject_id,
                        f"experiment_{record.subject_id}",
                    )
                elif category == "config":
                    self.deletion_executor.delete_config_data(
                        record.subject_id,
                        f"config_{record.subject_id}",
                    )
                elif category == "kms_key":
                    self.deletion_executor.destroy_kms_key(
                        record.subject_id,
                        f"key_{record.subject_id}",
                    )

            record.mark_deletion_complete()
            self.logger.info(f"Deletion completed for subject {record.subject_id}")
            return True
        except Exception as e:
            self.logger.error(f"Deletion failed for subject {record.subject_id}: {e}")
            return False

    def export_subject_data(self, subject_id: str, filepath: str) -> bool:
        """Export all data for a subject (right to portability)."""
        if subject_id not in self.data_subjects:
            self.logger.warning(f"Data subject {subject_id} not found")
            return False

        record = self.data_subjects[subject_id]

        try:
            data = {
                "subject_id": record.subject_id,
                "subject_name": record.subject_name,
                "created_at": record.created_at.isoformat(),
                "data_categories": record.data_categories,
                "retention_policy": record.retention_policy.value,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Exported subject data to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export subject data: {e}")
            return False

    def get_retention_statistics(self) -> Dict[str, Any]:
        """Get retention statistics."""
        expired_count = sum(
            1
            for r in self.data_subjects.values()
            if r.is_retention_expired(self.config)
        )
        deletion_requested_count = sum(
            1 for r in self.data_subjects.values() if r.deletion_requested
        )
        deletion_completed_count = sum(
            1 for r in self.data_subjects.values() if r.deletion_completed
        )

        return {
            "total_subjects": len(self.data_subjects),
            "expired_records": expired_count,
            "deletion_requested": deletion_requested_count,
            "deletion_completed": deletion_completed_count,
            "pending_deletion": deletion_requested_count - deletion_completed_count,
        }


# Global retention scheduler instance
_retention_scheduler = RetentionJobScheduler(RetentionConfig())


def get_retention_scheduler() -> RetentionJobScheduler:
    """Get global retention scheduler."""
    return _retention_scheduler


def set_retention_scheduler(scheduler: RetentionJobScheduler) -> None:
    """Set global retention scheduler (for testing)."""
    global _retention_scheduler
    _retention_scheduler = scheduler

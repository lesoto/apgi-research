"""
Immutable Audit Sink for APGI System

Provides append-only audit logging for all critical operations.
Implements signed action logs and audit trail integrity.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib
import hmac
import uuid

from apgi_logging import get_logger


class AuditEventType(Enum):
    """Types of audit events."""

    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    CONFIGURATION_CHANGED = "configuration_changed"
    DATA_EXPORTED = "data_exported"
    DATA_DELETED = "data_deleted"
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"
    SECURITY_VIOLATION = "security_violation"
    HYPOTHESIS_APPROVED = "hypothesis_approved"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    PLAN_EXECUTED = "plan_executed"
    OPERATOR_REGISTERED = "operator_registered"
    OPERATOR_DEACTIVATED = "operator_deactivated"


@dataclass
class AuditEvent:
    """Immutable audit event record."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.EXPERIMENT_STARTED
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operator_id: str = ""
    operator_name: str = ""
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # success, failure, denied
    error_message: Optional[str] = None

    # Integrity fields
    sequence_number: int = 0
    previous_hash: str = ""  # Hash of previous event
    event_hash: str = ""  # Hash of this event
    signature: str = ""  # HMAC signature

    def compute_hash(self) -> str:
        """Compute SHA256 hash of event content."""
        content = json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "timestamp": self.timestamp.isoformat(),
                "operator_id": self.operator_id,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "action": self.action,
                "details": self.details,
                "status": self.status,
                "sequence_number": self.sequence_number,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def sign(self, secret_key: str) -> None:
        """Sign event with HMAC."""
        self.event_hash = self.compute_hash()
        self.signature = hmac.new(
            secret_key.encode(),
            self.event_hash.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify_signature(self, secret_key: str) -> bool:
        """Verify event signature."""
        expected_signature = hmac.new(
            secret_key.encode(),
            self.event_hash.encode(),
            hashlib.sha256,
        ).hexdigest()
        return self.signature == expected_signature

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "status": self.status,
            "error_message": self.error_message,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "signature": self.signature,
        }


class ImmutableAuditSink:
    """Append-only audit sink with integrity guarantees."""

    def __init__(self, secret_key: Optional[str] = None):
        self.logger = get_logger("apgi.audit")
        self.secret_key = secret_key or self._get_default_key()
        self.events: List[AuditEvent] = []
        self.sequence_counter = 0

    def _get_default_key(self) -> str:
        """
        Get signing key from secure provisioning (fail-closed).

        Raises:
            RuntimeError: If APGI_AUDIT_KEY environment variable is not set
                        or if the key has insufficient entropy.
        """
        import os

        key = os.environ.get("APGI_AUDIT_KEY")
        if not key:
            raise RuntimeError(
                "APGI_AUDIT_KEY environment variable must be set for audit signing. "
                "Generate a secure key with: openssl rand -hex 32"
            )

        # Validate minimum entropy (256 bits = 32 bytes = 64 hex chars)
        if len(key.encode()) < 32:
            raise RuntimeError(
                f"APGI_AUDIT_KEY has insufficient entropy ({len(key.encode())} bytes). "
                f"Minimum required: 32 bytes (256 bits). "
                "Generate with: openssl rand -hex 32"
            )

        # Warn if using potentially weak key patterns
        weak_patterns = ["default", "test", "dev", "local", "changeme", "password"]
        for pattern in weak_patterns:
            if pattern in key.lower():
                raise RuntimeError(
                    f"APGI_AUDIT_KEY appears to use a weak pattern: '{pattern}'. "
                    "Generate a secure random key with: openssl rand -hex 32"
                )

        return key

    def record_event(
        self,
        event_type: AuditEventType,
        operator_id: str,
        operator_name: str,
        resource_type: str,
        resource_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """Record an audit event (append-only)."""
        event = AuditEvent(
            event_type=event_type,
            operator_id=operator_id,
            operator_name=operator_name,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            status=status,
            error_message=error_message,
            sequence_number=self.sequence_counter,
        )

        # Link to previous event for integrity chain
        if self.events:
            event.previous_hash = self.events[-1].event_hash

        # Sign event
        event.sign(self.secret_key)

        # Append to immutable log
        self.events.append(event)
        self.sequence_counter += 1

        self.logger.info(
            f"Audit event recorded: {event_type.value} by {operator_name} "
            f"on {resource_type}/{resource_id}"
        )

        return event

    def verify_integrity(self) -> bool:
        """Verify integrity of entire audit trail."""
        if not self.events:
            return True

        for i, event in enumerate(self.events):
            # Verify signature
            if not event.verify_signature(self.secret_key):
                self.logger.error(f"Signature verification failed for event {i}")
                return False

            # Verify hash chain
            if i > 0:
                if event.previous_hash != self.events[i - 1].event_hash:
                    self.logger.error(f"Hash chain broken at event {i}")
                    return False

            # Verify sequence
            if event.sequence_number != i:
                self.logger.error(f"Sequence number mismatch at event {i}")
                return False

        self.logger.info("Audit trail integrity verified")
        return True

    def get_events(
        self,
        limit: int = 100,
        event_type: Optional[AuditEventType] = None,
        operator_id: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit events with optional filtering."""
        events = self.events

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if operator_id:
            events = [e for e in events if e.operator_id == operator_id]
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]

        # Return most recent events
        events = events[-limit:]
        return [e.to_dict() for e in events]

    def export_audit_trail(self, filepath: str) -> None:
        """Export audit trail to JSON file."""
        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_events": len(self.events),
            "integrity_verified": self.verify_integrity(),
            "events": [e.to_dict() for e in self.events],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Audit trail exported to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        event_counts: Dict[str, int] = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        operator_counts: Dict[str, int] = {}
        for event in self.events:
            op_name = event.operator_name
            operator_counts[op_name] = operator_counts.get(op_name, 0) + 1

        return {
            "total_events": len(self.events),
            "event_types": event_counts,
            "operators": operator_counts,
            "integrity_verified": self.verify_integrity(),
        }


# Global audit sink instance - initialized lazily
_audit_sink: Optional[ImmutableAuditSink] = None


def get_audit_sink() -> ImmutableAuditSink:
    """Get global audit sink."""
    global _audit_sink
    if _audit_sink is None:
        _audit_sink = ImmutableAuditSink()
    return _audit_sink


def set_audit_sink(sink: ImmutableAuditSink) -> None:
    """Set global audit sink (for testing)."""
    global _audit_sink
    _audit_sink = sink

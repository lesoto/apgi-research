"""
Injectable Security Adapters for APGI System

Replaces global monkeypatching with explicit, context-aware security controls.
Implements deny-by-default allowlists with telemetry and audit logging.

Key Features:
- Per-context subprocess allowlists
- Explicit pickle/serialization controls
- Telemetry and deny metrics
- Audit logging for security events
- KMS-backed secret support
"""

import hashlib
import hmac
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from apgi_logging import get_logger


class SecurityLevel(Enum):
    """Security enforcement levels."""

    PERMISSIVE = "permissive"  # Log only
    STANDARD = "standard"  # Enforce with audit
    STRICT = "strict"  # Enforce with denial


@dataclass
class SecurityContext:
    """Per-context security configuration."""

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operator_id: Optional[str] = None
    role: str = "user"
    subprocess_allowlist: Set[str] = field(
        default_factory=lambda: {"git", "pytest", "python"}
    )
    pickle_allowed: bool = False
    serialization_format: str = "json"  # json, msgpack, protobuf
    security_level: SecurityLevel = SecurityLevel.STANDARD
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate context configuration."""
        if not self.operator_id:
            self.operator_id = "anonymous"
        if self.security_level == SecurityLevel.STRICT and self.pickle_allowed:
            raise ValueError("Pickle not allowed in STRICT security level")


@dataclass
class SecurityEvent:
    """Audit event for security operations."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context_id: str = ""
    operator_id: str = ""
    event_type: str = ""  # subprocess_call, pickle_attempt, config_validation, etc.
    resource: str = ""
    action: str = ""  # allowed, denied, logged
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "context_id": self.context_id,
            "operator_id": self.operator_id,
            "event_type": self.event_type,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
        }


class SecurityMetrics:
    """Tracks security telemetry and deny metrics."""

    def __init__(self) -> None:
        self.allowed_operations: Dict[str, int] = {}
        self.denied_operations: Dict[str, int] = {}
        self.audit_events: List[SecurityEvent] = []
        self.logger = get_logger("apgi.security.metrics")

    def record_allowed(self, operation_type: str, resource: str) -> None:
        """Record allowed operation."""
        key = f"{operation_type}:{resource}"
        self.allowed_operations[key] = self.allowed_operations.get(key, 0) + 1

    def record_denied(self, operation_type: str, resource: str, reason: str) -> None:
        """Record denied operation."""
        key = f"{operation_type}:{resource}"
        self.denied_operations[key] = self.denied_operations.get(key, 0) + 1
        self.logger.warning(f"Denied {operation_type} on {resource}: {reason}")

    def record_event(self, event: SecurityEvent) -> None:
        """Record audit event."""
        self.audit_events.append(event)
        if len(self.audit_events) > 10000:  # Keep last 10k events
            self.audit_events = self.audit_events[-10000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return {
            "allowed_operations": self.allowed_operations,
            "denied_operations": self.denied_operations,
            "total_allowed": sum(self.allowed_operations.values()),
            "total_denied": sum(self.denied_operations.values()),
            "audit_events_count": len(self.audit_events),
        }


class SubprocessSecurityAdapter:
    """Manages secure subprocess execution with allowlists."""

    def __init__(self, metrics: SecurityMetrics):
        self.metrics = metrics
        self.logger = get_logger("apgi.security.subprocess")
        self._original_popen = subprocess.Popen

    def create_secure_popen(self, context: SecurityContext) -> Callable:
        """Create a secure Popen wrapper for the given context."""

        def secure_popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
            """Secure subprocess.Popen wrapper."""
            # Extract command
            cmd = self._extract_command(args, kwargs)

            # Check allowlist
            is_allowed = self._check_allowlist(cmd, context.subprocess_allowlist)

            # Create audit event
            event = SecurityEvent(
                context_id=context.context_id,
                operator_id=context.operator_id or "",
                event_type="subprocess_call",
                resource=cmd,
                action="allowed" if is_allowed else "denied",
                details={"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
            )
            self.metrics.record_event(event)

            if is_allowed:
                self.metrics.record_allowed("subprocess", cmd)
                self.logger.info(f"Allowed subprocess: {cmd}")
                return self._original_popen(*args, **kwargs)
            else:
                self.metrics.record_denied("subprocess", cmd, "not in allowlist")

                if context.security_level == SecurityLevel.PERMISSIVE:
                    self.logger.warning(f"Permissive mode: allowing {cmd}")
                    return self._original_popen(*args, **kwargs)
                else:
                    raise PermissionError(
                        f"Subprocess command '{cmd}' not in allowlist for context {context.context_id}"
                    )

        return secure_popen

    def _extract_command(self, args: tuple, kwargs: dict) -> str:
        """Extract command name from subprocess arguments."""
        if args and isinstance(args[0], (list, tuple)):
            return str(args[0][0])
        elif args and isinstance(args[0], str):
            return args[0].split()[0]
        else:
            cmd_args = kwargs.get("args", ["unknown"])
            return str(cmd_args[0]) if cmd_args else "unknown"

    def _check_allowlist(self, cmd: str, allowlist: Set[str]) -> bool:
        """Check if command is in allowlist."""
        # Extract base command name
        base_cmd = cmd.split("/")[-1].split()[0]
        return base_cmd in allowlist


class SerializationSecurityAdapter:
    """Manages secure serialization with format controls."""

    def __init__(self, metrics: SecurityMetrics):
        self.metrics = metrics
        self.logger = get_logger("apgi.security.serialization")

    def create_secure_loads(self, context: SecurityContext) -> Callable:
        """Create a secure loads wrapper for the given context."""

        def secure_loads(data: bytes) -> Any:
            """Secure deserialization wrapper."""
            if (
                context.pickle_allowed
                and context.security_level == SecurityLevel.PERMISSIVE
            ):
                # Pickle allowed in permissive mode
                import pickle

                return pickle.loads(data)

            # Default: JSON only
            try:
                data_str = data.decode("utf-8") if isinstance(data, bytes) else data
                result = json.loads(data_str)

                event = SecurityEvent(
                    context_id=context.context_id,
                    operator_id=context.operator_id or "",
                    event_type="deserialization",
                    resource="json",
                    action="allowed",
                )
                self.metrics.record_event(event)
                self.metrics.record_allowed("deserialization", "json")

                return result
            except Exception as e:
                event = SecurityEvent(
                    context_id=context.context_id,
                    operator_id=context.operator_id or "",
                    event_type="deserialization",
                    resource="pickle",
                    action="denied",
                    details={"error": str(e)},
                )
                self.metrics.record_event(event)
                self.metrics.record_denied(
                    "deserialization", "pickle", "disabled for security"
                )

                raise ValueError(
                    f"Pickle is disabled for security in context {context.context_id}. "
                    f"Use JSON serialization instead. Error: {e}"
                )

        return secure_loads


class ConfigChecksumAdapter:
    """Manages configuration integrity with KMS-backed secrets."""

    def __init__(self, metrics: SecurityMetrics, kms_key: Optional[str] = None):
        self.metrics = metrics
        self.logger = get_logger("apgi.security.checksum")
        self.kms_key = kms_key or self._get_default_key()

    def _get_default_key(self) -> str:
        """Get KMS key from environment variable."""
        import os

        key = os.environ.get("APGI_KMS_KEY")
        if key is None:
            raise ValueError(
                "APGI_KMS_KEY environment variable must be set. "
                "Using static default secrets is not allowed for security reasons."
            )
        return key

    def validate_config_checksum(
        self,
        config_dict: Dict[str, Any],
        expected_hash: str,
        context: SecurityContext,
    ) -> bool:
        """Validate configuration checksum with KMS-backed secret."""
        config_str = json.dumps(config_dict, sort_keys=True)

        # Use HMAC for stronger integrity guarantee
        computed_hash = hmac.new(
            self.kms_key.encode("utf-8"),
            config_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        is_valid = computed_hash == expected_hash

        event = SecurityEvent(
            context_id=context.context_id,
            operator_id=context.operator_id or "",
            event_type="config_validation",
            resource="config_checksum",
            action="allowed" if is_valid else "denied",
            details={"config_size": len(config_str)},
        )
        self.metrics.record_event(event)

        if is_valid:
            self.metrics.record_allowed("config_validation", "checksum")
        else:
            self.metrics.record_denied("config_validation", "checksum", "hash mismatch")

        return is_valid


class SecurityAdapterFactory:
    """Factory for creating security adapters with shared metrics."""

    def __init__(self) -> None:
        self.metrics = SecurityMetrics()
        self.subprocess_adapter = SubprocessSecurityAdapter(self.metrics)
        self.serialization_adapter = SerializationSecurityAdapter(self.metrics)
        self._checksum_adapter: Optional[ConfigChecksumAdapter] = None
        self.logger = get_logger("apgi.security.factory")

    def create_context(
        self,
        operator_id: Optional[str] = None,
        role: str = "user",
        subprocess_allowlist: Optional[Set[str]] = None,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ) -> SecurityContext:
        """Create a new security context."""
        return SecurityContext(
            operator_id=operator_id,
            role=role,
            subprocess_allowlist=subprocess_allowlist or {"git", "pytest", "python"},
            security_level=security_level,
        )

    def get_secure_popen(self, context: SecurityContext) -> Callable:
        """Get secure Popen for context."""
        return self.subprocess_adapter.create_secure_popen(context)

    def get_secure_loads(self, context: SecurityContext) -> Callable:
        """Get secure loads for context."""
        return self.serialization_adapter.create_secure_loads(context)

    @property
    def checksum_adapter(self) -> ConfigChecksumAdapter:
        """Lazily initialize checksum adapter."""
        if self._checksum_adapter is None:
            self._checksum_adapter = ConfigChecksumAdapter(self.metrics)
        return self._checksum_adapter

    def validate_config(
        self,
        config_dict: Dict[str, Any],
        expected_hash: str,
        context: SecurityContext,
    ) -> bool:
        """Validate config checksum."""
        return self.checksum_adapter.validate_config_checksum(
            config_dict, expected_hash, context
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return self.metrics.get_metrics()

    def get_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events."""
        events = self.metrics.audit_events[-limit:]
        return [event.to_dict() for event in events]


# Global factory instance (can be replaced for testing)
_security_factory = SecurityAdapterFactory()


def get_security_factory() -> SecurityAdapterFactory:
    """Get global security adapter factory."""
    return _security_factory


def set_security_factory(factory: SecurityAdapterFactory) -> None:
    """Set global security adapter factory (for testing)."""
    global _security_factory
    _security_factory = factory

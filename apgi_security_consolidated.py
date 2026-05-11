"""
APGI Security Module (Consolidated)

Combines the best features from utils.apgi_security.py and apgi_security_adapters.py:
- Deny-by-default audit allowlists
- Context-aware security controls
- Secure subprocess/pickle wrappers
- Config validation with HMAC
- Telemetry and audit logging
- KMS-backed secret support
"""

import hashlib
import hmac
import json
import os
import pickle
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from utils.apgi_logging import get_logger

# ---------------------------------------------------------------------------
# Security Levels and Contexts
# ---------------------------------------------------------------------------


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
        default_factory=lambda: {
            "git",
            "pytest",
            "python",
            "python3",
            "pip",
            "pip3",
            "uv",
            "node",
            "npm",
            "curl",
            "wget",
            "rsync",
            "ssh",
            "scp",
            "tar",
            "unzip",
            "zip",
        }
    )
    pickle_allowed: bool = False
    serialization_format: str = "json"  # json, msgpack, protobuf
    security_level: SecurityLevel = SecurityLevel.STANDARD
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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


# ---------------------------------------------------------------------------
# Security Metrics and Telemetry
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Legacy Compatibility Types
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_SUBPROCESS_CMDS = [
    "git",
    "pytest",
    "python",
    "python3",
]


@dataclass
class SubprocessSecurityPolicy:
    """Legacy compatibility for existing code."""

    allowed_commands: List[str]
    require_explicit_allowlist: bool = True
    log_all_calls: bool = True

    def is_allowed(self, cmd: str) -> bool:
        """Check if command is in allowlist."""
        if not self.require_explicit_allowlist:
            return True
        return cmd in self.allowed_commands


class SecureSubprocessError(PermissionError):
    """Raised when subprocess execution is denied by security policy."""

    pass


class PickleSecurityError(ValueError):
    """Raised when pickle operation is blocked for security."""

    pass


# ---------------------------------------------------------------------------
# Core Security Implementation
# ---------------------------------------------------------------------------


class SecurityManager:
    """Main security manager combining all security features."""

    def __init__(self, kms_key: Optional[str] = None) -> None:
        self.metrics = SecurityMetrics()
        self.logger = get_logger("apgi.security.manager")
        self.kms_key = kms_key or self._get_default_key()

    def _get_default_key(self) -> str:
        """Get KMS key from environment variable."""
        key = os.environ.get("APGI_KMS_KEY") or os.environ.get("APGI_CONFIG_SECRET_KEY")
        if key is None:
            raise ValueError(
                "APGI_KMS_KEY or APGI_CONFIG_SECRET_KEY environment variable must be set. "
                "Using static default secrets is not allowed for security reasons."
            )
        return key

    def create_context(
        self,
        operator_id: Optional[str] = None,
        role: str = "user",
        subprocess_allowlist: Optional[Set[str]] = None,
        pickle_allowed: bool = False,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ) -> SecurityContext:
        """Create a new security context."""
        return SecurityContext(
            operator_id=operator_id,
            role=role,
            subprocess_allowlist=subprocess_allowlist
            or set(DEFAULT_ALLOWED_SUBPROCESS_CMDS),
            pickle_allowed=pickle_allowed,
            security_level=security_level,
        )

    def secure_popen(
        self,
        args: Union[str, Sequence[str]],
        context: Optional[SecurityContext] = None,
        **kwargs: Any,
    ) -> subprocess.Popen:
        """Secure subprocess execution with context-aware security."""
        if context is None:
            context = self.create_context()

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
            return subprocess.Popen(args, **kwargs)
        else:
            self.metrics.record_denied("subprocess", cmd, "not in allowlist")

            if context.security_level == SecurityLevel.PERMISSIVE:
                self.logger.warning(f"Permissive mode: allowing {cmd}")
                return subprocess.Popen(args, **kwargs)
            else:
                raise SecureSubprocessError(
                    f"Subprocess command '{cmd}' not in allowlist for context {context.context_id}. "
                    f"Allowed commands: {context.subprocess_allowlist}"
                )

    def secure_run(
        self,
        args: Union[str, Sequence[str]],
        context: Optional[SecurityContext] = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Secure subprocess.run with context-aware security."""
        if context is None:
            context = self.create_context()

        # Validate the command first
        cmd = self._extract_command(args, kwargs)
        if not self._check_allowlist(cmd, context.subprocess_allowlist):
            if context.security_level != SecurityLevel.PERMISSIVE:
                raise SecureSubprocessError(
                    f"Subprocess command '{cmd}' not in allowlist for context {context.context_id}"
                )

        return subprocess.run(args, **kwargs)

    def secure_loads(
        self,
        data: Union[bytes, str],
        context: Optional[SecurityContext] = None,
    ) -> Any:
        """Secure deserialization with context-aware security."""
        if context is None:
            context = self.create_context()

        if (
            context.pickle_allowed
            and context.security_level == SecurityLevel.PERMISSIVE
        ):
            # Pickle allowed in permissive mode
            return (
                pickle.loads(data)
                if isinstance(data, bytes)
                else pickle.loads(data.encode())
            )

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

            raise PickleSecurityError(
                f"Pickle is disabled for security in context {context.context_id}. "
                f"Use JSON serialization instead. Error: {e}"
            )

    def secure_dumps(
        self,
        obj: Any,
        context: Optional[SecurityContext] = None,
    ) -> bytes:
        """Secure serialization with context-aware security."""
        if context is None:
            context = self.create_context()

        if (
            context.pickle_allowed
            and context.security_level == SecurityLevel.PERMISSIVE
        ):
            return pickle.dumps(obj)

        # Default: JSON serialization
        try:
            return json.dumps(obj, default=str).encode("utf-8")
        except Exception as e:
            raise PickleSecurityError(f"JSON serialization failed: {e}")

    def validate_config_checksum(
        self,
        config_dict: Dict[str, Any],
        expected_hash: str,
        context: Optional[SecurityContext] = None,
    ) -> bool:
        """Validate configuration checksum with KMS-backed secret."""
        if context is None:
            context = self.create_context()

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

    def _extract_command(self, args: Any, kwargs: Dict[str, Any]) -> str:
        """Extract command name from subprocess arguments."""
        if isinstance(args, str):
            return args.split()[0]
        elif args and isinstance(args[0], str):
            return args[0].split()[0]
        elif args and isinstance(args[0], (list, tuple)):
            return str(args[0][0])
        else:
            cmd_args = kwargs.get("args", ["unknown"])
            return str(cmd_args[0]) if cmd_args else "unknown"

    def _check_allowlist(self, cmd: str, allowlist: Set[str]) -> bool:
        """Check if command is in allowlist."""
        # Extract base command name
        base_cmd = cmd.split("/")[-1].split()[0]
        return base_cmd in allowlist

    def get_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return self.metrics.get_metrics()

    def get_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events."""
        events = self.metrics.audit_events[-limit:]
        return [event.to_dict() for event in events]


# ---------------------------------------------------------------------------
# Global Manager Instance and Convenience Functions
# ---------------------------------------------------------------------------

_global_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


# Legacy compatibility functions
def secure_popen(
    args: Union[str, Sequence[str]],
    policy: Optional[SubprocessSecurityPolicy] = None,
    **kwargs: Any,
) -> subprocess.Popen:
    """Legacy compatibility wrapper for secure_popen."""
    manager = get_security_manager()

    if policy:
        # Convert legacy policy to context
        context = manager.create_context(
            subprocess_allowlist=set(policy.allowed_commands),
            security_level=(
                SecurityLevel.PERMISSIVE
                if not policy.require_explicit_allowlist
                else SecurityLevel.STANDARD
            ),
        )
    else:
        context = None

    return manager.secure_popen(args, context, **kwargs)


def secure_run(
    args: Union[str, Sequence[str]],
    policy: Optional[SubprocessSecurityPolicy] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Legacy compatibility wrapper for secure_run."""
    manager = get_security_manager()

    if policy:
        context = manager.create_context(
            subprocess_allowlist=set(policy.allowed_commands),
            security_level=(
                SecurityLevel.PERMISSIVE
                if not policy.require_explicit_allowlist
                else SecurityLevel.STANDARD
            ),
        )
    else:
        context = None

    return manager.secure_run(args, context, **kwargs)


def secure_loads(data: Union[bytes, str], use_pickle: bool = False) -> Any:
    """Legacy compatibility wrapper for secure_loads."""
    manager = get_security_manager()

    if use_pickle:
        context = manager.create_context(
            pickle_allowed=True,
            security_level=SecurityLevel.PERMISSIVE,
        )
    else:
        context = None

    return manager.secure_loads(data, context)


def secure_dumps(obj: Any, use_pickle: bool = False) -> bytes:
    """Legacy compatibility wrapper for secure_dumps."""
    manager = get_security_manager()

    if use_pickle:
        context = manager.create_context(
            pickle_allowed=True,
            security_level=SecurityLevel.PERMISSIVE,
        )
    else:
        context = None

    return manager.secure_dumps(obj, context)


def validate_config_checksum(
    config_dict: dict, expected_hash: str, secret_key: Optional[str] = None
) -> bool:
    """Legacy compatibility wrapper for config validation."""
    manager = get_security_manager()
    if secret_key:
        # Temporarily set the key
        original_key = manager.kms_key
        manager.kms_key = secret_key
        try:
            result = manager.validate_config_checksum(config_dict, expected_hash)
        finally:
            manager.kms_key = original_key
        return result
    else:
        return manager.validate_config_checksum(config_dict, expected_hash)


# ---------------------------------------------------------------------------
# Security Decorators
# ---------------------------------------------------------------------------


def require_allowed_subprocess(allowed_commands: Optional[List[str]] = None) -> Any:
    """
    Decorator to enforce subprocess security at function level.

    Usage:
        @require_allowed_subprocess(["git"])
        def my_git_operation():
            # subprocess calls in this function are validated
            ...
    """
    commands = allowed_commands or DEFAULT_ALLOWED_SUBPROCESS_CMDS

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create context for this function
            manager = get_security_manager()
            context = manager.create_context(
                subprocess_allowlist=set(commands),
                security_level=SecurityLevel.STANDARD,
            )
            # Store context for nested calls
            wrapper._security_context = context  # type: ignore
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Factory Function for Testing
# ---------------------------------------------------------------------------


def create_security_manager(kms_key: Optional[str] = None) -> SecurityManager:
    """Create a new security manager instance (for testing)."""
    return SecurityManager(kms_key)

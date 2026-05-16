"""
Compatibility shim: utils.apgi_audit

The audit implementation lives at repo root (`apgi_audit.py`). Some tests and
legacy imports expect it under `utils.*`. Re-export the public API here.
"""

from apgi_audit import (
    AuditEvent,
    AuditEventType,
    ImmutableAuditSink,
    get_audit_sink,
    set_audit_sink,
)

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "ImmutableAuditSink",
    "get_audit_sink",
    "set_audit_sink",
]

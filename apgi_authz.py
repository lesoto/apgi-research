"""
Authorization and Identity Management for APGI System

Implements role-based access control (RBAC) and operator identity tracking.
Provides fine-grained permission checks for GUI and autonomous agent actions.
"""

from typing import Dict, Set, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from apgi_logging import get_logger


class Role(Enum):
    """Operator roles with hierarchical permissions."""

    ADMIN = "admin"  # Full access
    OPERATOR = "operator"  # Run experiments, view results
    ANALYST = "analyst"  # View results only
    AUTONOMOUS_AGENT = "agent"  # Automated experiment execution
    GUEST = "guest"  # Read-only access


class Permission(Enum):
    """Fine-grained permissions."""

    # Experiment permissions
    RUN_EXPERIMENT = "run_experiment"
    MODIFY_EXPERIMENT = "modify_experiment"
    DELETE_EXPERIMENT = "delete_experiment"
    VIEW_EXPERIMENT = "view_experiment"

    # Configuration permissions
    MODIFY_CONFIG = "modify_config"
    VIEW_CONFIG = "view_config"

    # Data permissions
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    VIEW_DATA = "view_data"

    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOG = "view_audit_log"
    MODIFY_SECURITY = "modify_security"

    # Autonomous agent permissions
    APPROVE_HYPOTHESIS = "approve_hypothesis"
    EXECUTE_PLAN = "execute_plan"


# Role-to-permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.OPERATOR: {
        Permission.RUN_EXPERIMENT,
        Permission.MODIFY_EXPERIMENT,
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_CONFIG,
        Permission.EXPORT_DATA,
        Permission.VIEW_DATA,
        Permission.APPROVE_HYPOTHESIS,
        Permission.EXECUTE_PLAN,
    },
    Role.ANALYST: {
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_CONFIG,
        Permission.EXPORT_DATA,
        Permission.VIEW_DATA,
    },
    Role.AUTONOMOUS_AGENT: {
        Permission.RUN_EXPERIMENT,
        Permission.MODIFY_EXPERIMENT,
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_CONFIG,
        Permission.EXPORT_DATA,
        Permission.VIEW_DATA,
        Permission.EXECUTE_PLAN,
    },
    Role.GUEST: {
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_DATA,
    },
}


@dataclass
class OperatorIdentity:
    """Represents an operator's identity and credentials."""

    operator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    role: Role = Role.GUEST
    email: Optional[str] = None
    department: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def has_permission(self, permission: Permission) -> bool:
        """Check if operator has permission."""
        if not self.is_active:
            return False
        permissions = ROLE_PERMISSIONS.get(self.role, set())
        return permission in permissions

    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this operator."""
        if not self.is_active:
            return set()
        return ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class AuthorizationContext:
    """Context for authorization decisions."""

    operator: OperatorIdentity
    resource_type: str  # experiment, config, data, etc.
    resource_id: str
    action: Permission
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AuthorizationManager:
    """Manages authorization checks and decisions."""

    def __init__(self) -> None:
        self.logger = get_logger("apgi.authz")
        self.operators: Dict[str, OperatorIdentity] = {}
        self.authorization_log: List[Dict] = []

    def register_operator(
        self,
        username: str,
        role: Role,
        email: Optional[str] = None,
        department: Optional[str] = None,
    ) -> OperatorIdentity:
        """Register a new operator."""
        operator = OperatorIdentity(
            username=username,
            role=role,
            email=email,
            department=department,
        )
        self.operators[operator.operator_id] = operator
        self.logger.info(f"Registered operator {username} with role {role.value}")
        return operator

    def get_operator(self, operator_id: str) -> Optional[OperatorIdentity]:
        """Get operator by ID."""
        return self.operators.get(operator_id)

    def check_permission(
        self,
        operator: OperatorIdentity,
        permission: Permission,
    ) -> bool:
        """Check if operator has permission."""
        has_perm = operator.has_permission(permission)

        if not has_perm:
            self.logger.warning(
                f"Permission denied: {operator.username} lacks {permission.value}"
            )

        return has_perm

    def authorize_action(
        self,
        context: AuthorizationContext,
    ) -> bool:
        """Authorize an action with full context."""
        has_perm = context.operator.has_permission(context.action)

        # Log authorization decision
        log_entry = {
            "request_id": context.request_id,
            "timestamp": context.timestamp.isoformat(),
            "operator_id": context.operator.operator_id,
            "operator_name": context.operator.username,
            "resource_type": context.resource_type,
            "resource_id": context.resource_id,
            "action": context.action.value,
            "decision": "allowed" if has_perm else "denied",
        }
        self.authorization_log.append(log_entry)

        if not has_perm:
            self.logger.warning(
                f"Authorization denied: {context.operator.username} "
                f"cannot {context.action.value} on {context.resource_type}/{context.resource_id}"
            )
        else:
            self.logger.info(
                f"Authorization granted: {context.operator.username} "
                f"can {context.action.value} on {context.resource_type}/{context.resource_id}"
            )

        return has_perm

    def get_authorization_log(self, limit: int = 100) -> List[Dict]:
        """Get recent authorization log entries."""
        return self.authorization_log[-limit:]

    def deactivate_operator(self, operator_id: str) -> bool:
        """Deactivate an operator."""
        operator = self.operators.get(operator_id)
        if operator:
            operator.is_active = False
            self.logger.info(f"Deactivated operator {operator.username}")
            return True
        return False

    def update_operator_role(self, operator_id: str, new_role: Role) -> bool:
        """Update operator's role."""
        operator = self.operators.get(operator_id)
        if operator:
            old_role = operator.role
            operator.role = new_role
            self.logger.info(
                f"Updated operator {operator.username} role from {old_role.value} to {new_role.value}"
            )
            return True
        return False


# Global authorization manager instance
_authz_manager = AuthorizationManager()


def get_authz_manager() -> AuthorizationManager:
    """Get global authorization manager."""
    return _authz_manager


def set_authz_manager(manager: AuthorizationManager) -> None:
    """Set global authorization manager (for testing)."""
    global _authz_manager
    _authz_manager = manager

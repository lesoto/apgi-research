"""
Standardized CLI Entry Point Framework for APGI Experiments

Provides unified CLI handling, authorization enforcement, and logging setup
for all experiment entry points.

Usage:
    from apgi_cli import cli_entrypoint, require_auth

    @require_auth(Permission.RUN_EXPERIMENT)
    def run_my_experiment():
        ...

    if __name__ == "__main__":
        cli_entrypoint(run_my_experiment)
"""

import argparse
import logging
import sys
import os
from typing import Callable, Optional, Any, List
from functools import wraps

from apgi_logging import get_logger, APGIContextLogger
from apgi_authz import (
    get_authz_manager,
    AuthorizationContext,
    OperatorIdentity,
    Permission,
    Role,
)
from apgi_audit import get_audit_sink, AuditEventType


class CLIError(Exception):
    """Base exception for CLI-related errors."""

    pass


class AuthorizationError(CLIError):
    """Raised when authorization check fails."""

    pass


class ConfigurationError(CLIError):
    """Raised when configuration is invalid."""

    pass


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def create_standard_parser(description: str) -> argparse.ArgumentParser:
    """Create a standardized argument parser for experiment runners."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Standard arguments for all experiments
    parser.add_argument(
        "--trials",
        "-n",
        type=int,
        default=None,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results",
    )
    parser.add_argument(
        "--apgi-enabled",
        action="store_true",
        default=None,
        help="Enable APGI integration",
    )
    parser.add_argument(
        "--apgi-disabled",
        action="store_true",
        help="Disable APGI integration",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Operator ID for authorization",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="operator",
        choices=[r.value for r in Role],
        help="Role for authorization",
    )

    return parser


# ---------------------------------------------------------------------------
# Authorization Enforcement
# ---------------------------------------------------------------------------


def require_auth(
    permission: Permission,
    resource_type: str = "experiment",
    resource_id: Optional[str] = None,
) -> Callable:
    """
    Decorator to enforce authorization at CLI boundaries.

    Usage:
        @require_auth(Permission.RUN_EXPERIMENT)
        def run_experiment():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get operator identity from environment or arguments
            operator = _get_operator_identity()

            # Determine resource ID
            res_id = resource_id or func.__name__

            # Create authorization context
            context = AuthorizationContext(
                operator=operator,
                resource_type=resource_type,
                resource_id=res_id,
                action=permission,
            )

            # Check authorization
            authz = get_authz_manager()
            if not authz.authorize_action(context):
                # Record denial in audit log
                audit = get_audit_sink()
                audit.record_event(
                    event_type=AuditEventType.AUTHORIZATION_DENIED,
                    operator_id=operator.operator_id,
                    operator_name=operator.username,
                    resource_type=resource_type,
                    resource_id=res_id,
                    action=f"{permission.value}_denied",
                    status="denied",
                    error_message=f"Operator lacks {permission.value} permission",
                )
                raise AuthorizationError(
                    f"Authorization denied: {operator.username} lacks {permission.value} "
                    f"permission for {resource_type}/{res_id}"
                )

            # Record successful authorization
            audit = get_audit_sink()
            audit.record_event(
                event_type=AuditEventType.AUTHORIZATION_GRANTED,
                operator_id=operator.operator_id,
                operator_name=operator.username,
                resource_type=resource_type,
                resource_id=res_id,
                action=f"{permission.value}_granted",
                status="success",
            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_operator_identity() -> OperatorIdentity:
    """Get or create operator identity for CLI execution."""
    # Try to get from environment
    operator_id = os.environ.get("APGI_OPERATOR_ID")
    username = os.environ.get("APGI_OPERATOR_NAME", "cli_user")
    role_str = os.environ.get("APGI_OPERATOR_ROLE", "operator")

    try:
        role = Role(role_str)
    except ValueError:
        role = Role.OPERATOR

    # Check for pre-registered operator
    authz = get_authz_manager()
    if operator_id:
        existing = authz.get_operator(operator_id)
        if existing:
            return existing

    # Create new operator
    return authz.register_operator(
        username=username,
        role=role,
    )


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------


def setup_cli_logging(verbose: bool = False) -> APGIContextLogger:
    """Set up structured logging for CLI execution."""
    logger = get_logger("apgi.cli")

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return APGIContextLogger(logger)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def cli_entrypoint(
    main_func: Callable,
    parser: Optional[argparse.ArgumentParser] = None,
    add_common_args: bool = True,
) -> None:
    """
    Standardized entry point wrapper for experiment scripts.

    Handles:
    - Argument parsing
    - Logging setup
    - Error handling
    - Exit codes

    Usage:
        def main(parsed_args):
            # Your experiment code
            return results

        if __name__ == "__main__":
            cli_entrypoint(main)
    """
    if parser is None and add_common_args:
        parser = create_standard_parser(main_func.__doc__ or "APGI Experiment")

    try:
        # Parse arguments if parser provided
        if parser is not None:
            args = parser.parse_args()
        else:
            args = None

        # Determine verbosity
        verbose = getattr(args, "verbose", False) if args else False

        # Setup logging
        logger = setup_cli_logging(verbose)
        logger.info(f"Starting {main_func.__name__}")

        # Run main function
        if args is not None:
            result = main_func(args)
        else:
            result = main_func()

        logger.info(f"Completed {main_func.__name__} successfully")

        # Output results if returned
        if result is not None:
            import json

            print(json.dumps(result, indent=2, default=str))

        sys.exit(0)

    except AuthorizationError as e:
        logger.error(f"Authorization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(77)  # EX_CONFIG - permission issue

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(78)  # EX_CONFIG

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)  # Exit code for Ctrl+C

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Legacy Compatibility
# ---------------------------------------------------------------------------


def standardized_main(
    run_func: Callable,
    experiment_name: str,
    args: Optional[List[str]] = None,
) -> int:
    """
    Legacy-compatible standardized main function.

    For backward compatibility with existing run_*.py files.
    """
    parser = create_standard_parser(f"Run {experiment_name} experiment")
    parsed_args = parser.parse_args(args)

    try:
        setup_cli_logging(parsed_args.verbose)
        result = run_func(parsed_args)
        if result is not None:
            import json

            print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# Convenience exports
__all__ = [
    "cli_entrypoint",
    "require_auth",
    "create_standard_parser",
    "setup_cli_logging",
    "standardized_main",
    "CLIError",
    "AuthorizationError",
    "ConfigurationError",
]

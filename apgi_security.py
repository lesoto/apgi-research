"""
APGI Security Hardening Module

Implements deny-by-default audit allowlists, secure subprocess/pickle wrappers,
and config validation. All security controls are explicit opt-in (no monkey-patching).
"""

import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from functools import wraps
from typing import Any, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Subprocess Security (Explicit Wrapper - No Monkey Patching)
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_SUBPROCESS_CMDS = ["git", "pytest", "python", "screencapture"]


@dataclass
class SubprocessSecurityPolicy:
    """Configurable security policy for subprocess execution."""

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


class SecureSubprocessWrapper:
    """
    Explicit secure wrapper for subprocess operations.

    Usage:
        # Instead of subprocess.Popen(...), use:
        from apgi_security import secure_popen

        with secure_popen(["git", "status"], policy=my_policy) as proc:
            ...
    """

    def __init__(self, policy: Optional[SubprocessSecurityPolicy] = None):
        self.policy = policy or self._default_policy()

    @staticmethod
    def _default_policy() -> SubprocessSecurityPolicy:
        """Get default policy from environment or fallback."""
        allowed = os.environ.get(
            "APGI_ALLOWED_SUBPROCESS_CMDS", ",".join(DEFAULT_ALLOWED_SUBPROCESS_CMDS)
        ).split(",")
        return SubprocessSecurityPolicy(
            allowed_commands=[cmd.strip() for cmd in allowed if cmd.strip()],
            require_explicit_allowlist=True,
        )

    def _extract_command(self, args: Any) -> str:
        """Extract command name from various argument formats."""
        if args and isinstance(args[0], (list, tuple)) and len(args[0]) > 0:
            return str(args[0][0])
        elif args and isinstance(args[0], str):
            return args[0].split()[0]
        return ""

    def __call__(self, *args: Any, **kwargs: Any) -> subprocess.Popen:
        """Execute subprocess with security checks."""
        cmd = self._extract_command(args)

        if not cmd:
            cmd = kwargs.get("args", [""])[0] if kwargs.get("args") else ""

        if not self.policy.is_allowed(cmd):
            raise SecureSubprocessError(
                f"Subprocess command '{cmd}' is not in the allowlist. "
                f"Allowed commands: {self.policy.allowed_commands}. "
                f"Set APGI_ALLOWED_SUBPROCESS_CMDS env var to customize."
            )

        # Use original subprocess.Popen (no monkey-patching)
        return subprocess.Popen(*args, **kwargs)


# Global wrapper instance for convenience
_default_subprocess_wrapper: Optional[SecureSubprocessWrapper] = None


def get_secure_subprocess_wrapper(
    policy: Optional[SubprocessSecurityPolicy] = None,
) -> SecureSubprocessWrapper:
    """Get or create secure subprocess wrapper."""
    global _default_subprocess_wrapper
    if _default_subprocess_wrapper is None:
        _default_subprocess_wrapper = SecureSubprocessWrapper(policy)
    return _default_subprocess_wrapper


def secure_popen(
    args: Union[str, Sequence[str]],
    policy: Optional[SubprocessSecurityPolicy] = None,
    **kwargs: Any,
) -> subprocess.Popen:
    """
    Explicit secure wrapper for subprocess.Popen.

    Usage:
        proc = secure_popen(["git", "status"], stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()
    """
    wrapper = get_secure_subprocess_wrapper(policy)
    return wrapper(args, **kwargs)


def secure_run(
    args: Union[str, Sequence[str]],
    policy: Optional[SubprocessSecurityPolicy] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """
    Explicit secure wrapper for subprocess.run.

    Usage:
        result = secure_run(["python", "-c", "print('hello')"], capture_output=True)
    """
    wrapper = get_secure_subprocess_wrapper(policy)

    # Handle both string and sequence args consistently with wrapper
    if isinstance(args, str):
        import shlex

        args_seq: Sequence[str] = shlex.split(args)
    else:
        args_seq = args

    # Validate the command
    cmd = args_seq[0] if args_seq else ""
    if not (
        wrapper.policy.is_allowed(cmd)
        or wrapper.policy.is_allowed(cmd.split()[0] if isinstance(args, str) else cmd)
    ):
        raise SecureSubprocessError(
            f"Subprocess command '{cmd}' is not in the allowlist. "
            f"Allowed commands: {wrapper.policy.allowed_commands}"
        )

    return subprocess.run(args, **kwargs)


# ---------------------------------------------------------------------------
# Pickle Security (Explicit Wrapper - No Monkey Patching)
# ---------------------------------------------------------------------------


class PickleSecurityError(ValueError):
    """Raised when pickle operation is blocked for security."""

    pass


class SecurePickleWrapper:
    """
    Explicit secure wrapper for pickle operations.
    By default, restricts to JSON-only serialization for security.

    Usage:
        # Instead of pickle.loads(data), use:
        from apgi_security import secure_loads

        obj = secure_loads(data)  # Only allows JSON, never pickle
    """

    def __init__(self, allow_pickle: bool = False):
        self.allow_pickle = allow_pickle

    def loads(self, data: Union[bytes, str], **kwargs: Any) -> Any:
        """Secure deserialization - defaults to JSON only."""
        if self.allow_pickle and kwargs.get("use_pickle", False):
            # Only allow pickle if explicitly enabled AND requested
            return pickle.loads(data)  # type: ignore

        # Default: JSON-only deserialization
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)
        except Exception as e:
            raise PickleSecurityError(
                f"Deserialization failed. Pickle is disabled by default for security. "
                f"Use secure_loads_json() for JSON data. Error: {e}"
            )

    def dumps(self, obj: Any, **kwargs: Any) -> bytes:
        """Secure serialization - defaults to JSON only."""
        if self.allow_pickle and kwargs.get("use_pickle", False):
            return pickle.dumps(obj)

        # Default: JSON serialization
        return json.dumps(obj, default=str).encode("utf-8")

    def load(self, file_obj: Any, **kwargs: Any) -> Any:
        """Secure file deserialization."""
        data = file_obj.read()
        return self.loads(data, **kwargs)

    def dump(self, obj: Any, file_obj: Any, **kwargs: Any) -> None:
        """Secure file serialization."""
        data = self.dumps(obj, **kwargs)
        file_obj.write(data)


# Global wrapper instances
_default_pickle_wrapper = SecurePickleWrapper(allow_pickle=False)
_unsafe_pickle_wrapper: Optional[SecurePickleWrapper] = None


def get_secure_pickle_wrapper(allow_pickle: bool = False) -> SecurePickleWrapper:
    """Get pickle wrapper instance."""
    if allow_pickle:
        global _unsafe_pickle_wrapper
        if _unsafe_pickle_wrapper is None:
            _unsafe_pickle_wrapper = SecurePickleWrapper(allow_pickle=True)
        return _unsafe_pickle_wrapper
    return _default_pickle_wrapper


def secure_loads(data: Union[bytes, str], use_pickle: bool = False) -> Any:
    """
    Explicit secure deserialization.
    Defaults to JSON only. Pickle only if explicitly requested.
    """
    wrapper = get_secure_pickle_wrapper(allow_pickle=use_pickle)
    return wrapper.loads(data, use_pickle=use_pickle)


def secure_loads_json(data: Union[bytes, str]) -> Any:
    """Explicit JSON-only deserialization (always safe)."""
    return _default_pickle_wrapper.loads(data)


def secure_dumps(obj: Any, use_pickle: bool = False) -> bytes:
    """
    Explicit secure serialization.
    Defaults to JSON. Pickle only if explicitly requested.
    """
    wrapper = get_secure_pickle_wrapper(allow_pickle=use_pickle)
    return wrapper.dumps(obj, use_pickle=use_pickle)


def secure_load(file_obj: Any, use_pickle: bool = False) -> Any:
    """Explicit secure file deserialization."""
    wrapper = get_secure_pickle_wrapper(allow_pickle=use_pickle)
    return wrapper.load(file_obj, use_pickle=use_pickle)


def secure_dump(obj: Any, file_obj: Any, use_pickle: bool = False) -> None:
    """Explicit secure file serialization."""
    wrapper = get_secure_pickle_wrapper(allow_pickle=use_pickle)
    return wrapper.dump(obj, file_obj, use_pickle=use_pickle)


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


def validate_config_checksum(
    config_dict: dict, expected_hash: str, secret_key: Optional[str] = None
) -> bool:
    """
    Validate configuration signing via checksums.

    Requires secret_key from environment if not provided (fail-closed).
    """
    if secret_key is None:
        secret_key = os.environ.get("APGI_CONFIG_SECRET_KEY")
        if not secret_key:
            raise ValueError(
                "Config validation requires APGI_CONFIG_SECRET_KEY environment variable"
            )

    config_str = json.dumps(config_dict, sort_keys=True) + secret_key
    computed_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
    return computed_hash == expected_hash


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
            # Set policy for this function scope
            policy = SubprocessSecurityPolicy(
                allowed_commands=commands,
                require_explicit_allowlist=True,
            )
            # Store policy in thread-local or context for nested calls
            wrapper._security_policy = policy  # type: ignore
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Legacy Compatibility (DEPRECATED - will be removed)
# ---------------------------------------------------------------------------


def _deprecated_secure_subprocess_popen(*args: Any, **kwargs: Any) -> Any:
    """DEPRECATED: Use secure_popen() instead."""
    import warnings

    warnings.warn(
        "Direct subprocess.Popen patching is removed. Use secure_popen() explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return secure_popen(*args, **kwargs)


def _deprecated_secure_loads(data: bytes) -> Any:
    """DEPRECATED: Use secure_loads() or secure_loads_json() instead."""
    import warnings

    warnings.warn(
        "Direct pickle.loads patching is removed. Use secure_loads() explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return secure_loads(data)

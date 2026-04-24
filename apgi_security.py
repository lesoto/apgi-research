"""
APGI Security Hardening Module

Implements deny-by-default audit allowlists, removes pickle, and config validation.
"""

import subprocess
import pickle
import hashlib
import json
from typing import Any

ALLOWED_SUBPROCESS_CMDS = ["git", "pytest", "python", "screencapture"]


def _secure_subprocess_popen(*args, **kwargs):
    if args and isinstance(args[0], (list, tuple)):
        cmd = args[0][0]
    elif args and isinstance(args[0], str):
        cmd = args[0].split()[0]
    else:
        cmd = kwargs.get("args", [""])[0]

    if cmd not in ALLOWED_SUBPROCESS_CMDS:
        raise PermissionError(f"Subprocess command '{cmd}' is not allowed.")
    return _original_popen(*args, **kwargs)


_original_popen = subprocess.Popen
subprocess.Popen = _secure_subprocess_popen  # type: ignore


def secure_loads(data: bytes) -> Any:
    """Secure unpickling alternative restricted to standard formats."""
    try:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return json.loads(data)
    except Exception as e:
        raise ValueError(
            f"Pickle is disabled for security. Attempted to load data: {e}"
        )


# Strip pickle mapping overrides directly on loads/load to restrict usage.
pickle.loads = secure_loads  # type: ignore
pickle.load = secure_loads  # type: ignore


def validate_config_checksum(
    config_dict: dict, expected_hash: str, secret_key: str = "default_secure_key"
) -> bool:
    """Validate configuration signing via checksums."""
    config_str = json.dumps(config_dict, sort_keys=True) + secret_key
    computed_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
    return computed_hash == expected_hash

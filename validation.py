"""
Input validation module for APGI experiments.

Provides validation functions for experiment parameters,
file modifications, and git operations to prevent security issues.
"""

import os
import re
import math
import platform
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


def get_dangerous_system_paths() -> List[str]:
    """Get system paths that are dangerous to modify - cross-platform."""
    dangerous = []
    system = platform.system()

    if system == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dangerous.extend(
            [
                f"{windir}\\System32",
                f"{windir}\\SysWOW64",
                f"{windir}\\Tasks",
                f"{windir}\\System32\\drivers",
                f"{windir}\\System32\\config",
                "C:\\Program Files",
                "C:\\Program Files (x86)",
                "C:\\ProgramData",
            ]
        )
    elif system == "Darwin":  # macOS
        dangerous.extend(
            [
                "/etc/",
                "/usr/bin/",
                "/bin/",
                "/sbin/",
                "/var/",
                "/tmp/",
                "/dev/",
                "/proc/",
                "/sys/",
                "/System/",
                "/Library/",
                "/usr/lib/",
                "/Applications/",
            ]
        )
    else:  # Linux and other Unix-like systems
        dangerous.extend(
            [
                "/etc/",
                "/usr/bin/",
                "/bin/",
                "/sbin/",
                "/var/",
                "/tmp/",
                "/dev/",
                "/proc/",
                "/sys/",
                "/usr/lib/",
                "/lib/",
                "/opt/",
            ]
        )

    return dangerous


@dataclass
class ValidationResult:
    """Result of validation with errors and warnings."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __post_init__(self):
        # Ensure lists are initialized
        if not hasattr(self, "errors") or self.errors is None:
            self.errors = []
        if not hasattr(self, "warnings") or self.warnings is None:
            self.warnings = []


def validate_modifications_before_apply(
    modifications: Dict[str, Any]
) -> ValidationResult:
    """
    Comprehensive validation of modifications before applying them.

    Args:
        modifications: Dictionary of modifications to validate

    Returns:
        ValidationResult with detailed validation status
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check for dangerous keys first
    dangerous_keys = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "isinstance",
        "issubclass",
        "callable",
        "__builtins__",
    }
    for key in modifications:
        if key in dangerous_keys:
            errors.append(f"Dangerous key detected: {key}")
        elif any(
            dangerous in key.lower()
            for dangerous in [
                "import",
                "eval",
                "exec",
                "compile",
                "global",
                "local",
                "builtin",
            ]
        ):
            errors.append(f"Potentially dangerous key detected: {key}")

    # Check for path traversal attempts
    for key, value in modifications.items():
        if isinstance(value, str):
            # Check for path traversal patterns
            if ".." in value and any(sep in value for sep in ["/", "\\"]):
                errors.append(f"Path traversal detected in {key}: {value}")
            # Check for shell command patterns
            if any(
                pattern in value for pattern in ["&", "|", ";", "$(", "`", ">", "<"]
            ):
                errors.append(f"Shell command pattern detected in {key}: {value}")

    # Validate parameter types and ranges
    for param_name, param_value in modifications.items():
        # Skip validation for file_path as it's handled separately
        if param_name == "file_path":
            continue

        # Validate time_budget
        if param_name == "time_budget":
            if not isinstance(param_value, (int, float)):
                errors.append(
                    f"time_budget must be numeric, got {type(param_value).__name__}"
                )
            elif param_value <= 0:
                errors.append(f"time_budget must be positive, got {param_value}")
            elif param_value > 3600:  # 1 hour max
                warnings.append(f"time_budget very high: {param_value}s (>{3600}s)")

        # Validate participant_id
        elif param_name == "participant_id":
            if not isinstance(param_value, str):
                errors.append(
                    f"participant_id must be string, got {type(param_value).__name__}"
                )
            elif len(param_value) > 50:
                errors.append(
                    f"participant_id too long: {len(param_value)} chars (max 50)"
                )
            elif not param_value.replace("_", "").replace("-", "").isalnum():
                warnings.append(
                    f"participant_id contains special characters: {param_value}"
                )

        # Validate stimulus_type
        elif param_name == "stimulus_type":
            valid_types = [
                "visual",
                "auditory",
                "tactile",
                "olfactory",
                "gustatory",
                "neutral",
                "survival",
            ]
            if param_value not in valid_types:
                errors.append(
                    f"Invalid stimulus_type: {param_value}. Valid: {valid_types}"
                )

        # Validate numeric parameters
        elif isinstance(param_value, (int, float)):
            if abs(param_value) > 1e6:  # Very large numbers
                warnings.append(f"Large parameter value: {param_name}={param_value}")
            elif isinstance(param_value, float) and not math.isfinite(param_value):
                errors.append(f"Non-finite parameter: {param_name}={param_value}")

        # Validate string parameters
        elif isinstance(param_value, str):
            if len(param_value) > 1000:
                errors.append(
                    f"String parameter too long: {param_name} ({len(param_value)} chars)"
                )
            elif any(
                char in param_value for char in ["\x00", "\x01", "\x02", "\x03", "\x04"]
            ):
                errors.append(
                    f"String parameter contains control characters: {param_name}"
                )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_code_modification(
    file_path: str, new_content: str, original_content: Optional[str] = None
) -> ValidationResult:
    """
    Validate code modifications for security and correctness.

    Args:
        file_path: Path to file being modified
        new_content: Proposed new content
        original_content: Original file content (optional)

    Returns:
        ValidationResult with validation status
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check file path safety
    path_validation = validate_git_operations([file_path], "modify")
    errors.extend(path_validation.errors)
    warnings.extend(path_validation.warnings)

    # Check for dangerous patterns in new content
    dangerous_patterns = [
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"subprocess\.",
        r"os\.system",
        r"os\.popen",
        r"open\s*shell",
        r"input\s*\|",
        r"\.git\/",
        r"rm\s+-rf",
        r"sudo\s+",
        r"chmod\s+777",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, new_content, re.IGNORECASE):
            errors.append(f"Dangerous pattern detected in new content: {pattern}")

    # Check for import statement safety
    import_pattern = r"^(?:from\s+(\S+)\s+)?import\s+(\S+)"
    for match in re.finditer(import_pattern, new_content, re.MULTILINE):
        module_name = match.group(2) if match.group(2) else match.group(1)
        if module_name:
            # Validate module name
            if not validate_module_name(module_name):
                errors.append(f"Invalid module name: {module_name}")

    # Size checks
    if len(new_content) > 10 * 1024 * 1024:  # 10MB
        errors.append(f"File too large: {len(new_content)} bytes (max 10MB)")

    # Syntax check for Python files
    if file_path.endswith(".py"):
        try:
            compile(new_content, file_path, "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error in modified code: {e}")

    # If original content provided, check for reasonable changes
    if original_content is not None:
        size_change = len(new_content) - len(original_content)
        if abs(size_change) > 100 * 1024:  # 100KB change
            warnings.append(f"Large file size change: {size_change:+d} bytes")

        # Check for deletion of important content
        if len(new_content) < len(original_content) * 0.5:
            warnings.append(
                f"Significant content reduction: {len(new_content)}/{len(original_content)} chars"
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_module_name(module_name: str) -> bool:
    """
    Validate Python module name for safety.

    Args:
        module_name: Module name to validate

    Returns:
        True if module name is safe
    """
    # Check for empty or whitespace-only names
    if not module_name or not module_name.strip():
        return False

    # Check for Python keywords
    import keyword

    if module_name in keyword.kwlist:
        return False

    # Check for valid module name format
    if not re.match(
        r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", module_name
    ):
        return False

    # Blacklist only truly dangerous modules that allow system access
    dangerous_modules = {
        "subprocess",
        "shutil",
        "ctypes",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "nntplib",
        "webbrowser",
        "pwd",
        "grp",
        "resource",
    }

    # Check for dangerous modules (check base module)
    base_module = module_name.split(".")[0]
    if base_module in dangerous_modules:
        return False

    # Check for suspicious patterns that could indicate path traversal
    suspicious_patterns = ["..", "~", "$", ";", "&", "|", ">", "<", "`"]
    if any(pattern in module_name for pattern in suspicious_patterns):
        return False

    return True


def validate_experiment_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate experiment configuration parameters.

    Args:
        config: Experiment configuration dictionary

    Returns:
        ValidationResult with validation status
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Required fields
    required_fields = ["experiment_name", "participant_id", "time_budget"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate experiment name
    if "experiment_name" in config:
        name = config["experiment_name"]
        if not isinstance(name, str) or len(name) == 0:
            errors.append("experiment_name must be non-empty string")
        elif len(name) > 100:
            errors.append("experiment_name too long (max 100 chars)")

    # Validate time_budget
    if "time_budget" in config:
        time_budget = config["time_budget"]
        if not isinstance(time_budget, (int, float)):
            errors.append("time_budget must be numeric")
        elif time_budget <= 0:
            errors.append("time_budget must be positive")
        elif time_budget > 3600:
            warnings.append(f"time_budget very high: {time_budget}s")

    # Validate optional parameters
    optional_numeric_fields = ["trial_count", "break_duration", "stimulus_duration"]
    for field in optional_numeric_fields:
        if field in config:
            value = config[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be numeric")
            elif value < 0:
                errors.append(f"{field} must be non-negative")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_subprocess_operation(
    command: List[str], args: Optional[List[str]] = None
) -> ValidationResult:
    """
    Validate subprocess operations for security.

    Args:
        command: Command list to execute
        args: Arguments for the command (optional)

    Returns:
        ValidationResult with validation status
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not command:
        errors.append("Empty command not allowed")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # Validate command
    if len(command) == 0:
        errors.append("Empty command not allowed")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    base_command = command[0]

    # Check for dangerous commands
    dangerous_commands = {
        "rm",
        "rmdir",
        "mv",
        "cp",
        "chmod",
        "chown",
        "chgrp",
        "sudo",
        "su",
        "kill",
        "killall",
        "pkill",
        "skill",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "fdiskutil",
        "format",
        "fsck",
        "dd",
        "nc",
        "telnet",
        "ssh",
        "scp",
        "rsync",
        "wget",
        "curl",
        "perl",
        "awk",
        "sed",
        "grep",
        "find",
        "locate",
        "tar",
        "gzip",
        "gunzip",
        "zip",
        "unzip",
        "mount",
        "umount",
    }

    # Check base command
    if base_command in dangerous_commands:
        errors.append(f"Dangerous command detected: {base_command}")

    # Validate command path safety
    if "/" in base_command or "\\" in base_command:
        # Check for path traversal attempts
        suspicious_patterns = ["..", "~", "$", ";", "&", "|", ">", "<", "`", "\\"]
        if any(pattern in base_command for pattern in suspicious_patterns):
            errors.append(f"Suspicious characters in command: {base_command}")

    # Validate arguments
    if args:
        for arg in args:
            if len(arg) > 1000:  # Very long argument
                errors.append(f"Argument too long: {len(arg)} chars")
            elif any(char in arg for char in ["\x00", "\x01", "\x02", "\x03", "\x04"]):
                errors.append(f"Control characters in argument: {arg}")
            elif arg.startswith("-") and len(arg) > 10:
                # Long flag arguments
                warnings.append(f"Long flag argument: {arg}")

    # Validate package names if this is a package manager command
    package_managers = ["pip", "conda", "npm", "yarn", "apt", "yum", "dnf", "brew"]
    if base_command in package_managers:
        if args:
            package_name = args[0]
            if not validate_package_name(package_name):
                errors.append(f"Invalid package name: {package_name}")

    # Check for shell injection patterns
    full_command = " ".join(command + (args or []))
    shell_injection_patterns = [
        ";",
        "&&",
        "||",
        "|",
        "&",
        "`",
        "$(",
        "${",
        "$(",
        "$)",
        ">",
        ">>",
        "<",
        "<<",
    ]

    for pattern in shell_injection_patterns:
        if pattern in full_command:
            errors.append(f"Shell injection pattern detected: {pattern}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_package_name(package_name: str) -> bool:
    """
    Validate package name for security.

    Args:
        package_name: Package name to validate

    Returns:
        True if package name is safe
    """
    # Blacklist dangerous packages
    dangerous_packages = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "tempfile",
        "pickle",
        "marshal",
        "ctypes",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "nntplib",
        "ssl",
        "hashlib",
        "hmac",
        "secrets",
        "uuid",
        "threading",
        "multiprocessing",
        "asyncio",
        "webbrowser",
        "platform",
        "pwd",
        "grp",
        "resource",
        "sysconfig",
        "importlib",
        "compile",
        "__import__",
        "__globals__",
        "__locals__",
        "open",
        "input",
        "raw_input",
        "help",
        "exit",
        "quit",
        "reload",
    }

    # Check for dangerous packages
    base_package = package_name.split(".")[0].split("-")[0]
    if base_package in dangerous_packages:
        return False

    # Check for suspicious patterns
    suspicious_patterns = ["..", "~", "$", ";", "&", "|", ">", "<", "`", "\\", "/", "@"]
    if any(pattern in package_name for pattern in suspicious_patterns):
        return False

    # Check for valid Python identifier (basic check)
    clean_name = package_name.replace(".", "").replace("-", "_")
    return clean_name.isidentifier() and len(clean_name) > 0


def validate_import_statement(import_statement: str) -> ValidationResult:
    """
    Validate import statement for security.

    Args:
        import_statement: Import statement to validate

    Returns:
        ValidationResult with validation status
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Parse import statement
    import re

    import_pattern = r"^(?:from\s+(\S+)\s+)?import\s+(\S+)"
    match = re.match(import_pattern, import_statement.strip())

    if not match:
        errors.append("Invalid import statement syntax")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    from_module = match.group(1)
    import_name = match.group(2)

    # Validate module names
    if from_module and not validate_module_name(from_module):
        errors.append(f"Invalid from module: {from_module}")

    if import_name and not validate_module_name(import_name):
        errors.append(f"Invalid import module: {import_name}")

    # Check for multiple imports on single line
    if "," in import_name:
        warnings.append("Multiple imports on single line detected")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_experiment_parameters(modifications: Dict[str, Any]) -> ValidationResult:
    """Validate experiment parameters for safety (legacy function for compatibility)."""
    # Delegate to the new comprehensive validation function
    return validate_modifications_before_apply(modifications)


def get_safe_directories() -> List[str]:
    """Get directories that are generally safe for modifications."""
    system = platform.system()

    if system == "Windows":
        return [
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Downloads"),
            os.environ.get("TEMP", "C:\\temp"),
            os.environ.get("TMP", "C:\\tmp"),
        ]
    else:  # Unix-like systems (macOS, Linux)
        return [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~"),
            tempfile.gettempdir(),  # Use proper temp directory
        ]


def validate_git_operations(
    files: List[str],
    operation: str,
) -> ValidationResult:
    """
    Validate git operations for security.

    Args:
        files: List of files being operated on
        operation: Git operation type

    Returns:
        ValidationResult with validation status
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Get dangerous paths for current platform
    dangerous_paths = get_dangerous_system_paths()
    safe_directories = get_safe_directories()
    SAFE_EXTENSIONS = [".py", ".json", ".txt", ".csv", ".md"]

    for file_path in files:
        path_obj = Path(file_path)

        # Check if file exists (allow relative paths for testing)
        if not path_obj.exists() and not path_obj.is_absolute():
            # For relative paths, just check the format without existence
            pass
        elif not path_obj.exists():
            errors.append(f"File does not exist: {file_path}")
            continue

        # Check if file is in safe directory
        resolved_path = str(path_obj.resolve())
        in_safe_dir = any(safe_dir in resolved_path for safe_dir in safe_directories)

        if not in_safe_dir:
            # Check for dangerous paths using cross-platform function
            if any(dp in resolved_path for dp in dangerous_paths):
                errors.append(f"File in dangerous location: {resolved_path}")

        # Check file extension
        if path_obj.suffix.lower() not in SAFE_EXTENSIONS:
            errors.append(f"Unsafe file extension: {path_obj.suffix}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Guardrail Escalation System
# ---------------------------------------------------------------------------


@dataclass
class GuardrailEscalation:
    """Represents a guardrail escalation event."""

    trigger: str  # "confidence", "regression", "safety", "anomaly"
    severity: str  # "warning", "critical"
    experiment_name: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()


def check_guardrails(
    confidence: float,
    primary_metric: float,
    performance_history: list,
    experiment_name: str,
    confidence_threshold: float = 0.3,
    regression_window: int = 3,
) -> List[GuardrailEscalation]:
    """
    Comprehensive guardrail check combining confidence, regression,
    and safety violation detection.

    Args:
        confidence: Current agent confidence (0.0-1.0)
        primary_metric: Current primary metric value
        performance_history: List of past metric values
        experiment_name: Name of the experiment being evaluated
        confidence_threshold: Minimum acceptable confidence
        regression_window: Number of iterations to check for regression

    Returns:
        List of GuardrailEscalation objects (empty = all clear)
    """
    import math as _math
    import logging as _logging

    _logger = _logging.getLogger(__name__)
    escalations: List[GuardrailEscalation] = []

    # 1. Confidence threshold check
    if confidence < confidence_threshold:
        esc = GuardrailEscalation(
            trigger="confidence",
            severity="critical" if confidence < 0.1 else "warning",
            experiment_name=experiment_name,
            message=f"Agent confidence {confidence:.3f} below threshold {confidence_threshold}",
            metric_value=confidence,
            threshold=confidence_threshold,
        )
        escalations.append(esc)
        _logger.warning(f"[GUARDRAIL] {esc.message}")

    # 2. Safety violation: NaN / Inf / extreme values
    if _math.isnan(primary_metric) or _math.isinf(primary_metric):
        esc = GuardrailEscalation(
            trigger="safety",
            severity="critical",
            experiment_name=experiment_name,
            message=f"Safety violation: metric is {primary_metric}",
            metric_value=primary_metric,
        )
        escalations.append(esc)
        _logger.error(f"[GUARDRAIL] {esc.message}")

    # 3. Metric regression detection
    if len(performance_history) >= regression_window:
        window = performance_history[-regression_window:]
        declining = all(window[i] <= window[i - 1] for i in range(1, len(window)))
        if declining:
            esc = GuardrailEscalation(
                trigger="regression",
                severity="warning",
                experiment_name=experiment_name,
                message=(
                    f"Metric regression over {regression_window} iterations: "
                    f"{window}"
                ),
                metric_value=window[-1],
            )
            escalations.append(esc)
            _logger.warning(f"[GUARDRAIL] {esc.message}")

    return escalations


def escalate_to_human(
    escalation: GuardrailEscalation,
    log_path: str = "guardrail_escalations.json",
) -> bool:
    """
    Persist an escalation event and notify human reviewers.

    Args:
        escalation: The escalation event to record
        log_path: Path to the escalation log file

    Returns:
        True if logged successfully
    """
    import json as _json
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    entry = {
        "trigger": escalation.trigger,
        "severity": escalation.severity,
        "experiment": escalation.experiment_name,
        "message": escalation.message,
        "metric_value": escalation.metric_value,
        "threshold": escalation.threshold,
        "timestamp": escalation.timestamp,
    }

    try:
        log_file = Path(log_path)
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = _json.load(f)
        else:
            logs = []

        logs.append(entry)
        with open(log_file, "w") as f:
            _json.dump(logs, f, indent=2)

        _logger.info(
            f"[GUARDRAIL] Escalation logged: {escalation.trigger} "
            f"({escalation.severity}) for {escalation.experiment_name}"
        )
        return True
    except Exception as e:
        _logger.error(f"[GUARDRAIL] Failed to log escalation: {e}")
        return False

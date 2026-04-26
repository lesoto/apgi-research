"""
Enhanced validation and monitoring module for APGI autonomous agent.

Provides:
- Pre-validation of modifications before applying
- Rollback on git operation failures
- Experiment progress tracking
- Performance trending and memory monitoring
- Input validation for __import__ calls
- Package name validation for subprocess operations
"""

import json
import logging
import os
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, cast

import psutil

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Whitelisted safe import modules
SAFE_IMPORT_MODULES = {
    # Standard library
    "os",
    "sys",
    "json",
    "time",
    "math",
    "random",
    "re",
    "datetime",
    "pathlib",
    "typing",
    "dataclasses",
    "functools",
    "itertools",
    "collections",
    "statistics",
    "hashlib",
    "uuid",
    "inspect",
    "copy",
    "pickle",
    "csv",
    "io",
    "warnings",
    "string",
    "numbers",
    "decimal",
    "fractions",
    "typing",
    "enum",
    "abc",
    "types",
    # Scientific computing
    "numpy",
    "numpy.random",
    "numpy.linalg",
    "numpy.fft",
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    "scipy.signal",
    # ML/AI
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils",
    # Experiment-specific
    "apgi_integration",
    "experiment_apgi_integration",
}

# Dangerous import patterns to reject
DANGEROUS_IMPORT_PATTERNS = [
    r"__import__\s*\(\s*['\"]os['\"]",  # Dynamic os import
    r"__import__\s*\(\s*['\"]sys['\"]",  # Dynamic sys import
    r"__import__\s*\(\s*['\"]subprocess['\"]",  # Dynamic subprocess
    r"__import__\s*\(.*eval",  # Import eval
    r"__import__\s*\(.*exec",  # Import exec
]

# Safe subprocess package patterns
SAFE_PACKAGE_PATTERNS = {
    "pip",
    "python",
    "git",
    "pytest",
    "black",
    "flake8",
    "mypy",
    "conda",
    "mamba",
    "uv",
    "pipenv",
    "poetry",
}

# Package name validation regex
VALID_PACKAGE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")

# File backup extension
BACKUP_EXTENSION = ".backup"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ValidationReport:
    """Comprehensive validation report for modifications."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    modifications_validated: List[str] = field(default_factory=list)
    modifications_rejected: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_validated(self, param: str) -> None:
        self.modifications_validated.append(param)

    def add_rejected(self, param: str) -> None:
        self.modifications_rejected.append(param)


@dataclass
class ProgressSnapshot:
    """Snapshot of experiment progress."""

    timestamp: float
    iteration: int
    total_iterations: int
    metric_value: Optional[float] = None
    status: str = "running"  # running, paused, completed, failed
    checkpoint_path: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance and resource metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    experiment_runtime_s: float
    metric_trend: str = "stable"  # improving, degrading, stable
    metric_change_rate: float = 0.0


@dataclass
class ModificationBackup:
    """Backup information for modifications."""

    original_file_path: str
    backup_file_path: str
    modifications: Dict[str, Any]
    timestamp: float
    commit_hash_before: Optional[str] = None


# =============================================================================
# MODIFICATION VALIDATOR
# =============================================================================


class ModificationValidator:
    """Validates parameter modifications before applying."""

    # Parameter value constraints
    PARAMETER_CONSTRAINTS = {
        # Numeric ranges
        "BASE_DETECTION_RATE": (0.0, 1.0),
        "TARGET_DETECTION_RATE": (0.0, 1.0),
        "DETECTION_THRESHOLD": (0.0, 1.0),
        "STIMULUS_DURATION": (1, 10000),  # ms
        "INTER_STIMULUS_INTERVAL": (0, 5000),  # ms
        "MASK_DURATION": (1, 1000),  # ms
        "SOA_DURATION": (0, 2000),  # ms
        "CUE_DURATION": (1, 1000),  # ms
        "TARGET_DURATION": (1, 2000),  # ms
        "RESPONSE_WINDOW": (100, 5000),  # ms
        "TIMEOUT_DURATION": (1000, 60000),  # ms
        "MAX_RESPONSE_TIME": (100, 10000),  # ms
        "TRIALS_PER_BLOCK": (1, 500),
        "NUM_BLOCKS": (1, 50),
        "NUM_TRIALS": (1, 1000),
        "LEARNING_RATE": (0.0001, 1.0),
        "MOMENTUM": (0.0, 1.0),
        "WEIGHT_DECAY": (0.0, 1.0),
        "BATCH_SIZE": (1, 512),
        "HIDDEN_UNITS": (1, 4096),
        "NUM_LAYERS": (1, 20),
        "DROPOUT_RATE": (0.0, 0.9),
        "REGULARIZATION": (0.0, 1.0),
        # Boolean parameters (no range, just type check)
        "USE_ADAPTIVE_STIMULUS": bool,
        "USE_FEEDBACK": bool,
        "USE_PRACTICE_TRIALS": bool,
        "USE_RANDOMIZATION": bool,
        "USE_CUEING": bool,
        "USE_MASKING": bool,
        "USE_TIMEOUT": bool,
        "VERBOSE": bool,
        "DEBUG": bool,
        "SAVE_RESULTS": bool,
        "PLOT_RESULTS": bool,
    }

    @classmethod
    def validate_modifications(
        cls, modifications: Dict[str, Any], file_path: Optional[str] = None
    ) -> ValidationReport:
        """
        Validate modifications before applying.

        Args:
            modifications: Dictionary of parameter modifications
            file_path: Optional path to file being modified

        Returns:
            ValidationReport with validation results
        """
        report = ValidationReport(is_valid=True)

        for param_name, param_value in modifications.items():
            # Check if parameter is known
            if param_name not in cls.PARAMETER_CONSTRAINTS:
                report.add_warning(f"Unknown parameter: {param_name}")
                report.add_validated(param_name)
                continue

            constraint = cls.PARAMETER_CONSTRAINTS[param_name]

            # Validate based on constraint type
            if constraint is bool:
                if not isinstance(param_value, bool):
                    report.add_error(
                        f"Parameter {param_name} must be boolean, got {type(param_value)}"
                    )
                    report.add_rejected(param_name)
                else:
                    report.add_validated(param_name)

            elif isinstance(constraint, tuple):
                min_val, max_val = constraint

                # Type checking
                if not isinstance(param_value, (int, float)):
                    report.add_error(
                        f"Parameter {param_name} must be numeric, got {type(param_value)}"
                    )
                    report.add_rejected(param_name)
                    continue

                # Range checking
                if param_value < min_val or param_value > max_val:
                    report.add_error(
                        f"Parameter {param_name}={param_value} out of range "
                        f"[{min_val}, {max_val}]"
                    )
                    report.add_rejected(param_name)
                else:
                    report.add_validated(param_name)

            # Validate file path if provided
            if file_path:
                path_validation = cls._validate_file_path(file_path)
                if not path_validation.is_valid:
                    for error in path_validation.errors:
                        report.add_error(error)

        return report

    @classmethod
    def _validate_file_path(cls, file_path: str) -> ValidationReport:
        """Validate that file path is safe for modification."""
        report = ValidationReport(is_valid=True)
        path = Path(file_path).resolve()

        # Check for dangerous paths
        dangerous_patterns = [
            "/etc/",
            "/usr/bin/",
            "/bin/",
            "/sbin/",
            "/var/",
            "/dev/",
            "/proc/",
            "/sys/",
            "/lib/",
            "/lib64/",
        ]

        path_str = str(path)
        for pattern in dangerous_patterns:
            if pattern in path_str:
                report.add_error(f"Cannot modify system path: {path_str}")
                break

        # Check if file exists and is writable
        if path.exists() and not os.access(path, os.W_OK):
            report.add_error(f"File not writable: {path_str}")

        return report


# =============================================================================
# IMPORT VALIDATOR
# =============================================================================


class ImportValidator:
    """Validates __import__ calls for security."""

    @classmethod
    def validate_import_call(
        cls, import_code: str, allowed_modules: Optional[set] = None
    ) -> ValidationReport:
        """
        Validate an __import__ call for security.

        Args:
            import_code: The code containing the import
            allowed_modules: Optional set of allowed module names

        Returns:
            ValidationReport with validation results
        """
        report = ValidationReport(is_valid=True)
        allowed = allowed_modules or SAFE_IMPORT_MODULES

        # Check for dangerous patterns
        for pattern in DANGEROUS_IMPORT_PATTERNS:
            if re.search(pattern, import_code):
                report.add_error(f"Dangerous import pattern detected: {pattern}")

        # Extract module name from __import__ call
        import_match = re.search(r'__import__\s*\(\s*["\']([^"\']+)["\']', import_code)
        if import_match:
            module_name = import_match.group(1)

            # Check if module is in whitelist
            if module_name not in allowed:
                report.add_warning(f"Module not in whitelist: {module_name}")

            # Check for relative imports
            if module_name.startswith("."):
                report.add_error(f"Relative imports not allowed: {module_name}")

            # Check for submodule access
            base_module = module_name.split(".")[0]
            if base_module not in allowed:
                report.add_error(f"Base module not allowed: {base_module}")

        return report

    @classmethod
    def validate_importlib_usage(cls, code: str) -> ValidationReport:
        """Validate importlib usage patterns."""
        report = ValidationReport(is_valid=True)

        # Check for importlib.import_module
        if "importlib.import_module" in code:
            # Extract the module name being imported
            match = re.search(
                r'importlib\.import_module\s*\(\s*["\']([^"\']+)["\']', code
            )
            if match:
                module_name = match.group(1)
                if module_name not in SAFE_IMPORT_MODULES:
                    report.add_warning(
                        f"importlib usage with non-whitelisted module: {module_name}"
                    )

        return report


# =============================================================================
# SUBPROCESS VALIDATOR
# =============================================================================


class SubprocessValidator:
    """Validates subprocess operations and package names."""

    @classmethod
    def validate_package_name(cls, package_name: str) -> ValidationReport:
        """
        Validate a package name for subprocess operations.

        Args:
            package_name: Name of the package/command

        Returns:
            ValidationReport with validation results
        """
        report = ValidationReport(is_valid=True)

        # Check against safe patterns
        if package_name not in SAFE_PACKAGE_PATTERNS:
            report.add_warning(f"Package not in safe list: {package_name}")

        # Validate characters
        if not VALID_PACKAGE_NAME_REGEX.match(package_name):
            report.add_error(
                f"Invalid package name characters: {package_name}. "
                "Only alphanumeric, underscore, and hyphen allowed."
            )

        # Check for path traversal attempts
        if "/" in package_name or "\\" in package_name:
            report.add_error(
                f"Path separators not allowed in package name: {package_name}"
            )

        # Check for shell metacharacters
        shell_metacharacters = [";", "|", "&", "$", "`", "(", ")", "<", ">"]
        for char in shell_metacharacters:
            if char in package_name:
                report.add_error(f"Shell metacharacter not allowed: {char}")

        return report

    @classmethod
    def validate_subprocess_call(
        cls, command: List[str], shell: bool = False
    ) -> ValidationReport:
        """
        Validate a subprocess call.

        Args:
            command: Command and arguments list
            shell: Whether shell is being used

        Returns:
            ValidationReport with validation results
        """
        report = ValidationReport(is_valid=True)

        # Shell should be False for security
        if shell:
            report.add_error("Subprocess with shell=True is not allowed")

        if not command:
            report.add_error("Empty command")
            return report

        # Validate the main command
        cmd = command[0]
        package_validation = cls.validate_package_name(cmd)
        if not package_validation.is_valid:
            report.errors.extend(package_validation.errors)
            report.warnings.extend(package_validation.warnings)

        # Check for dangerous arguments
        dangerous_args = ["-c", "--command", "eval", "exec", "|", ";", "&&", "||"]
        for arg in command[1:]:
            if arg in dangerous_args:
                report.add_error(f"Dangerous argument detected: {arg}")

        return report


# =============================================================================
# MODIFICATION BACKUP MANAGER
# =============================================================================


class ModificationBackupManager:
    """Manages backups for safe rollback."""

    def __init__(self, backup_dir: str = ".apgi_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.backups: List[ModificationBackup] = []

    def create_backup(
        self,
        file_path: str,
        modifications: Dict[str, Any],
        commit_hash: Optional[str] = None,
    ) -> ModificationBackup:
        """Create a backup before applying modifications."""
        original_path = Path(file_path)

        if not original_path.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")

        # Create backup filename with timestamp
        timestamp = time.time()
        backup_name = (
            f"{original_path.stem}_{timestamp}{BACKUP_EXTENSION}{original_path.suffix}"
        )
        backup_path = self.backup_dir / backup_name

        # Copy original file
        shutil.copy2(file_path, backup_path)

        backup = ModificationBackup(
            original_file_path=file_path,
            backup_file_path=str(backup_path),
            modifications=modifications,
            timestamp=timestamp,
            commit_hash_before=commit_hash,
        )

        self.backups.append(backup)
        return backup

    def restore_backup(self, backup: ModificationBackup) -> bool:
        """Restore from a backup."""
        try:
            backup_path = Path(backup.backup_file_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            shutil.copy2(backup.backup_file_path, backup.original_file_path)
            logger.info(f"Restored backup to {backup.original_file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

    def cleanup_old_backups(self, max_age_hours: float = 24.0) -> None:
        """Remove backups older than specified hours."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for backup in self.backups[:]:
            if current_time - backup.timestamp > max_age_seconds:
                backup_path = Path(backup.backup_file_path)
                if backup_path.exists():
                    backup_path.unlink()
                self.backups.remove(backup)


# =============================================================================
# PROGRESS TRACKER
# =============================================================================


class ExperimentProgressTracker:
    """Tracks experiment progress with checkpoints."""

    def __init__(self, experiment_name: str, total_iterations: int):
        self.experiment_name = experiment_name
        self.total_iterations = total_iterations
        self.snapshots: List[ProgressSnapshot] = []
        self.start_time = time.time()
        self.current_iteration = 0
        self._checkpoint_dir = Path(".apgi_checkpoints")
        self._checkpoint_dir.mkdir(exist_ok=True)

    def update_progress(
        self,
        iteration: int,
        status: str = "running",
        metric_value: Optional[float] = None,
    ) -> ProgressSnapshot:
        """Update progress and create snapshot."""
        self.current_iteration = iteration

        snapshot = ProgressSnapshot(
            timestamp=time.time(),
            iteration=iteration,
            total_iterations=self.total_iterations,
            metric_value=metric_value,
            status=status,
        )

        self.snapshots.append(snapshot)

        # Save checkpoint every 10 iterations
        if iteration % 10 == 0:
            self._save_checkpoint(snapshot)

        return snapshot

    def _save_checkpoint(self, snapshot: ProgressSnapshot) -> None:
        """Save progress checkpoint to disk."""
        checkpoint_file = (
            self._checkpoint_dir / f"{self.experiment_name}_{snapshot.iteration}.json"
        )

        checkpoint_data = {
            "experiment_name": self.experiment_name,
            "iteration": snapshot.iteration,
            "total_iterations": snapshot.total_iterations,
            "metric_value": snapshot.metric_value,
            "status": snapshot.status,
            "timestamp": snapshot.timestamp,
            "elapsed_time": time.time() - self.start_time,
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        snapshot.checkpoint_path = str(checkpoint_file)

    def load_checkpoint(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Load a saved checkpoint."""
        checkpoint_file = (
            self._checkpoint_dir / f"{self.experiment_name}_{iteration}.json"
        )

        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, "r") as f:
            return cast(Optional[Dict[str, Any]], json.load(f))

    def get_progress_percentage(self) -> float:
        """Get progress as percentage."""
        if self.total_iterations == 0:
            return 100.0
        return (self.current_iteration / self.total_iterations) * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def get_estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.current_iteration == 0:
            return None

        elapsed = self.get_elapsed_time()
        avg_time_per_iteration = elapsed / self.current_iteration
        remaining_iterations = self.total_iterations - self.current_iteration

        return avg_time_per_iteration * remaining_iterations

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        return {
            "experiment_name": self.experiment_name,
            "progress_percentage": self.get_progress_percentage(),
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "elapsed_time": self.get_elapsed_time(),
            "estimated_remaining": self.get_estimated_remaining_time(),
            "snapshots_count": len(self.snapshots),
        }


# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================


class PerformanceMonitor:
    """Monitors system performance and resource usage."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.process = psutil.Process()
        self.metric_history: List[float] = []

    def capture_metrics(
        self, current_metric: Optional[float] = None
    ) -> PerformanceMetrics:
        """Capture current performance metrics."""
        timestamp = time.time()

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Calculate metric trend
        metric_trend = "stable"
        metric_change_rate = 0.0

        if current_metric is not None:
            self.metric_history.append(current_metric)

            if len(self.metric_history) >= 3:
                recent_avg = sum(self.metric_history[-3:]) / 3
                previous_avg = (
                    sum(self.metric_history[-6:-3]) / 3
                    if len(self.metric_history) >= 6
                    else recent_avg
                )

                if previous_avg != 0:
                    change = (recent_avg - previous_avg) / abs(previous_avg)
                    metric_change_rate = change

                    if change > 0.05:
                        metric_trend = "improving"
                    elif change < -0.05:
                        metric_trend = "degrading"

        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            experiment_runtime_s=timestamp - self.start_time,
            metric_trend=metric_trend,
            metric_change_rate=metric_change_rate,
        )

        self.metrics_history.append(metrics)
        return metrics

    def check_resource_limits(self) -> List[str]:
        """Check if resource usage exceeds safe limits."""
        warnings: List[str] = []

        if not self.metrics_history:
            return warnings

        latest = self.metrics_history[-1]

        # Memory limit: 90%
        if latest.memory_percent > 90:
            warnings.append(f"High memory usage: {latest.memory_percent:.1f}%")

        # CPU limit: 95% sustained
        if latest.cpu_percent > 95:
            warnings.append(f"High CPU usage: {latest.cpu_percent:.1f}%")

        # Disk limit: 95%
        if latest.disk_usage_percent > 95:
            warnings.append(f"Low disk space: {latest.disk_usage_percent:.1f}% used")

        return warnings

    def get_trend_summary(self, window_size: int = 10) -> Dict[str, Any]:
        """Get trend summary over recent window."""
        if len(self.metrics_history) < window_size:
            window = self.metrics_history
        else:
            window = self.metrics_history[-window_size:]

        if not window:
            return {}

        avg_cpu = sum(m.cpu_percent for m in window) / len(window)
        avg_memory = sum(m.memory_percent for m in window) / len(window)

        # Memory trend
        memory_trend = "stable"
        if len(window) >= 3:
            recent = sum(m.memory_percent for m in window[-3:]) / 3
            older = sum(m.memory_percent for m in window[:3]) / 3
            if recent > older * 1.1:
                memory_trend = "increasing"
            elif recent < older * 0.9:
                memory_trend = "decreasing"

        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "memory_trend": memory_trend,
            "latest_metric_trend": window[-1].metric_trend if window else "unknown",
            "runtime_seconds": window[-1].experiment_runtime_s if window else 0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]
        trend = self.get_trend_summary()
        warnings = self.check_resource_limits()

        return {
            "current": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_used_mb": latest.memory_used_mb,
                "disk_usage_percent": latest.disk_usage_percent,
                "runtime_s": latest.experiment_runtime_s,
            },
            "trends": trend,
            "warnings": warnings,
            "samples_count": len(self.metrics_history),
        }


# =============================================================================
# ROLLBACK MANAGER
# =============================================================================


class RollbackManager:
    """Manages rollback operations for git failures."""

    def __init__(self) -> None:
        self.failed_operations: List[Dict[str, Any]] = []
        self.rollback_hooks: List[Callable] = []

    def register_rollback_hook(self, hook: Callable) -> None:
        """Register a function to call during rollback."""
        self.rollback_hooks.append(hook)

    def record_failure(
        self, operation: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a failed operation."""
        self.failed_operations.append(
            {
                "operation": operation,
                "error": str(error),
                "timestamp": time.time(),
                "context": context or {},
            }
        )

    def execute_rollback(self, backup_manager: ModificationBackupManager) -> bool:
        """Execute rollback of failed operations."""
        success = True

        # Execute rollback hooks first
        for hook in self.rollback_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Rollback hook failed: {e}")
                success = False

        # Restore from backups
        for failure in self.failed_operations:
            context = failure.get("context", {})
            backup = context.get("backup")

            if backup:
                if not backup_manager.restore_backup(backup):
                    success = False

        return success


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================


@contextmanager
def validated_modifications(
    file_path: str,
    modifications: Dict[str, Any],
    backup_manager: ModificationBackupManager,
    commit_hash: Optional[str] = None,
) -> Iterator[ValidationReport]:
    """
    Context manager for validated modifications with automatic rollback.

    Usage:
        with validated_modifications(file_path, mods, backup_mgr) as result:
            # Apply modifications
            pass
    """
    # Pre-validate
    validator = ModificationValidator()
    validation_report = validator.validate_modifications(modifications, file_path)

    if not validation_report.is_valid:
        logger.error(f"Validation failed: {validation_report.errors}")
        raise ValueError(f"Modification validation failed: {validation_report.errors}")

    # Create backup
    backup = backup_manager.create_backup(file_path, modifications, commit_hash)

    try:
        yield validation_report

    except Exception as e:
        # Rollback on failure
        logger.error(f"Modification failed, rolling back: {e}")
        backup_manager.restore_backup(backup)
        raise


@contextmanager
def git_operation_guard(rollback_manager: RollbackManager) -> Iterator[None]:
    """
    Context manager for git operations with rollback support.

    Usage:
        with git_operation_guard(rollback_mgr):
            # Git operation
            pass
    """
    try:
        yield

    except Exception as e:
        # Record failure and trigger rollback
        rollback_manager.record_failure("git_operation", e)
        rollback_manager.execute_rollback(ModificationBackupManager())
        raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def validate_file_modifications(
    file_path: str, modifications: Dict[str, Any]
) -> ValidationReport:
    """Convenience function to validate file modifications."""
    return ModificationValidator.validate_modifications(modifications, file_path)


def validate_import_safety(code: str) -> ValidationReport:
    """Convenience function to validate import safety."""
    return ImportValidator.validate_import_call(code)


def validate_subprocess_safety(command: List[str]) -> ValidationReport:
    """Convenience function to validate subprocess safety."""
    return SubprocessValidator.validate_subprocess_call(command)


__all__ = [
    # Data classes
    "ValidationReport",
    "ProgressSnapshot",
    "PerformanceMetrics",
    "ModificationBackup",
    # Validators
    "ModificationValidator",
    "ImportValidator",
    "SubprocessValidator",
    # Managers
    "ModificationBackupManager",
    "ExperimentProgressTracker",
    "PerformanceMonitor",
    "RollbackManager",
    # Context managers
    "validated_modifications",
    "git_operation_guard",
    # Utilities
    "validate_file_modifications",
    "validate_import_safety",
    "validate_subprocess_safety",
    # Constants
    "SAFE_IMPORT_MODULES",
    "SAFE_PACKAGE_PATTERNS",
]

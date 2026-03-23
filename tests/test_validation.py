"""
Comprehensive tests for validation module.
"""

import os
import pytest
from unittest.mock import patch

from validation import (
    get_dangerous_system_paths,
    ValidationResult,
    validate_modifications_before_apply,
    validate_code_modification,
    validate_module_name,
    validate_experiment_config,
    validate_subprocess_operation,
    validate_package_name,
    validate_import_statement,
    validate_experiment_parameters,
    get_safe_directories,
    validate_git_operations,
)


class TestGetDangerousSystemPaths:
    """Tests for get_dangerous_system_paths function."""

    def test_get_dangerous_system_paths_windows(self):
        """Test getting dangerous paths on Windows."""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"WINDIR": "C:\\Windows"}, clear=True):
                paths = get_dangerous_system_paths()

                assert isinstance(paths, list)
                assert len(paths) > 0
                assert any("System32" in path for path in paths)
                assert any("Program Files" in path for path in paths)

    def test_get_dangerous_system_paths_macos(self):
        """Test getting dangerous paths on macOS."""
        with patch("platform.system", return_value="Darwin"):
            paths = get_dangerous_system_paths()

            assert isinstance(paths, list)
            assert len(paths) > 0
            assert "/etc/" in paths
            assert "/usr/bin/" in paths
            assert "/System/" in paths
            assert "/Applications/" in paths

    def test_get_dangerous_system_paths_linux(self):
        """Test getting dangerous paths on Linux."""
        with patch("platform.system", return_value="Linux"):
            paths = get_dangerous_system_paths()

            assert isinstance(paths, list)
            assert len(paths) > 0
            assert "/etc/" in paths
            assert "/usr/bin/" in paths
            assert "/opt/" in paths

    def test_get_dangerous_system_paths_unix_default(self):
        """Test getting dangerous paths on Unix-like systems."""
        with patch("platform.system", return_value="Unix"):
            paths = get_dangerous_system_paths()

            assert isinstance(paths, list)
            assert len(paths) > 0
            assert "/etc/" in paths
            assert "/usr/bin/" in paths


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_initialization_valid(self):
        """Test ValidationResult initialization with valid result."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_initialization_invalid(self):
        """Test ValidationResult initialization with invalid result."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]
        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        assert result.is_valid is False
        assert result.errors == errors
        assert result.warnings == warnings

    def test_post_init_empty_lists(self):
        """Test post_init with empty lists."""
        result = ValidationResult(is_valid=True)
        result.__post_init__()
        assert result.errors == []
        assert result.warnings == []

    def test_post_init_none_lists(self):
        """Test post_init with None lists."""
        result = ValidationResult(is_valid=True, errors=None, warnings=None)
        result.__post_init__()
        assert result.errors == []
        assert result.warnings == []


class TestValidateModificationsBeforeApply:
    """Tests for validate_modifications_before_apply function."""

    def test_validate_safe_modifications(self):
        """Test validating safe modifications."""
        modifications = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "use_dropout": True,
        }
        result = validate_modifications_before_apply(modifications)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_dangerous_keys(self):
        """Test validating modifications with dangerous keys."""
        modifications = {
            "__import__": "os.system",
            "eval": "dangerous_code",
            "exec": "dangerous_code",
        }
        result = validate_modifications_before_apply(modifications)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("dangerous" in error.lower() for error in result.errors)

    def test_validate_path_traversal(self):
        """Test validating modifications with path traversal."""
        modifications = {
            "file_path": "../../../etc/passwd",
            "output_dir": "..\\..\\..\\Windows\\System32",
        }
        result = validate_modifications_before_apply(modifications)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("path traversal" in error.lower() for error in result.errors)

    def test_validate_shell_commands(self):
        """Test validating modifications with shell commands."""
        modifications = {"command": "rm -rf /", "script": "curl | sh"}
        result = validate_modifications_before_apply(modifications)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("shell" in error.lower() for error in result.errors)

    def test_validate_empty_modifications(self):
        """Test validating empty modifications."""
        modifications = {}
        result = validate_modifications_before_apply(modifications)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_large_values(self):
        """Test validating modifications with suspiciously large values."""
        modifications = {
            "batch_size": 1000000,  # Suspiciously large
            "memory_limit": -1,  # Negative memory
            "timeout": 0,  # Zero timeout
        }
        result = validate_modifications_before_apply(modifications)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestValidateCodeModification:
    """Tests for validate_code_modification function."""

    def test_validate_safe_code(self):
        """Test validating safe code modifications."""
        file_path = "test.py"
        new_content = "print('Hello, World!')"
        result = validate_code_modification(file_path, new_content)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_dangerous_code(self):
        """Test validating dangerous code modifications."""
        file_path = "test.py"
        new_content = "import os; os.system('rm -rf /')"
        result = validate_code_modification(file_path, new_content)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_dangerous_imports(self):
        """Test validating dangerous imports."""
        file_path = "test.py"
        new_content = (
            "import subprocess\nsubprocess.call(['rm', '-rf', '/'], shell=True)"
        )
        result = validate_code_modification(file_path, new_content)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_eval_usage(self):
        """Test validating eval usage."""
        file_path = "test.py"
        new_content = "result = eval(user_input)"
        result = validate_code_modification(file_path, new_content)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_exec_usage(self):
        """Test validating exec usage."""
        file_path = "test.py"
        new_content = "exec(user_code)"
        result = validate_code_modification(file_path, new_content)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_file_access(self):
        """Test validating dangerous file access."""
        file_path = "test.py"
        new_content = "with open('/etc/passwd', 'r') as f: print(f.read())"
        result = validate_code_modification(file_path, new_content)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_with_original_content(self):
        """Test validation with original content comparison."""
        file_path = "test.py"
        original = "x = 1"
        new_content = "x = 2"
        result = validate_code_modification(file_path, new_content, original)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True


class TestValidateModuleName:
    """Tests for validate_module_name function."""

    def test_validate_safe_module_names(self):
        """Test validating safe module names."""
        assert validate_module_name("numpy") is True
        assert validate_module_name("pandas") is True
        assert validate_module_name("my_module") is True
        assert validate_module_name("test123") is True

    def test_validate_unsafe_module_names(self):
        """Test validating unsafe module names."""
        assert validate_module_name("import") is False  # Python keyword
        assert validate_module_name("class") is False  # Python keyword
        assert validate_module_name("def") is False  # Python keyword
        assert validate_module_name("if") is False  # Python keyword
        assert validate_module_name("1module") is False  # Starts with digit
        assert validate_module_name("module-name") is False  # Contains hyphen
        assert validate_module_name("module$name") is False  # Contains special char

    def test_validate_empty_module_name(self):
        """Test validating empty module name."""
        assert validate_module_name("") is False
        assert validate_module_name("   ") is False

    def test_validate_module_with_dots(self):
        """Test validating module names with dots."""
        assert validate_module_name("os.path") is True
        assert validate_module_name("my.module.name") is True


class TestValidateExperimentConfig:
    """Tests for validate_experiment_config function."""

    def test_validate_valid_config(self):
        """Test validating valid experiment config."""
        config = {
            "experiment_name": "test_experiment",
            "trials": 100,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10,
        }
        result = validate_experiment_config(config)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_invalid_config(self):
        """Test validating invalid experiment config."""
        config = {
            "experiment_name": "",  # Empty name
            "trials": -1,  # Negative trials
            "learning_rate": 2.0,  # Too high learning rate
            "batch_size": 0,  # Zero batch size
            "epochs": 10000,  # Too many epochs
        }
        result = validate_experiment_config(config)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_missing_required_fields(self):
        """Test validating config with missing required fields."""
        config = {}  # Empty config
        result = validate_experiment_config(config)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_extreme_values(self):
        """Test validating config with extreme values."""
        config = {
            "experiment_name": "test",
            "trials": 1000000,  # Too many trials
            "timeout": 0.001,  # Too short timeout
            "memory_limit": -1,  # Negative memory limit
        }
        result = validate_experiment_config(config)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestValidateSubprocessOperation:
    """Tests for validate_subprocess_operation function."""

    def test_validate_safe_command(self):
        """Test validating safe subprocess command."""
        command = ["python", "script.py", "--verbose"]
        result = validate_subprocess_operation(command)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_dangerous_command(self):
        """Test validating dangerous subprocess command."""
        command = ["rm", "-rf", "/"]
        result = validate_subprocess_operation(command)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_shell_usage(self):
        """Test validating subprocess with shell usage."""
        command = ["python", "-c", "import os; os.system('rm -rf /')"]
        result = validate_subprocess_operation(command)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_dangerous_args(self):
        """Test validating dangerous command arguments."""
        command = ["python", "script.py"]
        args = ["|", "sh", "rm", "-rf"]
        result = validate_subprocess_operation(command, args)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_empty_command(self):
        """Test validating empty command."""
        command = []
        result = validate_subprocess_operation(command)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_system_commands(self):
        """Test validating system commands."""
        command = ["sudo", "rm", "-rf", "/"]
        result = validate_subprocess_operation(command)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestValidatePackageName:
    """Tests for validate_package_name function."""

    def test_validate_safe_package_names(self):
        """Test validating safe package names."""
        assert validate_package_name("numpy") is True
        assert validate_package_name("pandas") is True
        assert validate_package_name("requests") is True
        assert validate_package_name("my-package") is True

    def test_validate_unsafe_package_names(self):
        """Test validating unsafe package names."""
        assert validate_package_name("package-name") is False  # Contains hyphen
        assert validate_package_name("package$name") is False  # Contains special char
        assert validate_package_name("1package") is False  # Starts with digit
        assert validate_package_name("") is False
        assert validate_package_name("   ") is False

    def test_validate_package_with_underscores(self):
        """Test validating package names with underscores."""
        assert validate_package_name("my_package") is True
        assert validate_package_name("test_package_123") is True


class TestValidateImportStatement:
    """Tests for validate_import_statement function."""

    def test_validate_safe_imports(self):
        """Test validating safe import statements."""
        imports = [
            "import numpy",
            "import pandas as pd",
            "from sklearn.model_selection import train_test_split",
            "import matplotlib.pyplot as plt",
        ]

        for imp in imports:
            result = validate_import_statement(imp)
            assert result.is_valid is True

    def test_validate_dangerous_imports(self):
        """Test validating dangerous import statements."""
        dangerous_imports = [
            "import os; os.system('rm -rf /')",
            "import subprocess; subprocess.call(['rm', '-rf', '/'], shell=True)",
            'eval(\'__import__("os").system("rm -rf /")\')',
        ]

        for imp in dangerous_imports:
            result = validate_import_statement(imp)
            assert result.is_valid is False
            assert len(result.errors) > 0

    def test_validate_from_import_dangerous(self):
        """Test validating dangerous from-import statements."""
        dangerous_imports = [
            "from os import system; system('rm -rf /')",
            "from subprocess import call; call(['rm', '-rf', '/'], shell=True)",
            'from builtins import eval; eval(\'__import__("os").system("rm -rf /")\')',
        ]

        for imp in dangerous_imports:
            result = validate_import_statement(imp)
            assert result.is_valid is False
            assert len(result.errors) > 0

    def test_validate_import_star(self):
        """Test validating star imports."""
        result = validate_import_statement("from module import *")
        assert result.is_valid is True  # Star imports are generally allowed

    def test_validate_relative_imports(self):
        """Test validating relative imports."""
        result = validate_import_statement("from .module import something")
        assert result.is_valid is True  # Relative imports are allowed


class TestValidateExperimentParameters:
    """Tests for validate_experiment_parameters function."""

    def test_validate_safe_parameters(self):
        """Test validating safe experiment parameters."""
        params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "dropout_rate": 0.5,
        }
        result = validate_experiment_parameters(params)

        # This should delegate to validate_modifications_before_apply
        assert isinstance(result, ValidationResult)

    def test_validate_unsafe_parameters(self):
        """Test validating unsafe experiment parameters."""
        params = {"__import__": "os.system", "eval": "dangerous"}
        result = validate_experiment_parameters(params)

        # This should delegate to validate_modifications_before_apply
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False


class TestGetSafeDirectories:
    """Tests for get_safe_directories function."""

    def test_get_safe_directories_windows(self):
        """Test getting safe directories on Windows."""
        with patch("platform.system", return_value="Windows"):
            dirs = get_safe_directories()

            assert isinstance(dirs, list)
            assert len(dirs) > 0
            assert any("Desktop" in dir for dir in dirs)
            assert any("Documents" in dir for dir in dirs)

    def test_get_safe_directories_macos(self):
        """Test getting safe directories on macOS."""
        with patch("platform.system", return_value="Darwin"):
            dirs = get_safe_directories()

            assert isinstance(dirs, list)
            assert len(dirs) > 0
            assert any("Desktop" in dir for dir in dirs)
            assert any("Documents" in dir for dir in dirs)
            assert any("Downloads" in dir for dir in dirs)

    def test_get_safe_directories_linux(self):
        """Test getting safe directories on Linux."""
        with patch("platform.system", return_value="Linux"):
            dirs = get_safe_directories()

            assert isinstance(dirs, list)
            assert len(dirs) > 0
            assert any("home" in dir for dir in dirs)
            assert any("tmp" in dir for dir in dirs)


class TestValidateGitOperations:
    """Tests for validate_git_operations function."""

    def test_validate_safe_git_operations(self):
        """Test validating safe git operations."""
        files = ["src/main.py", "tests/test_main.py"]
        operation = "add"
        result = validate_git_operations(files, operation)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_dangerous_git_files(self):
        """Test validating git operations on dangerous files."""
        files = ["/etc/passwd", "/usr/bin/python"]
        operation = "add"
        result = validate_git_operations(files, operation)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_dangerous_git_operations(self):
        """Test validating dangerous git operations."""
        files = ["src/main.py"]
        dangerous_ops = ["reset --hard", "clean -fd", "filter-branch --force"]

        for op in dangerous_ops:
            result = validate_git_operations(files, op)
            assert result.is_valid is False
            assert len(result.errors) > 0

    def test_validate_git_operation_with_empty_files(self):
        """Test validating git operation with empty file list."""
        files = []
        operation = "add"
        result = validate_git_operations(files, operation)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_git_operation_delete_files(self):
        """Test validating git delete operations."""
        files = ["/etc/passwd"]
        operation = "rm"
        result = validate_git_operations(files, operation)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestModuleIntegration:
    """Integration tests for the validation module."""

    def test_validation_workflow_comprehensive(self):
        """Test complete validation workflow."""
        # Test dangerous paths detection
        dangerous_paths = get_dangerous_system_paths()
        assert isinstance(dangerous_paths, list)

        # Test validation results
        result = ValidationResult(is_valid=False, errors=["Test error"])
        assert result.is_valid is False
        assert result.errors == ["Test error"]

        # Test parameter validation
        safe_params = {"learning_rate": 0.01}
        result = validate_modifications_before_apply(safe_params)
        assert isinstance(result, ValidationResult)

    def test_cross_platform_functionality(self):
        """Test that functions work across platforms."""
        # Test with different platforms
        platforms = ["Windows", "Darwin", "Linux"]

        for platform_name in platforms:
            with patch("platform.system", return_value=platform_name):
                # Test dangerous paths
                paths = get_dangerous_system_paths()
                assert isinstance(paths, list)

                # Test safe directories
                dirs = get_safe_directories()
                assert isinstance(dirs, list)

    def test_error_handling(self):
        """Test error handling in validation functions."""
        # Test with None inputs where appropriate
        try:
            validate_module_name(None)
            assert False, "Should return False for None input"
        except (TypeError, AttributeError):
            pass  # Expected for None input

        try:
            validate_package_name("")
            assert False, "Should return False for empty string"
        except (TypeError, AttributeError):
            pass  # Expected for empty string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

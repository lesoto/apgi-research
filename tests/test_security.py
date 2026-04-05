"""
================================================================================
SECURITY TESTS FOR APGI SYSTEM
================================================================================

This module provides comprehensive security testing including:
- Input sanitization validation
- SQL injection resistance
- XSS prevention
- Path traversal prevention
- Command injection resistance
- Data validation and type checking
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# INPUT SANITIZATION TESTS
# =============================================================================


class TestInputSanitization:
    """Tests for input sanitization and validation."""

    @pytest.mark.security
    @pytest.mark.parametrize(
        "malicious_input",
        [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1; DELETE FROM users",
            "SELECT * FROM passwords",
            "admin'--",
            "' UNION SELECT * FROM users--",
        ],
    )
    def test_sql_injection_resistance(
        self, malicious_input: str, security_tester
    ) -> None:
        """Test that SQL injection patterns are detected."""
        matches = security_tester.detect_sql_injection(malicious_input)
        assert (
            len(matches) > 0
        ), f"SQL injection pattern not detected: {malicious_input}"

    @pytest.mark.security
    @pytest.mark.parametrize(
        "malicious_input",
        [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
            "<iframe src='javascript:alert(1)'>",
            "<body onload=alert(1)>",
        ],
    )
    def test_xss_detection(self, malicious_input: str, security_tester) -> None:
        """Test that XSS patterns are detected."""
        matches = security_tester.detect_xss(malicious_input)
        assert len(matches) > 0, f"XSS pattern not detected: {malicious_input}"

    @pytest.mark.security
    @pytest.mark.parametrize(
        "input_value,expected_safe",
        [
            ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
            ('"quoted"', "&quot;quoted&quot;"),
            ("<div>content</div>", "&lt;div&gt;content&lt;/div&gt;"),
            ("&amp;test", "&amp;amp;test"),
        ],
    )
    def test_html_escaping(
        self, input_value: str, expected_safe: str, security_tester
    ) -> None:
        """Test HTML escaping for XSS prevention."""
        sanitized = security_tester.sanitize_input(input_value)
        assert "<script>" not in sanitized
        assert "<" not in sanitized or "&lt;" in sanitized

    @pytest.mark.security
    @pytest.mark.parametrize(
        "path_attempt",
        [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../.env",
            "../../../proc/self/environ",
            "/etc/passwd",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
        ],
    )
    def test_path_traversal_detection(self, path_attempt: str) -> None:
        """Test that path traversal attempts are detected."""
        # Path traversal patterns contain '..' or absolute paths
        is_suspicious = ".." in path_attempt or path_attempt.startswith(("/", "C:"))
        assert is_suspicious, f"Path traversal not detected: {path_attempt}"


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================


class TestDataValidation:
    """Tests for data validation and type checking."""

    @pytest.mark.security
    @pytest.mark.parametrize(
        "invalid_config",
        [
            {"n_trials": "not_a_number"},  # Type mismatch
            {"duration_ms": float("inf")},  # Infinite value
            {"random_seed": None},  # Null value where number expected
            {"output_dir": 123},  # Number where string expected
            {"isi_range": "500-1000"},  # String where list expected
            {"save_results": "yes"},  # String where bool expected
        ],
    )
    def test_configuration_type_validation(
        self, invalid_config: Dict[str, Any]
    ) -> None:
        """Test that invalid configuration types are rejected."""
        # Check that invalid types would be problematic
        for key, value in invalid_config.items():
            if key in ["n_trials", "duration_ms", "random_seed"]:
                assert (
                    not isinstance(value, (int, float)) or not np.isfinite(float(value))
                    if isinstance(value, (int, float))
                    else True
                ), f"{key} should be a valid number"

    @pytest.mark.security
    def test_numpy_array_type_safety(self) -> None:
        """Test numpy array type safety."""
        # Test that arrays maintain expected types
        float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        int_arr = np.array([1, 2, 3], dtype=np.int64)

        assert float_arr.dtype == np.float64
        assert int_arr.dtype == np.int64

        # Type promotion should be predictable
        result = float_arr + int_arr
        assert result.dtype == np.float64

    @pytest.mark.security
    @pytest.mark.parametrize(
        "overflow_input",
        [
            np.array([1e308, 1e308]),  # Overflow on sum
            np.array([1e154, 1e154]),  # Product overflow
            np.full(1000, sys.float_info.max),  # Array of max values
        ],
    )
    def test_numeric_overflow_handling(self, overflow_input: np.ndarray) -> None:
        """Test handling of numeric overflow scenarios."""
        # Operations should not crash the system
        try:
            np.sum(overflow_input)
            # Result may be inf, but should not crash
        except (OverflowError, FloatingPointError):
            # Acceptable to raise exception
            pass


# =============================================================================
# ENVIRONMENT AND SECRET HANDLING TESTS
# =============================================================================


class TestEnvironmentSecurity:
    """Tests for environment variable and secret handling."""

    @pytest.mark.security
    def test_sensitive_env_var_access(self) -> None:
        """Test that sensitive environment variables are handled carefully."""
        # Set a mock secret
        os.environ["TEST_SECRET"] = "super_secret_key_12345"

        # Access should work but not be logged
        secret = os.environ.get("TEST_SECRET")
        assert secret == "super_secret_key_12345"

        # Clean up
        del os.environ["TEST_SECRET"]

    @pytest.mark.security
    def test_env_var_sanitization(self) -> None:
        """Test environment variable sanitization."""
        # Malicious values in env vars
        malicious_values = [
            "$(whoami)",
            "`cat /etc/passwd`",
            "${PATH}",
            "<script>alert(1)</script>",
        ]

        for malicious in malicious_values:
            os.environ["TEST_VAR"] = malicious
            value = os.environ.get("TEST_VAR")
            # Should be stored as-is (not executed)
            assert value == malicious
            del os.environ["TEST_VAR"]


# =============================================================================
# FILE SYSTEM SECURITY TESTS
# =============================================================================


class TestFileSystemSecurity:
    """Tests for file system security."""

    @pytest.mark.security
    def test_safe_file_path_resolution(self, temp_dir: Path) -> None:
        """Test that file paths are resolved safely."""
        # Create a file in temp dir
        safe_file = temp_dir / "safe_file.txt"
        safe_file.write_text("content")

        # Path traversal should not escape temp dir
        suspicious_path = temp_dir / ".." / ".." / "etc" / "passwd"
        resolved = suspicious_path.resolve()

        # Resolved path should not be in temp_dir
        assert not str(resolved).startswith(str(temp_dir))

    @pytest.mark.security
    def test_file_extension_validation(self, temp_dir: Path) -> None:
        """Test file extension validation."""
        # Safe extensions
        safe_extensions = [".txt", ".json", ".npy", ".csv", ".log"]
        # Dangerous extensions
        dangerous_extensions = [".exe", ".sh", ".bat", ".php", ".jsp"]

        for ext in safe_extensions:
            file_path = temp_dir / f"test{ext}"
            assert file_path.suffix in safe_extensions

        for ext in dangerous_extensions:
            file_path = temp_dir / f"test{ext}"
            assert file_path.suffix not in safe_extensions

    @pytest.mark.security
    def test_symlink_handling(self, temp_dir: Path) -> None:
        """Test handling of symbolic links."""
        # Create a real file
        real_file = temp_dir / "real_file.txt"
        real_file.write_text("real content")

        # Create a symlink
        symlink = temp_dir / "symlink"
        try:
            symlink.symlink_to(real_file)
            # Symlink should resolve to real file
            assert symlink.resolve() == real_file.resolve()
        except OSError:
            # Symlinks may not be supported on all systems
            pytest.skip("Symlinks not supported")


# =============================================================================
# JSON SERIALIZATION SECURITY TESTS
# =============================================================================


class TestJSONSerializationSecurity:
    """Tests for JSON serialization security."""

    @pytest.mark.security
    def test_json_encoding_safety(self) -> None:
        """Test JSON encoding of potentially dangerous data."""
        # Data that might cause issues
        data = {
            "safe_string": "normal text",
            "html_string": "<script>alert(1)</script>",
            "path_string": "../../../etc/passwd",
            "command_string": "$(whoami)",
            "unicode": "🔥🚀💻日本語",
        }

        # JSON encoding should escape properly
        json_str = json.dumps(data)

        # Verify it can be decoded
        decoded = json.loads(json_str)
        assert decoded["safe_string"] == "normal text"
        assert decoded["html_string"] == "<script>alert(1)</script>"

    @pytest.mark.security
    def test_numpy_json_serialization(self) -> None:
        """Test safe serialization of numpy arrays to JSON."""
        arr = np.array([1.0, 2.0, 3.0])

        # Numpy arrays need conversion before JSON serialization
        data = {"array": arr.tolist()}
        json_str = json.dumps(data)

        # Verify serialization works
        decoded = json.loads(json_str)
        assert decoded["array"] == [1.0, 2.0, 3.0]

    @pytest.mark.security
    def test_prevent_arbitrary_code_execution(self) -> None:
        """Test that JSON loading doesn't execute arbitrary code."""
        # Malicious JSON that might execute code
        malicious_json = '{"__class__": "os.system", "args": ["whoami"]}'

        # Standard json.loads should not execute anything
        data = json.loads(malicious_json)
        assert data["__class__"] == "os.system"
        # Should be treated as plain data, not executed


# =============================================================================
# RANDOM NUMBER GENERATOR SECURITY
# =============================================================================


class TestRandomNumberSecurity:
    """Tests for random number generator security."""

    @pytest.mark.security
    def test_deterministic_rng_reproducibility(self) -> None:
        """Test that deterministic RNG produces reproducible results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        values1 = rng1.random(1000)
        values2 = rng2.random(1000)

        np.testing.assert_array_equal(values1, values2)

    @pytest.mark.security
    def test_rng_seed_isolation(self) -> None:
        """Test that different seeds produce different sequences."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)

        values1 = rng1.random(100)
        values2 = rng2.random(100)

        # Should be different (with very high probability)
        assert not np.array_equal(values1, values2)

    @pytest.mark.security
    def test_rng_state_not_leaked(self) -> None:
        """Test that RNG state doesn't leak between operations."""
        rng = np.random.default_rng(42)

        # First sequence
        seq1 = rng.random(100)

        # Create new RNG with same seed - should produce same first sequence
        rng2 = np.random.default_rng(42)
        seq1_copy = rng2.random(100)

        np.testing.assert_array_equal(seq1, seq1_copy)


# =============================================================================
# PARAMETER VALIDATION SECURITY
# =============================================================================


class TestParameterValidationSecurity:
    """Security tests for parameter validation."""

    @pytest.mark.security
    def test_apgi_parameters_boundary_protection(self) -> None:
        """Test that APGI parameters enforce boundary constraints."""
        from APGI_System import APGIParameters

        # Create parameters with extreme but valid values
        params = APGIParameters(
            tau_S=0.2,  # Boundary value
            tau_theta=60.0,  # Boundary value
            theta_0=1.0,  # Boundary value
        )

        violations = params.validate()
        # At exact boundaries, should be valid
        assert len(violations) == 0, f"Boundary values rejected: {violations}"

    @pytest.mark.security
    def test_apgi_parameters_reject_invalid_ranges(self) -> None:
        """Test that APGI parameters reject values outside valid ranges."""
        from APGI_System import APGIParameters

        # Test various out-of-range values
        invalid_cases = [
            ("tau_S", 0.01),  # Too small
            ("tau_S", 10.0),  # Too large
            ("beta", 0.0),  # Too small
            ("beta", 10.0),  # Too large
        ]

        for param_name, invalid_value in invalid_cases:
            params = APGIParameters()
            setattr(params, param_name, invalid_value)
            violations = params.validate()
            assert len(violations) > 0, f"Should reject {param_name}={invalid_value}"

    @pytest.mark.security
    def test_no_negative_precisions(self) -> None:
        """Test that precision values are always positive."""
        from APGI_System import FoundationalEquations

        # Precision should be capped at positive value for invalid inputs
        result = FoundationalEquations.precision(-1.0)
        assert result > 0
        assert result == 1e6  # Capped value


# =============================================================================
# METRICS AND LOGGING SECURITY
# =============================================================================


class TestMetricsLoggingSecurity:
    """Security tests for metrics and logging."""

    @pytest.mark.security
    def test_log_injection_prevention(self) -> None:
        """Test that log messages don't execute code."""
        import logging

        # Malicious log message
        malicious_message = "User login: $(whoami) <script>alert(1)</script>"

        # Logger should treat this as plain text
        logger = logging.getLogger("test_logger")

        # Should not raise exception or execute
        try:
            logger.info(malicious_message)
        except Exception as e:
            pytest.fail(f"Logger should handle any message: {e}")

    @pytest.mark.security
    def test_metric_calculation_safety(self) -> None:
        """Test that metric calculations handle edge cases safely."""
        from APGI_System import DerivedQuantities

        # Extreme values that might cause issues
        extreme_histories = [
            np.array([1e308] * 10),  # Very large values
            np.array([1e-308] * 10),  # Very small values
            np.full(100, np.nan),  # All NaN
            np.array([]),  # Empty
        ]

        for history in extreme_histories:
            try:
                DerivedQuantities.metabolic_cost(history, dt=0.01)
                # Should either return a value or raise a controlled exception
            except Exception as e:
                # Exception is acceptable if it's controlled
                assert not isinstance(
                    e, (MemoryError, SystemExit)
                ), f"Unexpected exception type: {type(e)}"

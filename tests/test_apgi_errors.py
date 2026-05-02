"""
Comprehensive tests for apgi_errors.py - Error taxonomy module.
"""

import pytest

from apgi_errors import (
    APGIConfigurationError,
    APGIDataValidationError,
    APGIError,
    APGIIntegrationError,
    APGIRuntimeError,
    APGITimeoutError,
)


class TestAPGIError:
    """Tests for base APGIError class."""

    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = APGIError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_error_with_context(self):
        """Test error with context dictionary."""
        context = {"key": "value", "number": 42}
        error = APGIError("Error with context", context=context)
        assert error.context == context

    def test_error_with_empty_context(self):
        """Test error with empty context (default)."""
        error = APGIError("Simple error")
        assert error.context == {}

    def test_error_message_accessible(self):
        """Test that error message is accessible via message attribute."""
        error = APGIError("Custom message")
        assert error.message == "Custom message"

    def test_error_inheritance(self):
        """Test that APGIError inherits from Exception."""
        assert issubclass(APGIError, Exception)

    def test_error_catchable_as_exception(self):
        """Test that APGIError can be caught as Exception."""
        try:
            raise APGIError("Test")
        except Exception as e:
            assert isinstance(e, APGIError)
            assert str(e) == "Test"


class TestAPGIConfigurationError:
    """Tests for APGIConfigurationError."""

    def test_configuration_error_creation(self):
        """Test configuration error creation."""
        error = APGIConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert isinstance(error, APGIError)

    def test_configuration_error_with_context(self):
        """Test configuration error with context."""
        error = APGIConfigurationError(
            "Missing parameter", context={"parameter": "tau_S"}
        )
        assert error.context["parameter"] == "tau_S"

    def test_configuration_error_inheritance(self):
        """Test that APGIConfigurationError inherits from APGIError."""
        assert issubclass(APGIConfigurationError, APGIError)


class TestAPGIRuntimeError:
    """Tests for APGIRuntimeError."""

    def test_runtime_error_creation(self):
        """Test runtime error creation."""
        error = APGIRuntimeError("Runtime failure")
        assert str(error) == "Runtime failure"
        assert isinstance(error, APGIError)

    def test_runtime_error_with_context(self):
        """Test runtime error with context."""
        error = APGIRuntimeError(
            "Simulation failed", context={"trial": 5, "error_code": 123}
        )
        assert error.context["trial"] == 5

    def test_runtime_error_inheritance(self):
        """Test that APGIRuntimeError inherits from APGIError."""
        assert issubclass(APGIRuntimeError, APGIError)


class TestAPGIDataValidationError:
    """Tests for APGIDataValidationError."""

    def test_validation_error_creation(self):
        """Test data validation error creation."""
        error = APGIDataValidationError("Invalid data format")
        assert str(error) == "Invalid data format"
        assert isinstance(error, APGIError)

    def test_validation_error_with_context(self):
        """Test validation error with context."""
        error = APGIDataValidationError(
            "Out of range", context={"field": "tau_S", "value": 999}
        )
        assert error.context["field"] == "tau_S"

    def test_validation_error_inheritance(self):
        """Test that APGIDataValidationError inherits from APGIError."""
        assert issubclass(APGIDataValidationError, APGIError)


class TestAPGIIntegrationError:
    """Tests for APGIIntegrationError."""

    def test_integration_error_creation(self):
        """Test integration error creation."""
        error = APGIIntegrationError("External system failure")
        assert str(error) == "External system failure"
        assert isinstance(error, APGIError)

    def test_integration_error_with_context(self):
        """Test integration error with context."""
        error = APGIIntegrationError(
            "API call failed", context={"endpoint": "/api/v1/data", "status": 500}
        )
        assert error.context["endpoint"] == "/api/v1/data"

    def test_integration_error_inheritance(self):
        """Test that APGIIntegrationError inherits from APGIError."""
        assert issubclass(APGIIntegrationError, APGIError)


class TestAPGITimeoutError:
    """Tests for APGITimeoutError."""

    def test_timeout_error_creation(self):
        """Test timeout error creation."""
        error = APGITimeoutError("Operation timed out")
        assert str(error) == "Operation timed out"
        assert isinstance(error, APGIError)

    def test_timeout_error_with_context(self):
        """Test timeout error with context."""
        error = APGITimeoutError(
            "Experiment timeout", context={"timeout_seconds": 300, "elapsed": 305}
        )
        assert error.context["timeout_seconds"] == 300

    def test_timeout_error_inheritance(self):
        """Test that APGITimeoutError inherits from APGIError."""
        assert issubclass(APGITimeoutError, APGIError)


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Test that all error types inherit from APGIError."""
        errors = [
            APGIConfigurationError,
            APGIRuntimeError,
            APGIDataValidationError,
            APGIIntegrationError,
            APGITimeoutError,
        ]
        for error_class in errors:
            assert issubclass(error_class, APGIError)

    def test_all_errors_catchable_as_apgi(self):
        """Test that all errors can be caught as APGIError."""
        errors = [
            APGIConfigurationError("Test"),
            APGIRuntimeError("Test"),
            APGIDataValidationError("Test"),
            APGIIntegrationError("Test"),
            APGITimeoutError("Test"),
        ]
        for error in errors:
            assert isinstance(error, APGIError)


class TestErrorEdgeCases:
    """Edge case tests for errors."""

    def test_error_with_none_message(self):
        """Test error with None message."""
        error = APGIError(None)
        assert str(error) == "None"
        assert error.message is None

    def test_error_with_empty_message(self):
        """Test error with empty message."""
        error = APGIError("")
        assert str(error) == ""
        assert error.message == ""

    def test_error_with_unicode_message(self):
        """Test error with unicode message."""
        error = APGIError("Error: 你好世界 🌍")
        assert "你好世界" in str(error)

    def test_error_with_long_message(self):
        """Test error with very long message."""
        long_msg = "A" * 10000
        error = APGIError(long_msg)
        assert len(str(error)) == 10000

    def test_error_with_nested_context(self):
        """Test error with nested context dictionary."""
        nested_context = {
            "level1": {"level2": {"level3": "deep_value"}},
            "list": [1, 2, 3],
        }
        error = APGIError("Nested", context=nested_context)
        assert error.context["level1"]["level2"]["level3"] == "deep_value"

    def test_error_raise_and_catch_specific(self):
        """Test raising and catching specific error types."""
        error_types = [
            (APGIConfigurationError, "config"),
            (APGIRuntimeError, "runtime"),
            (APGIDataValidationError, "validation"),
            (APGIIntegrationError, "integration"),
            (APGITimeoutError, "timeout"),
        ]

        for error_class, msg in error_types:
            with pytest.raises(error_class):
                raise error_class(f"Test {msg}")

    def test_error_chaining(self):
        """Test error chaining with from clause."""
        original = ValueError("Original error")
        chained = APGIError("Wrapped error")
        chained.__cause__ = original

        assert chained.__cause__ is original


class TestErrorUseCases:
    """Real-world use case tests for errors."""

    def test_configuration_error_for_invalid_parameter(self):
        """Test using configuration error for invalid parameter."""
        param_name = "tau_S"
        invalid_value = -1.0

        error = APGIConfigurationError(
            f"Invalid value for {param_name}: {invalid_value}",
            context={
                "parameter": param_name,
                "value": invalid_value,
                "expected": "> 0",
            },
        )

        assert param_name in str(error)
        assert error.context["parameter"] == param_name

    def test_validation_error_for_data_format(self):
        """Test using validation error for data format issues."""
        error = APGIDataValidationError(
            "Input data must be 2D array",
            context={"shape": (100,), "expected": "(n_trials, n_features)"},
        )

        assert "2D array" in str(error)
        assert error.context["expected"] == "(n_trials, n_features)"

    def test_timeout_error_for_experiment(self):
        """Test using timeout error for experiment timeout."""
        error = APGITimeoutError(
            "Experiment exceeded time budget",
            context={
                "experiment": "stroop_effect",
                "timeout_seconds": 600,
                "elapsed_seconds": 605.3,
            },
        )

        assert "time budget" in str(error)
        assert error.context["experiment"] == "stroop_effect"

    def test_integration_error_for_api_failure(self):
        """Test using integration error for API failures."""
        error = APGIIntegrationError(
            "Failed to fetch experiment data",
            context={
                "endpoint": "/api/experiments/123",
                "method": "GET",
                "status_code": 503,
            },
        )

        assert error.context["status_code"] == 503

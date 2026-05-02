"""
Comprehensive tests for apgi_protocols.py - Protocol definitions module.
"""

from typing import Any, Dict

import pytest

from apgi_protocols import (
    APGIModelProtocol,
    BaseAPGIRunner,
    ExperimentRunnerProtocol,
    deprecated,
)


class ValidRunner:
    """Valid experiment runner implementation."""

    def run_experiment(self) -> Dict[str, Any]:
        return {"success": True}


class InvalidRunner:
    """Invalid experiment runner (missing run_experiment)."""

    def execute(self):
        pass


class ValidAPGIModel:
    """Valid APGI model implementation."""

    def process_trial(self, observed: float, predicted: float) -> Dict[str, float]:
        return {"error": abs(observed - predicted)}

    def reset(self) -> None:
        pass


class InvalidAPGIModel:
    """Invalid APGI model (missing required methods)."""

    def other_method(self):
        pass


class TestExperimentRunnerProtocol:
    """Tests for ExperimentRunnerProtocol."""

    def test_runner_protocol_compliance(self):
        """Test that valid runner satisfies protocol."""
        assert isinstance(ValidRunner(), ExperimentRunnerProtocol)

    def test_invalid_runner_not_protocol(self):
        """Test that invalid runner does not satisfy protocol."""
        assert not isinstance(InvalidRunner(), ExperimentRunnerProtocol)

    def test_protocol_runtime_checkable(self):
        """Test that ExperimentRunnerProtocol is runtime checkable."""
        # Should not raise when checking isinstance
        try:
            isinstance(ValidRunner(), ExperimentRunnerProtocol)
            isinstance(InvalidRunner(), ExperimentRunnerProtocol)
        except Exception as e:
            pytest.fail(f"Runtime protocol check raised exception: {e}")


class TestAPGIModelProtocol:
    """Tests for APGIModelProtocol."""

    def test_valid_model_protocol(self):
        """Test that valid model satisfies protocol."""
        assert isinstance(ValidAPGIModel(), APGIModelProtocol)

    def test_invalid_model_not_protocol(self):
        """Test that invalid model does not satisfy protocol."""
        assert not isinstance(InvalidAPGIModel(), APGIModelProtocol)

    def test_model_protocol_methods(self):
        """Test that protocol methods are accessible."""
        model = ValidAPGIModel()
        result = model.process_trial(1.0, 1.5)
        assert "error" in result

    def test_model_reset_method(self):
        """Test that reset method exists and is callable."""
        model = ValidAPGIModel()
        # Should not raise
        model.reset()


class TestBaseAPGIRunner:
    """Tests for BaseAPGIRunner class."""

    def test_run_experiment_not_implemented(self):
        """Test that BaseAPGIRunner.run_experiment raises NotImplementedError."""
        runner = BaseAPGIRunner()
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            runner.run_experiment()

    def test_execute_deprecated(self):
        """Test that execute() is deprecated and calls run_experiment."""

        class DummyRunner(BaseAPGIRunner):
            def run_experiment(self) -> Dict[str, Any]:
                return {"result": 42}

        runner = DummyRunner()
        with pytest.warns(DeprecationWarning, match="execute is deprecated"):
            res = runner.execute()
            assert res["result"] == 42

    def test_execute_returns_run_experiment_result(self):
        """Test that execute() returns result from run_experiment()."""

        class TestRunner(BaseAPGIRunner):
            def run_experiment(self) -> Dict[str, Any]:
                return {"test": "data", "number": 123}

        runner = TestRunner()
        with pytest.warns(DeprecationWarning):
            result = runner.execute()
            assert result == {"test": "data", "number": 123}


class TestDeprecatedDecorator:
    """Tests for deprecated decorator."""

    def test_deprecated_emits_warning(self):
        """Test that deprecated decorator emits DeprecationWarning."""

        @deprecated("Use new_function instead")
        def old_function():
            return "old"

        with pytest.warns(DeprecationWarning, match="old_function is deprecated"):
            result = old_function()
            assert result == "old"

    def test_deprecated_includes_reason(self):
        """Test that deprecated decorator includes reason in warning."""

        @deprecated("This function is obsolete")
        def obsolete_function():
            return None

        with pytest.warns(DeprecationWarning) as warning_list:
            obsolete_function()

        assert len(warning_list) == 1
        assert "obsolete" in str(warning_list[0].message)

    def test_deprecated_preserves_function(self):
        """Test that deprecated decorator preserves function behavior."""

        @deprecated("Use something else")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        with pytest.warns(DeprecationWarning):
            result = add(2, 3)
            assert result == 5

        # Check that function metadata is preserved
        assert add.__name__ == "add"
        assert add.__doc__ == "Add two numbers."

    def test_deprecated_with_args(self):
        """Test deprecated decorator with function arguments."""

        @deprecated("Use new_calc instead")
        def calculate(x: int, y: int, operation: str = "add") -> int:
            if operation == "add":
                return x + y
            return x - y

        with pytest.warns(DeprecationWarning):
            result = calculate(5, 3, operation="subtract")
            assert result == 2


class TestProtocolEdgeCases:
    """Edge case tests for protocols."""

    def test_none_not_protocol(self):
        """Test that None is not a valid protocol implementation."""
        assert not isinstance(None, ExperimentRunnerProtocol)

    def test_object_not_protocol(self):
        """Test that plain object is not a valid protocol implementation."""
        assert not isinstance(object(), ExperimentRunnerProtocol)

    def test_empty_class_not_protocol(self):
        """Test that empty class is not a valid protocol implementation."""

        class Empty:
            pass

        assert not isinstance(Empty(), ExperimentRunnerProtocol)

    def test_partial_model_not_protocol(self):
        """Test that partial model implementation is not valid."""

        class PartialModel:
            def process_trial(
                self, observed: float, predicted: float
            ) -> Dict[str, float]:
                return {}

            # Missing reset method

        assert not isinstance(PartialModel(), APGIModelProtocol)

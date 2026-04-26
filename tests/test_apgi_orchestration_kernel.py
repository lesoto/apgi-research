"""
Comprehensive tests for apgi_orchestration_kernel.py module.
Aiming for 100% code coverage.
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from apgi_orchestration_kernel import (
    APGIOrchestrationKernel,
    ExperimentRunConfig,
    TrialMetrics,
    TrialTransformer,
    _kernel,
    get_orchestration_kernel,
    set_orchestration_kernel,
)
from apgi_security_adapters import SecurityLevel


class ConcreteTrialTransformer(TrialTransformer):
    """Concrete implementation for testing."""

    def transform_trial(self, trial_data):
        return {"transformed": True, **trial_data}

    def extract_prediction_error(self, trial_data):
        return trial_data.get("error", 0.0)

    def extract_precision(self, trial_data):
        return trial_data.get("precision", 1.0)


class TestTrialMetrics:
    """Test TrialMetrics dataclass."""

    def test_default_creation(self):
        """Test default creation."""
        metrics = TrialMetrics()

        assert metrics.trial_number == 0
        assert metrics.prediction_error == 0.0
        assert metrics.precision == 1.0
        assert metrics.somatic_marker == 0.0
        assert metrics.ignition_probability == 0.0
        assert metrics.experiment_metrics == {}
        assert metrics.operator_id == ""
        assert metrics.experiment_name == ""
        assert metrics.trial_id is not None
        assert isinstance(metrics.timestamp, float)

    def test_custom_creation(self):
        """Test custom creation."""
        metrics = TrialMetrics(
            trial_number=5,
            prediction_error=0.5,
            precision=2.0,
            somatic_marker=0.1,
            ignition_probability=0.8,
            operator_id="op_001",
            experiment_name="test_exp",
        )

        assert metrics.trial_number == 5
        assert metrics.prediction_error == 0.5
        assert metrics.precision == 2.0
        assert metrics.somatic_marker == 0.1
        assert metrics.ignition_probability == 0.8
        assert metrics.operator_id == "op_001"
        assert metrics.experiment_name == "test_exp"

    def test_trial_id_unique(self):
        """Test that trial IDs are unique."""
        m1 = TrialMetrics()
        m2 = TrialMetrics()

        assert m1.trial_id != m2.trial_id


class TestExperimentRunConfig:
    """Test ExperimentRunConfig dataclass."""

    def test_creation(self):
        """Test creation with required fields."""
        mock_config = Mock()
        mock_config.tau_S = 0.1
        mock_config.beta = 0.5
        mock_config.theta_0 = 1.0
        mock_config.alpha = 0.3

        config = ExperimentRunConfig(
            experiment_name="test_exp",
            operator_id="op_001",
            operator_name="Test Operator",
            apgi_config=mock_config,
        )

        assert config.experiment_name == "test_exp"
        assert config.operator_id == "op_001"
        assert config.operator_name == "Test Operator"
        assert config.apgi_config == mock_config
        assert config.timeout_seconds == 600
        assert config.enable_hierarchical is True
        assert config.enable_precision_gap is True
        assert config.security_level == "standard"
        assert config.trial_callback is None
        assert config.completion_callback is None

    def test_custom_creation(self):
        """Test creation with custom values."""
        mock_config = Mock()
        callback = Mock()

        config = ExperimentRunConfig(
            experiment_name="test_exp",
            operator_id="op_001",
            operator_name="Test",
            apgi_config=mock_config,
            timeout_seconds=300,
            enable_hierarchical=False,
            enable_precision_gap=False,
            security_level="strict",
            trial_callback=callback,
            completion_callback=callback,
        )

        assert config.timeout_seconds == 300
        assert config.enable_hierarchical is False
        assert config.security_level == "strict"
        assert config.trial_callback == callback


class TestAPGIOrchestrationKernel:
    """Test APGIOrchestrationKernel class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        with patch("apgi_orchestration_kernel.get_logger") as mock_logger:
            with patch("apgi_orchestration_kernel.get_authz_manager") as mock_authz:
                with patch("apgi_orchestration_kernel.get_audit_sink") as mock_audit:
                    with patch(
                        "apgi_orchestration_kernel.get_security_factory"
                    ) as mock_sec:
                        with patch(
                            "apgi_orchestration_kernel.APGIIntegration"
                        ) as mock_apgi:
                            with patch("apgi_orchestration_kernel.APGIContextLogger"):
                                yield {
                                    "logger": mock_logger,
                                    "authz": mock_authz,
                                    "audit": mock_audit,
                                    "security": mock_sec,
                                    "apgi": mock_apgi,
                                }

    @pytest.fixture
    def kernel(self, mock_dependencies):
        """Create kernel with mocked dependencies."""
        kernel = APGIOrchestrationKernel()
        return kernel

    @pytest.fixture
    def run_config(self):
        """Create run configuration."""
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.5
        mock_apgi_config.theta_0 = 1.0
        mock_apgi_config.alpha = 0.3

        return ExperimentRunConfig(
            experiment_name="test_experiment",
            operator_id="op_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=60,
        )

    def test_init(self, kernel):
        """Test initialization."""
        assert kernel.active_runs == {}
        assert kernel.completed_runs == []

    def test_create_run_context(self, kernel, run_config):
        """Test creating run context."""
        context = kernel.create_run_context(run_config)

        assert "run_id" in context
        assert context["config"] == run_config
        assert "security_context" in context
        assert "logger" in context
        assert "apgi" in context
        assert context["status"] == "running"
        assert context["trial_count"] == 0
        assert context["trial_metrics"] == []

        # Check audit was called
        kernel.audit_sink.record_event.assert_called_once()

    def test_create_run_context_security_level(self, kernel, run_config):
        """Test creating run context with different security levels."""
        run_config.security_level = "strict"

        with patch.object(kernel.security_factory, "create_context") as mock_create:
            kernel.create_run_context(run_config)

            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[1]["security_level"] == SecurityLevel.STRICT

    def test_process_trial_success(self, kernel, run_config):
        """Test processing a trial successfully."""
        run_context = kernel.create_run_context(run_config)
        trial_data = {"error": 0.5, "precision": 2.0}
        transformer = ConcreteTrialTransformer()

        # Mock APGI response
        kernel.active_runs[run_context["run_id"]][
            "apgi"
        ].compute_ignition_probability.return_value = 0.75

        result = kernel.process_trial(run_context, trial_data, transformer)

        assert isinstance(result, TrialMetrics)
        assert result.trial_number == 0
        assert result.prediction_error == 0.5
        assert result.precision == 2.0
        assert result.ignition_probability == 0.75

        # Check trial was recorded
        assert len(run_context["trial_metrics"]) == 1
        assert run_context["trial_count"] == 1

    def test_process_trial_with_callback(self, kernel, run_config):
        """Test processing trial with callback."""
        callback = Mock()
        run_config.trial_callback = callback

        run_context = kernel.create_run_context(run_config)
        trial_data = {"error": 0.5, "precision": 2.0}
        transformer = ConcreteTrialTransformer()

        kernel.active_runs[run_context["run_id"]][
            "apgi"
        ].compute_ignition_probability.return_value = 0.75

        kernel.process_trial(run_context, trial_data, transformer)

        callback.assert_called_once()

    def test_process_trial_callback_failure(self, kernel, run_config):
        """Test processing trial with failing callback."""
        callback = Mock(side_effect=Exception("Callback error"))
        run_config.trial_callback = callback

        run_context = kernel.create_run_context(run_config)
        trial_data = {"error": 0.5, "precision": 2.0}
        transformer = ConcreteTrialTransformer()

        kernel.active_runs[run_context["run_id"]][
            "apgi"
        ].compute_ignition_probability.return_value = 0.75

        # Should not raise
        result = kernel.process_trial(run_context, trial_data, transformer)
        assert result is not None

    def test_process_trial_timeout(self, kernel, run_config):
        """Test processing trial that times out."""
        run_context = kernel.create_run_context(run_config)

        # Set start time in the past to trigger timeout
        run_context["start_time"] = time.time() - 100

        trial_data = {"error": 0.5}
        transformer = ConcreteTrialTransformer()

        from apgi_errors import APGITimeoutError

        with pytest.raises(APGITimeoutError) as exc_info:
            kernel.process_trial(run_context, trial_data, transformer)

        assert "exceeded timeout" in str(exc_info.value)

    def test_process_trial_transform_failure(self, kernel, run_config):
        """Test processing trial with transform failure."""
        run_context = kernel.create_run_context(run_config)
        trial_data: dict[str, Any] = {}

        class FailingTransformer(ConcreteTrialTransformer):
            def transform_trial(self, trial_data):
                raise ValueError("Transform error")

        transformer = FailingTransformer()

        with pytest.raises(Exception) as exc_info:
            kernel.process_trial(run_context, trial_data, transformer)

        assert "Failed to transform trial data" in str(exc_info.value)

    def test_process_trial_apgi_failure(self, kernel, run_config):
        """Test processing trial with APGI failure."""
        run_context = kernel.create_run_context(run_config)
        trial_data = {"error": 0.5, "precision": 2.0}
        transformer = ConcreteTrialTransformer()

        # Mock APGI to raise exception
        kernel.active_runs[run_context["run_id"]][
            "apgi"
        ].compute_ignition_probability.side_effect = Exception("APGI error")

        from apgi_errors import APGIRuntimeError

        with pytest.raises(APGIRuntimeError) as exc_info:
            kernel.process_trial(run_context, trial_data, transformer)

        assert "APGI processing failed" in str(exc_info.value)

    def test_finalize_run(self, kernel, run_config):
        """Test finalizing run."""
        run_context = kernel.create_run_context(run_config)

        # Add some trial metrics
        run_context["trial_count"] = 5
        run_context["trial_metrics"] = [TrialMetrics(trial_number=i) for i in range(5)]

        # Mock APGI finalize
        kernel.active_runs[run_context["run_id"]]["apgi"].finalize.return_value = {
            "summary": "test"
        }

        result = kernel.finalize_run(run_context)

        assert result["run_id"] == run_context["run_id"]
        assert result["experiment_name"] == "test_experiment"
        assert result["operator_id"] == "op_001"
        assert result["status"] == "completed"
        assert result["trial_count"] == 5
        assert "elapsed_seconds" in result
        assert "apgi_summary" in result

        # Check run moved to completed
        assert run_context["run_id"] not in kernel.active_runs
        assert len(kernel.completed_runs) == 1

    def test_finalize_run_with_callback(self, kernel, run_config):
        """Test finalizing run with callback."""
        callback = Mock()
        run_config.completion_callback = callback

        run_context = kernel.create_run_context(run_config)
        kernel.active_runs[run_context["run_id"]]["apgi"].finalize.return_value = {}

        kernel.finalize_run(run_context)

        callback.assert_called_once()

    def test_finalize_run_callback_failure(self, kernel, run_config):
        """Test finalizing run with failing callback."""
        callback = Mock(side_effect=Exception("Callback error"))
        run_config.completion_callback = callback

        run_context = kernel.create_run_context(run_config)
        kernel.active_runs[run_context["run_id"]]["apgi"].finalize.return_value = {}

        # Should not raise
        result = kernel.finalize_run(run_context)
        assert result is not None

    def test_get_run_status_running(self, kernel, run_config):
        """Test getting status of running run."""
        run_context = kernel.create_run_context(run_config)

        status = kernel.get_run_status(run_context["run_id"])

        assert status is not None
        assert status["status"] == "running"
        assert status["trial_count"] == 0
        assert "elapsed_seconds" in status

    def test_get_run_status_completed(self, kernel, run_config):
        """Test getting status of completed run."""
        run_context = kernel.create_run_context(run_config)
        run_context["trial_count"] = 10

        kernel.active_runs[run_context["run_id"]]["apgi"].finalize.return_value = {}
        kernel.finalize_run(run_context)

        status = kernel.get_run_status(run_context["run_id"])

        assert status is not None
        assert status["status"] == "completed"
        assert status["trial_count"] == 10

    def test_get_run_status_not_found(self, kernel):
        """Test getting status of non-existent run."""
        status = kernel.get_run_status("nonexistent")

        assert status is None

    def test_get_metrics(self, kernel, run_config):
        """Test getting kernel metrics."""
        # Create and complete a run
        run_context = kernel.create_run_context(run_config)
        run_context["trial_count"] = 5
        kernel.active_runs[run_context["run_id"]]["apgi"].finalize.return_value = {}
        kernel.finalize_run(run_context)

        # Create another active run
        kernel.create_run_context(run_config)

        # Mock security and audit
        kernel.security_factory.get_metrics.return_value = {"security": "test"}
        kernel.audit_sink.events = [Mock(), Mock()]

        metrics = kernel.get_metrics()

        assert metrics["active_runs"] == 1
        assert metrics["completed_runs"] == 1
        assert metrics["total_trials_processed"] == 5
        assert "security_metrics" in metrics
        assert "audit_events" in metrics


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_orchestration_kernel(self):
        """Test getting global orchestration kernel."""
        kernel = get_orchestration_kernel()
        assert isinstance(kernel, APGIOrchestrationKernel)

    def test_set_orchestration_kernel(self):
        """Test setting global orchestration kernel."""
        new_kernel = APGIOrchestrationKernel()
        set_orchestration_kernel(new_kernel)

        assert get_orchestration_kernel() is new_kernel

        # Reset to original
        set_orchestration_kernel(_kernel)


class TestTrialTransformerABC:
    """Test TrialTransformer abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that TrialTransformer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrialTransformer()  # type: ignore

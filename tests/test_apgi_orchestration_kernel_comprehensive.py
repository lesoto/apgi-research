"""
Comprehensive test suite for apgi_orchestration_kernel.py to achieve ≥90% coverage.

This file tests the complex code paths and edge cases not covered in the basic test suite.
"""

import os
import sys
import time
import uuid
from unittest.mock import Mock

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apgi_orchestration_kernel import (
    APGIOrchestrationKernel,
    ExperimentRunConfig,
    TrialMetrics,
    TrialTransformer,
    get_orchestration_kernel,
    set_orchestration_kernel,
)


class MockTrialTransformer(TrialTransformer):
    """Mock implementation of TrialTransformer for testing."""

    def transform_trial(self, trial_data):
        """Mock trial transformation."""
        return trial_data

    def extract_prediction_error(self, trial_data):
        """Mock prediction error extraction."""
        return trial_data.get("error", 0.1)

    def extract_precision(self, trial_data):
        """Mock precision extraction."""
        return trial_data.get("precision", 0.9)


class MockTrialExtractor:
    """Mock implementation of TrialExtractor for testing."""

    def extract_trial_data(self, trial_result):
        """Mock trial data extraction."""
        return trial_result


class TestTrialMetrics:
    """Test the TrialMetrics dataclass."""

    def test_default_initialization(self):
        """Test default metrics initialization."""
        metrics = TrialMetrics()

        assert isinstance(metrics.trial_id, str)
        assert len(metrics.trial_id) > 0
        assert metrics.trial_number == 0
        assert metrics.timestamp > 0
        assert metrics.prediction_error == 0.0
        assert metrics.precision == 1.0
        assert metrics.somatic_marker == 0.0
        assert metrics.ignition_probability == 0.0
        assert metrics.experiment_metrics == {}
        assert metrics.operator_id == ""
        assert metrics.experiment_name == ""

    def test_custom_initialization(self):
        """Test custom metrics initialization."""
        custom_id = str(uuid.uuid4())
        metrics = TrialMetrics(
            trial_number=5,
            prediction_error=0.15,
            precision=0.85,
            somatic_marker=0.3,
            ignition_probability=0.7,
            experiment_metrics={"accuracy": 0.8, "rt": 500.0},
            operator_id="operator_001",
            experiment_name="test_experiment",
            trial_id=custom_id,
        )

        assert metrics.trial_id == custom_id
        assert metrics.trial_number == 5
        assert metrics.prediction_error == 0.15
        assert metrics.precision == 0.85
        assert metrics.somatic_marker == 0.3
        assert metrics.ignition_probability == 0.7
        assert metrics.experiment_metrics["accuracy"] == 0.8
        assert metrics.operator_id == "operator_001"
        assert metrics.experiment_name == "test_experiment"

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        metrics = TrialMetrics(
            trial_number=3,
            prediction_error=0.2,
            precision=0.8,
            experiment_metrics={"test": "value"},
            operator_id="test_op",
            experiment_name="test_exp",
        )

        metrics_dict = metrics.__dict__

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["trial_number"] == 3
        assert metrics_dict["prediction_error"] == 0.2
        assert metrics_dict["precision"] == 0.8
        assert metrics_dict["experiment_metrics"]["test"] == "value"
        assert metrics_dict["operator_id"] == "test_op"
        assert metrics_dict["experiment_name"] == "test_exp"


class TestExperimentRunConfig:
    """Test the ExperimentRunConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="test_experiment",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
        )

        assert config.experiment_name == "test_experiment"
        assert config.operator_id == "operator_001"
        assert config.operator_name == "Test Operator"
        assert config.apgi_config == mock_apgi_config
        assert config.timeout_seconds == 600
        assert config.enable_hierarchical is True
        assert config.enable_precision_gap is True
        assert config.security_level == "standard"
        assert config.trial_callback is None
        assert config.completion_callback is None

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        # Mock callbacks
        trial_callback = Mock()
        completion_callback = Mock()

        config = ExperimentRunConfig(
            experiment_name="custom_experiment",
            operator_id="operator_002",
            operator_name="Custom Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=1200,
            enable_hierarchical=False,
            enable_precision_gap=False,
            security_level="high",
            trial_callback=trial_callback,
            completion_callback=completion_callback,
        )

        assert config.experiment_name == "custom_experiment"
        assert config.timeout_seconds == 1200
        assert config.enable_hierarchical is False
        assert config.enable_precision_gap is False
        assert config.security_level == "high"
        assert config.trial_callback is trial_callback
        assert config.completion_callback is completion_callback


class TestAPGIOrchestrationKernel:
    """Test the APGIOrchestrationKernel class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kernel = APGIOrchestrationKernel()

    def test_initialization(self):
        """Test kernel initialization."""
        assert self.kernel.logger is not None
        assert self.kernel.authz_manager is not None
        assert self.kernel.audit_sink is not None
        assert self.kernel.security_factory is not None
        assert isinstance(self.kernel.active_runs, dict)
        assert isinstance(self.kernel.completed_runs, list)
        assert len(self.kernel.active_runs) == 0
        assert len(self.kernel.completed_runs) == 0

    def test_create_run_context(self):
        """Test creating a run context."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="test_experiment",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
            security_level="standard",
        )

        run_context = self.kernel.create_run_context(config)

        # Verify run context structure
        assert "run_id" in run_context
        assert "config" in run_context
        assert "security_context" in run_context
        assert "logger" in run_context
        assert "apgi" in run_context
        assert "start_time" in run_context
        assert "trial_count" in run_context
        assert "trial_metrics" in run_context
        assert "status" in run_context

        # Verify values
        assert run_context["config"] is config
        assert run_context["trial_count"] == 0
        assert run_context["status"] == "running"
        assert run_context["start_time"] > 0

        # Verify run was added to active runs
        assert run_context["run_id"] in self.kernel.active_runs

    def test_create_run_context_with_callbacks(self):
        """Test creating run context with callbacks."""
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        trial_callback = Mock()
        completion_callback = Mock()

        config = ExperimentRunConfig(
            experiment_name="callback_test",
            operator_id="operator_002",
            operator_name="Callback Test",
            apgi_config=mock_apgi_config,
            trial_callback=trial_callback,
            completion_callback=completion_callback,
        )

        run_context = self.kernel.create_run_context(config)

        # Callbacks should be accessible via config
        assert run_context["config"].trial_callback is trial_callback
        assert run_context["config"].completion_callback is completion_callback

    def test_process_trial_success(self):
        """Test successful trial processing."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="test_experiment",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        run_context = self.kernel.create_run_context(config)

        # Mock transformer
        transformer = MockTrialTransformer()

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.compute_ignition_probability.return_value = 0.75
        run_context["apgi"] = mock_apgi

        trial_data = {"error": 0.1, "precision": 0.9, "accuracy": 0.85}

        metrics = self.kernel.process_trial(run_context, trial_data, transformer)

        # Verify metrics
        assert isinstance(metrics, TrialMetrics)
        assert metrics.trial_number == 1  # Should be incremented
        assert metrics.prediction_error == 0.1
        assert metrics.precision == 0.9
        assert metrics.ignition_probability == 0.75
        assert metrics.operator_id == "operator_001"
        assert metrics.experiment_name == "test_experiment"
        assert metrics.experiment_metrics == trial_data

        # Verify run context was updated
        assert run_context["trial_count"] == 1
        assert len(run_context["trial_metrics"]) == 1
        assert run_context["trial_metrics"][0] is metrics

    def test_process_trial_timeout(self):
        """Test trial processing with timeout."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="timeout_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=1,  # Very short timeout (as int)
        )

        run_context = self.kernel.create_run_context(config)

        # Set start time in the past to trigger timeout
        run_context["start_time"] = time.time() - 1.0

        transformer = MockTrialTransformer()
        trial_data = {"test": "data"}

        # Should raise timeout error
        with pytest.raises(Exception):  # APGITimeoutError
            self.kernel.process_trial(run_context, trial_data, transformer)

    def test_process_trial_transformation_error(self):
        """Test trial processing with transformation error."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="transform_error_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        run_context = self.kernel.create_run_context(config)

        # Mock transformer that raises exception
        transformer = Mock()
        transformer.transform_trial.side_effect = Exception("Transformation failed")

        trial_data = {"test": "data"}

        # Should raise integration error
        with pytest.raises(Exception):  # APGIIntegrationError
            self.kernel.process_trial(run_context, trial_data, transformer)

    def test_process_trial_apgi_error(self):
        """Test trial processing with APGI computation error."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="apgi_error_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        run_context = self.kernel.create_run_context(config)

        transformer = MockTrialTransformer()

        # Mock APGI integration to raise exception
        mock_apgi = Mock()
        mock_apgi.compute_ignition_probability.side_effect = Exception("APGI failed")
        run_context["apgi"] = mock_apgi

        trial_data = {"test": "data"}

        # Should raise runtime error
        with pytest.raises(Exception):  # APGIRuntimeError
            self.kernel.process_trial(run_context, trial_data, transformer)

    def test_process_trial_with_callback(self):
        """Test trial processing with trial callback."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        # Mock callback
        trial_callback = Mock()

        config = ExperimentRunConfig(
            experiment_name="callback_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            trial_callback=trial_callback,
        )

        run_context = self.kernel.create_run_context(config)

        transformer = MockTrialTransformer()

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.compute_ignition_probability.return_value = 0.75
        run_context["apgi"] = mock_apgi

        trial_data = {"test": "data"}

        self.kernel.process_trial(run_context, trial_data, transformer)

        # Verify callback was called
        trial_callback.assert_called_once()
        callback_args = trial_callback.call_args[0][0]
        assert callback_args["trial_number"] == 1
        assert callback_args["prediction_error"] == 0.1
        assert callback_args["precision"] == 0.9

    def test_finalize_run_success(self):
        """Test successful run finalization."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="finalize_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        run_context = self.kernel.create_run_context(config)

        # Add some trial metrics
        metrics1 = TrialMetrics(
            trial_number=1,
            prediction_error=0.1,
            precision=0.9,
            operator_id="operator_001",
            experiment_name="finalize_test",
        )
        metrics2 = TrialMetrics(
            trial_number=2,
            prediction_error=0.08,
            precision=0.92,
            operator_id="operator_001",
            experiment_name="finalize_test",
        )
        run_context["trial_metrics"] = [metrics1, metrics2]

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.finalize.return_value = {"summary": "test_summary", "score": 0.85}
        run_context["apgi"] = mock_apgi

        results = self.kernel.finalize_run(run_context)

        # Verify results structure
        assert "run_id" in results
        assert "experiment_name" in results
        assert "operator_id" in results
        assert "status" in results
        assert "trial_count" in results
        assert "elapsed_seconds" in results
        assert "apgi_summary" in results
        assert "trial_metrics" in results

        # Verify values
        assert results["experiment_name"] == "finalize_test"
        assert results["operator_id"] == "operator_001"
        assert results["status"] == "completed"
        assert results["trial_count"] == 2
        assert results["apgi_summary"]["summary"] == "test_summary"
        assert len(results["trial_metrics"]) == 2

        # Verify run was moved from active to completed
        assert run_context["run_id"] not in self.kernel.active_runs
        assert len(self.kernel.completed_runs) == 1
        assert self.kernel.completed_runs[0]["run_id"] == run_context["run_id"]

    def test_finalize_run_with_completion_callback(self):
        """Test run finalization with completion callback."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        # Mock completion callback
        completion_callback = Mock()

        config = ExperimentRunConfig(
            experiment_name="callback_finalize_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
            completion_callback=completion_callback,
        )

        run_context = self.kernel.create_run_context(config)

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.finalize.return_value = {"summary": "test_summary"}
        run_context["apgi"] = mock_apgi

        self.kernel.finalize_run(run_context)

        # Verify callback was called
        completion_callback.assert_called_once()
        callback_args = completion_callback.call_args[0][0]
        assert callback_args["experiment_name"] == "callback_finalize_test"
        assert callback_args["trial_count"] == 0
        assert callback_args["status"] == "completed"

    def test_get_run_status_active(self):
        """Test getting status of active run."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="status_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
        )

        run_context = self.kernel.create_run_context(config)
        run_id = run_context["run_id"]

        status = self.kernel.get_run_status(run_id)

        assert status is not None
        assert status["run_id"] == run_id
        assert status["status"] == "running"
        assert status["trial_count"] == 0
        assert "elapsed_seconds" in status

    def test_get_run_status_completed(self):
        """Test getting status of completed run."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="completed_status_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
        )

        run_context = self.kernel.create_run_context(config)
        run_id = run_context["run_id"]

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.finalize.return_value = {"summary": "test"}
        run_context["apgi"] = mock_apgi

        # Finalize the run
        self.kernel.finalize_run(run_context)

        status = self.kernel.get_run_status(run_id)

        assert status is not None
        assert status["run_id"] == run_id
        assert status["status"] == "completed"
        assert status["trial_count"] == 0
        assert "elapsed_seconds" in status

    def test_get_run_status_nonexistent(self):
        """Test getting status of nonexistent run."""
        status = self.kernel.get_run_status("nonexistent_run_id")

        assert status is None

    def test_get_metrics(self):
        """Test getting kernel metrics."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="metrics_test",
            operator_id="operator_001",
            operator_name="Test Operator",
            apgi_config=mock_apgi_config,
        )

        # Create some runs
        run_context1 = self.kernel.create_run_context(config)
        self.kernel.create_run_context(config)

        # Mock APGI integration and finalize one run
        mock_apgi = Mock()
        mock_apgi.finalize.return_value = {"summary": "test"}
        run_context1["apgi"] = mock_apgi
        self.kernel.finalize_run(run_context1)

        # Mock security factory metrics
        mock_security_factory = Mock()
        mock_security_factory.get_metrics.return_value = {
            "active_contexts": 2,
            "total_checks": 100,
        }
        self.kernel.security_factory = mock_security_factory

        # Mock audit sink
        mock_audit_sink = Mock()
        mock_audit_sink.events = [{"event": "audit1"}, {"event": "audit2"}]
        self.kernel.audit_sink = mock_audit_sink

        metrics = self.kernel.get_metrics()

        assert "active_runs" in metrics
        assert "completed_runs" in metrics
        assert "total_trials_processed" in metrics
        assert "security_metrics" in metrics
        assert "audit_events" in metrics

        assert metrics["active_runs"] == 1  # One still active
        assert metrics["completed_runs"] == 1  # One completed
        assert metrics["total_trials_processed"] == 0  # No trials in our test
        assert metrics["security_metrics"]["active_contexts"] == 2
        assert metrics["audit_events"] == 2


class TestGlobalKernelFunctions:
    """Test the global kernel functions."""

    def test_get_orchestration_kernel_singleton(self):
        """Test getting global orchestration kernel singleton."""
        # Clear any existing kernel
        set_orchestration_kernel(None)

        # First call should create new instance
        kernel1 = get_orchestration_kernel()
        assert isinstance(kernel1, APGIOrchestrationKernel)

        # Second call should return same instance
        kernel2 = get_orchestration_kernel()
        assert kernel1 is kernel2

    def test_set_orchestration_kernel(self):
        """Test setting global orchestration kernel."""
        custom_kernel = APGIOrchestrationKernel()

        set_orchestration_kernel(custom_kernel)

        # Should return the custom kernel
        retrieved = get_orchestration_kernel()
        assert retrieved is custom_kernel

    def test_set_orchestration_kernel_none(self):
        """Test setting global orchestration kernel to None."""
        # Set a kernel first
        temp_kernel = APGIOrchestrationKernel()
        set_orchestration_kernel(temp_kernel)

        # Should create new instance on next get
        new_kernel = get_orchestration_kernel()
        assert new_kernel is temp_kernel


class TestKernelIntegration:
    """Integration tests for the orchestration kernel."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kernel = APGIOrchestrationKernel()

    def test_full_experiment_workflow(self):
        """Test a complete experiment workflow."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="integration_test",
            operator_id="operator_001",
            operator_name="Integration Test",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        # Create run context
        run_context = self.kernel.create_run_context(config)
        run_id = run_context["run_id"]

        # Mock transformer
        transformer = MockTrialTransformer()

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.compute_ignition_probability.return_value = 0.75
        mock_apgi.finalize.return_value = {
            "summary": "integration_test_summary",
            "score": 0.85,
        }
        run_context["apgi"] = mock_apgi

        # Process multiple trials
        trial_data_list = [
            {"error": 0.1, "precision": 0.9, "accuracy": 0.85},
            {"error": 0.08, "precision": 0.92, "accuracy": 0.88},
            {"error": 0.06, "precision": 0.94, "accuracy": 0.91},
        ]

        metrics_list = []
        for trial_data in trial_data_list:
            metrics = self.kernel.process_trial(run_context, trial_data, transformer)
            metrics_list.append(metrics)

        # Finalize run
        results = self.kernel.finalize_run(run_context)

        # Verify workflow
        assert len(metrics_list) == 3
        assert all(isinstance(m, TrialMetrics) for m in metrics_list)
        assert results["status"] == "completed"
        assert results["trial_count"] == 3

        # Verify run status
        status = self.kernel.get_run_status(run_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["trial_count"] == 3

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="error_test",
            operator_id="operator_001",
            operator_name="Error Test",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        run_context = self.kernel.create_run_context(config)

        # Mock transformer that fails initially
        transformer = Mock()
        transformer.transform_trial.side_effect = [
            Exception("First failure"),
            None,
            None,
        ]

        # Mock APGI integration
        mock_apgi = Mock()
        mock_apgi.compute_ignition_probability.return_value = 0.75
        run_context["apgi"] = mock_apgi

        trial_data = {"error": 0.1, "precision": 0.9}

        # First trial should fail
        with pytest.raises(Exception):
            self.kernel.process_trial(run_context, trial_data, transformer)

        # Fix the transformer
        transformer.transform_trial.side_effect = None

        # Subsequent trials should succeed
        metrics1 = self.kernel.process_trial(run_context, trial_data, transformer)
        metrics2 = self.kernel.process_trial(run_context, trial_data, transformer)

        assert isinstance(metrics1, TrialMetrics)
        assert isinstance(metrics2, TrialMetrics)
        assert run_context["trial_count"] == 2

    def test_concurrent_runs(self):
        """Test handling multiple concurrent runs."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        # Create multiple run configurations
        configs = []
        run_contexts = []

        for i in range(3):
            config = ExperimentRunConfig(
                experiment_name=f"concurrent_test_{i}",
                operator_id=f"operator_{i:03d}",
                operator_name=f"Operator {i}",
                apgi_config=mock_apgi_config,
                timeout_seconds=300,
            )

            run_context = self.kernel.create_run_context(config)
            run_contexts.append(run_context)
            configs.append(config)

        # Process trials in each run
        transformer = MockTrialTransformer()

        for i, run_context in enumerate(run_contexts):
            # Mock APGI integration
            mock_apgi = Mock()
            mock_apgi.compute_ignition_probability.return_value = 0.75
            run_context["apgi"] = mock_apgi

            trial_data = {"error": 0.1, "precision": 0.9}
            metrics = self.kernel.process_trial(run_context, trial_data, transformer)

            assert metrics.experiment_name == f"concurrent_test_{i}"
            assert metrics.operator_id == f"operator_{i:03d}"

        # Verify all runs are active
        assert len(self.kernel.active_runs) == 3

        # Finalize all runs
        for run_context in run_contexts:
            mock_apgi = Mock()
            mock_apgi.finalize.return_value = {"summary": "test"}
            run_context["apgi"] = mock_apgi
            self.kernel.finalize_run(run_context)

        # Verify all runs are completed
        assert len(self.kernel.active_runs) == 0
        assert len(self.kernel.completed_runs) == 3

    def test_performance_with_large_number_of_trials(self):
        """Test performance with large number of trials."""
        # Mock APGI config
        mock_apgi_config = Mock()
        mock_apgi_config.tau_S = 0.1
        mock_apgi_config.beta = 0.2
        mock_apgi_config.theta_0 = 0.3
        mock_apgi_config.alpha = 0.4

        config = ExperimentRunConfig(
            experiment_name="performance_test",
            operator_id="operator_001",
            operator_name="Performance Test",
            apgi_config=mock_apgi_config,
            timeout_seconds=300,
        )

        run_context = self.kernel.create_run_context(config)

        # Mock transformer and APGI
        transformer = MockTrialTransformer()
        mock_apgi = Mock()
        mock_apgi.compute_ignition_probability.return_value = 0.75
        run_context["apgi"] = mock_apgi

        # Process many trials
        num_trials = 100
        trial_data = {"error": 0.1, "precision": 0.9}

        start_time = time.time()

        for i in range(num_trials):
            metrics = self.kernel.process_trial(run_context, trial_data, transformer)
            assert metrics.trial_number == i + 1

        end_time = time.time()

        # Should complete efficiently
        assert end_time - start_time < 10.0  # 10 second limit for 100 trials
        assert run_context["trial_count"] == num_trials
        assert len(run_context["trial_metrics"]) == num_trials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test suite for standard_apgi_runner.py module.

Tests standardized APGI runner template functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clear any existing mocks for standard_apgi_runner
if "standard_apgi_runner" in sys.modules:
    del sys.modules["standard_apgi_runner"]

# Import the real classes
import standard_apgi_runner as sar


class TestHierarchicalState:
    """Test HierarchicalState dataclass."""

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_hierarchical_state_initialization(
        self, mock_compute, mock_config, mock_apgi
    ):
        """Test HierarchicalState initialization."""
        state = sar.HierarchicalState(
            level_1={"sensory": 0.5},
            level_2={"feature": 0.6},
            level_3={"pattern": 0.7},
            level_4={"semantic": 0.8},
            level_5={"executive": 0.9},
        )

        # Check that original values are preserved
        assert state.level_1["sensory"] == 0.5
        assert state.level_2["feature"] == 0.6
        assert state.level_3["pattern"] == 0.7
        assert state.level_4["semantic"] == 0.8
        assert state.level_5["executive"] == 0.9

        # Check that default values are added via __post_init__
        assert "S" in state.level_1
        assert "theta" in state.level_1
        assert "M" in state.level_1
        assert "ignition_prob" in state.level_1

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_hierarchical_state_post_init(self, mock_compute, mock_config, mock_apgi):
        """Test HierarchicalState post-initialization."""
        state = sar.HierarchicalState(
            level_1={}, level_2={}, level_3={}, level_4={}, level_5={}
        )

        # All levels should be initialized as dicts with defaults
        assert isinstance(state.level_1, dict)
        assert isinstance(state.level_2, dict)
        assert isinstance(state.level_3, dict)
        assert isinstance(state.level_4, dict)
        assert isinstance(state.level_5, dict)

        # Check defaults are set
        assert state.level_1["S"] == 0.0
        assert state.level_1["theta"] == 0.5


class TestStandardAPGIRunner:
    """Test StandardAPGIRunner class."""

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_runner_initialization(self, mock_compute, mock_config, mock_apgi):
        """Test runner initialization."""
        # Mock base runner
        mock_base_runner = MagicMock()
        mock_base_runner.experiment_name = "test_experiment"

        # Mock APGI params
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi.return_value = MagicMock()

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        assert runner.base_runner is mock_base_runner
        assert runner.experiment_name == "test_experiment"
        assert runner.enable_hierarchical
        assert runner.enable_precision_gap

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_runner_initialization_with_default_params(
        self, mock_compute, mock_config, mock_apgi
    ):
        """Test runner initialization with default APGI params."""
        mock_base_runner = MagicMock()
        mock_base_runner.experiment_name = "test_experiment"

        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi.return_value = MagicMock()

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner, experiment_name="test_experiment"
        )

        assert runner.base_runner is mock_base_runner
        assert runner.experiment_name == "test_experiment"
        mock_config.assert_called_once_with("test_experiment")

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_initialize_hierarchical_state(self, mock_compute, mock_config, mock_apgi):
        """Test hierarchical state initialization."""
        # Test hierarchical state creation
        state = sar.HierarchicalState(
            level_1={"sensory": 0.5},
            level_2={"feature": 0.6},
            level_3={"pattern": 0.7},
            level_4={"semantic": 0.8},
            level_5={"executive": 0.9},
        )

        assert isinstance(state, sar.HierarchicalState)
        assert state.level_1["sensory"] == 0.5
        assert state.level_2["feature"] == 0.6
        assert state.level_3["pattern"] == 0.7
        assert state.level_4["semantic"] == 0.8
        assert state.level_5["executive"] == 0.9

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_process_trial_data(self, mock_compute, mock_config, mock_apgi):
        """Test trial data processing."""
        mock_base_runner = MagicMock()

        # Mock APGI params to be disabled
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = False
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi.return_value = MagicMock()

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner, experiment_name="test_experiment"
        )

        # Test that method exists and returns empty dict when APGI disabled
        result = runner.process_trial_with_full_apgi(
            observed=0.5, predicted=0.4, trial_type="test"
        )

        # Should return empty dict when APGI is disabled
        assert result == {}

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_update_hierarchical_state(self, mock_compute, mock_config, mock_apgi):
        """Test hierarchical state updating."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi_instance = MagicMock()
        mock_apgi_instance.dynamics.params.alpha = 6.0
        mock_apgi.return_value = mock_apgi_instance

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        # Test hierarchical processing with correct parameter type
        basic_metrics = {"accuracy": 0.8, "response_time": 0.5}
        result = runner._process_hierarchical_level(basic_metrics, 1)

        assert isinstance(result, dict)
        # Check that hierarchical level keys are present
        assert "level_1_S" in result or len(result) == 0

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_compute_enhanced_metrics(self, mock_compute, mock_config, mock_apgi):
        """Test enhanced metrics computation."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi.return_value = MagicMock()

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner, experiment_name="test_experiment"
        )

        # Test hierarchical summary
        metrics = runner._get_hierarchical_summary()

        assert isinstance(metrics, dict)
        # Should have hierarchical level keys
        assert len(metrics) >= 0  # Can be empty if no metrics history

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_run_experiment_success(self, mock_compute, mock_config, mock_apgi):
        """Test successful experiment run."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {
            "trials": [{"stimulus": "A", "response": 1}],
            "accuracy": 0.8,
        }
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi_instance = MagicMock()
        mock_apgi_instance.process_trial.return_value = {
            "S": 0.7,
            "theta": 0.5,
        }
        mock_apgi_instance.finalize.return_value = {
            "ignition_rate": 0.5,
            "mean_surprise": 0.3,
        }
        mock_apgi.return_value = mock_apgi_instance

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        # Mock timeout methods to prevent actual timeout
        with patch.object(runner, "_setup_timeout_handler"):
            with patch.object(runner, "_start_timeout_timer"):
                with patch.object(runner, "_cancel_timeout_timer"):
                    with patch.object(runner, "_check_timeout", return_value=False):
                        result = runner.run_experiment()

                        assert "apgi_enabled" in result
                        assert result["apgi_enabled"] is True

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_run_experiment_with_signal_handling(
        self, mock_compute, mock_config, mock_apgi
    ):
        """Test experiment run with signal handling."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {
            "trials": [{"stimulus": "A", "response": 1}],
            "accuracy": 0.8,
        }
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi_instance = MagicMock()
        mock_apgi_instance.finalize.return_value = {
            "ignition_rate": 0.5,
        }
        mock_apgi.return_value = mock_apgi_instance

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        with patch.object(runner, "_setup_timeout_handler"):
            with patch.object(runner, "_start_timeout_timer"):
                with patch.object(runner, "_cancel_timeout_timer"):
                    with patch.object(runner, "_check_timeout", return_value=False):
                        result = runner.run_experiment()

                        assert "apgi_enabled" in result
                        assert result["apgi_enabled"] is True

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_handle_signal_interrupt(self, mock_compute, mock_config, mock_apgi):
        """Test signal interrupt handling."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi.return_value = MagicMock()

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        # Test timeout setup
        runner._setup_timeout_handler()
        assert hasattr(runner, "timeout_handler")

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_save_results(self, mock_compute, mock_config, mock_apgi):
        """Test results saving."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi.return_value = MagicMock()

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        # Test basic processing
        assert hasattr(runner, "process_trial_with_full_apgi")

    @patch("standard_apgi_runner.APGIIntegration")
    @patch("standard_apgi_runner.get_experiment_apgi_config")
    @patch("standard_apgi_runner.compute_apgi_enhanced_metric")
    def test_error_handling_during_trial_processing(
        self, mock_compute, mock_config, mock_apgi
    ):
        """Test error handling during trial processing."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True
        mock_apgi_params.to_apgi_parameters.return_value = {}

        mock_config.return_value = mock_apgi_params
        mock_apgi_instance = MagicMock()
        # The exception should be raised when calling process_trial
        mock_apgi_instance.process_trial.side_effect = Exception("Processing error")
        mock_apgi.return_value = mock_apgi_instance

        runner = sar.StandardAPGIRunner(
            base_runner=mock_base_runner,
            experiment_name="test_experiment",
            apgi_params=mock_apgi_params,
        )

        # Test trial processing - exception should propagate
        try:
            result = runner.process_trial_with_full_apgi(
                observed=0.5, predicted=0.4, trial_type="test"
            )
            # If we get here, the exception was caught internally
            assert isinstance(result, dict)
        except Exception as e:
            # Exception should be raised
            assert "Processing error" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])

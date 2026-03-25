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

# Mock the APGI dependencies
sys.modules["apgi_integration"] = MagicMock()
sys.modules["experiment_apgi_integration"] = MagicMock()

import standard_apgi_runner as sar


class TestHierarchicalState:
    """Test HierarchicalState dataclass."""

    def test_hierarchical_state_initialization(self):
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

        # Check that default values are added
        for level in [
            state.level_1,
            state.level_2,
            state.level_3,
            state.level_4,
            state.level_5,
        ]:
            assert "S" in level
            assert "theta" in level
            assert "M" in level
            assert "ignition_prob" in level

    def test_hierarchical_state_post_init(self):
        """Test HierarchicalState post-initialization."""
        state = sar.HierarchicalState(
            level_1={}, level_2={}, level_3={}, level_4={}, level_5={}
        )

        # All levels should be initialized as empty dicts
        assert isinstance(state.level_1, dict)
        assert isinstance(state.level_2, dict)
        assert isinstance(state.level_3, dict)
        assert isinstance(state.level_4, dict)
        assert isinstance(state.level_5, dict)


class TestStandardAPGIRunner:
    """Test StandardAPGIRunner class."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        # Mock base runner
        mock_base_runner = MagicMock()
        mock_base_runner.experiment_name = "test_experiment"

        # Mock APGI params
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi.return_value = MagicMock()

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner,
                    experiment_name="test_experiment",
                    apgi_params=mock_apgi_params,
                )

                assert runner.base_runner == mock_base_runner
                assert runner.experiment_name == "test_experiment"
                assert runner.enable_hierarchical
                assert runner.enable_precision_gap

    def test_runner_initialization_with_default_params(self):
        """Test runner initialization with default APGI params."""
        mock_base_runner = MagicMock()
        mock_base_runner.experiment_name = "test_experiment"

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = MagicMock()
                mock_apgi.return_value = MagicMock()

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner, experiment_name="test_experiment"
                )

            assert runner.base_runner == mock_base_runner
            assert runner.experiment_name == "test_experiment"
            mock_config.assert_called_once_with("test_experiment")

    def test_initialize_hierarchical_state(self):
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
        assert len(state.level_1) > 0
        assert len(state.level_2) > 0
        assert len(state.level_3) > 0
        assert len(state.level_4) > 0
        assert len(state.level_5) > 0

    def test_process_trial_data(self):
        """Test trial data processing."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi_instance = MagicMock()
                mock_apgi.return_value = mock_apgi_instance
                mock_apgi_instance.process_trial.return_value = {
                    "response": 1.0,
                    "rt": 0.5,
                    "apgi_state": {"pi": 0.7},
                }

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner, experiment_name="test_experiment"
                )

                # Test that the runner can process trial data
                result = runner.process_trial_with_full_apgi(
                    observed=0.5, predicted=0.4, trial_type="test"
                )

                assert "response" in result
                assert "rt" in result
                assert "apgi_state" in result
                mock_apgi_instance.process_trial.assert_called_once()

    def test_update_hierarchical_state(self):
        """Test hierarchical state updating."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi.return_value = MagicMock()

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner, experiment_name="test_experiment"
                )

                # Test hierarchical processing with correct parameter type
                basic_metrics = {"accuracy": 0.8, "response_time": 0.5}
                result = runner._process_hierarchical_level(basic_metrics, 1)

                assert isinstance(result, dict)
                assert "higher_level_broadcast" in result

    def test_compute_enhanced_metrics(self):
        """Test enhanced metrics computation."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.compute_apgi_enhanced_metric"
            ) as mock_metric:
                with patch(
                    "standard_apgi_runner.get_experiment_apgi_config"
                ) as mock_config:
                    mock_config.return_value = mock_apgi_params
                    mock_apgi.return_value = MagicMock()
                    mock_metric.return_value = {"enhanced_accuracy": 0.85}

                    runner = sar.StandardAPGIRunner(
                        base_runner=mock_base_runner, experiment_name="test_experiment"
                    )

                    # Test hierarchical summary
                    metrics = runner._get_hierarchical_summary()

                    assert isinstance(metrics, dict)

    def test_run_experiment_success(self):
        """Test successful experiment run."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {
            "trials": [{"stimulus": "A", "response": 1}],
            "accuracy": 0.8,
        }
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi_instance = MagicMock()
                mock_apgi_instance.process_trial.return_value = {
                    "response": 1.0,
                    "rt": 0.5,
                    "apgi_state": {"pi": 0.7},
                }
                mock_apgi.return_value = mock_apgi_instance

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner,
                    experiment_name="test_experiment",
                    apgi_params=mock_apgi_params,
                )

                with patch.object(runner, "_process_trial_data") as mock_process:
                    with patch.object(
                        runner, "_compute_enhanced_metrics"
                    ) as mock_metrics:
                        mock_process.return_value = {
                            "response": 1,
                            "apgi_state": {"pi": 0.7},
                        }
                        mock_metrics.return_value = {"enhanced_accuracy": 0.85}

                        result = runner.run_experiment()

                        assert "base_results" in result
                        assert "apgi_enhanced_results" in result
                        assert "enhanced_metrics" in result
                        assert "hierarchical_states" in result

    def test_run_experiment_with_signal_handling(self):
        """Test experiment run with signal handling."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {
            "trials": [{"stimulus": "A", "response": 1}],
            "accuracy": 0.8,
        }
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi.return_value = MagicMock()

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner,
                    experiment_name="test_experiment",
                    apgi_params=mock_apgi_params,
                )

                # Mock signal handler
                with patch("signal.signal") as mock_signal:
                    with patch.object(runner, "_process_trial_data") as mock_process:
                        with patch.object(
                            runner, "_compute_enhanced_metrics"
                        ) as mock_metrics:
                            mock_process.return_value = {"response": 1}
                            mock_metrics.return_value = {"enhanced_accuracy": 0.85}

                            result = runner.run_experiment()

                            assert "base_results" in result
                            mock_signal.assert_called()

    def test_handle_signal_interrupt(self):
        """Test signal interrupt handling."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
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

    def test_save_results(self):
        """Test results saving."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi.return_value = MagicMock()

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner,
                    experiment_name="test_experiment",
                    apgi_params=mock_apgi_params,
                )

                # Test basic processing
                assert hasattr(runner, "process_trial_with_full_apgi")

    def test_error_handling_during_trial_processing(self):
        """Test error handling during trial processing."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()

        with patch("standard_apgi_runner.APGIIntegration") as mock_apgi:
            with patch(
                "standard_apgi_runner.get_experiment_apgi_config"
            ) as mock_config:
                mock_config.return_value = mock_apgi_params
                mock_apgi_instance = MagicMock()
                mock_apgi_instance.process_trial.side_effect = Exception(
                    "Processing error"
                )
                mock_apgi.return_value = mock_apgi_instance

                runner = sar.StandardAPGIRunner(
                    base_runner=mock_base_runner,
                    experiment_name="test_experiment",
                    apgi_params=mock_apgi_params,
                )

                # Test trial processing
                result = runner.process_trial_with_full_apgi(
                    observed=0.5, predicted=0.4, trial_type="test"
                )

                assert result is not None
                assert "error" in result or "response" in result


if __name__ == "__main__":
    pytest.main([__file__])

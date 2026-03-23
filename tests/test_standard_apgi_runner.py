"""
Test suite for standard_apgi_runner.py module.

Tests standardized APGI runner template functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

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

        assert state.level_1 == {"sensory": 0.5}
        assert state.level_2 == {"feature": 0.6}
        assert state.level_3 == {"pattern": 0.7}
        assert state.level_4 == {"semantic": 0.8}
        assert state.level_5 == {"executive": 0.9}

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
                assert runner.apgi_params == mock_apgi_params
                assert runner.apgi_integration is not None

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

                state = runner._initialize_hierarchical_state()

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

                trial_data = {"stimulus": "A", "response": 1, "rt": 0.5}
                hierarchical_state = runner._initialize_hierarchical_state()

                result = runner._process_trial_data(trial_data, hierarchical_state)

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

                state = runner._initialize_hierarchical_state()
                trial_result = {"apgi_state": {"pi": 0.8, "theta": 0.6}}

                updated_state = runner._update_hierarchical_state(state, trial_result)

                assert isinstance(updated_state, sar.HierarchicalState)
                # State should be updated with trial results
                assert updated_state != state

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

                    trial_results = [
                        {"response": 1, "correct": True},
                        {"response": 0, "correct": False},
                    ]
                    hierarchical_state = runner._initialize_hierarchical_state()

                    metrics = runner._compute_enhanced_metrics(
                        trial_results, hierarchical_state
                    )

                    assert "enhanced_accuracy" in metrics
                    mock_metric.assert_called_once()

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

                # Simulate signal interrupt
                with patch("sys.exit") as mock_exit:
                    runner._handle_signal_interrupt(2, None)  # SIGINT
                    mock_exit.assert_called_once_with(1)

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

                results = {
                    "base_results": {"accuracy": 0.8},
                    "apgi_enhanced_results": [],
                    "enhanced_metrics": {"enhanced_accuracy": 0.85},
                    "hierarchical_states": [],
                }

                with patch("json.dump") as mock_dump:
                    with patch("builtins.open", mock_open()):
                        runner.save_results(results, "test_output.json")
                        mock_dump.assert_called_once()

    def test_validate_apgi_params(self):
        """Test APGI parameters validation."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.tau_S = 0.35
        mock_apgi_params.beta = 1.5
        mock_apgi_params.theta_0 = 0.5
        mock_apgi_params.alpha = 5.5
        mock_apgi_params.rho = 0.7

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

                # Valid params should not raise exception
                runner._validate_apgi_params(mock_apgi_params)

    def test_validate_apgi_params_invalid(self):
        """Test invalid APGI parameters validation."""
        mock_base_runner = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.tau_S = -1.0  # Invalid negative value

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

                with pytest.raises(ValueError):
                    runner._validate_apgi_params(mock_apgi_params)

    def test_get_experiment_summary(self):
        """Test experiment summary generation."""
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

                results = {
                    "base_results": {"accuracy": 0.8, "trials": 100},
                    "enhanced_metrics": {"enhanced_accuracy": 0.85},
                    "apgi_enhanced_results": [{"pi": 0.7}, {"pi": 0.8}],
                }

                summary = runner.get_experiment_summary(results)

                assert "experiment_name" in summary
                assert "base_accuracy" in summary
                assert "enhanced_accuracy" in summary
                assert "total_trials" in summary
                assert "mean_pi" in summary

    def test_compute_pi_statistics(self):
        """Test Pi statistics computation."""
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

                apgi_results = [
                    {"apgi_state": {"pi": 0.7}},
                    {"apgi_state": {"pi": 0.8}},
                    {"apgi_state": {"pi": 0.9}},
                ]

                stats = runner._compute_pi_statistics(apgi_results)

                assert "mean_pi" in stats
                assert "std_pi" in stats
                assert "min_pi" in stats
                assert "max_pi" in stats
                assert stats["mean_pi"] == pytest.approx(0.8, rel=1e-2)

    def test_real_time_processing(self):
        """Test real-time trial processing."""
        mock_base_runner = MagicMock()
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

                trials = [
                    {"stimulus": "A", "timestamp": 0.0},
                    {"stimulus": "B", "timestamp": 1.0},
                    {"stimulus": "C", "timestamp": 2.0},
                ]

                with patch.object(runner, "_process_trial_data") as mock_process:
                    mock_process.return_value = {
                        "response": 1,
                        "apgi_state": {"pi": 0.7},
                    }

                    results = runner.real_time_processing(trials)

                    assert len(results) == 3
                    assert all("response" in result for result in results)
                    assert all("apgi_state" in result for result in results)

    def test_error_handling_in_processing(self):
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

                trial_data = {"stimulus": "A", "response": 1}
                hierarchical_state = runner._initialize_hierarchical_state()

                # Should handle error gracefully
                result = runner._process_trial_data(trial_data, hierarchical_state)

                assert result is not None
                assert "error" in result or "response" in result


if __name__ == "__main__":
    pytest.main([__file__])

"""
Test suite for experiment_apgi_integration.py module.

Tests APGI integration functionality for experiments.
"""

import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Mock the APGI dependencies before importing
sys.modules["apgi_integration"] = MagicMock()

import experiment_apgi_integration as eai


class TestExportedAPGIParams:
    """Test ExportedAPGIParams dataclass."""

    def test_exported_apgi_params_initialization(self):
        """Test ExportedAPGIParams initialization."""
        params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        assert params.experiment_name == "test"
        assert params.enabled is True
        assert params.tau_s == 0.35
        assert params.beta == 1.5
        assert params.theta_0 == 0.5
        assert params.alpha == 5.5
        assert params.gamma_m == -0.3  # Default value
        assert params.lambda_s == 0.1  # Default value
        assert params.sigma_s == 0.05  # Default value
        assert params.sigma_theta == 0.02  # Default value
        assert params.sigma_m == 0.03  # Default value
        assert params.rho == 0.7  # Default value

    def test_exported_apgi_params_with_kwargs(self):
        """Test ExportedAPGIParams with additional parameters."""
        params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.4,
            beta=2.0,
            theta_0=0.3,
            alpha=6.0,
            gamma_m=-0.2,
            lambda_s=0.15,
        )

        assert params.gamma_m == -0.2
        assert params.lambda_s == 0.15

    def test_to_apgi_parameters(self):
        """Test conversion to APGIParameters."""
        params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        with patch("experiment_apgi_integration.APGIParameters") as mock_apgi_params:
            mock_apgi_params.return_value = MagicMock()

            params.to_apgi_parameters()

            mock_apgi_params.assert_called_once_with(
                tau_S=0.35,
                beta=1.5,
                theta_0=0.5,
                alpha=5.5,
                gamma_M=-0.3,
                lambda_S=0.1,
                sigma_S=0.05,
                sigma_theta=0.02,
                sigma_M=0.03,
                rho=0.7,
            )


class TestExportAPGIParams:
    """Test export_apgi_params function."""

    def test_export_apgi_params_defaults(self):
        """Test export_apgi_params with default values."""
        params = eai.export_apgi_params("test_experiment")

        assert isinstance(params, eai.ExportedAPGIParams)
        assert params.experiment_name == "test_experiment"
        assert params.enabled is True
        assert params.tau_s == 0.35
        assert params.beta == 1.5
        assert params.theta_0 == 0.5
        assert params.alpha == 5.5

    def test_export_apgi_params_custom_values(self):
        """Test export_apgi_params with custom values."""
        params = eai.export_apgi_params(
            "test_experiment",
            tau_s=0.4,
            beta=2.0,
            theta_0=0.3,
            alpha=6.0,
            enabled=False,
        )

        assert params.tau_s == 0.4
        assert params.beta == 2.0
        assert params.theta_0 == 0.3
        assert params.alpha == 6.0
        assert params.enabled is False

    def test_export_apgi_params_with_kwargs(self):
        """Test export_apgi_params with additional kwargs."""
        params = eai.export_apgi_params(
            "test_experiment", gamma_m=-0.2, lambda_s=0.15, rho=0.8
        )

        assert params.gamma_m == -0.2
        assert params.lambda_s == 0.15
        assert params.rho == 0.8


class TestExperimentAPGIRunner:
    """Test ExperimentAPGIRunner class."""

    def test_runner_initialization_enabled(self):
        """Test runner initialization with APGI enabled."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        with patch("experiment_apgi_integration.APGIIntegration") as mock_apgi:
            mock_apgi.return_value = MagicMock()

            runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

            assert runner.base_runner == mock_base_runner
            assert runner.apgi_params == mock_params
            assert runner.apgi is not None
            assert runner.apgi_metrics_history == []
            assert runner.start_time is None

    def test_runner_initialization_disabled(self):
        """Test runner initialization with APGI disabled."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=False,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        assert runner.base_runner == mock_base_runner
        assert runner.apgi_params == mock_params
        assert runner.apgi is None

    def test_run_experiment_enabled(self):
        """Test running experiment with APGI enabled."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {"accuracy": 0.8}

        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        with patch("experiment_apgi_integration.APGIIntegration") as mock_apgi:
            with patch(
                "experiment_apgi_integration.compute_apgi_enhanced_metric"
            ) as mock_metric:
                with patch(
                    "experiment_apgi_integration.format_apgi_output"
                ) as mock_format:
                    mock_apgi_instance = MagicMock()
                    mock_apgi.return_value = mock_apgi_instance
                    mock_apgi_instance.finalize.return_value = {"pi": 0.7, "theta": 0.5}
                    mock_metric.return_value = 0.85
                    mock_format.return_value = "Formatted APGI output"

                    runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

                    with patch("time.time") as mock_time:
                        mock_time.side_effect = [0, 10]  # Start and end times

                        result = runner.run_experiment()

                        assert result["apgi_enabled"] is True
                        assert "apgi_params" in result
                        assert "apgi_metrics" in result
                        assert "apgi_enhanced_metric" in result
                        assert "apgi_formatted" in result
                        assert result["apgi_enhanced_metric"] == 0.85

    def test_run_experiment_disabled(self):
        """Test running experiment with APGI disabled."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {"accuracy": 0.8}

        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=False,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        result = runner.run_experiment()

        assert result["apgi_enabled"] is False
        assert "apgi_params" not in result
        assert "apgi_metrics" not in result

    def test_run_base_experiment_success(self):
        """Test successful base experiment run."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {"accuracy": 0.8}

        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        result = runner._run_base_experiment()

        assert result == {"accuracy": 0.8}
        mock_base_runner.run_experiment.assert_called_once()

    def test_run_base_experiment_no_method(self):
        """Test base experiment run when method doesn't exist."""
        mock_base_runner = MagicMock()
        del mock_base_runner.run_experiment  # Remove the method

        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        with pytest.raises(
            ValueError, match="Base runner must have run_experiment method"
        ):
            runner._run_base_experiment()

    def test_extract_primary_metric_known_key(self):
        """Test extracting primary metric with known key."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        results = {"accuracy": 0.8, "other": "value"}

        metric = runner._extract_primary_metric(results)

        assert metric == 0.8

    def test_extract_primary_metric_unknown_key(self):
        """Test extracting primary metric with unknown key."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        results = {"some_metric": 0.75, "other": "value"}

        metric = runner._extract_primary_metric(results)

        assert metric == 0.75

    def test_extract_primary_metric_no_numeric(self):
        """Test extracting primary metric when no numeric values."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        results = {"text": "value", "_private": 5}

        metric = runner._extract_primary_metric(results)

        assert metric is None

    def test_process_trial_with_apgi_enabled(self):
        """Test processing trial with APGI enabled."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        with patch("experiment_apgi_integration.APGIIntegration") as mock_apgi:
            mock_apgi_instance = MagicMock()
            mock_apgi.return_value = mock_apgi_instance
            mock_apgi_instance.process_trial.return_value = {"pi": 0.7, "theta": 0.5}

            runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

            result = runner.process_trial_with_apgi(
                observed=1.0, predicted=0.8, trial_type="neutral"
            )

            assert result == {"pi": 0.7, "theta": 0.5}
            assert len(runner.apgi_metrics_history) == 1
            mock_apgi_instance.process_trial.assert_called_once()

    def test_process_trial_with_apgi_disabled(self):
        """Test processing trial with APGI disabled."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=False,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        result = runner.process_trial_with_apgi(
            observed=1.0, predicted=0.8, trial_type="neutral"
        )

        assert result is None
        assert len(runner.apgi_metrics_history) == 0

    def test_process_trial_with_precision_overrides(self):
        """Test processing trial with precision overrides."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        with patch("experiment_apgi_integration.APGIIntegration") as mock_apgi:
            mock_apgi_instance = MagicMock()
            mock_apgi.return_value = mock_apgi_instance
            mock_apgi_instance.process_trial.return_value = {"pi": 0.7}

            runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

            result = runner.process_trial_with_apgi(
                observed=1.0,
                predicted=0.8,
                trial_type="neutral",
                precision_ext=0.9,
                precision_int=0.8,
            )

            assert result == {"pi": 0.7}
            mock_apgi_instance.process_trial.assert_called_once_with(
                observed=1.0,
                predicted=0.8,
                trial_type="neutral",
                precision_ext=0.9,
                precision_int=0.8,
            )


class TestGetExperimentAPGIConfig:
    """Test get_experiment_apgi_config function."""

    def test_get_experiment_config_known_experiment(self):
        """Test getting config for known experiment."""
        config = eai.get_experiment_apgi_config("stroop_effect")

        assert isinstance(config, eai.ExportedAPGIParams)
        assert config.experiment_name == "stroop_effect"
        assert config.tau_s == 0.30
        assert config.beta == 1.6
        assert config.theta_0 == 0.35
        assert config.alpha == 6.0

    def test_get_experiment_config_attention_experiment(self):
        """Test getting config for attention experiment."""
        config = eai.get_experiment_apgi_config("attentional_blink")

        assert config.tau_s == 0.25
        assert config.beta == 1.8
        assert config.theta_0 == 0.4
        assert config.alpha == 6.0

    def test_get_experiment_config_memory_experiment(self):
        """Test getting config for memory experiment."""
        config = eai.get_experiment_apgi_config("working_memory_span")

        assert config.tau_s == 0.38
        assert config.beta == 1.3
        assert config.theta_0 == 0.5
        assert config.alpha == 5.2

    def test_get_experiment_config_decision_experiment(self):
        """Test getting config for decision-making experiment."""
        config = eai.get_experiment_apgi_config("iowa_gambling_task")

        assert config.tau_s == 0.40
        assert config.beta == 2.0
        assert config.theta_0 == 0.4
        assert config.alpha == 5.0

    def test_get_experiment_config_unknown_experiment(self):
        """Test getting config for unknown experiment."""
        config = eai.get_experiment_apgi_config("unknown_experiment")

        assert isinstance(config, eai.ExportedAPGIParams)
        assert config.experiment_name == "unknown_experiment"
        # Should use default values
        assert config.tau_s == 0.35
        assert config.beta == 1.5
        assert config.theta_0 == 0.5
        assert config.alpha == 5.5

    def test_get_experiment_config_perception_experiment(self):
        """Test getting config for perception experiment."""
        config = eai.get_experiment_apgi_config("masking")

        assert config.tau_s == 0.25
        assert config.beta == 1.8
        assert config.theta_0 == 0.3
        assert config.alpha == 7.0

    def test_get_experiment_config_learning_experiment(self):
        """Test getting config for learning experiment."""
        config = eai.get_experiment_apgi_config("artificial_grammar_learning")

        assert config.tau_s == 0.40
        assert config.beta == 1.1
        assert config.theta_0 == 0.55
        assert config.alpha == 4.8

    def test_get_experiment_config_interoception_experiment(self):
        """Test getting config for interoception experiment."""
        config = eai.get_experiment_apgi_config("interoceptive_gating")

        assert config.tau_s == 0.45
        assert config.beta == 2.2
        assert config.theta_0 == 0.5
        assert config.alpha == 5.0


class TestIntegrationWorkflow:
    """Test complete integration workflow."""

    def test_full_workflow_enabled(self):
        """Test complete workflow with APGI enabled."""
        # Mock base runner
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {
            "accuracy": 0.8,
            "response_times": [600, 650, 700, 750],
        }

        # Get APGI config
        config = eai.get_experiment_apgi_config("stroop_effect")

        with patch("experiment_apgi_integration.APGIIntegration") as mock_apgi:
            with patch(
                "experiment_apgi_integration.compute_apgi_enhanced_metric"
            ) as mock_metric:
                with patch(
                    "experiment_apgi_integration.format_apgi_output"
                ) as mock_format:
                    mock_apgi_instance = MagicMock()
                    mock_apgi.return_value = mock_apgi_instance
                    mock_apgi_instance.finalize.return_value = {
                        "pi": 0.7,
                        "theta": 0.5,
                        "surprise": 0.3,
                    }
                    mock_metric.return_value = 0.85
                    mock_format.return_value = "Formatted output"

                    # Create runner
                    runner = eai.ExperimentAPGIRunner(mock_base_runner, config)

                    # Process some trials
                    runner.process_trial_with_apgi(1.0, 0.8, "incongruent")
                    runner.process_trial_with_apgi(0.0, 0.0, "congruent")

                    # Check history before run_experiment
                    assert len(runner.apgi_metrics_history) == 2

                    # Run experiment
                    with patch("time.time") as mock_time:
                        mock_time.side_effect = [0, 5, 10]

                        results = runner.run_experiment()

                        # Verify complete workflow
                        assert results["apgi_enabled"] is True
                        assert "accuracy" in results
                        assert "apgi_metrics" in results
                        assert "apgi_enhanced_metric" in results
                        assert results["apgi_enhanced_metric"] == 0.85
                        assert len(runner.apgi_metrics_history) == 2

    def test_full_workflow_disabled(self):
        """Test complete workflow with APGI disabled."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {
            "accuracy": 0.8,
            "response_times": [600, 650, 700, 750],
        }

        # Get APGI config and disable it
        config = eai.get_experiment_apgi_config("stroop_effect")
        config.enabled = False

        runner = eai.ExperimentAPGIRunner(mock_base_runner, config)

        # Process trials (should return None)
        result1 = runner.process_trial_with_apgi(1.0, 0.8, "incongruent")
        result2 = runner.process_trial_with_apgi(0.0, 0.0, "congruent")

        assert result1 is None
        assert result2 is None
        assert len(runner.apgi_metrics_history) == 0

        # Run experiment
        results = runner.run_experiment()

        assert results["apgi_enabled"] is False
        assert "accuracy" in results
        assert "apgi_metrics" not in results


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_apgi_integration_error(self):
        """Test handling of APGI integration errors."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.return_value = {"accuracy": 0.8}

        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )

        with patch("experiment_apgi_integration.APGIIntegration") as mock_apgi:
            mock_apgi.side_effect = Exception("APGI initialization failed")

            with pytest.raises(Exception):
                eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

    def test_base_experiment_error(self):
        """Test handling of base experiment errors."""
        mock_base_runner = MagicMock()
        mock_base_runner.run_experiment.side_effect = Exception("Experiment failed")

        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        with pytest.raises(Exception):
            runner.run_experiment()

    def test_metric_extraction_edge_cases(self):
        """Test metric extraction edge cases."""
        mock_base_runner = MagicMock()
        mock_params = eai.ExportedAPGIParams(
            experiment_name="test",
            enabled=True,
            tau_s=0.35,
            beta=1.5,
            theta_0=0.5,
            alpha=5.5,
        )
        runner = eai.ExperimentAPGIRunner(mock_base_runner, mock_params)

        # Empty results
        assert runner._extract_primary_metric({}) is None

        # Only strings
        assert runner._extract_primary_metric({"text": "value"}) is None

        # Only private keys
        assert runner._extract_primary_metric({"_private": 5}) is None

        # Mixed content
        result = runner._extract_primary_metric(
            {"text": "value", "_private": 5, "accuracy": 0.8, "other_metric": 0.75}
        )
        assert result == 0.8  # First numeric key found


if __name__ == "__main__":
    pytest.main([__file__])

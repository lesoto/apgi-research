"""
Test suite for run_change_blindness_full_apgi.py module.

Tests change blindness experiment runner with full APGI integration.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path to import the module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the dependencies before importing
sys.modules["standard_apgi_runner"] = MagicMock()
sys.modules["experiment_apgi_integration"] = MagicMock()
sys.modules["prepare_change_blindness"] = MagicMock()

import run_change_blindness_full_apgi as runner


class TestConstants:
    """Test module constants."""

    def test_time_budget(self):
        """Test TIME_BUDGET constant."""
        assert runner.TIME_BUDGET == 600

    def test_num_trials_config(self):
        """Test NUM_TRIALS_CONFIG constant."""
        assert runner.NUM_TRIALS_CONFIG == 60

    def test_change_probability(self):
        """Test CHANGE_PROBABILITY constant."""
        assert runner.CHANGE_PROBABILITY == 0.50


class TestChangeBlindnessExperimentRunner:
    """Test ChangeBlindnessExperimentRunner class."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        with patch(
            "run_change_blindness_full_apgi.StandardAPGIRunner"
        ) as mock_runner_class:
            mock_runner_class.return_value = MagicMock()

            runner_instance = runner.ChangeBlindnessExperimentRunner()

            assert isinstance(runner_instance, mock_runner_class.return_value)

    def test_run_experiment_with_apgi_enabled(self):
        """Test running experiment with APGI enabled."""
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = True

        # Mock APGI runner
        mock_apgi_runner = MagicMock()

        with patch(
            "run_change_blindness_full_apgi.StandardAPGIRunner"
        ) as mock_runner_class:
            mock_runner_class.return_value = mock_apgi_runner

            runner_instance = runner.ChangeBlindnessExperimentRunner()

            # Mock experiment methods
            mock_experiment = MagicMock()
            mock_experiment.create_trial_sequence.return_value = [
                {
                    "stimulus": "word",
                    "mask": "mask",
                    "trial_type": runner.TrialType.MASKED,
                },
                {
                    "stimulus": "word",
                    "mask": "",
                    "trial_type": runner.TrialType.UNMASKED,
                },
            ]
            mock_experiment.run_trial_sequence.return_value = {
                "detection_rates": [0.8, 0.6],
                "response_times": [600, 650, 700, 750],
            }

            mock_apgi_runner.run_experiment.return_value = {
                "base_results": {
                    "detection_rates": [0.8, 0.6],
                    "response_times": [600, 650, 700, 750],
                },
                "apgi_metrics": {
                    "mean_pi": 0.7,
                    "hierarchical_state": {"level_1": {"sensory": 0.8}},
                },
            }

            # Run experiment
            with patch("time.time") as mock_time:
                mock_time.side_effect = [0, 10]  # Start and end times

                result = runner_instance.run_experiment()

                assert "base_results" in result
                assert "apgi_metrics" in result
                assert result["apgi_enabled"] is True
                mock_apgi_runner.run_experiment.assert_called_once()

    def test_run_experiment_with_apgi_disabled(self):
        """Test running experiment with APGI disabled."""
        mock_experiment = MagicMock()
        mock_apgi_params = MagicMock()
        mock_apgi_params.enabled = False

        # Mock APGI runner as None
        with patch(
            "run_change_blindness_full_apgi.StandardAPGIRunner"
        ) as mock_runner_class:
            mock_runner_class.return_value = None

            runner_instance = runner.ChangeBlindnessExperimentRunner()

            # Mock experiment methods
            mock_experiment.create_trial_sequence.return_value = [
                {
                    "stimulus": "word",
                    "mask": "mask",
                    "trial_type": runner.TrialType.MASKED,
                },
                {
                    "stimulus": "word",
                    "mask": "",
                    "trial_type": runner.TrialType.UNMASKED,
                },
            ]
            mock_experiment.run_trial_sequence.return_value = {
                "detection_rates": [0.8, 0.6],
                "response_times": [600, 650, 700, 750],
            }

            result = runner_instance.run_experiment()

            assert "base_results" in result
            assert result["apgi_enabled"] is False
            assert "apgi_metrics" not in result

    def test_calculate_detection_rate(self):
        """Test detection rate calculation."""
        trial_results = [
            {"trial_type": "masked", "detected": True},
            {"trial_type": "masked", "detected": False},
            {"trial_type": "unmasked", "detected": True},
            {"trial_type": "unmasked", "detected": False},
        ]

        rate = runner._calculate_detection_rate(trial_results)

        expected = (1.0 + 0.0 + 1.0) / 4  # 2/4 = 0.5
        assert rate == expected

    def test_calculate_detection_rate_no_trials(self):
        """Test detection rate calculation with no trials."""
        rate = runner._calculate_detection_rate([])

        assert rate == 0.0

    def test_get_experiment_summary(self):
        """Test experiment summary generation."""
        results = {
            "base_results": {
                "detection_rates": [0.8, 0.6],
                "response_times": [600, 650, 700, 750],
            },
            "apgi_metrics": {
                "mean_pi": 0.7,
                "hierarchical_state": {"level_1": {"sensory": 0.8}},
            },
        }

        summary = runner.get_experiment_summary(results)

        assert "detection_rate" in summary
        assert "mean_response_time" in summary
        assert "apgi_enabled" in summary
        assert "mean_pi" in summary

    def test_save_results(self):
        """Test results saving."""
        results = {
            "detection_rate": 0.75,
            "response_times": [600, 650, 700],
            "apgi_metrics": {"pi": 0.7},
        }

        with patch("builtins.open", mock_open()) as mock_open_file:
            with patch("json.dump") as mock_dump:
                runner.save_results(results, "test_results.json")

                mock_open_file.assert_called_once_with("test_results.json", "w")
                mock_dump.assert_called_once()

    def test_main_function_success(self):
        """Test successful main function execution."""
        mock_runner_class = MagicMock()
        mock_runner_instance = MagicMock()
        mock_runner_class.return_value = mock_runner_instance

        mock_runner_instance.run_experiment.return_value = {
            "detection_rate": 0.75,
            "total_time": 120,
        }

        mock_format = MagicMock()
        mock_format.return_value = "Formatted output"

        with patch(
            "run_change_blindness_full_apgi.ChangeBlindnessExperimentRunner"
        ) as mock_runner:
            mock_runner.return_value = mock_runner_instance

            with patch("sys.exit") as mock_exit:
                runner.main()

                mock_runner.assert_called_once()
                mock_runner_instance.run_experiment.assert_called_once()
                mock_exit.assert_called_once_with(0)

    def test_main_function_with_error(self):
        """Test main function with error handling."""
        mock_runner_class = MagicMock()
        mock_runner_instance = MagicMock()
        mock_runner_class.return_value = mock_runner_instance
        mock_runner_instance.run_experiment.side_effect = Exception("Experiment failed")

        with patch(
            "run_change_blindness_full_apgi.ChangeBlindnessExperimentRunner"
        ) as mock_runner:
            mock_runner.return_value = mock_runner_instance

            with patch("sys.exit") as mock_exit:
                runner.main()

                mock_runner.assert_called_once()
                mock_runner_instance.run_experiment.assert_called_once()
                mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    pytest.main([__file__])

"""
Test suite for run_stroop_effect.py module.

Tests Stroop effect experiment runner functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path to import the module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the dependencies before importing
sys.modules["apgi_integration"] = MagicMock()
sys.modules["ultimate_apgi_template"] = MagicMock()
sys.modules["prepare_stroop_effect"] = MagicMock()

import run_stroop_effect as runner


class TestConstants:
    """Test module constants."""

    def test_time_budget(self):
        """Test TIME_BUDGET constant."""
        assert runner.TIME_BUDGET == 600

    def test_num_trials_config(self):
        """Test NUM_TRIALS_CONFIG constant."""
        assert runner.NUM_TRIALS_CONFIG == 80

    def test_inter_trial_interval(self):
        """Test INTER_TRIAL_INTERVAL_MS constant."""
        assert runner.INTER_TRIAL_INTERVAL_MS == 1000

    def test_feedback_delay(self):
        """Test FEEDBACK_DELAY_MS constant."""
        assert runner.FEEDBACK_DELAY_MS == 500


class TestStroopExperimentRunner:
    """Test Stroop experiment runner functionality."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        with patch("run_stroop_effect.StroopExperiment") as mock_exp:
            with patch("run_stroop_effect.APGIIntegration") as mock_apgi:
                with patch("run_stroop_effect.HierarchicalProcessor") as mock_hp:
                    with patch(
                        "run_stroop_effect.PrecisionExpectationState"
                    ) as mock_pes:
                        mock_exp_instance = MagicMock()
                        mock_apgi_instance = MagicMock()
                        mock_hp_instance = MagicMock()
                        mock_pes_instance = MagicMock()

                        mock_exp.return_value = mock_exp_instance
                        mock_apgi.return_value = mock_apgi_instance
                        mock_hp.return_value = mock_hp_instance
                        mock_pes.return_value = mock_pes_instance

                        exp_runner = runner.EnhancedStroopRunner()

                        assert exp_runner.experiment == mock_exp_instance
                        assert exp_runner.apgi == mock_apgi_instance
                        assert exp_runner.hierarchical == mock_hp_instance
                        assert exp_runner.precision_gap == mock_pes_instance

    def test_create_trial_sequence(self):
        """Test trial sequence creation."""
        mock_runner = MagicMock()
        mock_runner.experiment = MagicMock()
        mock_runner.experiment.trial_types = [
            runner.TrialType.CONGRUENT,
            runner.TrialType.INCONGRUENT,
        ] * 40  # 80 trials

        sequence = runner.EnhancedStroopRunner._create_trial_sequence(mock_runner)

        assert len(sequence) == 80
        assert all(trial in mock_runner.experiment.trial_types for trial in sequence)

    def test_run_single_trial(self):
        """Test running a single trial."""
        mock_runner = MagicMock()
        mock_runner.experiment = MagicMock()
        mock_runner.apgi = MagicMock()
        mock_runner.hierarchical = MagicMock()
        mock_runner.precision_gap = MagicMock()
        mock_runner.neuromodulators = {"ACh": 1.0, "NE": 1.0, "DA": 1.0, "HT5": 1.0}

        # Mock experiment methods
        mock_runner.experiment.get_next_trial.return_value = MagicMock(
            trial_type=runner.TrialType.INCONGRUENT, ink_color="blue"
        )

        # Mock APGI processing
        mock_runner.apgi.process_trial.return_value = {
            "pi": 0.7,
            "theta": 0.5,
            "surprise": 0.3,
        }

        # Mock hierarchical processing
        mock_runner.hierarchical.process_level.return_value = MagicMock(S=0.8)

        # Mock precision state
        mock_runner.precision_gap.Pi_e_actual = 1.5
        mock_runner.precision_gap.Pi_i_actual = 1.5

        with patch("run_stroop_effect.time.sleep"):
            # Set up participant mock directly on the runner
            mock_participant_instance = MagicMock()
            mock_participant_instance.process_trial.return_value = (True, 750)
            mock_runner.participant = mock_participant_instance

            runner.EnhancedStroopRunner._run_single_trial(mock_runner, 0)

            # Verify the method was called
            mock_runner.experiment.get_next_trial.assert_called()

    def test_calculate_interference_effect(self):
        """Test interference effect calculation."""
        mock_runner = MagicMock()

        # Mock trial results
        trial_results = [
            {"trial_type": runner.TrialType.CONGRUENT, "response_time": 600},
            {"trial_type": runner.TrialType.CONGRUENT, "response_time": 650},
            {"trial_type": runner.TrialType.INCONGRUENT, "response_time": 800},
            {"trial_type": runner.TrialType.INCONGRUENT, "response_time": 850},
        ]

        effect = runner.StroopExperimentRunner._calculate_interference_effect(
            mock_runner, trial_results
        )

        # Should calculate difference between incongruent and congruent RTs
        congruent_mean = (600 + 650) / 2
        incongruent_mean = (800 + 850) / 2
        expected_effect = incongruent_mean - congruent_mean

        assert effect == expected_effect

    def test_calculate_interference_effect_no_trials(self):
        """Test interference effect calculation with no trials."""
        mock_runner = MagicMock()

        effect = runner.StroopExperimentRunner._calculate_interference_effect(
            mock_runner, []
        )

        assert effect == 0

    def test_calculate_interference_effect_one_type(self):
        """Test interference effect with only one trial type."""
        mock_runner = MagicMock()

        trial_results = [
            {"trial_type": runner.TrialType.CONGRUENT, "response_time": 600},
            {"trial_type": runner.TrialType.CONGRUENT, "response_time": 650},
        ]

        effect = runner.StroopExperimentRunner._calculate_interference_effect(
            mock_runner, trial_results
        )

        assert effect == 0

    def test_run_experiment(self):
        """Test full experiment run."""
        # Create a real runner instance
        runner_instance = runner.EnhancedStroopRunner()
        runner_instance.enable_apgi = False  # Disable APGI for simpler test

        # Mock the experiment summary
        runner_instance.experiment = MagicMock()
        runner_instance.experiment.get_summary.return_value = {
            "interference_effect_ms": 200.0,
            "congruent_rt_ms": 600.0,
            "incongruent_rt_ms": 800.0,
            "accuracy": 0.9,
        }
        runner_instance.experiment.trials = [MagicMock()] * 80

        with patch("run_stroop_effect.time.time") as mock_time:
            mock_time.side_effect = [0] + [i * 0.5 for i in range(1, 100)]

            # Mock the _run_single_trial to avoid actual execution
            with patch.object(runner_instance, "_run_single_trial"):
                results = runner_instance.run_experiment()

                assert "interference_effect_ms" in results
                assert "completion_time_s" in results
                assert results["interference_effect_ms"] == 200.0

    def test_run_experiment_time_budget(self):
        """Test experiment run with time budget limit."""
        mock_runner = MagicMock()
        mock_runner.experiment = MagicMock()
        mock_runner.apgi = MagicMock()
        mock_runner.hierarchical_processor = MagicMock()
        mock_runner.precision_state = MagicMock()

        with patch.object(mock_runner, "_create_trial_sequence") as mock_sequence:
            with patch.object(mock_runner, "_run_single_trial") as mock_trial:
                with patch.object(
                    mock_runner, "_calculate_interference_effect"
                ) as mock_effect:
                    with patch("run_stroop_effect.time.time") as mock_time:
                        # Mock time budget exceeded
                        mock_sequence.return_value = [runner.TrialType.CONGRUENT] * 100
                        mock_trial.return_value = {"response_time": 600}
                        mock_effect.return_value = 100
                        mock_time.side_effect = [0, 300, 700]  # Exceeds TIME_BUDGET

                        results = runner.StroopExperimentRunner.run_experiment(
                            mock_runner
                        )

                        # Should stop early due to time budget
                        assert len(results["trial_results"]) < len(
                            mock_sequence.return_value
                        )

    def test_get_experiment_summary(self):
        """Test experiment summary generation."""
        mock_runner = MagicMock()

        results = {
            "num_trials": 80,
            "interference_effect_ms": 200,
            "completion_time_s": 120,
            "congruent_rt_ms": 600.0,
            "incongruent_rt_ms": 800.0,
        }

        summary = runner.EnhancedStroopRunner.get_experiment_summary(
            mock_runner, results
        )

        assert "interference_effect_ms" in summary
        assert "total_trials" in summary
        assert "completion_time_s" in summary

    def test_analyze_apgi_dynamics(self):
        """Test APGI dynamics analysis."""
        mock_runner = MagicMock()
        mock_runner.apgi = MagicMock()
        mock_runner.apgi.get_current_state.return_value = {
            "pi": 0.7,
            "theta": 0.5,
        }

        trial_results = [
            {"apgi_state": {"pi": 0.7, "theta": 0.5}},
            {"apgi_state": {"pi": 0.8, "theta": 0.6}},
            {"apgi_state": {"pi": 0.9, "theta": 0.7}},
        ]

        analysis = runner.EnhancedStroopRunner._analyze_apgi_dynamics(
            mock_runner, trial_results
        )

        assert "apgi_state" in analysis

    def test_analyze_hierarchical_processing(self):
        """Test hierarchical processing analysis."""
        mock_runner = MagicMock()

        trial_results = [
            {
                "hierarchical_state": {
                    "level_1": {"sensory": 0.8},
                    "level_2": {"feature": 0.6},
                }
            },
            {
                "hierarchical_state": {
                    "level_1": {"sensory": 0.9},
                    "level_2": {"feature": 0.7},
                }
            },
        ]

        analysis = runner.EnhancedStroopRunner._analyze_hierarchical_processing(
            mock_runner, trial_results
        )

        assert "level_1" in analysis
        assert "level_2" in analysis

    def test_save_results(self):
        """Test results saving."""
        mock_runner = MagicMock()

        results = {
            "interference_effect_ms": 200,
            "trial_results": [{"trial": 1}, {"trial": 2}],
        }

        with patch("builtins.open", mock_open()):
            with patch("json.dump") as mock_dump:
                runner.StroopExperimentRunner.save_results(
                    mock_runner, results, "test.json"
                )
                mock_dump.assert_called_once()

    def test_validate_experiment_config(self):
        """Test experiment configuration validation."""
        mock_runner = MagicMock()
        mock_runner.experiment = MagicMock()
        mock_runner.experiment.trial_types = [
            runner.TrialType.CONGRUENT,
            runner.TrialType.INCONGRUENT,
        ]

        # Valid config should pass
        assert runner.StroopExperimentRunner._validate_experiment_config(mock_runner)

    def test_validate_experiment_config_invalid(self):
        """Test invalid experiment configuration validation."""
        mock_runner = MagicMock()
        mock_runner.experiment = MagicMock()
        mock_runner.experiment.trial_types = []  # No trial types

        # Invalid config should fail
        assert not runner.StroopExperimentRunner._validate_experiment_config(
            mock_runner
        )

    def test_calculate_additional_metrics(self):
        """Test additional metrics calculation."""
        mock_runner = MagicMock()

        trial_results = [
            {
                "trial_type": runner.TrialType.CONGRUENT,
                "response_time": 600,
                "correct": True,
            },
            {
                "trial_type": runner.TrialType.CONGRUENT,
                "response_time": 650,
                "correct": True,
            },
            {
                "trial_type": runner.TrialType.INCONGRUENT,
                "response_time": 800,
                "correct": True,
            },
            {
                "trial_type": runner.TrialType.INCONGRUENT,
                "response_time": 850,
                "correct": False,
            },
        ]

        metrics = runner.EnhancedStroopRunner._calculate_additional_metrics(
            mock_runner, trial_results
        )

        assert "enhanced_accuracy" in metrics
        assert "response_variance" in metrics
        assert "learning_rate" in metrics


class TestMainFunction:
    """Test main function and entry points."""

    @patch("run_stroop_effect.EnhancedStroopRunner")
    @patch("run_stroop_effect.format_apgi_output")
    def test_main_function(self, mock_format, mock_runner_class):
        """Test main function execution."""
        mock_runner_instance = MagicMock()
        mock_runner_class.return_value = mock_runner_instance

        results = {
            "interference_effect_ms": 200,
            "num_trials": 80,
            "completion_time_s": 120,
            "congruent_rt_ms": 600.0,
            "incongruent_rt_ms": 800.0,
            "accuracy": 0.9,
        }
        mock_runner_instance.run_experiment.return_value = results
        mock_format.return_value = "Formatted output"

        with patch("builtins.print") as mock_print:
            runner.main()

            mock_runner_instance.run_experiment.assert_called_once()
            mock_print.assert_called()

    @patch("run_stroop_effect.StroopExperimentRunner")
    def test_main_function_with_error(self, mock_runner_class):
        """Test main function with error handling."""
        mock_runner_instance = MagicMock()
        mock_runner_class.return_value = mock_runner_instance
        mock_runner_instance.run_experiment.side_effect = Exception("Test error")

        with patch("builtins.print"):
            with pytest.raises(Exception):
                runner.main()

    def test_experiment_factory(self):
        """Test experiment factory function."""
        with patch("run_stroop_effect.EnhancedStroopRunner") as mock_runner_class:
            mock_instance = MagicMock()
            mock_runner_class.return_value = mock_instance

            experiment = runner.create_experiment()

            assert experiment == mock_instance
            mock_runner_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])

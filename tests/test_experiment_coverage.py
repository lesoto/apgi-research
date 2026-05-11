"""
Focused tests for experiment modules to improve coverage.

Tests the key constants and classes across experiment files.
"""

import sys
from pathlib import Path

import pytest

# Add experiments directory to path
experiments_dir = Path(__file__).parent.parent / "experiments"
sys.path.insert(0, str(experiments_dir))

# Import specific experiment modules that we know exist
import prepare_ai_benchmarking
import prepare_attentional_blink
import prepare_go_no_go
import prepare_stroop_effect


class TestPrepareAIBenchmarking:
    """Test prepare_ai_benchmarking module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(prepare_ai_benchmarking, "TIME_BUDGET")
        assert hasattr(prepare_ai_benchmarking, "NUM_TRIALS")
        assert hasattr(prepare_ai_benchmarking, "APGI_ENABLED")

    def test_constants_values(self):
        """Test constant values."""
        assert prepare_ai_benchmarking.TIME_BUDGET == 600
        assert prepare_ai_benchmarking.NUM_TRIALS == 50
        assert prepare_ai_benchmarking.APGI_ENABLED is True

    def test_benchmark_type_enum(self):
        """Test BenchmarkType enum."""
        assert hasattr(prepare_ai_benchmarking, "BenchmarkType")
        benchmark_types = list(prepare_ai_benchmarking.BenchmarkType)
        assert prepare_ai_benchmarking.BenchmarkType.REASONING in benchmark_types
        assert prepare_ai_benchmarking.BenchmarkType.MEMORY in benchmark_types
        assert prepare_ai_benchmarking.BenchmarkType.ATTENTION in benchmark_types

    def test_difficulty_enum(self):
        """Test Difficulty enum."""
        assert hasattr(prepare_ai_benchmarking, "Difficulty")
        difficulties = list(prepare_ai_benchmarking.Difficulty)
        assert prepare_ai_benchmarking.Difficulty.EASY in difficulties
        assert prepare_ai_benchmarking.Difficulty.MEDIUM in difficulties
        assert prepare_ai_benchmarking.Difficulty.HARD in difficulties

    def test_ai_benchmark_generator_class(self):
        """Test AIBenchmarkGenerator class."""
        assert hasattr(prepare_ai_benchmarking, "AIBenchmarkGenerator")

        generator = prepare_ai_benchmarking.AIBenchmarkGenerator()
        assert hasattr(generator, "create_trial")
        assert hasattr(generator, "reset")
        assert callable(generator.create_trial)
        assert callable(generator.reset)

    def test_ai_benchmark_trial_dataclass(self):
        """Test AIBenchmarkTrial dataclass."""
        assert hasattr(prepare_ai_benchmarking, "AIBenchmarkTrial")

        # Test creating a trial with correct parameters
        trial = prepare_ai_benchmarking.AIBenchmarkTrial(
            trial_number=1,
            benchmark_type=prepare_ai_benchmarking.BenchmarkType.REASONING,
            difficulty=prepare_ai_benchmarking.Difficulty.MEDIUM,
            task_description="Test task",
            correct_answer="4",
        )
        assert trial.trial_number == 1
        assert trial.benchmark_type == prepare_ai_benchmarking.BenchmarkType.REASONING
        assert trial.difficulty == prepare_ai_benchmarking.Difficulty.MEDIUM
        assert trial.task_description == "Test task"
        assert trial.correct_answer == "4"

    def test_ai_benchmark_experiment_class(self):
        """Test AIBenchmarkExperiment class."""
        assert hasattr(prepare_ai_benchmarking, "AIBenchmarkExperiment")

        experiment = prepare_ai_benchmarking.AIBenchmarkExperiment(num_trials=10)
        assert experiment.num_trials == 10
        assert hasattr(experiment, "get_next_trial")
        assert hasattr(experiment, "run_trial")
        assert callable(experiment.get_next_trial)

    def test_generator_create_trial(self):
        """Test trial creation."""
        generator = prepare_ai_benchmarking.AIBenchmarkGenerator(seed=42)
        trial = generator.create_trial(1)

        assert isinstance(trial, prepare_ai_benchmarking.AIBenchmarkTrial)
        assert trial.trial_number == 1
        assert isinstance(trial.benchmark_type, prepare_ai_benchmarking.BenchmarkType)
        assert isinstance(trial.difficulty, prepare_ai_benchmarking.Difficulty)
        assert trial.task_description != ""
        assert trial.correct_answer != ""

    def test_experiment_get_next_trial(self):
        """Test getting next trial from experiment."""
        experiment = prepare_ai_benchmarking.AIBenchmarkExperiment(
            num_trials=5, seed=42
        )

        trial = experiment.get_next_trial()
        assert trial is not None
        assert isinstance(trial, prepare_ai_benchmarking.AIBenchmarkTrial)

        # Get all trials
        trials = []
        while trial is not None:
            trials.append(trial)
            trial = experiment.get_next_trial()

        assert len(trials) == 5


class TestPrepareAttentionalBlink:
    """Test prepare_attentional_blink module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(prepare_attentional_blink, "TIME_BUDGET")
        assert hasattr(prepare_attentional_blink, "NUM_TRIALS")
        assert hasattr(prepare_attentional_blink, "APGI_ENABLED")

    def test_constants_values(self):
        """Test constant values."""
        assert prepare_attentional_blink.TIME_BUDGET == 600
        assert prepare_attentional_blink.NUM_TRIALS == 100
        assert prepare_attentional_blink.APGI_ENABLED is True


class TestPrepareGoNoGo:
    """Test prepare_go_no_go module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(prepare_go_no_go, "TIME_BUDGET")
        assert hasattr(prepare_go_no_go, "NUM_TRIALS")
        assert hasattr(prepare_go_no_go, "APGI_ENABLED")

    def test_constants_values(self):
        """Test constant values."""
        assert prepare_go_no_go.TIME_BUDGET == 600
        assert prepare_go_no_go.NUM_TRIALS == 120
        assert prepare_go_no_go.APGI_ENABLED is True


class TestPrepareStroopEffect:
    """Test prepare_stroop_effect module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(prepare_stroop_effect, "TIME_BUDGET")
        assert hasattr(prepare_stroop_effect, "NUM_TRIALS")
        assert hasattr(prepare_stroop_effect, "APGI_ENABLED")

    def test_constants_values(self):
        """Test constant values."""
        assert prepare_stroop_effect.TIME_BUDGET == 600
        assert prepare_stroop_effect.NUM_TRIALS == 150
        assert prepare_stroop_effect.APGI_ENABLED is True


class TestExperimentIntegration:
    """Test integration between prepare modules."""

    def test_parameter_consistency(self):
        """Test that parameters are consistent between prepare modules."""
        # Check that TIME_BUDGET is consistent
        assert prepare_ai_benchmarking.TIME_BUDGET == 600
        assert prepare_attentional_blink.TIME_BUDGET == 600
        assert prepare_go_no_go.TIME_BUDGET == 600
        assert prepare_stroop_effect.TIME_BUDGET == 600

    def test_apgi_enabled_all_experiments(self):
        """Test that APGI is enabled in all experiment modules."""
        prepare_modules = [
            prepare_ai_benchmarking,
            prepare_attentional_blink,
            prepare_go_no_go,
            prepare_stroop_effect,
        ]

        for module in prepare_modules:
            assert hasattr(module, "APGI_ENABLED")
            assert module.APGI_ENABLED is True

    def test_all_modules_have_cli_integration(self):
        """Test that all modules have CLI integration."""
        prepare_modules = [
            prepare_ai_benchmarking,
            prepare_attentional_blink,
            prepare_go_no_go,
            prepare_stroop_effect,
        ]

        for module in prepare_modules:
            # Check that CLI functions are imported
            assert hasattr(module, "cli_entrypoint")
            assert hasattr(module, "create_standard_parser")


if __name__ == "__main__":
    pytest.main([__file__])

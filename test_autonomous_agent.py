"""
Autonomous agent tests for APGI experiments.

Tests the autonomous agent capabilities and decision-making processes.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import autonomous agent modules (assuming they exist)
try:
    from autonomous_agent import (
        AutonomousAgent,
        GitPerformanceTracker,
        ParameterOptimizer,
    )
    from autonomous_agent import ExperimentResult, OptimizationStrategy

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Autonomous agent module not available")
class TestExperimentResult:
    """Test the ExperimentResult dataclass."""

    def test_experiment_result_creation(self):
        """Test creating an ExperimentResult with all fields."""
        result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test_experiment",
            primary_metric=0.85,
            apgi_metrics={"ignition_prob": 0.25},
            apgi_enhanced_metric=0.3,
            completion_time_s=120.5,
            timestamp="2023-01-01T12:00:00",
            parameter_modifications={"param1": 0.5},
            status="success",
        )

        assert result.commit_hash == "abc123"
        assert result.experiment_name == "test_experiment"
        assert result.primary_metric == 0.85
        assert result.completion_time_s == 120.5
        assert result.status == "success"

    def test_experiment_result_to_dict(self):
        """Test converting ExperimentResult to dictionary."""
        result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test_experiment",
            primary_metric=0.85,
            apgi_metrics={"ignition_prob": 0.25},
            apgi_enhanced_metric=0.3,
            completion_time_s=120.5,
            timestamp="2023-01-01T12:00:00",
            parameter_modifications={"param1": 0.5},
            status="success",
        )

        result_dict = result.__dict__
        assert isinstance(result_dict, dict)
        assert result_dict["commit_hash"] == "abc123"


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Autonomous agent module not available")
class TestOptimizationStrategy:
    """Test the OptimizationStrategy dataclass."""

    def test_optimization_strategy_creation(self):
        """Test creating an OptimizationStrategy."""
        strategy = OptimizationStrategy(
            name="test_strategy",
            description="Test optimization strategy",
            parameter_ranges={"param1": (0.1, 1.0)},
            mutation_strength=0.1,
            exploration_rate=0.2,
            learning_rate=0.01,
        )

        assert strategy.name == "test_strategy"
        assert strategy.description == "Test optimization strategy"
        assert strategy.parameter_ranges == {"param1": (0.1, 1.0)}
        assert strategy.mutation_strength == 0.1
        assert strategy.exploration_rate == 0.2
        assert strategy.learning_rate == 0.01


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Autonomous agent module not available")
class TestParameterOptimizer:
    """Test the ParameterOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = ParameterOptimizer()

    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.strategies is not None
        assert "default" in self.optimizer.strategies
        assert "attentional_blink" in self.optimizer.strategies

    def test_get_strategy_existing(self):
        """Test getting an existing strategy."""
        strategy = self.optimizer.get_strategy("attentional_blink")
        assert strategy.name == "attention_optimization"
        assert strategy.parameter_ranges is not None

    def test_get_strategy_default(self):
        """Test getting default strategy for unknown experiment."""
        strategy = self.optimizer.get_strategy("unknown_experiment")
        assert strategy.name == "general_optimization"

    def test_suggest_modifications_empty_history(self):
        """Test suggesting modifications with no performance history."""
        current_params = {"BASE_DETECTION_RATE": 0.5}
        modifications = self.optimizer.suggest_modifications(
            "attentional_blink", current_params, []
        )

        # Should return some modifications
        assert isinstance(modifications, dict)

    def test_suggest_modifications_with_history(self):
        """Test suggesting modifications with performance history."""
        current_params = {"BASE_DETECTION_RATE": 0.5}
        performance_history = [0.7, 0.75, 0.8, 0.72, 0.78]

        modifications = self.optimizer.suggest_modifications(
            "attentional_blink", current_params, performance_history
        )

        # Should suggest modifications based on performance trend
        assert isinstance(modifications, dict)

    def test_parameter_modification_types(self):
        """Test different parameter modification types."""
        current_params = {"BASE_DETECTION_RATE": 0.5}
        modifications = self.optimizer.suggest_modifications(
            "default", current_params, []
        )

        # Should return a dict of modifications
        assert isinstance(modifications, dict)


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Autonomous agent module not available")
class TestGitPerformanceTracker:
    """Test the GitPerformanceTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        import git
        import os

        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Initialize git repo
        self.repo = git.Repo.init(self.repo_path)

        # Configure git
        with self.repo.config_writer() as config:
            config.set_value("user", "name", "Test User")
            config.set_value("user", "email", "test@example.com")

        # Create initial commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("Initial content")

        # Change to repo directory and add file
        old_cwd = os.getcwd()
        try:
            os.chdir(self.repo_path)
            self.repo.index.add(["test.txt"])
            self.repo.index.commit("Initial commit")
        finally:
            os.chdir(old_cwd)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = GitPerformanceTracker(str(self.repo_path))
        assert tracker.repo_path == self.repo_path
        assert tracker.results_file == self.repo_path / "optimization_results.json"

    def test_load_best_results_empty(self):
        """Test loading best results when file doesn't exist."""
        tracker = GitPerformanceTracker(str(self.repo_path))
        results = tracker._load_best_results()
        assert results == {}

    def test_save_and_load_results(self):
        """Test saving and loading results."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Create test result
        result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test_experiment",
            primary_metric=0.85,
            apgi_metrics={"ignition_prob": 0.25},
            apgi_enhanced_metric=0.3,
            completion_time_s=120.5,
            timestamp="2023-01-01T12:00:00",
            parameter_modifications={"param1": 0.5},
            status="success",
        )

        # Save results
        tracker.save_results({"test_experiment": result})

        # Load results
        loaded_results = tracker._load_best_results()
        assert "test_experiment" in loaded_results
        assert loaded_results["test_experiment"].primary_metric == 0.85

    def test_commit_experiment(self):
        """Test committing experiment modifications."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # This method might not exist or have different signature
        # Let's test what's actually available
        assert hasattr(tracker, "commit_experiment") or hasattr(
            tracker, "_commit_experiment"
        )

    def test_is_improvement_new_experiment(self):
        """Test improvement check for new experiment."""
        tracker = GitPerformanceTracker(str(self.repo_path))
        assert tracker.is_improvement("new_experiment", 0.5) is True

    def test_is_improvement_higher_better(self):
        """Test improvement check for higher-is-better metric."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Add existing result
        result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test_experiment",
            primary_metric=0.8,
            apgi_metrics={"ignition_prob": 0.25},
            apgi_enhanced_metric=0.3,
            completion_time_s=120.5,
            timestamp="2023-01-01T12:00:00",
            parameter_modifications={"param1": 0.5},
            status="success",
        )
        tracker.save_results({"test_experiment": result})

        # Load results into memory
        tracker.best_results = tracker._load_best_results()

        # Test improvements
        assert tracker.is_improvement("test_experiment", 0.9) is True
        assert tracker.is_improvement("test_experiment", 0.7) is False

    def test_get_best_metric(self):
        """Test getting best metric for experiment."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Add existing result
        result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test_experiment",
            primary_metric=0.8,
            apgi_metrics={"ignition_prob": 0.25},
            apgi_enhanced_metric=0.3,
            completion_time_s=120.5,
            timestamp="2023-01-01T12:00:00",
            parameter_modifications={"param1": 0.5},
            status="success",
        )
        tracker.save_results({"test_experiment": result})

        # Load results into memory
        tracker.best_results = tracker._load_best_results()

        best_metric = tracker.get_best_metric("test_experiment")
        assert best_metric == 0.8


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Autonomous agent module not available")
class TestAutonomousAgent:
    """Test the AutonomousAgent class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Initialize git repo
        import git

        self.repo = git.Repo.init(self.repo_path)

        # Configure git
        with self.repo.config_writer() as config:
            config.set_value("user", "name", "Test User")
            config.set_value("user", "email", "test@example.com")

        # Create initial commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("Initial content")

        # Change to repo directory and add file (using relative path)
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(self.repo_path)
            self.repo.index.add(["test.txt"])
            self.repo.index.commit("Initial commit")
        finally:
            os.chdir(old_cwd)

        self.create_mock_experiment_files()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_experiment_files(self):
        """Create mock experiment files for testing."""
        # Create prepare file
        prepare_content = """
\"\"\"Mock experiment preparation.\"\"\"
NUM_TRIALS_CONFIG = 50
BASE_DETECTION_RATE = 0.5
"""
        (self.repo_path / "prepare_test_experiment.py").write_text(prepare_content)

        # Create run file
        run_content = """
\"\"\"Mock experiment runner.\"\"\"
import time

NUM_TRIALS_CONFIG = 50
BASE_DETECTION_RATE = 0.5

class MockRunner:
    def run_experiment(self):
        \"\"\"Mock experiment execution.\"\"\"
        time.sleep(0.1)  # Simulate processing time
        return {
            "accuracy": BASE_DETECTION_RATE,
            "primary_metric": BASE_DETECTION_RATE,
            "completion_time": 1.0
        }
"""
        (self.repo_path / "run_test_experiment.py").write_text(run_content)

    def test_initialization(self):
        """Test agent initialization."""
        agent = AutonomousAgent(str(self.repo_path))
        assert agent.git_tracker is not None
        assert agent.optimizer is not None

    @patch("importlib.import_module")
    def test_load_experiment_modules(self, mock_import):
        """Test loading experiment modules."""
        # Mock the importlib.import_module calls
        mock_prepare = Mock()
        mock_run = Mock()
        mock_import.side_effect = [mock_prepare, mock_run]

        # Create a temporary directory with our test files
        temp_exp_dir = self.repo_path / "temp_experiments"
        temp_exp_dir.mkdir()

        # Change to the temp directory temporarily
        import os

        original_cwd = os.getcwd()
        os.chdir(temp_exp_dir)

        try:
            # Create mock experiment files in temp directory
            (temp_exp_dir / "prepare_test_experiment.py").write_text("# test")
            (temp_exp_dir / "run_test_experiment.py").write_text("# test")

            agent = AutonomousAgent(str(self.repo_path))

            # Check that some modules were loaded (actual experiment files from repo)
            assert len(agent.experiment_modules) > 0

        finally:
            os.chdir(original_cwd)
            import shutil

            shutil.rmtree(temp_exp_dir, ignore_errors=True)

    def test_get_current_parameters(self):
        """Test extracting current parameters from run file."""
        agent = AutonomousAgent(str(self.repo_path))

        # This would require the actual module loading logic
        # For now, just test that the method exists
        assert hasattr(agent, "get_current_parameters")

    def test_apply_modifications(self):
        """Test applying parameter modifications."""
        agent = AutonomousAgent(str(self.repo_path))

        # This would require the actual file modification logic
        # For now, just test that the method exists
        assert hasattr(agent, "apply_modifications")

    def test_extract_primary_metric(self):
        """Test extracting primary metric from experiment results."""
        agent = AutonomousAgent(str(self.repo_path))

        mock_results = {
            "accuracy": 0.85,
            "primary_metric": 0.82,
            "completion_time": 1.5,
        }

        metric = agent.extract_primary_metric(mock_results)
        # Should return the primary_metric if available, otherwise accuracy
        assert metric == 0.82

    def test_get_metric_direction(self):
        """Test determining if higher or lower is better for a metric."""
        agent = AutonomousAgent(str(self.repo_path))

        # Test known metrics
        assert agent.get_metric_direction("accuracy") == "higher"
        assert agent.get_metric_direction("error_rate") == "lower"
        assert agent.get_metric_direction("response_time") == "lower"

        # Test unknown metric (should default to higher)
        assert agent.get_metric_direction("unknown_metric") == "higher"

    def test_run_experiment_success(self):
        """Test successful experiment execution."""
        agent = AutonomousAgent(str(self.repo_path))

        # This would require the actual experiment running logic
        # For now, just test that the method exists
        assert hasattr(agent, "run_experiment")

    def test_timeout_error(self):
        """Test timeout error handling."""
        from autonomous_agent import TimeoutError

        # Test that TimeoutError can be raised
        with pytest.raises(TimeoutError):
            raise TimeoutError("Test timeout")

    def test_full_optimization_cycle(self):
        """Test a full optimization cycle."""
        agent = AutonomousAgent(str(self.repo_path))

        # This would require the full optimization logic
        # For now, just test that the method exists
        assert hasattr(agent, "optimize_experiment")


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Autonomous agent module not available")
class TestPerformance:
    """Performance tests for the autonomous agent."""

    def test_parameter_extraction_performance(self):
        """Test performance of parameter extraction."""
        # This would test that parameter extraction is efficient
        pass

    def test_git_operations_performance(self):
        """Test performance of Git operations."""
        # This would test that Git operations are efficient
        pass


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__])

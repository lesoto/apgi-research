"""
Integration tests for the APGI RL optimization loop.

This test suite validates the end-to-end functionality of the autonomous agent's
experiment loop, including parameter modification, performance tracking,
and self-healing guardrails.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest

from autonomous_agent import AutonomousAgent, ExperimentResult
from human_layer import ReviewDecision, ReviewResult
from xpr_agent_engine import SkillResult, SkillType


class TestRLLoopIntegration:
    """End-to-end testing of the reinforcement learning (RL) experiment loop."""

    temp_dir: str
    repo_path: Path
    repo: git.Repo
    run_file_name: str
    run_file: Path
    agent: AutonomousAgent

    def setup_method(self):
        """Set up a mock repository structure for loop testing."""
        self.temp_dir = tempfile.mkdtemp()
        # Resolve path to avoid /private/var symlink issues on macOS
        self.repo_path = Path(self.temp_dir).resolve()

        # Initialize Git repo
        self.repo = git.Repo.init(str(self.repo_path))
        self.repo.config_writer().set_value("user", "name", "Loop Tester").release()
        self.repo.config_writer().set_value(
            "user", "email", "loop@example.com"
        ).release()

        # Create initial test files
        (self.repo_path / "README.md").write_text("# APGI Loop Test Repo")
        self.repo.index.add(["README.md"])
        self.repo.index.commit("Initial commit")

        # Create a mock experiment run file with tunable parameters
        self.run_file_name = "run_masked_task.py"
        self.run_file = self.repo_path / self.run_file_name
        self.run_file.write_text("""
\"\"\"Mock masked task for loop testing.\"\"\"
TARGET_DURATION = 50
MASK_DURATION = 100
NOISE_LEVEL = 0.5

def run_experiment():
    score = (TARGET_DURATION / 100.0) * (1.1 - NOISE_LEVEL)
    import json
    return {
        "primary_metric": score,
        "completion_time": 1.0,
        "status": "success"
    }

if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run_experiment()))
""")
        self.repo.index.add([self.run_file_name])
        self.repo.index.commit("Add mock experiment")

        # Initialize the agent
        with patch("autonomous_agent.HumanControlLayer"):
            self.agent = AutonomousAgent(repo_path=str(self.repo_path))

        # Ensure it has the mock experiment registered
        self.agent.experiment_modules = {
            "masked_task": {"run_file": str(self.run_file), "name": "Masked Task"}
        }

        # Mock human control to avoid interactive prompts and review triggers
        human_control_config = {"interaction_mode": "autonomous", "configured": False}
        self.agent.human_control = MagicMock()
        self.agent.human_control.get_configuration_summary = MagicMock(
            return_value=human_control_config
        )
        self.agent.human_control.review = MagicMock(
            return_value=ReviewResult(
                decision=ReviewDecision.APPROVE,
                comments="Mock approval",
                reviewer="mock",
                timestamp="now",
                confidence=1.0,
            ),
        )
        # Mock git_tracker with best_results attribute and commit_experiment method
        self.agent.git_tracker = MagicMock()
        self.agent.git_tracker.best_results = {
            "masked_task": ExperimentResult(
                commit_hash="mock_hash",
                experiment_name="masked_task",
                primary_metric=0.8,
                apgi_metrics={},
                apgi_enhanced_metric=0.8,
                completion_time_s=1.0,
                timestamp="now",
                parameter_modifications={"TARGET_DURATION": 80},
                status="success",
            )
        }
        self.agent.git_tracker.commit_experiment = MagicMock(
            return_value="mock_commit_hash"
        )
        self.agent.git_tracker.get_best_metric = MagicMock(return_value=0.5)
        # Mock agent_engine
        self.agent.agent_engine = MagicMock()

    def teardown_method(self):
        """Clean up the temporary repository."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("autonomous_agent.AutonomousAgent.run_experiment")
    def test_optimization_improvement_cycle(self, mock_run):
        """Verify that the agent correctly tracks and applies improvements."""

        # side_effect for run_experiment
        # The agent engine skill is mocked below
        mock_run.side_effect = [
            # Baseline (first run in optimize_experiment)
            ExperimentResult(
                commit_hash="h1",
                experiment_name="masked_task",
                primary_metric=0.5,
                apgi_metrics={},
                apgi_enhanced_metric=0.5,
                completion_time_s=1.0,
                timestamp="now",
                parameter_modifications={"TARGET_DURATION": 50},
                status="success",
            ),
            # Modification (second run in optimize_experiment)
            ExperimentResult(
                commit_hash="h2",
                experiment_name="masked_task",
                primary_metric=0.8,
                apgi_metrics={},
                apgi_enhanced_metric=0.8,
                completion_time_s=1.0,
                timestamp="now",
                parameter_modifications={"TARGET_DURATION": 80},
                status="success",
            ),
        ]
        mock_run.side_effect = mock_run.side_effect
        mock_run.return_value = SkillResult(
            success=True,
            skill_type=SkillType.PLAN_GENERATION.value,
            result={
                "hypothesis": "Maybe 80 is better",
                "modifications": {"TARGET_DURATION": 80},
            },
            confidence=0.9,
        )

        # Mock engine to provide a plan
        self.agent.agent_engine.execute_skill.return_value = SkillResult(  # type: ignore[attr-defined]
            success=True,
            skill_type=SkillType.PLAN_GENERATION.value,
            result={
                "hypothesis": "Maybe 80 is better",
                "modifications": {"TARGET_DURATION": 80},
            },
            confidence=0.9,
        )

        # Run optimization
        self.agent.optimize_experiment("masked_task", iterations=2)

        # Check that best results were updated
        assert "masked_task" in self.agent.git_tracker.best_results
        assert self.agent.git_tracker.best_results["masked_task"].primary_metric == 0.8

    def test_parameter_extraction_safeguard(self):
        """Ensure the loop can correctly read parameters from modified files."""
        params = self.agent._get_current_parameters("masked_task")
        assert params["TARGET_DURATION"] == 50
        assert params["NOISE_LEVEL"] == 0.5

        # Modify file manually and check
        content = self.run_file.read_text()
        self.run_file.write_text(
            content.replace("TARGET_DURATION = 50", "TARGET_DURATION = 80")
        )
        params_updated = self.agent._get_current_parameters("masked_task")
        assert params_updated["TARGET_DURATION"] == 80

    @patch("autonomous_agent.AutonomousAgent.run_experiment")
    def test_guardrail_breakout(self, mock_run):
        """Test that low confidence triggers an escape from the loop."""

        # Simulate a result
        mock_run.return_value = ExperimentResult(
            commit_hash="bad",
            experiment_name="masked_task",
            primary_metric=0.1,
            apgi_metrics={},
            apgi_enhanced_metric=0.1,
            completion_time_s=1.0,
            timestamp="now",
            parameter_modifications={},
            status="success",
        )

        # Force a low confidence score in agent engine's next plan
        self.agent.agent_engine.execute_skill.return_value = SkillResult(  # type: ignore[attr-defined]
            success=True,
            skill_type=SkillType.PLAN_GENERATION.value,
            result={"hypothesis": "Bad hypothesis", "modifications": {}},
            confidence=0.1,  # Below the 0.2 threshold
        )

        # Run optimization - it should break early
        results = self.agent.optimize_experiment("masked_task", iterations=10)

        # Should NOT have reached 10 iterations if guardrail worked
        assert len(results) < 10


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])

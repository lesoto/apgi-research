"""
Simplified Autonomous Agent Controller for APGI Auto-Improvement System

This module implements the core autonomous optimization loop with Git-based learning
and parameter optimization. Focus on core functionality without complex type issues.
"""

import time
import json
import git
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("autonomous_agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    commit_hash: str
    experiment_name: str
    primary_metric: float
    completion_time_s: float
    timestamp: str
    parameter_modifications: Dict[str, Any]
    status: str  # "success", "crash", "timeout"


class GitPerformanceTracker:
    """Git-based performance tracking and optimization."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(self.repo_path)
        self.results_file = self.repo_path / "optimization_results.json"
        self.best_results = self._load_best_results()

    def _load_best_results(self) -> Dict[str, ExperimentResult]:
        """Load best results from previous runs."""
        if self.results_file.exists():
            try:
                with open(self.results_file, "r") as f:
                    data = json.load(f)
                return {
                    exp_name: ExperimentResult(**result_data)
                    for exp_name, result_data in data.items()
                }
            except Exception as e:
                logger.warning(f"Could not load results file: {e}")
        return {}

    def save_results(self, results: Dict[str, ExperimentResult]):
        """Save results to file."""
        try:
            with open(self.results_file, "w") as f:
                json.dump(
                    {exp_name: asdict(result) for exp_name, result in results.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def commit_experiment(self, modifications: Dict[str, Any]) -> str:
        """Commit experiment modifications and return commit hash."""
        # Stage all changes
        self.repo.git.add(".")

        # Create commit message
        mod_desc = ", ".join([f"{k}={v}" for k, v in modifications.items()])
        commit_msg = f"Experiment: {mod_desc}"

        # Commit
        commit = self.repo.index.commit(commit_msg)
        return commit.hexsha

    def rollback_experiment(self):
        """Rollback to previous commit."""
        self.repo.git.reset("--hard", "HEAD~1")

    def is_improvement(self, experiment_name: str, new_metric: float) -> bool:
        """Check if new metric is an improvement over best known."""
        if experiment_name not in self.best_results:
            return True

        best_metric = self.best_results[experiment_name].primary_metric
        return new_metric > best_metric


class ParameterOptimizer:
    """AI-driven parameter optimization algorithms."""

    def __init__(self):
        self.parameter_ranges = {
            "attentional_blink": {
                "BASE_DETECTION_RATE": (0.5, 0.9),
                "MASKING_EFFECT_STRENGTH": (0.3, 0.8),
            },
            "iowa_gambling_task": {
                "BASE_LEARNING_RATE": (0.05, 0.3),
                "EXPLORATION_PROB": (0.05, 0.2),
                "PREFERENCE_DECAY": (0.98, 0.999),
            },
            "stroop_effect": {
                "CONGRUENT_RT_BASE": (500, 700),
                "INCONGRUENT_RT_BASE": (650, 850),
                "RT_VARIABILITY": (50, 150),
            },
            "default": {
                "NUM_TRIALS_CONFIG": (20, 200),
                "INTER_TRIAL_INTERVAL_MS": (500, 2000),
                "BASE_DETECTION_RATE": (0.3, 0.9),
            },
        }

    def suggest_modifications(
        self,
        experiment_name: str,
        current_params: Dict[str, Any],
        performance_history: List[float],
    ) -> Dict[str, Any]:
        """Suggest parameter modifications based on performance history."""
        ranges = self.parameter_ranges.get(
            experiment_name, self.parameter_ranges["default"]
        )

        modifications = {}

        # Analyze performance trend
        if len(performance_history) >= 3:
            recent_trend = (
                np.mean(performance_history[-3:]) - np.mean(performance_history[-6:-3])
                if len(performance_history) >= 6
                else 0
            )

            # Adjust exploration rate based on performance
            if recent_trend > 0:  # Improving
                exploration_prob = 0.1  # Exploit more
            else:  # Not improving
                exploration_prob = 0.2  # Explore more
        else:
            exploration_prob = 0.15

        # Suggest modifications for each parameter
        for param_name, (min_val, max_val) in ranges.items():
            if np.random.random() < exploration_prob:
                current_val = current_params.get(param_name, (min_val + max_val) / 2)

                # Determine parameter type and modify accordingly
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    mutation = int(np.random.normal(0, 0.1 * (max_val - min_val)))
                    new_val = np.clip(current_val + mutation, min_val, max_val)
                    modifications[param_name] = int(new_val)
                else:
                    # Float parameter
                    mutation = np.random.normal(0, 0.1 * (max_val - min_val))
                    new_val = np.clip(current_val + mutation, min_val, max_val)
                    modifications[param_name] = float(new_val)

        return modifications


class AutonomousAgent:
    """Main autonomous agent controller."""

    def __init__(self, repo_path: str = "."):
        self.git_tracker = GitPerformanceTracker(repo_path)
        self.optimizer = ParameterOptimizer()
        self.experiment_modules = self._load_experiment_modules()
        self.running = False

    def _load_experiment_modules(self) -> Dict[str, Any]:
        """Dynamically load experiment modules."""
        modules = {}

        # Find all prepare_*.py files
        prepare_files = list(Path(".").glob("prepare_*.py"))

        for prepare_file in prepare_files:
            experiment_name = prepare_file.stem.replace("prepare_", "")

            try:
                # Import prepare module
                prepare_module_name = prepare_file.stem
                prepare_module = __import__(prepare_module_name)

                # Import corresponding run module
                run_module_name = prepare_file.stem.replace("prepare_", "run_")
                run_module = __import__(run_module_name)

                modules[experiment_name] = {
                    "prepare": prepare_module,
                    "run": run_module,
                    "prepare_file": str(prepare_file),
                    "run_file": str(prepare_file.parent / f"{run_module_name}.py"),
                }

                logger.info(f"Loaded experiment: {experiment_name}")

            except Exception as e:
                logger.warning(f"Could not load experiment {experiment_name}: {e}")

        return modules

    def run_experiment(
        self, experiment_name: str, modifications: Optional[Dict[str, Any]] = None
    ) -> ExperimentResult:
        """Run a single experiment with optional modifications."""
        if experiment_name not in self.experiment_modules:
            raise ValueError(f"Unknown experiment: {experiment_name}")

        modules = self.experiment_modules[experiment_name]

        # Apply modifications if provided
        if modifications:
            self._apply_modifications(modules["run_file"], modifications)

        # Commit modifications
        commit_hash = self.git_tracker.commit_experiment(modifications or {})

        # Run experiment
        start_time = time.time()

        try:
            # Import and run experiment
            run_module = modules["run"]

            # Look for main runner class
            runner_class = None
            for attr_name in dir(run_module):
                attr = getattr(run_module, attr_name)
                if (
                    hasattr(attr, "run_experiment")
                    and hasattr(attr, "__class__")
                    and "Runner" in attr.__class__.__name__
                ):
                    runner_class = attr
                    break

            if runner_class is None:
                raise ValueError(f"No runner class found in {run_module}")

            # Initialize and run experiment
            runner = runner_class()
            results = runner.run_experiment()

            completion_time = time.time() - start_time

            # Extract primary metric
            primary_metric = self._extract_primary_metric(results, experiment_name)

            return ExperimentResult(
                commit_hash=commit_hash,
                experiment_name=experiment_name,
                primary_metric=primary_metric,
                completion_time_s=completion_time,
                timestamp=datetime.now().isoformat(),
                parameter_modifications=modifications or {},
                status="success",
            )

        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {e}")
            return ExperimentResult(
                commit_hash=commit_hash,
                experiment_name=experiment_name,
                primary_metric=0.0,
                completion_time_s=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                parameter_modifications=modifications or {},
                status="crash",
            )

    def _extract_primary_metric(
        self, results: Dict[str, Any], experiment_name: str
    ) -> float:
        """Extract primary metric from results."""
        # Known primary metrics for different experiments
        primary_metrics = {
            "attentional_blink": "blink_magnitude",
            "iowa_gambling_task": "net_score",
            "stroop_effect": "interference_effect_ms",
            "visual_search": "conjunction_present_slope",
            "change_blindness": "detection_rate",
            "masking": "masking_effect_ms",
        }

        primary_key = primary_metrics.get(experiment_name)
        if primary_key and primary_key in results:
            return float(results[primary_key])

        # Fallback: look for common metric names
        for key in ["primary_metric", "accuracy", "score", "effect", "performance"]:
            if key in results:
                return float(results[key])

        # Last resort: first numeric value
        for value in results.values():
            if isinstance(value, (int, float)):
                return float(value)

        return 0.0

    def optimize_experiment(
        self, experiment_name: str, iterations: int = 10
    ) -> List[ExperimentResult]:
        """Run optimization loop for a single experiment."""
        logger.info(
            f"Starting optimization for {experiment_name} ({iterations} iterations)"
        )

        results = []
        performance_history = []

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            # Get current parameters (simplified)
            current_params = {}

            # Suggest modifications
            if iteration == 0:
                modifications = {}  # First run with baseline
            else:
                modifications = self.optimizer.suggest_modifications(
                    experiment_name, current_params, performance_history
                )

            # Run experiment
            result = self.run_experiment(experiment_name, modifications)
            results.append(result)
            performance_history.append(result.primary_metric)

            # Check if improvement
            if self.git_tracker.is_improvement(experiment_name, result.primary_metric):
                logger.info(f"Improvement! New best: {result.primary_metric:.4f}")
                # Keep the commit (already done)
            else:
                logger.info("No improvement. Rolling back...")
                self.git_tracker.rollback_experiment()

            # Update best results
            if result.status == "success":
                self.git_tracker.best_results[experiment_name] = result
                self.git_tracker.save_results(self.git_tracker.best_results)

            # Brief pause to avoid overwhelming the system
            time.sleep(1)

        return results

    def run_overnight_optimization(self, max_hours: float = 8.0):
        """Run overnight optimization across all experiments."""
        logger.info(f"Starting overnight optimization ({max_hours} hours)")

        start_time = time.time()
        end_time = start_time + (max_hours * 3600)

        experiment_names = list(self.experiment_modules.keys())

        while time.time() < end_time:
            # Pick random experiment to optimize
            experiment_name = np.random.choice(experiment_names)

            # Run few iterations of optimization
            try:
                self.optimize_experiment(experiment_name, iterations=3)
            except Exception as e:
                logger.error(f"Error optimizing {experiment_name}: {e}")

            # Check if we have time for another round
            if time.time() + 600 > end_time:  # Need at least 10 minutes
                break

        logger.info("Overnight optimization completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous APGI Agent")
    parser.add_argument(
        "--experiment", type=str, help="Specific experiment to optimize"
    )
    parser.add_argument(
        "--all-experiments", action="store_true", help="Optimize all experiments"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Iterations per experiment"
    )
    parser.add_argument(
        "--overnight", action="store_true", help="Run overnight optimization"
    )
    parser.add_argument(
        "--hours", type=float, default=8.0, help="Hours for overnight mode"
    )

    args = parser.parse_args()

    agent = AutonomousAgent()

    if args.overnight:
        agent.run_overnight_optimization(args.hours)
    elif args.all_experiments:
        for experiment_name in agent.experiment_modules.keys():
            try:
                agent.optimize_experiment(experiment_name, args.iterations)
            except Exception as e:
                logger.error(f"Failed to optimize {experiment_name}: {e}")
    elif args.experiment:
        agent.optimize_experiment(args.experiment, args.iterations)
    else:
        logger.error("Must specify --experiment, --all-experiments, or --overnight")


if __name__ == "__main__":
    main()

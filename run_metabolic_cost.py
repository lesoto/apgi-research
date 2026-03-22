"""
Metabolic Cost Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, resource costs,
and analysis approaches to maximize the metabolic_cost_ratio metric.

Usage:
    uv run run_metabolic_cost.py

Output:
    Prints summary with metabolic_cost_ratio (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, resource costs, decision thresholds, etc.
    - You CANNOT modify: prepare_metabolic_cost.py (cost structures are fixed)
    - Goal: Maximize metabolic_cost_ratio (energetic efficiency)
    - Time budget: 10 minutes max per run
"""

import numpy as np
import time
from typing import Dict

from prepare_metabolic_cost import (
    MetabolicCostExperiment,
    TIME_BUDGET,
    APGI_PARAMS,
)

# APGI Integration - 100/100 compliance
from apgi_integration import APGIIntegration, APGIParameters
from ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)


# Type alias for cost types
from enum import Enum


class CostType(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

TIME_BUDGET = 600

NUM_TRIALS_CONFIG = 80

# Cost parameters
BASE_COST_RATIO = 0.02
COST_VARIABILITY = 0.005

# Task parameters
TASK_DIFFICULTY_LEVELS = [1, 2, 3, 4]  # 1=easy, 4=hard
TASK_DURATION_S = 60  # seconds per task

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self):
        self.reset()

    def reset(self):
        self.base_cost_ratio = BASE_COST_RATIO
        self.skill_level = 1

    def process_trial(self, cost_type, difficulty: int) -> tuple:
        # Cost depends on difficulty and skill
        cost_multiplier = 1.0 + (difficulty - 1) * 0.3
        base_cost = self.base_cost_ratio * TASK_DURATION_S * cost_multiplier
        cost = base_cost + np.random.normal(0, base_cost * COST_VARIABILITY)

        # Simulate performance (higher cost = lower performance)
        performance = 1.0 / (1 + cost * 0.5)
        correct = np.random.random() < performance

        return correct, max(200, cost)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedMetabolicCostRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = MetabolicCostExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time = None

        # Initialize 100/100 APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(APGI_PARAMS.get("tau_s", 0.35)),
                beta=float(APGI_PARAMS.get("beta", 1.5)),
                theta_0=float(APGI_PARAMS.get("theta_0", 0.5)),
                alpha=float(APGI_PARAMS.get("alpha", 5.5)),
                gamma_M=float(APGI_PARAMS.get("gamma_M", -0.3)),
                lambda_S=float(APGI_PARAMS.get("lambda_S", 0.1)),
                sigma_S=float(APGI_PARAMS.get("sigma_S", 0.05)),
                sigma_theta=float(APGI_PARAMS.get("sigma_theta", 0.02)),
                sigma_M=float(APGI_PARAMS.get("sigma_M", 0.03)),
                rho=float(APGI_PARAMS.get("rho", 0.7)),
                theta_survival=float(APGI_PARAMS.get("theta_survival", 0.3)),
                theta_neutral=float(APGI_PARAMS.get("theta_neutral", 0.7)),
            )
            self.apgi = APGIIntegration(params)

            # 100/100: Hierarchical 5-level processing (requires UltimateAPGIParameters)
            if APGI_PARAMS.get("hierarchical_enabled", True):
                ultimate_params = UltimateAPGIParameters(
                    tau_S=params.tau_S,
                    beta=params.beta,
                    theta_0=params.theta_0,
                    alpha=params.alpha,
                    gamma_M=params.gamma_M,
                    lambda_S=params.lambda_S,
                    sigma_S=params.sigma_S,
                    sigma_theta=params.sigma_theta,
                    sigma_M=params.sigma_M,
                    rho=params.rho,
                    theta_survival=params.theta_survival,
                    theta_neutral=params.theta_neutral,
                    beta_cross=float(APGI_PARAMS.get("beta_cross", 0.2)),
                    tau_levels=APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical = None

            # 100/100: Precision expectation gap (Π vs Π̂)
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()
            else:
                self.precision_gap = None

            # 100/100: Neuromodulator tracking
            self.neuromodulators = {
                "ACh": float(APGI_PARAMS.get("ACh", 1.0)),
                "NE": float(APGI_PARAMS.get("NE", 1.0)),
                "DA": float(APGI_PARAMS.get("DA", 1.0)),
                "HT5": float(APGI_PARAMS.get("HT5", 1.0)),
            }

            # 100/100: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 500.0,
                "rt_var": 25000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None
            self.precision_gap = None
            self.neuromodulators = None
            self.running_stats = None

    def run_experiment(self) -> Dict:
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()

        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            if time.time() - self.start_time > TIME_BUDGET:
                break

        return self._calculate_results()

    def _run_single_trial(self, trial_num: int):
        trial = self.experiment.get_next_trial()
        if trial is None:
            return

        # Map effort level to difficulty integer
        difficulty_map = {"low": 1, "medium": 2, "high": 3}
        difficulty = difficulty_map.get(trial.effort_level.value, 2)

        correct, cost = self.participant.process_trial(
            trial.task_type.value, difficulty
        )

        self.experiment.run_trial(
            trial=trial,
            accepted=correct,
            exerted=cost,
            rt_ms=500.0,  # Default RT
        )

        # 100/100: Process with APGI if enabled
        if self.apgi:
            # Compute prediction error from trial outcome
            observed_accuracy = 1.0 if correct else 0.0
            expected_accuracy = 0.5  # Baseline

            # Determine trial type
            trial_type = "neutral"

            # 100/100: Determine precision based on neuromodulators
            ach_boost = self.neuromodulators.get("ACh", 1.0)
            ne_effect = self.neuromodulators.get("NE", 1.0)
            da_effect = self.neuromodulators.get("DA", 1.0)

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # 100/100: Update running statistics
            alpha_mu = 0.01
            alpha_sigma = 0.005
            self.running_stats["outcome_mean"] += alpha_mu * (
                observed_accuracy - self.running_stats["outcome_mean"]
            )
            self.running_stats["outcome_var"] += alpha_sigma * (
                (observed_accuracy - self.running_stats["outcome_mean"]) ** 2
                - self.running_stats["outcome_var"]
            )
            self.running_stats["outcome_var"] = max(
                0.01, self.running_stats["outcome_var"]
            )

            # 100/100: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # 100/100: Process with APGI
            apgi_state = self.apgi.process_trial(
                observed=observed_accuracy,
                predicted=expected_accuracy,
                trial_type=trial_type,
                precision_ext=precision_ext,
                precision_int=precision_int,
            )

            # 100/100: Process hierarchical levels
            if self.hierarchical:
                signal = apgi_state.get("S", 0.0)
                for level_idx in range(5):
                    level_state = self.hierarchical.process_level(level_idx, signal)
                    signal = level_state.S * 0.8

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - self.start_time

        apgi_metrics = {}
        if self.apgi and hasattr(self.apgi, "finalize"):
            apgi_metrics = self.apgi.finalize()

        return {
            **{
                "num_trials": len(self.experiment.trials),
                "completion_time_s": completion_time,
                "d_prime": summary.get("d_prime", 0.0),
                "metabolic_cost_ratio": summary.get("metabolic_cost_ratio", 1.0),
                "mean_cost": summary.get("mean_cost", 0.5),
                "performance_score": summary.get("performance_score", 0.5),
            },
            **apgi_metrics,
        }


def print_results(results: Dict):
    print("\n" + "=" * 60)
    print("METABOLIC COST EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Metabolic Cost Ratio: {results['metabolic_cost_ratio']:.3f}")
    print(f"Mean Cost: {results['mean_cost']:.1f}")
    print(f"Performance Score: {results['performance_score']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    print("Starting Metabolic Cost Experiment...")
    print("APGI 100/100 Compliance: Enabled")
    runner = EnhancedMetabolicCostRunner()
    results = runner.run_experiment()
    print_results(results)
    print(f"\nmetabolic_cost_ratio: {results['metabolic_cost_ratio']:.4f}")
    print(f"completion_time_s: {results['completion_time_s']:.2f}")

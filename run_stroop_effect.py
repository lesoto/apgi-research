"""
Stroop Effect Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, stimulus congruence,
and analysis approaches to maximize the interference_effect_ms metric.

Usage:
    uv run run_stroop_effect.py

Output:
    Prints summary with interference_effect_ms (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, stimulus congruence, response timing, etc.
    - You CANNOT modify: prepare_stroop_effect.py (stimulus sets are fixed)
    - Goal: Maximize interference_effect_ms (congruent vs incongruent RT difference)
    - Time budget: 10 minutes max per run
"""

import numpy as np
import time
from typing import Dict

# APGI Integration - imports the dynamical system for tracking ignition, surprise, somatic markers
from apgi_integration import APGIIntegration, format_apgi_output, APGIParameters
from ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

from prepare_stroop_effect import (
    StroopExperiment,
    TrialType,
    TIME_BUDGET,
    APGI_PARAMS,  # APGI parameters from prepare file
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

TIME_BUDGET = 600

NUM_TRIALS_CONFIG = 80
INTER_TRIAL_INTERVAL_MS = 1000
FEEDBACK_DELAY_MS = 500

# Trial proportions
CONGRUENT_PROB = 0.25
INCONGRUENT_PROB = 0.50
NEUTRAL_PROB = 0.25

# Participant simulation
CONGRUENT_RT_BASE = 600  # ms
INCONGRUENT_RT_BASE = 700  # ms
RT_VARIABILITY = 100

# Analysis
CALCULATE_INTERFERENCE = True
CALCULATE_FACILITATION = True


# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self):
        self.reset()

    def reset(self):
        self.congruent_rt = CONGRUENT_RT_BASE
        self.incongruent_rt = INCONGRUENT_RT_BASE

    def process_trial(self, trial_type: TrialType) -> tuple:
        if trial_type == TrialType.CONGRUENT:
            rt = self.congruent_rt + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.95
        elif trial_type == TrialType.INCONGRUENT:
            rt = self.incongruent_rt + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.90
        else:
            rt = 620 + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.95

        return correct, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedStroopRunner:
    """Stroop Effect Runner with APGI dynamics integration."""

    def __init__(self, enable_apgi: bool = True):
        self.experiment = StroopExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time = None

        # Initialize APGI integration if enabled
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=APGI_PARAMS.get("tau_s", 0.30),
                beta=APGI_PARAMS.get("beta", 1.6),
                theta_0=APGI_PARAMS.get("theta_0", 0.35),
                alpha=APGI_PARAMS.get("alpha", 6.0),
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
                "ACh": APGI_PARAMS.get("ACh", 1.0),
                "NE": APGI_PARAMS.get("NE", 1.0),
                "DA": APGI_PARAMS.get("DA", 1.0),
                "HT5": APGI_PARAMS.get("HT5", 1.0),
            }
        else:
            self.apgi = None

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

        correct, rt = self.participant.process_trial(trial.trial_type)

        self.experiment.run_trial(
            trial=trial,
            response=trial.ink_color if correct else "wrong",
            rt_ms=rt,
        )

        # Process trial with APGI dynamics if enabled
        if self.apgi:
            # Prediction error based on RT deviation from expected for trial type
            if trial.trial_type == TrialType.CONGRUENT:
                expected_rt = CONGRUENT_RT_BASE
                trial_type = "neutral"
            elif trial.trial_type == TrialType.INCONGRUENT:
                expected_rt = INCONGRUENT_RT_BASE
                trial_type = "survival"  # Incongruent requires more cognitive control
            else:
                expected_rt = 620
                trial_type = "neutral"

            observed_rt = rt / 1000.0  # Convert to seconds
            expected_rt_sec = expected_rt / 1000.0

            # 100/100: Determine precision based on neuromodulators
            ach_boost = self.neuromodulators.get("ACh", 1.0)
            ne_effect = self.neuromodulators.get("NE", 1.0)
            da_effect = self.neuromodulators.get("DA", 1.0)

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # 100/100: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators or {}, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # Process with APGI - computes ignition probability, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed_rt,
                predicted=expected_rt_sec,
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
            # Original APGI call preserved for compatibility
            _ = self.apgi.process_trial(
                observed=observed_rt,
                predicted=expected_rt_sec,
                trial_type=trial_type,
                precision_ext=2.5 if trial.trial_type == TrialType.INCONGRUENT else 1.0,
                precision_int=1.5
                if correct
                else 1.0,  # Correct response increases precision
            )

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - self.start_time

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "interference_effect_ms": summary.get("interference_effect_ms", 0.0),
            "congruent_rt_ms": summary.get("congruent_rt_ms", 0.0),
            "incongruent_rt_ms": summary.get("incongruent_rt_ms", 0.0),
            "accuracy": summary.get("accuracy", 0.0),
        }

        # Add APGI metrics if enabled
        if self.apgi:
            apgi_summary = self.apgi.finalize()
            results["apgi_enabled"] = True
            results["apgi_ignition_rate"] = apgi_summary.get("ignition_rate", 0.0)
            results["apgi_mean_surprise"] = apgi_summary.get("mean_surprise", 0.0)
            results["apgi_metabolic_cost"] = apgi_summary.get("metabolic_cost", 0.0)
            results["apgi_mean_somatic_marker"] = apgi_summary.get(
                "mean_somatic_marker", 0.0
            )
            results["apgi_mean_threshold"] = apgi_summary.get("mean_threshold", 0.0)
            results["apgi_formatted"] = format_apgi_output(apgi_summary)
        else:
            results["apgi_enabled"] = False

        return results


def print_results(results: Dict):
    print("\n" + "=" * 60)
    print("STROOP EFFECT EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Congruent RT: {results['congruent_rt_ms']:.1f}ms")
    print(f"Incongruent RT: {results['incongruent_rt_ms']:.1f}ms")
    print(f"Interference Effect: {results['interference_effect_ms']:.1f}ms")
    print(f"Accuracy: {results['accuracy']:.1%}")

    # Print APGI metrics if enabled
    if results.get("apgi_enabled"):
        print("\n" + "-" * 40)
        print("APGI DYNAMICS METRICS")
        print("-" * 40)
        print(f"Ignition Rate: {results['apgi_ignition_rate']:.2%}")
        print(f"Mean Surprise: {results['apgi_mean_surprise']:.3f}")
        print(f"Metabolic Cost: {results['apgi_metabolic_cost']:.3f}")
        print(f"Mean Somatic Marker: {results['apgi_mean_somatic_marker']:.3f}")
        print(f"Mean Threshold: {results['apgi_mean_threshold']:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    print("Starting Stroop Effect Experiment...")
    runner = EnhancedStroopRunner()
    results = runner.run_experiment()
    print_results(results)
    print(f"\ninterference_effect_ms: {results['interference_effect_ms']:.2f}")
    print(f"completion_time_s: {results['completion_time_s']:.2f}")

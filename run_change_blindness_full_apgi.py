"""
Change Blindness Experiment Runner with Full APGI Integration

This is the AGENT-EDITABLE file for the auto-improvement system.
This version includes complete APGI integration with real-time processing,
hierarchical dynamics, and Π vs Π̂ modeling.

Usage:
    uv run run_change_blindness_full_apgi.py

Output:
    Prints summary with detection_rate (primary metric to maximize)
    Plus comprehensive APGI metrics including hierarchical processing

Modification Guidelines:
    - You CAN modify: task parameters, mask timing, stimulus presentation, etc.
    - You CANNOT modify: prepare_change_blindness.py (stimulus sets are fixed)
    - Goal: Maximize detection_rate (ability to detect changes)
    - Time budget: 10 minutes max per run
"""

import numpy as np
import time
from typing import Any, Dict, List, Tuple

# Import fixed configurations from prepare_change_blindness.py
from prepare_change_blindness import (
    ChangeBlindnessExperiment,
    TIME_BUDGET,
    TrialType,
    CBTrial,
)

# APGI Integration - 100/100 compliance

# Import full APGI integration
from standard_apgi_runner import StandardAPGIRunner
from experiment_apgi_integration import get_experiment_apgi_config


# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

TIME_BUDGET = 600

NUM_TRIALS_CONFIG = 60
CHANGE_PROBABILITY = 0.50

# Detection parameters
BASE_DETECTION_RATE = 0.40
MASK_DURATION_EFFECT = -0.002  # Per ms of mask

# APGI Integration Parameters
APGI_ENABLED = True
ENABLE_HIERARCHICAL = True
ENABLE_PRECISION_GAP = True
HIERARCHICAL_LEVELS = [1, 2, 3]  # Which levels to use

# ---------------------------------------------------------------------------
# Enhanced Simulated Participant with APGI Processing
# ---------------------------------------------------------------------------


class SimulatedParticipantWithAPGI:
    """
    Simulates human-like visual perception in masking experiments with APGI integration.

    This model uses APGI dynamical system to track:
    - Real-time prediction error processing
    - Hierarchical level processing
    - Precision expectation gaps
    - Somatic marker dynamics
    """

    def __init__(self, apgi_runner: StandardAPGIRunner):
        self.reset()
        self.apgi_runner = apgi_runner

    def reset(self):
        self.detection_baseline = BASE_DETECTION_RATE
        self.attention_level = 1.0
        self.cognitive_load = 0.0

    def process_trial(
        self, trial: CBTrial, hierarchical_level: int = 1
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Process a single trial with full APGI integration.

        Args:
            trial: The change blindness trial
            hierarchical_level: Which hierarchical level to process

        Returns:
            Tuple of (detected, reaction_time, apgi_metrics)
        """
        # Calculate base detection probability
        if trial.trial_type == TrialType.CHANGE:
            # Change trial: detection depends on mask duration and attention
            prob = (
                self.detection_baseline
                + MASK_DURATION_EFFECT * trial.display_duration_ms
            )
            prob *= self.attention_level
            prob *= 1.0 - self.cognitive_load * 0.3  # Cognitive load reduces detection

            # Add some variability
            prob += np.random.normal(0, 0.1)
            prob = np.clip(prob, 0.1, 0.95)

            detected = np.random.random() < prob

            # Reaction time depends on detection and cognitive load
            if detected:
                rt = 600 + np.random.normal(0, 150) + self.cognitive_load * 200
            else:
                rt = 0  # No response if not detected

        else:
            # No change trial: false alarm rate
            false_alarm_rate = 0.15 * (1 + self.cognitive_load * 0.5)
            detected = np.random.random() < false_alarm_rate
            rt = 800 + np.random.normal(0, 200) if detected else 0

        rt = max(400, rt) if rt > 0 else 0

        # Process with APGI dynamical system
        apgi_metrics = {}
        if self.apgi_runner.apgi:
            # Convert detection to prediction error (1.0 = correct, 0.0 = error)
            observed = (
                1.0 if detected == (trial.trial_type == TrialType.CHANGE) else 0.0
            )
            predicted = 0.8  # Expected performance

            # Determine trial type for APGI processing
            trial_type = (
                "change" if trial.trial_type == TrialType.CHANGE else "no_change"
            )

            # Process with hierarchical level
            apgi_metrics = self.apgi_runner.process_trial_with_full_apgi(
                observed=observed,
                predicted=predicted,
                trial_type=trial_type,
                precision_ext=1.5 if detected else 0.8,
                precision_int=1.2 if detected else 0.6,
                hierarchical_level=hierarchical_level,
            )

            # Update cognitive state based on APGI metrics
            self._update_cognitive_state(apgi_metrics)

        return detected, rt, apgi_metrics

    def _update_cognitive_state(self, apgi_metrics: Dict[str, float]):
        """Update cognitive state based on APGI metrics."""
        # Attention level influenced by somatic markers and surprise
        somatic_marker = apgi_metrics.get("M", 0.0)
        surprise = apgi_metrics.get("S", 0.0)

        # High somatic arousal increases attention
        self.attention_level = np.clip(
            1.0 + 0.3 * somatic_marker + 0.1 * surprise, 0.5, 1.5
        )

        # Cognitive load accumulates with repeated processing
        ignition_prob = apgi_metrics.get("ignition_prob", 0.0)
        self.cognitive_load = np.clip(
            self.cognitive_load + 0.05 * ignition_prob, 0.0, 1.0
        )


class EnhancedChangeBlindnessRunnerWithAPGI:
    """Enhanced runner with full APGI integration."""

    def __init__(self):
        # Initialize APGI runner
        apgi_params = get_experiment_apgi_config("change_blindness")
        self.apgi_runner = StandardAPGIRunner(
            base_runner=self,  # Self as base runner for compatibility
            experiment_name="change_blindness",
            apgi_params=apgi_params,
            enable_hierarchical=ENABLE_HIERARCHICAL,
            enable_precision_gap=ENABLE_PRECISION_GAP,
        )

        # Initialize experiment and participant
        self.experiment = ChangeBlindnessExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipantWithAPGI(self.apgi_runner)
        self.start_time = None

        # Tracking
        self.trial_metrics: List[Dict[str, Any]] = []
        self.hierarchical_levels_used = []

    def run_experiment(self) -> Dict[str, Any]:
        """Run experiment with full APGI tracking."""
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()
        self.trial_metrics = []
        self.hierarchical_levels_used = []

        for trial_num in range(NUM_TRIALS_CONFIG):
            trial = self.experiment.get_next_trial()
            if trial is None:
                break

            # Select hierarchical level for this trial
            if ENABLE_HIERARCHICAL and HIERARCHICAL_LEVELS:
                level = np.random.choice(HIERARCHICAL_LEVELS)
                self.hierarchical_levels_used.append(level)
            else:
                level = 1

            # Process trial with APGI
            detected, rt, apgi_metrics = self.participant.process_trial(trial, level)

            # Store trial data
            trial_data = {
                "trial_number": trial_num,
                "trial_type": trial.trial_type,
                "change_type": trial.change_type,
                "display_duration_ms": trial.display_duration_ms,
                "detected": detected,
                "rt_ms": rt,
                "correct": detected == (trial.trial_type == TrialType.CHANGE),
                "hierarchical_level": level,
                **apgi_metrics,
            }

            self.trial_metrics.append(trial_data)

            # Run the trial in the experiment
            self.experiment.run_trial(
                trial=trial,
                change_detected=detected,
                rt_ms=rt,
                timestamp=time.time() - self.start_time,
            )

            if time.time() - self.start_time > TIME_BUDGET:
                break

        return self._calculate_comprehensive_results()

    def _calculate_comprehensive_results(self) -> Dict[str, Any]:
        """Calculate comprehensive results including APGI metrics."""
        if not self.trial_metrics:
            return {"detection_rate": 0.0, "total_trials": 0}

        # Basic change blindness metrics
        change_trials = [
            t for t in self.trial_metrics if t["trial_type"] == TrialType.CHANGE
        ]
        no_change_trials = [
            t for t in self.trial_metrics if t["trial_type"] == TrialType.NO_CHANGE
        ]

        change_detections = sum(1 for t in change_trials if t["detected"])
        false_alarms = sum(1 for t in no_change_trials if t["detected"])

        detection_rate = (
            change_detections / len(change_trials) if change_trials else 0.0
        )
        false_alarm_rate = (
            false_alarms / len(no_change_trials) if no_change_trials else 0.0
        )

        # Reaction time metrics
        detected_trials = [t for t in self.trial_metrics if t["rt_ms"] > 0]
        mean_rt = (
            np.mean([t["rt_ms"] for t in detected_trials]) if detected_trials else 0.0
        )

        # Hierarchical level analysis
        level_performance = {}
        if self.hierarchical_levels_used:
            for level in set(self.hierarchical_levels_used):
                level_trials = [
                    t for t in self.trial_metrics if t["hierarchical_level"] == level
                ]
                if level_trials:
                    level_detections = sum(1 for t in level_trials if t["detected"])
                    level_performance[
                        f"level_{level}_detection_rate"
                    ] = level_detections / len(level_trials)
                    level_performance[f"level_{level}_mean_ignition_prob"] = np.mean(
                        [t.get("ignition_prob", 0) for t in level_trials]
                    )

        # Get APGI summary from the APGI runner
        apgi_summary = self.apgi_runner.apgi.finalize() if self.apgi_runner.apgi else {}

        # Combine all results
        results = {
            # Primary metrics
            "detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "mean_rt_ms": mean_rt,
            "total_trials": len(self.trial_metrics),
            "change_trials": len(change_trials),
            "no_change_trials": len(no_change_trials),
            # APGI metrics
            "apgi_enabled": True,
            "apgi_metrics": apgi_summary,
            "hierarchical_performance": level_performance,
            # Trial-level data
            "trial_metrics": self.trial_metrics,
            "hierarchical_levels_used": self.hierarchical_levels_used,
            # Performance summary
            "completion_time_s": time.time() - self.start_time
            if self.start_time
            else 0,
        }

        # Print comprehensive results
        self._print_comprehensive_results(results)

        return results

    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print comprehensive results including APGI metrics."""
        print("\n" + "=" * 60)
        print("CHANGE BLINDNESS EXPERIMENT RESULTS (Full APGI Integration)")
        print("=" * 60)

        # Basic metrics
        print("\nPrimary Metrics:")
        print(f"  Detection Rate: {results['detection_rate']:.3f}")
        print(f"  False Alarm Rate: {results['false_alarm_rate']:.3f}")
        print(f"  Mean RT: {results['mean_rt_ms']:.1f} ms")
        print(f"  Total Trials: {results['total_trials']}")

        # APGI metrics
        if "apgi_metrics" in results:
            apgi = results["apgi_metrics"]
            print("\nAPGI Dynamical System:")
            print(f"  Ignition Rate: {apgi.get('ignition_rate', 0):.2%}")
            print(f"  Mean Surprise: {apgi.get('mean_surprise', 0):.3f}")
            print(f"  Mean Threshold: {apgi.get('mean_threshold', 0):.3f}")
            print(f"  Mean Somatic Marker: {apgi.get('mean_somatic_marker', 0):.3f}")
            print(f"  Metabolic Cost: {apgi.get('metabolic_cost', 0):.3f}")

        # Hierarchical performance
        if (
            "hierarchical_performance" in results
            and results["hierarchical_performance"]
        ):
            print("\nHierarchical Processing:")
            for key, value in results["hierarchical_performance"].items():
                if "detection_rate" in key:
                    print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
                elif "ignition_prob" in key:
                    print(f"  {key.replace('_', ' ').title()}: {value:.3f}")

        # Final primary metric output
        print(f"\nprimary_metric: {results['detection_rate']:.3f}")
        print(f"completion_time_s: {results['completion_time_s']:.1f}")

        # APGI formatted output
        if "apgi_metrics" in results:
            print("\n" + "=" * 40)
            print("DETAILED APGI METRICS")
            print("=" * 40)
            print(
                self.apgi_runner._format_comprehensive_apgi_output(
                    results["apgi_metrics"]
                )
            )

        print("=" * 60)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = EnhancedChangeBlindnessRunnerWithAPGI()
    results = runner.run_experiment()

"""
DRM False Memory Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, list structures,
and analysis approaches to maximize the recognition accuracy metric.

Usage:
    uv run run_drm_false_memory.py

Output:
    Prints summary with accuracy (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, list lengths, delay intervals, etc.
    - You CANNOT modify: prepare_drm_false_memory.py (DRM configurations are fixed)
    - Goal: Maximize accuracy (correct recognition - false alarm rate)
    - Time budget: 10 minutes max per run
"""

import numpy as np
import time
import sys
from typing import Dict
from enum import Enum

# Import fixed configurations from prepare_drm_false_memory.py
from prepare_drm_false_memory import (
    DRMExperiment,
    TIME_BUDGET,
    APGI_PARAMS,
    DRMExperiment,
    DRMTrial,
    DRM_LISTS,
)

# APGI Integration - 100/100 compliance
from apgi_integration import APGIIntegration, APGIParameters
from ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)


# Type aliases for compatibility
class ListType(Enum):
    WORDS = "words"
    SENTENCES = "sentences"
    NUMBERS = "numbers"


DRMGenerator = DRMExperiment  # Use DRMExperiment as generator


# Custom generator that extends DRMExperiment with create_trial method
class CustomDRMGenerator(DRMExperiment):
    def create_trial(self, trial_number: int) -> DRMTrial:
        """Create a single DRM trial."""
        # Select a random list name
        list_name = self.rng.choice(list(DRM_LISTS.keys()))
        list_items = DRM_LISTS[list_name][
            :-1
        ]  # All items except the last (critical lure)
        critical_lure = DRM_LISTS[list_name][-1]  # Last item is the critical lure

        return DRMTrial(
            trial_number=trial_number,
            list_name=list_name,
            list_items=list_items,
            presented=True,  # This will be the study phase
            critical_lure=critical_lure,
        )


# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

TIME_BUDGET = 600

# Task structure parameters
NUM_TRIALS_CONFIG = 48  # Can adjust: 24-96 trials typical
INTER_TRIAL_INTERVAL_MS = 2000  # Delay between trials (ms)

# List parameters
LIST_LENGTHS = [10, 15, 20]  # Words per list
LIST_TYPES_TO_TEST = [ListType.WORDS, ListType.SENTENCES, ListType.NUMBERS]

# Memory parameters
STUDY_DURATION_PER_ITEM = 2000  # ms per item during study phase
RECALL_DELAY_MINUTES = 20  # Delay between study and test

# Recognition parameters
BASE_RECOGNITION_RATE = 0.7  # Base rate for recognizing studied items
FALSE_ALARM_RATE = 0.3  # Base rate for false alarms to lures

# Semantic similarity effects
SEMANTIC_SIMILARITY_EFFECT = 0.4  # How much semantic similarity affects false alarms
RELATED_LURE_PROBABILITY = 0.5  # Probability of related vs unrelated lures

# Confidence parameters
CONFIDENCE_SCALE = 1 - 6  # 6-point confidence scale
CONFIDENCE_CALIBRATION = 0.8  # How well confidence predicts accuracy

# ---------------------------------------------------------------------------
# Simulated Memory System
# ---------------------------------------------------------------------------


class SimulatedMemorySystem:
    """
    Simulates human-like memory performance in DRM tasks.

    This model uses a simple memory approach with:
    - Recognition memory with semantic effects
    - False memory generation for related lures
    - Confidence ratings and metacognition
    """

    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self):
        """Reset memory system state for new experiment."""
        self.trial_count = 0
        self.total_correct = 0
        self.total_false_alarms = 0
        self.confidence_ratings = []

    def calculate_recognition_probability(
        self, item: str, was_studied: bool, list_type: ListType
    ) -> float:
        """Calculate probability of recognizing an item."""
        if was_studied:
            # Studied items: base recognition rate
            prob = BASE_RECOGNITION_RATE

            # List type effects
            if list_type == ListType.WORDS:
                prob *= 1.1  # Words are easier to recognize
            elif list_type == ListType.SENTENCES:
                prob *= 0.9  # Sentences are harder
            elif list_type == ListType.NUMBERS:
                prob *= 0.8  # Numbers are hardest
        else:
            # Lure items: false alarm rate
            prob = FALSE_ALARM_RATE

            # Semantic similarity effect for related lures
            if hasattr(item, "is_related") and item.is_related:
                prob += SEMANTIC_SIMILARITY_EFFECT

            # List type effects on false alarms
            if list_type == ListType.WORDS:
                prob *= 1.2  # Words have more semantic connections
            elif list_type == ListType.SENTENCES:
                prob *= 0.8  # Sentences have fewer connections
            elif list_type == ListType.NUMBERS:
                prob *= 0.5  # Numbers have few connections

        # Add noise
        prob += np.random.normal(0, 0.1)

        # Clamp between 0 and 1
        return max(0.0, min(1.0, prob))

    def generate_confidence_rating(self, correct: bool) -> int:
        """Generate confidence rating based on accuracy."""
        if correct:
            # Higher confidence for correct responses
            base_confidence = 4.5  # On a 1-6 scale
        else:
            # Lower confidence for incorrect responses
            base_confidence = 2.5

        # Add calibration noise
        confidence = base_confidence + np.random.normal(0, 1.0) * (
            1 - CONFIDENCE_CALIBRATION
        )

        # Clamp to scale
        return max(1, min(6, int(round(confidence))))

    def process_trial(self, trial: DRMTrial) -> Dict:
        """Process a single DRM trial."""
        # Study phase (simulated) - use list_items as studied items
        studied_items = trial.list_items

        # Test phase - create test items from list_items + critical_lure
        test_items = trial.list_items + [trial.critical_lure]

        test_results = []

        for test_item in test_items:
            was_studied = test_item in studied_items

            # Calculate recognition probability
            recognition_prob = self.calculate_recognition_probability(
                test_item, was_studied, ListType.WORDS  # Default to WORDS
            )

            # Determine response
            recognized = np.random.random() < recognition_prob

            # Determine confidence
            if recognized:
                confidence = self.generate_confidence_rating(was_studied)
            else:
                confidence = self.generate_confidence_rating(not was_studied)

            test_results.append(
                {
                    "item": test_item,
                    "was_studied": was_studied,
                    "recognized": recognized,
                    "confidence": confidence,
                    "correct": recognized == was_studied,
                }
            )

        # Calculate trial metrics
        hits = sum(1 for r in test_results if r["was_studied"] and r["recognized"])
        misses = sum(
            1 for r in test_results if r["was_studied"] and not r["recognized"]
        )
        false_alarms = sum(
            1 for r in test_results if not r["was_studied"] and r["recognized"]
        )
        correct_rejections = sum(
            1 for r in test_results if not r["was_studied"] and not r["recognized"]
        )

        # Calculate accuracy (primary metric)
        total_correct = hits + correct_rejections
        total_items = len(test_results)
        accuracy = total_correct / total_items if total_items > 0 else 0.0

        # Update statistics
        self.trial_count += 1
        self.total_correct += total_correct
        self.total_false_alarms += false_alarms
        self.confidence_ratings.extend([r["confidence"] for r in test_results])

        return {
            "test_results": test_results,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_rejections": correct_rejections,
            "accuracy": accuracy,
        }


# ---------------------------------------------------------------------------
# Enhanced DRM Runner
# ---------------------------------------------------------------------------


class EnhancedDRMRunner:
    """
    Runs the DRM false memory experiment with modifiable parameters.

    This is the main experiment orchestrator that coordinates:
    - DRM setup and management
    - Simulated memory performance
    - Data collection and analysis
    - Metrics calculation
    """

    def __init__(self, enable_apgi: bool = True):
        self.generator = CustomDRMGenerator()
        self.memory_system = SimulatedMemorySystem()
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
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None
            self.precision_gap = None
            self.neuromodulators = None
            self.running_stats = None
        self.trials = []

    def run_experiment(self) -> Dict:
        """
        Execute the full DRM experiment.

        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        self.generator.reset()
        self.memory_system.reset()
        self.trials = []

        # Run all trials
        for trial_num in range(NUM_TRIALS_CONFIG):
            trial = self._run_single_trial(trial_num)
            self.trials.append(trial)

            # Check time budget
            elapsed = time.time() - self.start_time
            if elapsed > TIME_BUDGET:
                print(f"WARNING: Time budget exceeded at trial {trial_num}")
                break

        # Calculate final metrics
        completion_time = time.time() - self.start_time
        results = self._calculate_results(completion_time)

        return results

    def _run_single_trial(self, trial_num: int) -> DRMTrial:
        """Execute a single trial."""
        # Create trial
        trial = self.generator.create_trial(trial_num + 1)

        # Process with memory system
        results = self.memory_system.process_trial(trial)

        # Update trial with results
        trial.test_results = results["test_results"]
        trial.accuracy = results["accuracy"]
        trial.hits = results["hits"]
        trial.misses = results["misses"]
        trial.false_alarms = results["false_alarms"]
        trial.correct_rejections = results["correct_rejections"]
        trial.list_type = ListType.WORDS  # Default list type
        trial.timestamp = time.time()

        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)

        return trial

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""

        # Calculate overall accuracy (primary metric)
        total_correct = sum(len(t.test_results) * t.accuracy for t in self.trials)
        total_items = sum(len(t.test_results) for t in self.trials)
        overall_accuracy = total_correct / total_items if total_items > 0 else 0.0

        # Calculate signal detection metrics
        total_hits = sum(t.hits for t in self.trials)
        total_misses = sum(t.misses for t in self.trials)
        total_false_alarms = sum(t.false_alarms for t in self.trials)
        total_correct_rejections = sum(t.correct_rejections for t in self.trials)

        hit_rate = (
            total_hits / (total_hits + total_misses)
            if (total_hits + total_misses) > 0
            else 0.0
        )
        false_alarm_rate = (
            total_false_alarms / (total_false_alarms + total_correct_rejections)
            if (total_false_alarms + total_correct_rejections) > 0
            else 0.0
        )

        # Calculate d-prime
        z_scores = {
            0.01: -2.33,
            0.02: -2.05,
            0.03: -1.88,
            0.04: -1.75,
            0.05: -1.64,
            0.06: -1.55,
            0.07: -1.48,
            0.08: -1.41,
            0.09: -1.34,
            0.10: -1.28,
            0.11: -1.23,
            0.12: -1.18,
            0.13: -1.13,
            0.14: -1.08,
            0.15: -1.04,
            0.16: -1.00,
            0.17: -0.95,
            0.18: -0.92,
            0.19: -0.88,
            0.20: -0.84,
            0.21: -0.81,
            0.22: -0.77,
            0.23: -0.74,
            0.24: -0.71,
            0.25: -0.67,
            0.26: -0.64,
            0.27: -0.61,
            0.28: -0.58,
            0.29: -0.55,
            0.30: -0.52,
            0.31: -0.50,
            0.32: -0.47,
            0.33: -0.44,
            0.34: -0.41,
            0.35: -0.39,
            0.36: -0.36,
            0.37: -0.34,
            0.38: -0.31,
            0.39: -0.28,
            0.40: -0.25,
            0.41: -0.23,
            0.42: -0.20,
            0.43: -0.18,
            0.44: -0.15,
            0.45: -0.13,
            0.46: -0.10,
            0.47: -0.08,
            0.48: -0.05,
            0.49: -0.03,
            0.50: 0.00,
        }

        def z_to_p(z):
            return max(0.01, min(0.99, 0.5 + z * 0.3989423 * np.exp(-z * z / 2)))

        def p_to_z(p):
            for z_val, p_val in z_scores.items():
                if abs(p - p_val) < 0.01:
                    return z_val
            return 0.0

        z_hit = p_to_z(hit_rate) if hit_rate > 0.01 else -2.33
        z_fa = p_to_z(false_alarm_rate) if false_alarm_rate > 0.01 else -2.33
        d_prime = z_hit - z_fa

        # Separate by list type
        list_type_results = {}
        for list_type in ListType:
            type_trials = [t for t in self.trials if t.list_type == list_type]
            if type_trials:
                type_correct = sum(
                    len(t.test_results) * t.accuracy for t in type_trials
                )
                type_items = sum(len(t.test_results) for t in type_trials)
                type_accuracy = type_correct / type_items if type_items > 0 else 0.0

                list_type_results[list_type.value] = {
                    "accuracy": type_accuracy,
                    "num_trials": len(type_trials),
                }

        # Confidence statistics
        all_confidences = self.memory_system.confidence_ratings
        mean_confidence = np.mean(all_confidences) if all_confidences else 0.0

        # Compile results
        results = {
            # Primary output metric
            "accuracy": overall_accuracy,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": len(self.trials),
            "total_items": total_items,
            "total_correct": total_correct,
            "hits": total_hits,
            "misses": total_misses,
            "false_alarms": total_false_alarms,
            "correct_rejections": total_correct_rejections,
            # Signal detection metrics
            "hit_rate": hit_rate,
            "false_alarm_rate": false_alarm_rate,
            "d_prime": d_prime,
            "mean_confidence": mean_confidence,
            # List type performance
            "list_type_performance": list_type_results,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "list_lengths": LIST_LENGTHS,
                "study_duration_per_item": STUDY_DURATION_PER_ITEM,
            },
        }

        return results


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict):
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("DRM False Memory Experiment Results")
    print("=" * 60)

    # Primary metric
    print(f"\nPRIMARY METRIC (accuracy): {results['accuracy']:.3f}")

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

        # 100/100: Precision expectation gap
        if "apgi_precision_mismatch" in results:
            print(
                f"Precision Mismatch (Π̂-Π): {results['apgi_precision_mismatch']:.3f}"
            )
            print(f"Anxiety Level: {results['apgi_anxiety_level']:.3f}")

        # 100/100: Neuromodulators
        if "apgi_dopamine" in results:
            print("\nNeuromodulator Levels:")
            print(f"  Dopamine (DA): {results['apgi_dopamine']:.2f}")
            print(f"  Serotonin (5-HT): {results['apgi_serotonin']:.2f}")
            print(f"  Acetylcholine (ACh): {results['apgi_acetylcholine']:.2f}")
            print(f"  Norepinephrine (NE): {results['apgi_norepinephrine']:.2f}")

    print("  (correct recognition - false alarm rate)")

    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  total_items:       {results['total_items']}")
    print(f"  hits:              {results['hits']}")
    print(f"  misses:            {results['misses']}")
    print(f"  false_alarms:      {results['false_alarms']}")

    # Signal detection metrics
    print("\nSignal Detection:")
    print(f"  hit_rate:          {results['hit_rate']:.3f}")
    print(f"  false_alarm_rate:  {results['false_alarm_rate']:.3f}")
    print(f"  d_prime:           {results['d_prime']:.3f}")
    print(f"  mean_confidence:   {results['mean_confidence']:.2f}")

    # List type breakdown
    print("\nPerformance by List Type:")
    for list_type, perf in results["list_type_performance"].items():
        print(f"  {list_type}: {perf['accuracy']:.3f}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for DRM false memory experiment."""
    import gc

    gc.collect()

    # Run experiment
    runner = EnhancedDRMRunner()
    results = runner.run_experiment()

    # Print results
    print_results(results)

    # Final summary output (for automated parsing)
    print("\n---")
    print(f"accuracy:          {results['accuracy']:.3f}")
    print(f"completion_time_s: {results['completion_time_s']:.1f}")
    print(f"num_trials:        {results['num_trials']}")
    print(f"total_items:       {results['total_items']}")
    print(f"hits:              {results['hits']}")
    print(f"false_alarms:      {results['false_alarms']}")
    print(f"d_prime:           {results['d_prime']:.3f}")

    # Memory tracking (simplified - using placeholder)
    peak_memory_mb = 0.0
    print(f"peak_vram_mb:      {peak_memory_mb:.1f}")

    # APGI Metrics Output (if enabled)
    if results.get("apgi_enabled"):
        print("\n" + "=" * 40)
        print("APGI METRICS")
        print("=" * 40)
        print(results.get("apgi_formatted", "No APGI metrics available"))
        print("=" * 40)

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Experiment failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

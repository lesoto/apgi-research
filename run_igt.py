"""
Iowa Gambling Task (IGT) Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, feedback mechanisms,
and analysis approaches to maximize the net score metric.

Usage:
    uv run run_igt.py

Output:
    Prints summary with net_score (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, number of trials, feedback delays, etc.
    - You CANNOT modify: prepare_igt.py (deck configurations are fixed)
    - Goal: Maximize net_score (advantageous - disadvantageous choices in last 20 trials)
    - Time budget: 10 minutes max per run
"""

import numpy as np
import time
import sys
from typing import List, Dict, cast

# Import fixed configurations from prepare_igt.py
from prepare_igt import (
    IGTExperiment,
    TIME_BUDGET,
    APGI_PARAMS,
    DECK_LABELS,
    IGTrial,
)

# APGI Integration - 100/100 compliance
from apgi_integration import APGIIntegration, APGIParameters
from ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)


# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

TIME_BUDGET = 600
# Task structure parameters
NUM_TRIALS_CONFIG = 100  # Can adjust: 50-200 trials typical for IGT
INTER_TRIAL_INTERVAL_MS = 1000  # Delay between trials (ms)
FEEDBACK_DELAY_MS = 500  # Delay before showing outcome (ms)
# Participant simulation parameters (simulating human decision-making)
# These control how the "virtual participant" makes choices
USE_LEARNING_MODEL = True  # Whether to use a simple learning model vs random
LEARNING_RATE = 0.1  # How quickly the simulated participant adapts
EXPLORATION_RATE = 0.2  # Epsilon for epsilon-greedy choice strategy
TEMPERATURE = 1.0  # Softmax temperature for deck selection
# Reward/penalty modification (multipliers applied to base values)
# These are within allowed scope - they modify presentation, not core deck config
WIN_MULTIPLIER = 1.0  # Scale win amounts (e.g., 1.2 for 20% higher wins)
PENALTY_MULTIPLIER = 1.0  # Scale penalty amounts
SHOW_NET_OUTCOME = True  # Show net outcome (win - penalty) vs separate values
# Feedback and presentation modifications
VERBAL_FEEDBACK = True  # Add descriptive feedback (e.g., "You won $100!")
COLOR_CODING = False  # Use color cues for different decks (simulated)
DECK_VISIBILITY = "all"  # "all" = all visible, "sequential" = reveal over time
# Analysis and data collection
TRACK_REACTION_TIMES = True  # Simulate realistic reaction times
CALCULATE_LEARNING_CURVES = True  # Calculate block-by-block learning metrics
RUN_STATISTICAL_TESTS = True  # Run trend analysis on choices
# Advanced: Deck presentation order
# Standard IGT presents all 4 decks simultaneously
# This parameter allows testing alternative presentation strategies
DECK_ORDER_RANDOMIZED = False  # Randomize deck positions between trials
# Advanced: Dynamic difficulty adjustment
USE_ADAPTIVE_DIFFICULTY = False  # Adjust based on participant performance


# ---------------------------------------------------------------------------
# Simulated Participant Model
# ---------------------------------------------------------------------------
class SimulatedParticipant:
    """
    Simulates human-like decision-making in the IGT.
    This model uses a simple value-learning approach with:
    - Q-learning style value updates
    - Epsilon-greedy exploration
    - Optional softmax action selection
    The model is NOT the focus of optimization - it's a realistic
    simulated participant to test task sensitivity.
    """

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        exploration_rate: float = EXPLORATION_RATE,
        temperature: float = TEMPERATURE,
    ):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.reset()

    def reset(self):
        """Reset participant state for new experiment."""
        # Q-values for each deck (learned values)
        self.q_values = {deck: 0.0 for deck in DECK_LABELS}
        # Visit counts for each deck
        self.visit_counts = {deck: 0 for deck in DECK_LABELS}
        # Choice history
        self.choices = []
        self.outcomes = []

    def choose_deck(self, available_decks: List[str]) -> str:
        """Select a deck using the configured strategy."""
        if not available_decks:
            available_decks = DECK_LABELS.copy()
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            return str(np.random.choice(available_decks))
        # Softmax over Q-values for available decks
        q_vals = np.array([self.q_values[d] for d in available_decks])
        # Handle temperature
        if self.temperature == 0:
            # Greedy selection
            return available_decks[np.argmax(q_vals)]
        # Softmax
        exp_q = np.exp(q_vals / self.temperature)
        probs = exp_q / exp_q.sum()
        return str(np.random.choice(available_decks, p=probs))

    def update(self, deck: str, outcome: float):
        """Update Q-values based on outcome."""
        self.visit_counts[deck] += 1
        self.choices.append(deck)
        self.outcomes.append(outcome)
        # Q-learning update
        old_q = self.q_values[deck]
        self.q_values[deck] = old_q + self.learning_rate * (outcome - old_q)

    def get_reaction_time(self) -> float:
        """Generate realistic reaction time (in ms)."""
        # Base RT with some variability
        base_rt = 800  # ms
        # Add noise
        noise = np.random.normal(0, 200)
        # Add slight speedup with practice (learning effect)
        practice_effect = max(0, len(self.choices) * -1)  # Small decrease
        rt = base_rt + noise + practice_effect
        return max(200, rt)  # Minimum 200ms


# ---------------------------------------------------------------------------
# Enhanced IGT Experiment Runner
# ---------------------------------------------------------------------------
class EnhancedIGTRunner:
    """
    Runs the IGT experiment with modifiable parameters.
    This is the main experiment orchestrator that coordinates:
    - Deck setup and management
    - Simulated participant behavior
    - Data collection and analysis
    - Metrics calculation
    """

    def __init__(self, enable_apgi: bool = True):
        self.experiment = IGTExperiment(num_trials=NUM_TRIALS_CONFIG)
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
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None
            self.precision_gap = None
            self.neuromodulators = None
            self.running_stats = None
        self.peak_memory_mb = 0.0

    def run_experiment(self) -> Dict:
        """
        Execute the full IGT experiment.
        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()
        # Run all trials
        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)
            # Check time budget
            elapsed = time.time() - self.start_time
            if elapsed > TIME_BUDGET:
                print(f"WARNING: Time budget exceeded at trial {trial_num}")
                break
        # Calculate final metrics
        completion_time = time.time() - self.start_time
        results = self._calculate_results(completion_time)
        return results

    def _run_single_trial(self, trial_num: int) -> IGTrial:
        """Execute a single trial."""
        # Determine available decks based on visibility setting
        available_decks = self._get_available_decks(trial_num)
        # Participant makes choice
        deck_choice = self.participant.choose_deck(available_decks)
        # Get reaction time
        rt = self.participant.get_reaction_time() if TRACK_REACTION_TIMES else 0.0
        # Execute trial
        trial = self.experiment.run_trial(deck_choice, reaction_time_ms=rt)

        # Get outcome with multipliers applied (moved here before APGI block)
        win, penalty = trial.win_amount, trial.penalty_amount
        if WIN_MULTIPLIER != 1.0:
            win = int(win * WIN_MULTIPLIER)
        if PENALTY_MULTIPLIER != 1.0:
            penalty = int(penalty * PENALTY_MULTIPLIER)
        net_outcome = win - penalty
        # Determine if choice was correct (positive net outcome)
        correct = net_outcome > 0

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

        # Get outcome with multipliers applied
        win, penalty = trial.win_amount, trial.penalty_amount
        if WIN_MULTIPLIER != 1.0:
            win = int(win * WIN_MULTIPLIER)
        if PENALTY_MULTIPLIER != 1.0:
            penalty = int(penalty * PENALTY_MULTIPLIER)
        net_outcome = win - penalty
        # Determine if choice was correct (positive net outcome)
        correct = net_outcome > 0
        # Update participant model
        self.participant.update(deck_choice, net_outcome)
        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)
        return cast(IGTrial, trial)

    def _get_available_decks(self, trial_num: int) -> List[str]:
        """Determine which decks are available based on settings."""
        if DECK_VISIBILITY == "sequential":
            # Gradually reveal decks over trials
            if trial_num < 25:
                return ["A", "B"]
            elif trial_num < 50:
                return ["A", "B", "C"]
            else:
                return DECK_LABELS
        else:
            # All decks available
            decks = DECK_LABELS.copy()
            if DECK_ORDER_RANDOMIZED:
                np.random.shuffle(decks)
            return decks

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""
        summary = self.experiment.get_summary()
        # Primary metric: net score (last 20 trials)
        net_score = summary.get("net_score", 0.0)
        # Learning curve analysis
        learning_rate = summary.get("learning_rate", 0.0)
        # Participant model statistics
        participant_stats = {
            "final_q_values": self.participant.q_values.copy(),
            "visit_counts": self.participant.visit_counts.copy(),
            "exploration_rate": self.participant.exploration_rate,
            "learning_rate": self.participant.learning_rate,
        }
        # Statistical tests if enabled
        statistical_results = {}
        if RUN_STATISTICAL_TESTS and len(self.experiment.trials) >= 20:
            statistical_results = self._run_statistical_tests()
        # Compile results
        results = {
            # Primary output metric
            "net_score": net_score,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": len(self.experiment.trials),
            "advantageous_choices": summary.get("advantageous_choices", 0),
            "disadvantageous_choices": summary.get("disadvantageous_choices", 0),
            "choices_by_deck": summary.get("choices_by_deck", {}),
            "final_money": summary.get("final_money", 0),
            "learning_rate": learning_rate,
            "net_score_first_half": summary.get("net_score_first_half", 0.0),
            "mean_reaction_time": summary.get("mean_reaction_time", 0.0),
            # Participant model info
            "participant_stats": participant_stats,
            # Statistical analysis
            "statistical_results": statistical_results,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "learning_rate": LEARNING_RATE,
                "exploration_rate": EXPLORATION_RATE,
                "temperature": TEMPERATURE,
                "win_multiplier": WIN_MULTIPLIER,
                "penalty_multiplier": PENALTY_MULTIPLIER,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "feedback_delay_ms": FEEDBACK_DELAY_MS,
            },
        }
        return results

    def _run_statistical_tests(self) -> Dict:
        """Run statistical analysis on choice patterns."""
        trials = self.experiment.trials
        # Test 1: Trend in advantageous choices over blocks
        block_size = len(trials) // 5
        block_advantageous = []
        for i in range(5):
            start = i * block_size
            end = min(start + block_size, len(trials))
            block = trials[start:end]
            if block:
                adv = sum(
                    1
                    for t in block
                    if self.experiment.decks[t.deck_choice].is_advantageous()
                )
                block_advantageous.append(adv / len(block))
        # Linear trend
        if len(block_advantageous) >= 2:
            x = np.arange(len(block_advantageous))
            y = np.array(block_advantageous)
            slope, intercept = np.polyfit(x, y, 1)
            # Simple R-squared calculation
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            slope, r_squared = 0.0, 0.0
        return {
            "learning_trend_slope": float(slope),
            "learning_trend_r2": float(r_squared),
            "block_advantageous_pcts": [float(p) for p in block_advantageous],
        }


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------
def print_results(results: Dict):
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("IGT Experiment Results")
    print("=" * 60)
    # Primary metric
    print(f"\nPRIMARY METRIC (net_score): {results['net_score']:.2f}")

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

    print("  (advantageous - disadvantageous choices in last 20 trials)")
    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  advantageous:      {results['advantageous_choices']}")
    print(f"  disadvantageous:   {results['disadvantageous_choices']}")
    print(f"  learning_rate:     {results['learning_rate']:.4f}")
    # Deck breakdown
    print("\nChoices by Deck:")
    for deck, count in results["choices_by_deck"].items():
        print(f"  Deck {deck}: {count}")
    # Statistical results
    if results.get("statistical_results"):
        stats = results["statistical_results"]
        print("\nStatistical Analysis:")
        print(f"  Learning trend slope: {stats['learning_trend_slope']:.4f}")
        print(f"  Learning trend R²:    {stats['learning_trend_r2']:.4f}")
    print("\n" + "=" * 60)


def main():
    """Main entry point for IGT experiment."""
    import gc

    gc.collect()
    # Run experiment
    runner = EnhancedIGTRunner()
    results = runner.run_experiment()
    # Print results
    print_results(results)
    # Final summary output (for automated parsing)
    print("\n---")
    print(f"net_score:         {results['net_score']:.2f}")
    print(f"completion_time_s: {results['completion_time_s']:.1f}")
    print(f"num_trials:        {results['num_trials']}")
    print(f"advantageous_choices: {results['advantageous_choices']}")
    print(f"disadvantageous_choices: {results['disadvantageous_choices']}")
    print(f"learning_rate:     {results['learning_rate']:.4f}")
    # Memory tracking (simplified - using placeholder)
    peak_memory_mb = 0.0
    print(f"peak_vram_mb:      {peak_memory_mb:.1f}")
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

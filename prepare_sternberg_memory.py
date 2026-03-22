"""Fixed constants for Sternberg Memory experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TIME_BUDGET = 600
NUM_TRIALS = 100

# APGI Integration Parameters - 100/100 Compliance
# Optimized for sternberg memory dynamics
APGI_ENABLED = True

# Core dynamical system parameters
APGI_TAU_S = 0.35  # Surprise decay (200-500ms)
APGI_TAU_THETA = 30.0  # Threshold adaptation (5-60s)
APGI_TAU_M = 1.5  # Somatic marker timescale (1-2s)

# Threshold and sigmoid parameters
APGI_THETA_0 = 0.5  # Baseline ignition threshold (0.1-1.0 AU)
APGI_ALPHA = 5.5  # Sigmoid steepness (3.0-8.0)

# Somatic modulation
APGI_BETA = 1.5  # β_som: Somatic influence (0.5-2.5)
APGI_BETA_M = 1.0  # Marker sensitivity (0.5-2.0)
APGI_M_0 = 0.0  # Reference somatic marker level

# Sensitivities
APGI_GAMMA_M = -0.3  # Metabolic sensitivity (-0.5 to 0.5)
APGI_GAMMA_A = 0.1  # Arousal sensitivity (-0.3 to 0.3)
APGI_LAMBDA_S = 0.1  # Metabolic coupling strength

# Noise strengths
APGI_SIGMA_S = 0.05  # Surprise noise
APGI_SIGMA_THETA = 0.02  # Threshold noise
APGI_SIGMA_M = 0.03  # Somatic marker noise

# Domain-specific thresholds
APGI_THETA_SURVIVAL = 0.3  # Lower threshold for survival-relevant
APGI_THETA_NEUTRAL = 0.7  # Higher threshold for neutral content

# Reset dynamics
APGI_RHO = 0.7  # Reset fraction after ignition (0.3-0.9)

# Hierarchical processing (5-level)
APGI_HIERARCHICAL_ENABLED = True
APGI_BETA_CROSS = 0.2  # Cross-level coupling (0.1-0.5)
APGI_TAU_LEVELS = [0.1, 0.2, 0.4, 1.0, 5.0]  # Level-specific timescales

# Neuromodulator baselines
APGI_ACHT = 1.0  # Acetylcholine
APGI_NE = 1.0  # Norepinephrine
APGI_DA = 1.0  # Dopamine
APGI_HT5 = 1.0  # Serotonin

# Precision expectation gap (Π vs Π̂)
APGI_ENABLE_PRECISION_GAP = True
APGI_PI_E_EXPECTED = 1.0
APGI_PI_I_EXPECTED = 1.0

# Psychiatric profiles
APGI_GAD_PROFILE = False
APGI_MDD_PROFILE = False
APGI_PSYCHOSIS_PROFILE = False


class TrialType(Enum):
    POSITIVE = "positive"  # Probe was in memory set
    NEGATIVE = "negative"  # Probe was not in memory set


SET_SIZES = [1, 2, 4, 6, 8]
POSITIVE_PROB = 0.5
STIMULI = list("CDEFGHJKLMNPQRSTUVWXYZ")


@dataclass
class SternbergTrial:
    trial_number: int
    set_size: int
    memory_set: List[str]
    probe: str
    trial_type: TrialType
    response: Optional[str] = None
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class SternbergGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> SternbergTrial:
        set_size = int(self.rng.choice(SET_SIZES))
        is_positive = self.rng.random() < POSITIVE_PROB

        memory_set = list(self.rng.choice(STIMULI, size=set_size, replace=False))

        if is_positive:
            probe = self.rng.choice(memory_set)
        else:
            probe = self.rng.choice([s for s in STIMULI if s not in memory_set])

        return SternbergTrial(
            trial_number=trial_number,
            set_size=set_size,
            memory_set=memory_set,
            probe=probe,
            trial_type=TrialType.POSITIVE if is_positive else TrialType.NEGATIVE,
        )


class SternbergExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = SternbergGenerator(seed=seed)
        self.trials: List[SternbergTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[SternbergTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(
        self, trial: SternbergTrial, response: str, rt_ms: float
    ) -> SternbergTrial:
        trial.response = response
        trial.correct = (response == "yes") == (trial.trial_type == TrialType.POSITIVE)
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_slope(self) -> float:
        """Calculate slope (ms/item) - classic result ~38ms/item."""
        by_size = {size: [] for size in SET_SIZES}
        for t in self.trials:
            if t.correct:
                by_size[t.set_size].append(t.rt_ms)

        sizes = []
        rts = []
        for size in SET_SIZES:
            if by_size[size]:
                sizes.append(size)
                rts.append(np.mean(by_size[size]))

        if len(sizes) < 2:
            return 0.0
        slope, _ = np.polyfit(sizes, rts, 1)
        return slope

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "search_slope_ms_per_item": self.get_slope(),
            "accuracy": np.mean([t.correct for t in self.trials]),
            "positive_rt_ms": np.mean(
                [
                    t.rt_ms
                    for t in self.trials
                    if t.trial_type == TrialType.POSITIVE and t.correct
                ]
            ),
            "negative_rt_ms": np.mean(
                [
                    t.rt_ms
                    for t in self.trials
                    if t.trial_type == TrialType.NEGATIVE and t.correct
                ]
            ),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "set_size": t.set_size,
                    "memory_set": t.memory_set,
                    "probe": t.probe,
                    "trial_type": t.trial_type.value,
                    "response": t.response,
                    "correct": t.correct,
                    "rt_ms": t.rt_ms,
                    "timestamp": t.timestamp,
                }
                for t in self.trials
            ],
            "summary": self.get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# Standardized APGI Parameters Export (READ-ONLY)
# These parameters are used by the AGENT-EDITABLE run file for APGI integration
APGI_PARAMS = {
    # Core identification
    "experiment_name": "sternberg_memory",
    "enabled": APGI_ENABLED,
    # Dynamical system timescales
    "tau_s": APGI_TAU_S,
    "tau_theta": APGI_TAU_THETA,
    "tau_M": APGI_TAU_M,
    # Threshold and ignition parameters
    "theta_0": APGI_THETA_0,
    "alpha": APGI_ALPHA,
    "rho": APGI_RHO,
    # Somatic modulation
    "beta": APGI_BETA,
    "beta_M": APGI_BETA_M,
    "M_0": APGI_M_0,
    # Sensitivities
    "gamma_M": APGI_GAMMA_M,
    "gamma_A": APGI_GAMMA_A,
    "lambda_S": APGI_LAMBDA_S,
    # Noise strengths
    "sigma_S": APGI_SIGMA_S,
    "sigma_theta": APGI_SIGMA_THETA,
    "sigma_M": APGI_SIGMA_M,
    # Domain-specific thresholds
    "theta_survival": APGI_THETA_SURVIVAL,
    "theta_neutral": APGI_THETA_NEUTRAL,
    # Hierarchical processing
    "hierarchical_enabled": APGI_HIERARCHICAL_ENABLED,
    "beta_cross": APGI_BETA_CROSS,
    "tau_levels": APGI_TAU_LEVELS,
    # Neuromodulator baselines
    "ACh": APGI_ACHT,
    "NE": APGI_NE,
    "DA": APGI_DA,
    "HT5": APGI_HT5,
    # Precision expectation gap
    "precision_gap_enabled": APGI_ENABLE_PRECISION_GAP,
    "Pi_e_expected": APGI_PI_E_EXPECTED,
    "Pi_i_expected": APGI_PI_I_EXPECTED,
    # Psychiatric profiles
    "GAD_profile": APGI_GAD_PROFILE,
    "MDD_profile": APGI_MDD_PROFILE,
    "psychosis_profile": APGI_PSYCHOSIS_PROFILE,
}


def verify():
    print("Sternberg Memory - Configuration Verification")
    print(f"Set Sizes: {SET_SIZES}")
    print(f"Positive Probability: {POSITIVE_PROB}")


if __name__ == "__main__":
    verify()

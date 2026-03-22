"""Fixed constants for Inattentional Blindness experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TIME_BUDGET = 600
NUM_TRIALS = 20  # Fewer trials due to complex stimuli

# APGI Integration Parameters - 100/100 Compliance
# Optimized for inattentional blindness dynamics
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
    CRITICAL = "critical"  # Unexpected object appears
    CONTROL = "control"  # Object expected
    DISTRACTOR = "distractor"  # Standard trial


TASK_TYPES = ["count_passes", "track_ball", "monitor_players"]
UNEXPECTED_OBJECTS = ["gorilla", "umbrella_woman", "red_cross"]
ATTENTION_LOADS = ["high", "low"]


@dataclass
class IBTrial:
    trial_number: int
    trial_type: TrialType
    task_type: str
    unexpected_object: Optional[str]
    attention_load: str
    noticed_unexpected: bool = False
    task_accuracy: float = 0.0
    timestamp: float = 0.0


class InattentionalBlindnessGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> IBTrial:
        # 20% critical trials
        is_critical = self.rng.random() < 0.20
        trial_type = TrialType.CRITICAL if is_critical else TrialType.DISTRACTOR
        task = self.rng.choice(TASK_TYPES)
        unexpected = self.rng.choice(UNEXPECTED_OBJECTS) if is_critical else None
        load = self.rng.choice(ATTENTION_LOADS)

        return IBTrial(
            trial_number=trial_number,
            trial_type=trial_type,
            task_type=task,
            unexpected_object=unexpected,
            attention_load=load,
        )


class InattentionalBlindnessExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = InattentionalBlindnessGenerator(seed=seed)
        self.trials: List[IBTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[IBTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(self, trial: IBTrial, noticed: bool, task_accuracy: float) -> IBTrial:
        trial.noticed_unexpected = noticed
        trial.task_accuracy = task_accuracy
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_inattentional_blindness_rate(self) -> float:
        """Proportion who missed unexpected object."""
        critical = [t for t in self.trials if t.trial_type == TrialType.CRITICAL]
        if not critical:
            return 0.0
        return 1 - np.mean([t.noticed_unexpected for t in critical])

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        critical = [t for t in self.trials if t.trial_type == TrialType.CRITICAL]
        high_load = [t for t in critical if t.attention_load == "high"]
        low_load = [t for t in critical if t.attention_load == "low"]

        return {
            "num_trials": len(self.trials),
            "inattentional_blindness_rate": self.get_inattentional_blindness_rate(),
            "high_load_notice_rate": np.mean([t.noticed_unexpected for t in high_load])
            if high_load
            else 0,
            "low_load_notice_rate": np.mean([t.noticed_unexpected for t in low_load])
            if low_load
            else 0,
            "mean_task_accuracy": np.mean([t.task_accuracy for t in self.trials]),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "trial_type": t.trial_type.value,
                    "task_type": t.task_type,
                    "unexpected_object": t.unexpected_object,
                    "attention_load": t.attention_load,
                    "noticed": t.noticed_unexpected,
                    "task_accuracy": t.task_accuracy,
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
    "experiment_name": "inattentional_blindness",
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
    print("Inattentional Blindness - Configuration Verification")
    print(f"Task Types: {TASK_TYPES}")
    print(f"Unexpected Objects: {UNEXPECTED_OBJECTS}")


if __name__ == "__main__":
    verify()

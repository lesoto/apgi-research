"""Fixed constants for Go/No-Go experiments.

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
# Optimized for go no go dynamics
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
    GO = "go"
    NO_GO = "no_go"


STIMULI = {"circle": TrialType.GO, "square": TrialType.NO_GO}
GO_PROBABILITY = 0.75


@dataclass
class GoNoGoTrial:
    trial_number: int
    stimulus: str
    trial_type: TrialType
    response: Optional[str] = None  # "press" or "no_press"
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class GoNoGoGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> GoNoGoTrial:
        is_go = self.rng.random() < GO_PROBABILITY
        stimulus = "circle" if is_go else "square"
        return GoNoGoTrial(
            trial_number=trial_number,
            stimulus=stimulus,
            trial_type=TrialType.GO if is_go else TrialType.NO_GO,
        )


class GoNoGoExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = GoNoGoGenerator(seed=seed)
        self.trials: List[GoNoGoTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[GoNoGoTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(
        self, trial: GoNoGoTrial, responded: bool, rt_ms: float
    ) -> GoNoGoTrial:
        correct = (trial.trial_type == TrialType.GO and responded) or (
            trial.trial_type == TrialType.NO_GO and not responded
        )
        import time

        trial.response = "press" if responded else "no_press"
        trial.correct = correct
        trial.rt_ms = rt_ms
        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_d_prime(self) -> float:
        """Calculate sensitivity d' (hits vs false alarms)."""
        go_trials = [t for t in self.trials if t.trial_type == TrialType.GO]
        no_go_trials = [t for t in self.trials if t.trial_type == TrialType.NO_GO]
        hits = np.mean([t.correct for t in go_trials]) if go_trials else 0
        fa_rate = 1 - np.mean([t.correct for t in no_go_trials]) if no_go_trials else 0
        return hits - fa_rate

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        go_trials = [t for t in self.trials if t.trial_type == TrialType.GO]
        no_go_trials = [t for t in self.trials if t.trial_type == TrialType.NO_GO]
        return {
            "num_trials": len(self.trials),
            "go_accuracy": np.mean([t.correct for t in go_trials]) if go_trials else 0,
            "no_go_accuracy": np.mean([t.correct for t in no_go_trials])
            if no_go_trials
            else 0,
            "mean_go_rt_ms": np.mean([t.rt_ms for t in go_trials if t.correct])
            if go_trials
            else 0,
            "d_prime": self.get_d_prime(),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "stimulus": t.stimulus,
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
    "experiment_name": "go_no_go",
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
    print("Go/No-Go - Configuration Verification")
    print(f"Stimuli: {STIMULI}")
    print(f"Go Probability: {GO_PROBABILITY}")


if __name__ == "__main__":
    verify()

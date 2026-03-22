"""Fixed constants for Time Estimation experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TIME_BUDGET = 600
NUM_TRIALS = 50

# APGI Integration Parameters - 100/100 Compliance
# Optimized for time estimation dynamics
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


class EstimationMethod(Enum):
    REPRODUCTION = "reproduction"
    VERBAL = "verbal"
    COMPARISON = "comparison"


DURATION_RANGES = {
    "short": (400, 1000),  # ms
    "medium": (1000, 3000),
    "long": (2000, 8000),
}


@dataclass
class TimeEstTrial:
    trial_number: int
    method: EstimationMethod
    target_duration_ms: int
    estimated_duration_ms: int
    error_ms: float = 0.0
    error_percent: float = 0.0
    timestamp: float = 0.0


class TimeEstGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> TimeEstTrial:
        method = self.rng.choice(list(EstimationMethod))
        range_name = self.rng.choice(list(DURATION_RANGES.keys()))
        min_d, max_d = DURATION_RANGES[range_name]
        target = int(self.rng.randint(min_d, max_d + 1))

        return TimeEstTrial(
            trial_number=trial_number,
            method=method,
            target_duration_ms=target,
            estimated_duration_ms=0,
        )


class TimeEstExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = TimeEstGenerator(seed=seed)
        self.trials: List[TimeEstTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[TimeEstTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(self, trial: TimeEstTrial, estimated_ms: int) -> TimeEstTrial:
        trial.estimated_duration_ms = estimated_ms
        trial.error_ms = estimated_ms - trial.target_duration_ms
        trial.error_percent = (trial.error_ms / trial.target_duration_ms) * 100
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_mean_error(self, method: Optional[EstimationMethod] = None) -> float:
        trials = [t for t in self.trials if method is None or t.method == method]
        if not trials:
            return 0.0
        return np.mean([t.error_ms for t in trials])

    def get_variability(self) -> float:
        """Coefficient of variation."""
        if not self.trials:
            return 0.0
        return np.std([t.error_percent for t in self.trials]) / abs(
            np.mean([t.error_percent for t in self.trials])
        )

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "mean_error_ms": self.get_mean_error(),
            "mean_error_percent": np.mean([t.error_percent for t in self.trials]),
            "reproduction_error": self.get_mean_error(EstimationMethod.REPRODUCTION),
            "verbal_error": self.get_mean_error(EstimationMethod.VERBAL),
            "comparison_error": self.get_mean_error(EstimationMethod.COMPARISON),
            "variability_cv": self.get_variability(),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "method": t.method.value,
                    "target_ms": t.target_duration_ms,
                    "estimated_ms": t.estimated_duration_ms,
                    "error_ms": t.error_ms,
                    "error_percent": t.error_percent,
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
    "experiment_name": "time_estimation",
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
    print("Time Estimation - Configuration Verification")
    print(f"Duration Ranges: {DURATION_RANGES}")
    print(f"Estimation Methods: {[m.value for m in EstimationMethod]}")


if __name__ == "__main__":
    verify()

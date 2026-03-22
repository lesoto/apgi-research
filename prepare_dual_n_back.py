"""Fixed constants for Dual N-Back experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

TIME_BUDGET = 600
NUM_TRIALS = 80

# APGI Integration Parameters - 100/100 Compliance
# Optimized for dual n back dynamics
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

STIMULI = list("CDEFGHJKLMNPQRSTUVWXYZ")  # Letters only
N_BACK_LEVELS = [1, 2, 3, 4]


@dataclass
class NBackTrial:
    trial_number: int
    n_level: int
    stimulus: str
    is_match: bool  # Current stimulus matches n-back
    position_match: bool = False  # For spatial N-back
    stimulus_response: bool = False
    position_response: bool = False
    stimulus_correct: bool = False
    position_correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class DualNBackGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.stimulus_history: List[str] = []
        self.reset()

    def reset(self):
        self.trial_count = 0
        self.stimulus_history = []

    def create_trial(self, trial_number: int, n_level: int) -> NBackTrial:
        # 30% match probability
        is_match = self.rng.random() < 0.30

        if is_match and len(self.stimulus_history) >= n_level:
            stimulus = self.stimulus_history[-n_level]
        else:
            stimulus = self.rng.choice(STIMULI)

        self.stimulus_history.append(stimulus)

        return NBackTrial(
            trial_number=trial_number,
            n_level=n_level,
            stimulus=stimulus,
            is_match=is_match,
        )


class DualNBackExperiment:
    def __init__(
        self, num_trials: int = NUM_TRIALS, n_level: int = 2, seed: Optional[int] = None
    ):
        self.num_trials = num_trials
        self.n_level = n_level
        self.generator = DualNBackGenerator(seed=seed)
        self.trials: List[NBackTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[NBackTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1, self.n_level)
        return trial

    def run_trial(
        self, trial: NBackTrial, stimulus_response: bool, rt_ms: float
    ) -> NBackTrial:
        trial.stimulus_response = stimulus_response
        trial.stimulus_correct = stimulus_response == trial.is_match
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_d_prime(self) -> float:
        """Calculate d' sensitivity."""
        hits = (
            np.mean([t.stimulus_correct for t in self.trials if t.is_match])
            if self.trials
            else 0
        )
        correct_rejections = (
            np.mean([t.stimulus_correct for t in self.trials if not t.is_match])
            if self.trials
            else 0
        )
        return float(hits + correct_rejections - 1)

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        matches = [t for t in self.trials if t.is_match]
        non_matches = [t for t in self.trials if not t.is_match]
        return {
            "num_trials": len(self.trials),
            "n_level": self.n_level,
            "hit_rate": np.mean([t.stimulus_correct for t in matches])
            if matches
            else 0,
            "correct_rejection_rate": np.mean([t.stimulus_correct for t in non_matches])
            if non_matches
            else 0,
            "d_prime": self.get_d_prime(),
            "mean_rt_ms": np.mean([t.rt_ms for t in self.trials]),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "n_level": t.n_level,
                    "stimulus": t.stimulus,
                    "is_match": t.is_match,
                    "response": t.stimulus_response,
                    "correct": t.stimulus_correct,
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
    "experiment_name": "dual_n_back",
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
    print("Dual N-Back - Configuration Verification")
    print(f"Stimuli: {STIMULI[:5]}... (n={len(STIMULI)})")
    print(f"N-Back Levels: {N_BACK_LEVELS}")


if __name__ == "__main__":
    verify()

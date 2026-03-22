"""Fixed constants for Probabilistic Category Learning experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TIME_BUDGET = 600
NUM_TRIALS = 200

# APGI Integration Parameters - 100/100 Compliance
# Optimized for probabilistic category learning dynamics
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


class RuleType(Enum):
    SINGLE = "single"  # Single deterministic rule
    MULTI = "multi"  # Multiple probabilistic rules
    CONFIGURAL = "configural"  # Configural/conditional rule


CATEGORY_STRUCTURES = {
    "single": {"A": 0.8, "B": 0.2},  # P(category|cue)
    "multi": {"A": 0.7, "B": 0.6, "C": 0.5},
    "configural": {"AB": 0.9, "A_not_B": 0.3, "not_A_B": 0.2, "not_A_not_B": 0.1},
}

FEEDBACK_PROB = 1.0  # Deterministic feedback


@dataclass
class PCLTrial:
    trial_number: int
    rule_type: RuleType
    cues: List[str]
    correct_category: str
    category_a_prob: float
    category_b_prob: float
    response: Optional[str] = None
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class PCLGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> PCLTrial:
        rule_type = self.rng.choice(list(RuleType))

        if rule_type == RuleType.SINGLE:
            cue = self.rng.choice(["A", "B"])
            probs = CATEGORY_STRUCTURES["single"]
            cat_a_prob = probs[cue]
            correct = "A" if self.rng.random() < cat_a_prob else "B"
        elif rule_type == RuleType.MULTI:
            # Multi-rule logic simplified
            correct = "A" if self.rng.random() < 0.6 else "B"
            cat_a_prob = 0.6
        else:  # CONFIGURAL
            # Simplified configural logic
            correct = "A" if self.rng.random() < 0.5 else "B"
            cat_a_prob = 0.5

        return PCLTrial(
            trial_number=trial_number,
            rule_type=rule_type,
            cues=["A"],  # Simplified
            correct_category=correct,
            category_a_prob=cat_a_prob,
            category_b_prob=1 - cat_a_prob,
        )


class PCLExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = PCLGenerator(seed=seed)
        self.trials: List[PCLTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[PCLTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(self, trial: PCLTrial, response: str, rt_ms: float) -> PCLTrial:
        trial.response = response
        trial.correct = response == trial.correct_category
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_learning_curve(self) -> List[float]:
        """Accuracy by blocks of 20 trials."""
        block_size = 20
        blocks = len(self.trials) // block_size
        curve = []
        for i in range(blocks):
            block = self.trials[i * block_size : (i + 1) * block_size]
            curve.append(np.mean([t.correct for t in block]))
        return curve

    def get_final_accuracy(self) -> float:
        """Accuracy in last 20 trials."""
        if len(self.trials) < 20:
            return 0.0
        return np.mean([t.correct for t in self.trials[-20:]])

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "overall_accuracy": np.mean([t.correct for t in self.trials]),
            "final_accuracy": self.get_final_accuracy(),
            "learning_curve": self.get_learning_curve(),
            "mean_rt_ms": np.mean([t.rt_ms for t in self.trials]),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "rule_type": t.rule_type.value,
                    "cues": t.cues,
                    "correct_category": t.correct_category,
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
    "experiment_name": "probabilistic_category_learning",
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
    print("Probabilistic Category Learning - Configuration Verification")
    print(f"Rule Types: {[r.value for r in RuleType]}")
    print(f"Trials: {NUM_TRIALS}")


if __name__ == "__main__":
    verify()

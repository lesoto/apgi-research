"""Fixed constants for Working Memory Span experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TIME_BUDGET = 600
NUM_TRIALS = 20

# APGI Integration Parameters - 100/100 Compliance
# Optimized for working memory span dynamics
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


class SpanType(Enum):
    SIMPLE = "simple"  # Single processing task
    COMPLEX = "complex"  # Complex processing (reading/digit)


SPAN_LEVELS = [2, 3, 4, 5, 6, 7]
WORDS = [
    "apple",
    "table",
    "house",
    "chair",
    "water",
    "music",
    "light",
    "phone",
    "bread",
    "money",
    "paper",
    "world",
    "watch",
    "glass",
    "plant",
    "shoes",
    "smile",
    "beach",
    "horse",
    "cloud",
]
PROCESSING_TASKS = ["arithmetic", "sentence_verification"]


@dataclass
class WMSpanTrial:
    trial_number: int
    span_size: int
    span_type: SpanType
    to_remember: List[str]
    processing_items: List[Dict]
    recalled: List[str] = None
    recall_accuracy: float = 0.0
    processing_accuracy: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.recalled is None:
            self.recalled = []


class WorkingMemorySpanGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> WMSpanTrial:
        span_size = int(self.rng.choice(SPAN_LEVELS))
        span_type = self.rng.choice([SpanType.SIMPLE, SpanType.COMPLEX])
        to_remember = list(self.rng.choice(WORDS, size=span_size, replace=False))
        processing_items = []
        for _ in range(span_size):
            task = self.rng.choice(PROCESSING_TASKS)
            if task == "arithmetic":
                a, b = self.rng.randint(1, 20), self.rng.randint(1, 20)
                correct = a + b
                processing_items.append(
                    {
                        "task": "arithmetic",
                        "problem": f"{a}+{b}",
                        "correct_answer": correct,
                        "user_answer": None,
                    }
                )
            else:
                sentence = "The sky is blue."
                is_valid = True
                processing_items.append(
                    {
                        "task": "sentence",
                        "sentence": sentence,
                        "is_valid": is_valid,
                        "user_answer": None,
                    }
                )

        return WMSpanTrial(
            trial_number=trial_number,
            span_size=span_size,
            span_type=span_type,
            to_remember=to_remember,
            processing_items=processing_items,
        )


class WorkingMemorySpanExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = WorkingMemorySpanGenerator(seed=seed)
        self.trials: List[WMSpanTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[WMSpanTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(
        self, trial: WMSpanTrial, recalled: List[str], proc_answers: List[bool]
    ) -> WMSpanTrial:
        trial.recalled = recalled
        trial.recall_accuracy = (
            sum(1 for r in recalled if r in trial.to_remember) / len(trial.to_remember)
            if trial.to_remember
            else 0
        )
        trial.processing_accuracy = np.mean(proc_answers) if proc_answers else 0
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_span_capacity(self) -> float:
        """Estimate working memory span capacity."""
        if not self.trials:
            return 0.0
        perfect = [t.span_size for t in self.trials if t.recall_accuracy == 1.0]
        return (
            np.mean(perfect)
            if perfect
            else max((t.span_size * t.recall_accuracy for t in self.trials), default=0)
        )

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "span_capacity": self.get_span_capacity(),
            "mean_recall_accuracy": np.mean([t.recall_accuracy for t in self.trials]),
            "mean_processing_accuracy": np.mean(
                [t.processing_accuracy for t in self.trials]
            ),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "span_size": t.span_size,
                    "span_type": t.span_type.value,
                    "to_remember": t.to_remember,
                    "recalled": t.recalled,
                    "recall_accuracy": t.recall_accuracy,
                    "processing_accuracy": t.processing_accuracy,
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
    "experiment_name": "working_memory_span",
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
    print("Working Memory Span - Configuration Verification")
    print(f"Span Levels: {SPAN_LEVELS}")
    print(f"Word Pool Size: {len(WORDS)}")


if __name__ == "__main__":
    verify()

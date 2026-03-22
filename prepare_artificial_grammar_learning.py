"""
Fixed constants and data preparation for Artificial Grammar Learning experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed grammar configurations, rule sets, and evaluation metrics.

Usage:
    python prepare_artificial_grammar_learning.py  # Verify grammar configurations

Artificial Grammar Learning paradigms:
- Finite state grammar learning: Learn patterns from letter strings
- Rule acquisition: Discover underlying grammatical structure
- Generalization: Apply learned rules to novel strings
- Implicit learning: Unconscious pattern recognition
"""
import numpy as np
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)
NUM_TRIALS = 200

# APGI Integration Parameters - 100/100 Compliance
# Optimized for artificial grammar learning dynamics
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
    pass


GRAMMAR_STRINGS = [
    "ac",
    "acc",
    "accc",
    "acd",
    "acdd",
    "bd",
    "bdd",
    "bcc",
    "bccc",
    "bcd",
    "aac",
    "aacc",
    "aacd",
    "abbd",
    "bbd",
    "bbdd",
    "abcc",
    "abcdd",
    "aaccc",
    "abc",
]

NON_GRAMMAR_STRINGS = [
    "ca",
    "cb",
    "da",
    "db",
    "dc",
    "aa",
    "bb",
    "cc",
    "dd",
    "abab",
    "cd",
    "ba",
    "ad",
    "bc",
    "acb",
    "bdc",
    "cad",
    "dba",
    "aca",
    "bdb",
]

STUDY_ITEMS = 20

TEST_ITEMS = 40


@dataclass
class AGTrial:
    trial_number: int
    string: str
    is_grammatical: bool
    response: Optional[bool] = None
    correct: bool = False
    confidence: int = 0
    rt_ms: float = 0.0
    timestamp: float = 0.0


class AGExperiment:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.study_items: List[str] = []
        self.test_items: List[AGTrial] = []
        self.reset()

    def reset(self):
        self.study_items = []
        self.test_items = []

    def generate_study_phase(self) -> List[str]:
        """Generate strings for study phase (all grammatical)."""
        self.study_items = list(
            self.rng.choice(GRAMMAR_STRINGS, size=STUDY_ITEMS, replace=True)
        )
        return self.study_items

    def generate_test_phase(self) -> List[AGTrial]:
        """Generate test phase with grammatical and non-grammatical strings."""
        test_grammatical = list(
            self.rng.choice(GRAMMAR_STRINGS, size=TEST_ITEMS // 2, replace=True)
        )
        test_non_grammatical = list(
            self.rng.choice(NON_GRAMMAR_STRINGS, size=TEST_ITEMS // 2, replace=True)
        )

        for i, s in enumerate(test_grammatical):
            self.test_items.append(
                AGTrial(trial_number=i + 1, string=s, is_grammatical=True)
            )
        for i, s in enumerate(test_non_grammatical):
            self.test_items.append(
                AGTrial(
                    trial_number=TEST_ITEMS // 2 + i + 1, string=s, is_grammatical=False
                )
            )

        # Convert to list for shuffling
        trial_list = list(self.test_items)
        random.shuffle(trial_list)
        self.test_items = trial_list
        for i, t in enumerate(self.test_items):
            t.trial_number = i + 1
        return self.test_items

    def record_response(
        self, trial: AGTrial, response: bool, confidence: int, rt_ms: float
    ):
        trial.response = response
        trial.confidence = confidence
        trial.correct = response == trial.is_grammatical
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()

    def get_d_prime(self) -> float:
        """Calculate sensitivity d'."""
        hits = np.mean([t.correct for t in self.test_items if t.is_grammatical])
        fa_rate = 1 - np.mean(
            [t.correct for t in self.test_items if not t.is_grammatical]
        )
        return float(hits - fa_rate)

    def get_accuracy(self) -> float:
        return (
            float(np.mean([t.correct for t in self.test_items]))
            if self.test_items
            else 0.0
        )

    def get_summary(self) -> Dict:
        if not self.test_items:
            return {}
        return {
            "num_study_items": len(self.study_items),
            "num_test_items": len(self.test_items),
            "accuracy": self.get_accuracy(),
            "d_prime": self.get_d_prime(),
            "grammatical_accuracy": float(
                np.mean([t.correct for t in self.test_items if t.is_grammatical])
            ),
            "non_grammatical_accuracy": float(
                np.mean([t.correct for t in self.test_items if not t.is_grammatical])
            ),
        }

    def save_results(self, filepath: str):
        data = {
            "study_items": self.study_items,
            "test_results": [
                {
                    "string": t.string,
                    "is_grammatical": t.is_grammatical,
                    "response": t.response,
                    "correct": t.correct,
                    "confidence": t.confidence,
                }
                for t in self.test_items
            ],
            "summary": self.get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# Standardized APGI Parameters Export (READ-ONLY)
# These parameters are used by the AGENT-EDITABLE run file for APGI integration
APGI_PARAMS = {
    # Core identification
    "experiment_name": "artificial_grammar_learning",
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
    print("Artificial Grammar Learning - Configuration Verification")
    print(f"Grammar Strings: {len(GRAMMAR_STRINGS)}")
    print(f"Non-Grammar Strings: {len(NON_GRAMMAR_STRINGS)}")
    print(f"Study Items: {STUDY_ITEMS}, Test Items: {TEST_ITEMS}")


if __name__ == "__main__":
    verify()

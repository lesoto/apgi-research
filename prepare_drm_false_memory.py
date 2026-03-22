"""Fixed constants for DRM False Memory experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import random
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

TIME_BUDGET = 600
NUM_TRIALS = 12  # Lists per experiment

# APGI Integration Parameters - 100/100 Compliance
# Optimized for drm false memory dynamics
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

# DRM Lists (6 items each, last item is critical lure)
DRM_LISTS = {
    "sleep": ["bed", "rest", "awake", "tired", "dream", "sleep"],
    "sweet": ["sugar", "bitter", "candy", "taste", "good", "sweet"],
    "man": ["woman", "male", "girl", "boy", "person", "man"],
    "doctor": ["nurse", "sick", "lawyer", "medicine", "health", "doctor"],
    "cold": ["hot", "snow", "warm", "winter", "ice", "cold"],
    "slow": ["fast", "lethargic", "quick", "speed", "snail", "slow"],
    "window": ["door", "glass", "pane", "shade", "ledge", "window"],
    "bread": ["butter", "food", "eat", "sandwich", "rye", "bread"],
    "rough": ["smooth", "bumpy", "road", "tough", "sandpaper", "rough"],
    "mountain": ["hill", "valley", "climb", "peak", "ski", "mountain"],
    "moon": ["night", "dark", "star", "space", "planet", "moon"],
    "music": ["sound", "sing", "radio", "band", "melody", "music"],
}


@dataclass
class DRMTrial:
    trial_number: int
    list_name: str
    list_items: List[str]
    presented: bool  # True if presented during study
    critical_lure: str
    response_recognized: bool = False
    confidence: int = 0
    is_false_memory: bool = False
    timestamp: float = 0.0


class DRMExperiment:
    def __init__(self, num_lists: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_lists = num_lists
        self.rng = np.random.RandomState(seed)
        self.study_lists: List[DRMTrial] = []
        self.test_items: List[DRMTrial] = []
        self.reset()

    def reset(self):
        self.study_lists = []
        self.test_items = []

    def generate_study_lists(self) -> List[DRMTrial]:
        """Generate lists for study phase (without critical lures)."""
        list_names = list(
            self.rng.choice(
                list(DRM_LISTS.keys()),
                size=min(self.num_lists, len(DRM_LISTS)),
                replace=False,
            )
        )

        for i, name in enumerate(list_names):
            items = DRM_LISTS[name][:-1]  # All except critical lure
            trial = DRMTrial(
                trial_number=i + 1,
                list_name=name,
                list_items=items,
                presented=True,
                critical_lure=DRM_LISTS[name][-1],
            )
            self.study_lists.append(trial)
        return self.study_lists

    def generate_test_items(self) -> List[DRMTrial]:
        """Generate test phase items (studied + critical lures + distractors)."""
        test_items = []

        # Add studied items
        for trial in self.study_lists:
            for item in trial.list_items:
                test_items.append(
                    DRMTrial(
                        trial_number=0,
                        list_name=trial.list_name,
                        list_items=[item],
                        presented=True,
                        critical_lure=item,
                    )
                )

        # Add critical lures (not studied but related)
        for trial in self.study_lists:
            test_items.append(
                DRMTrial(
                    trial_number=0,
                    list_name=trial.list_name,
                    list_items=[],
                    presented=False,
                    critical_lure=trial.critical_lure,
                    is_false_memory=True,
                )
            )

        random.shuffle(test_items)
        self.test_items = test_items
        return test_items

    def record_recognition(self, trial: DRMTrial, recognized: bool, confidence: int):
        trial.response_recognized = recognized
        trial.confidence = confidence
        import time

        trial.timestamp = time.time()

    def get_false_memory_rate(self) -> float:
        """Proportion of critical lures falsely recognized."""
        lures = [t for t in self.test_items if t.is_false_memory]
        if not lures:
            return 0.0
        return float(np.mean([t.response_recognized for t in lures]))

    def get_hit_rate(self) -> float:
        """Correct recognition of studied items."""
        studied = [t for t in self.test_items if t.presented]
        if not studied:
            return 0.0
        return float(np.mean([t.response_recognized for t in studied]))

    def get_summary(self) -> Dict:
        if not self.test_items:
            return {}
        return {
            "num_lists": len(self.study_lists),
            "false_memory_rate": self.get_false_memory_rate(),
            "hit_rate": self.get_hit_rate(),
            "mean_confidence": np.mean([t.confidence for t in self.test_items]),
        }

    def save_results(self, filepath: str):
        data = {
            "study_lists": [
                {"list_name": t.list_name, "items": t.list_items}
                for t in self.study_lists
            ],
            "test_results": [
                {
                    "item": t.critical_lure,
                    "presented": t.presented,
                    "is_lure": t.is_false_memory,
                    "recognized": t.response_recognized,
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
    "experiment_name": "drm_false_memory",
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
    print("DRM False Memory - Configuration Verification")
    print(f"Number of DRM Lists: {len(DRM_LISTS)}")
    print("Items per List: 6 (5 associates + 1 critical lure)")


if __name__ == "__main__":
    verify()

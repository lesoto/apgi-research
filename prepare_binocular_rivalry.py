"""
Fixed constants and data preparation for Binocular Rivalry experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed rivalry configurations, stimulus types, and evaluation metrics.

Usage:
    python prepare_binocular_rivalry.py  # Verify rivalry configurations

Binocular Rivalry paradigms:
- Red-green rivalry: Color competition between eyes
- Horizontal-vertical rivalry: Orientation competition
- Face-house rivalry: Semantic competition
- Perceptual alternation: Spontaneous switching between percepts
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)

NUM_TRIALS = 60

# APGI Integration Parameters - 100/100 Compliance
# Optimized for binocular rivalry dynamics
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


class StimulusType(Enum):
    RED_GREEN = "red_green"
    HORIZONTAL_VERTICAL = "horizontal_vertical"
    FACE_HOUSE = "face_house"


RIVALRY_DURATION_S = 60  # Standard rivalry duration


@dataclass
class RivalryTrial:
    trial_number: int
    stimulus_type: StimulusType
    duration_s: int
    percepts: List[Dict]  # List of {percept: str, duration: float}
    alternation_count: int
    mean_duration_a: float
    mean_duration_b: float
    timestamp: float = 0.0


class BinocularRivalryGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> dict:
        """Returns trial config (actual rivalry simulated in runner)."""
        stim_type_list = [st.value for st in StimulusType]
        stim_type_str = self.rng.choice(stim_type_list)
        stim_type = StimulusType(stim_type_str)
        return {
            "trial_number": trial_number,
            "stimulus_type": stim_type,
            "duration_s": RIVALRY_DURATION_S,
        }


class BinocularRivalryExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = BinocularRivalryGenerator(seed=seed)
        self.trials: List[RivalryTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[dict]:
        if self.current_trial_idx >= self.num_trials:
            return None
        return self.generator.create_trial(self.current_trial_idx + 1)

    def run_trial(
        self,
        trial_number: int,
        stimulus_type: StimulusType,
        percepts: List[Dict],
        duration_s: int,
    ) -> RivalryTrial:
        """Record rivalry trial with simulated perceptual alternations."""
        alternations = len(percepts) - 1 if len(percepts) > 0 else 0
        a_percepts = [
            p
            for p in percepts
            if p.get("percept") in ["A", "red", "horizontal", "face"]
        ]
        b_percepts = [
            p
            for p in percepts
            if p.get("percept") in ["B", "green", "vertical", "house"]
        ]

        mean_a = float(
            np.mean([p.get("duration", 0) for p in a_percepts]) if a_percepts else 0.0
        )
        mean_b = float(
            np.mean([p.get("duration", 0) for p in b_percepts]) if b_percepts else 0.0
        )

        import time

        trial = RivalryTrial(
            trial_number=trial_number,
            stimulus_type=stimulus_type,
            duration_s=duration_s,
            percepts=percepts,
            alternation_count=alternations,
            mean_duration_a=mean_a,
            mean_duration_b=mean_b,
            timestamp=time.time(),
        )
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_alternation_rate(self) -> float:
        """Calculate mean alternations per minute."""
        if not self.trials:
            return 0.0
        total_alts = sum(t.alternation_count for t in self.trials)
        total_duration = sum(t.duration_s for t in self.trials) / 60  # minutes
        return float(total_alts / total_duration) if total_duration > 0 else 0.0

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "alternation_rate_per_min": self.get_alternation_rate(),
            "mean_duration_a_s": float(
                np.mean([t.mean_duration_a for t in self.trials])
            ),
            "mean_duration_b_s": float(
                np.mean([t.mean_duration_b for t in self.trials])
            ),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "stimulus_type": t.stimulus_type.value,
                    "duration_s": t.duration_s,
                    "alternation_count": t.alternation_count,
                    "mean_duration_a": t.mean_duration_a,
                    "mean_duration_b": t.mean_duration_b,
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
    "experiment_name": "binocular_rivalry",
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
    print("Binocular Rivalry - Configuration Verification")
    print(f"Stimulus Types: {[t.value for t in StimulusType]}")
    print(f"Rivalry Duration: {RIVALRY_DURATION_S}s")


if __name__ == "__main__":
    verify()

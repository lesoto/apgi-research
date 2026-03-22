"""Fixed constants for Somatic Marker Priming experiments.

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
# Optimized for somatic marker priming dynamics
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


class EmotionType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class PrimeType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TaskType(Enum):
    RISKY_CHOICE = "risky_choice"
    IOWA_GAMBLING = "iowa_gambling"
    DELAY_DISCOUNTING = "delay_discounting"


FEEDBACK_TYPES = ["skin_conductance", "heart_rate", "facial_emg", "subjective"]


@dataclass
class SomaticTrial:
    trial_number: int
    emotion_type: EmotionType
    task_type: TaskType
    feedback_type: str
    choice_options: List[Dict]
    selected_option: Optional[str] = None
    outcome_value: float = 0.0
    somatic_response: float = 0.0  # Physiological measure
    rt_ms: float = 0.0
    timestamp: float = 0.0


class SomaticMarkerGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> SomaticTrial:
        emotion = self.rng.choice(list(EmotionType))
        task = self.rng.choice(list(TaskType))
        feedback = self.rng.choice(FEEDBACK_TYPES)

        # Create choice options with expected values
        option_a = {"label": "A", "ev": 100, "risk": 0.3}
        option_b = {"label": "B", "ev": 80, "risk": 0.1}

        return SomaticTrial(
            trial_number=trial_number,
            emotion_type=emotion,
            task_type=task,
            feedback_type=feedback,
            choice_options=[option_a, option_b],
        )


class SomaticMarkerExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = SomaticMarkerGenerator(seed=seed)
        self.trials: List[SomaticTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[SomaticTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(
        self,
        trial: SomaticTrial,
        selected: str,
        outcome: float,
        somatic: float,
        rt_ms: float,
    ) -> SomaticTrial:
        trial.selected_option = selected
        trial.outcome_value = outcome
        trial.somatic_response = somatic
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_advantageous_rate(self, emotion: Optional[EmotionType] = None) -> float:
        """Proportion of trials selecting advantageous option."""
        trials = [
            t for t in self.trials if emotion is None or t.emotion_type == emotion
        ]
        if not trials:
            return 0.0
        # Advantageous = higher expected value option
        return np.mean([t.selected_option == "A" for t in trials])

    def get_somatic_correlation(self) -> float:
        """Correlation between somatic response and advantageous choice."""
        if not self.trials:
            return 0.0
        somatic = [t.somatic_response for t in self.trials]
        advantageous = [1 if t.selected_option == "A" else 0 for t in self.trials]
        if len(somatic) < 2:
            return 0.0
        return np.corrcoef(somatic, advantageous)[0, 1]

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "overall_advantageous_rate": self.get_advantageous_rate(),
            "positive_advantageous_rate": self.get_advantageous_rate(
                EmotionType.POSITIVE
            ),
            "negative_advantageous_rate": self.get_advantageous_rate(
                EmotionType.NEGATIVE
            ),
            "neutral_advantageous_rate": self.get_advantageous_rate(
                EmotionType.NEUTRAL
            ),
            "somatic_decision_correlation": self.get_somatic_correlation(),
            "mean_somatic_response": np.mean([t.somatic_response for t in self.trials]),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "emotion_type": t.emotion_type.value,
                    "task_type": t.task_type.value,
                    "feedback_type": t.feedback_type,
                    "selected_option": t.selected_option,
                    "outcome_value": t.outcome_value,
                    "somatic_response": t.somatic_response,
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
    "experiment_name": "somatic_marker_priming",
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
    print("Somatic Marker Priming - Configuration Verification")
    print(f"Emotion Types: {[e.value for e in EmotionType]}")
    print(f"Task Types: {[t.value for t in TaskType]}")


if __name__ == "__main__":
    verify()

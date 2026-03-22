"""Fixed constants for Multisensory Integration experiments.

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
# Optimized for multisensory integration dynamics
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


class Modality(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    BIMODAL = "bimodal"


class Congruency(Enum):
    CONGRUENT = "congruent"
    INCONGRUENT = "incongruent"


STIMULI = {
    "high_pitch": "up",
    "low_pitch": "down",
    "up_arrow": "up",
    "down_arrow": "down",
}
SOA_VALUES = [-200, -100, 0, 100, 200]  # ms (negative = visual first)


@dataclass
class MultisensoryTrial:
    trial_number: int
    modality: Modality
    visual_stim: Optional[str]
    auditory_stim: Optional[str]
    congruency: Congruency
    soa_ms: int
    correct_response: str
    response: Optional[str] = None
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class MultisensoryGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> MultisensoryTrial:
        modality = self.rng.choice(list(Modality))
        is_congruent = self.rng.random() < 0.5
        soa = int(self.rng.choice(SOA_VALUES))

        if modality == Modality.VISUAL:
            visual = "up_arrow" if self.rng.random() < 0.5 else "down_arrow"
            auditory = None
            correct = STIMULI[visual]
        elif modality == Modality.AUDITORY:
            auditory = "high_pitch" if self.rng.random() < 0.5 else "low_pitch"
            visual = None
            correct = STIMULI[auditory]
        else:  # BIMODAL
            if is_congruent:
                if self.rng.random() < 0.5:
                    visual, auditory = "up_arrow", "high_pitch"
                else:
                    visual, auditory = "down_arrow", "low_pitch"
            else:
                if self.rng.random() < 0.5:
                    visual, auditory = "up_arrow", "low_pitch"
                else:
                    visual, auditory = "down_arrow", "high_pitch"
            correct = STIMULI[visual]

        return MultisensoryTrial(
            trial_number=trial_number,
            modality=modality,
            visual_stim=visual,
            auditory_stim=auditory,
            congruency=Congruency.CONGRUENT if is_congruent else Congruency.INCONGRUENT,
            soa_ms=soa,
            correct_response=correct,
        )


class MultisensoryExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = MultisensoryGenerator(seed=seed)
        self.trials: List[MultisensoryTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[MultisensoryTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(
        self, trial: MultisensoryTrial, response: str, rt_ms: float
    ) -> MultisensoryTrial:
        trial.response = response
        trial.correct = response == trial.correct_response
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_mean_rt(self, modality: Modality) -> float:
        trials = [t for t in self.trials if t.modality == modality and t.correct]
        if not trials:
            return 0.0
        return np.mean([t.rt_ms for t in trials])

    def get_multisensory_gain(self) -> float:
        """Visual RT + Auditory RT - Bimodal RT (should be ~50-100ms)."""
        visual_rt = self.get_mean_rt(Modality.VISUAL)
        auditory_rt = self.get_mean_rt(Modality.AUDITORY)
        bimodal_rt = self.get_mean_rt(Modality.BIMODAL)
        return (visual_rt + auditory_rt) / 2 - bimodal_rt

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "visual_rt_ms": self.get_mean_rt(Modality.VISUAL),
            "auditory_rt_ms": self.get_mean_rt(Modality.AUDITORY),
            "bimodal_rt_ms": self.get_mean_rt(Modality.BIMODAL),
            "multisensory_gain_ms": self.get_multisensory_gain(),
            "accuracy": np.mean([t.correct for t in self.trials]),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "modality": t.modality.value,
                    "visual": t.visual_stim,
                    "auditory": t.auditory_stim,
                    "congruency": t.congruency.value,
                    "soa_ms": t.soa_ms,
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
    "experiment_name": "multisensory_integration",
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
    print("Multisensory Integration - Configuration Verification")
    print(f"Modalities: {[m.value for m in Modality]}")
    print(f"SOA Values: {SOA_VALUES}")


if __name__ == "__main__":
    verify()

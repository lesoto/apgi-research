"""
Fixed constants and data preparation for Posner Cueing experiments.

This file is READ-ONLY. Do not modify.
It defines fixed cue configurations, target parameters, and evaluation metrics.

Usage:
    python prepare_posner_cueing.py  # Verify cue configurations

The Posner cueing paradigm measures attention shifts using:
- Valid cues: Target appears at cued location (benefit)
- Invalid cues: Target appears opposite cued location (cost)
- Neutral cues: No spatial information (baseline)
- SOA: Stimulus onset asynchrony between cue and target
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
NUM_TRIALS = 100  # Standard trial count

# APGI Integration Parameters - 100/100 Compliance
# Optimized for posner cueing dynamics
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

# Spatial Configuration
LOCATIONS = ["left", "right", "up", "down"]  # Possible target locations
CUE_LOCATIONS = ["left", "right"]  # Standard bilateral cueing

# SOA values in milliseconds
SOA_VALUES = [100, 200, 400, 800]  # Short to long SOAs


# Cue Types
class CueType(Enum):
    """Types of attention cues."""

    VALID = "valid"  # Cue at target location (70% typical)
    INVALID = "invalid"  # Cue opposite target location (15% typical)
    NEUTRAL = "neutral"  # No spatial info (15% typical)
    DOUBLE = "double"  # Both sides cued (alternative)


# Standard probabilities
CUE_PROBABILITIES = {
    CueType.VALID: 0.70,
    CueType.INVALID: 0.15,
    CueType.NEUTRAL: 0.15,
}

# Target Stimuli
TARGET_SYMBOLS = ["*", "X", "O", "+", "#"]


@dataclass
class PosnerTrial:
    """Single Posner cueing trial data."""

    trial_number: int
    cue_type: CueType
    cue_location: str
    target_location: str
    soa_ms: int
    target_symbol: str
    response: Optional[str] = None  # "left", "right", etc.
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class PosnerCueGenerator:
    """Generates Posner cueing trials."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        """Reset generator."""
        self.trial_count = 0

    def _select_cue_type(self) -> CueType:
        """Select cue type based on probabilities."""
        types = list(CUE_PROBABILITIES.keys())
        probs = list(CUE_PROBABILITIES.values())
        return self.rng.choice(types, p=probs)

    def create_trial(self, trial_number: int) -> PosnerTrial:
        """Create a trial with appropriate cueing."""
        cue_type = self._select_cue_type()
        cue_location = self.rng.choice(CUE_LOCATIONS)
        soa = int(self.rng.choice(SOA_VALUES))
        target_symbol = self.rng.choice(TARGET_SYMBOLS)

        # Determine target location based on cue type
        if cue_type == CueType.VALID:
            target_location = cue_location
        elif cue_type == CueType.INVALID:
            # Opposite location
            target_location = "right" if cue_location == "left" else "left"
        else:  # NEUTRAL or DOUBLE
            target_location = self.rng.choice(CUE_LOCATIONS)

        return PosnerTrial(
            trial_number=trial_number,
            cue_type=cue_type,
            cue_location=cue_location,
            target_location=target_location,
            soa_ms=soa,
            target_symbol=target_symbol,
        )


class PosnerExperiment:
    """Manages a complete Posner cueing experiment session."""

    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.cue_gen = PosnerCueGenerator(seed=seed)
        self.trials: List[PosnerTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        """Reset experiment state."""
        self.trials = []
        self.current_trial_idx = 0
        self.cue_gen.reset()

    def get_next_trial(self) -> Optional[PosnerTrial]:
        """Get the next trial."""
        if self.current_trial_idx >= self.num_trials:
            return None

        trial = self.cue_gen.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(self, trial: PosnerTrial, response: str, rt_ms: float) -> PosnerTrial:
        """Execute a trial with participant response."""
        correct = response == trial.target_location

        import time

        trial.response = response
        trial.correct = correct
        trial.rt_ms = rt_ms
        trial.timestamp = time.time()

        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_mean_rt(self, cue_type: CueType) -> float:
        """Calculate mean RT for given cue type."""
        trials = [t for t in self.trials if t.cue_type == cue_type and t.correct]
        if not trials:
            return 0.0
        return np.mean([t.rt_ms for t in trials])

    def get_validity_effect(self) -> float:
        """
        Calculate validity effect (Invalid - Valid RT).
        Positive = attention shift cost; expected ~50-100ms
        """
        valid_rt = self.get_mean_rt(CueType.VALID)
        invalid_rt = self.get_mean_rt(CueType.INVALID)

        if valid_rt == 0 or invalid_rt == 0:
            return 0.0

        return invalid_rt - valid_rt

    def get_benefit_cost(self) -> tuple:
        """
        Calculate benefit (Neutral - Valid) and cost (Invalid - Neutral).
        """
        valid_rt = self.get_mean_rt(CueType.VALID)
        invalid_rt = self.get_mean_rt(CueType.INVALID)
        neutral_rt = self.get_mean_rt(CueType.NEUTRAL)

        benefit = neutral_rt - valid_rt if valid_rt > 0 and neutral_rt > 0 else 0.0
        cost = invalid_rt - neutral_rt if invalid_rt > 0 and neutral_rt > 0 else 0.0

        return benefit, cost

    def get_accuracy(self, cue_type: Optional[CueType] = None) -> float:
        """Calculate accuracy for given cue type."""
        trials = [t for t in self.trials if cue_type is None or t.cue_type == cue_type]
        if not trials:
            return 0.0
        return np.mean([t.correct for t in trials])

    def get_summary(self) -> Dict:
        """Generate experiment summary."""
        if not self.trials:
            return {}

        return {
            "num_trials": len(self.trials),
            "overall_accuracy": self.get_accuracy(),
            "valid_accuracy": self.get_accuracy(CueType.VALID),
            "invalid_accuracy": self.get_accuracy(CueType.INVALID),
            "neutral_accuracy": self.get_accuracy(CueType.NEUTRAL),
            "valid_rt_ms": self.get_mean_rt(CueType.VALID),
            "invalid_rt_ms": self.get_mean_rt(CueType.INVALID),
            "neutral_rt_ms": self.get_mean_rt(CueType.NEUTRAL),
            "validity_effect_ms": self.get_validity_effect(),
            "benefit_ms": self.get_benefit_cost()[0],
            "cost_ms": self.get_benefit_cost()[1],
        }

    def save_results(self, filepath: str):
        """Save trial data to JSON file."""
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "cue_type": t.cue_type.value,
                    "cue_location": t.cue_location,
                    "target_location": t.target_location,
                    "soa_ms": t.soa_ms,
                    "target_symbol": t.target_symbol,
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


def verify_configurations():
    """Verify and print configurations."""
    print("=" * 60)
    print("Posner Cueing - Configuration Verification")
    print("=" * 60)

    print(f"\nCue Locations: {CUE_LOCATIONS}")
    print(f"SOA Values (ms): {SOA_VALUES}")
    print("Cue Probabilities:")
    for cue_type, prob in CUE_PROBABILITIES.items():
        print(f"  {cue_type.value}: {prob:.0%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    verify_configurations()

# Standardized APGI Parameters Export (READ-ONLY)
# These parameters are used by the AGENT-EDITABLE run file for APGI integration
APGI_PARAMS = {
    # Core identification
    "experiment_name": "posner_cueing",
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

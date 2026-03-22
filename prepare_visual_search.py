"""
Fixed constants and data preparation for Visual Search experiments.

This file is READ-ONLY. Do not modify.
It defines fixed stimulus sets, display parameters, and evaluation metrics.

Usage:
    python prepare_visual_search.py  # Verify stimulus configurations

Visual Search paradigms:
- Feature search: Find target defined by single feature (e.g., red among green)
- Conjunction search: Target defined by feature conjunction (e.g., red X among green X and red O)
- Set size effect: Search time increases with display size (n)
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)
NUM_TRIALS = 80  # Standard trial count

# APGI Integration Parameters - 100/100 Compliance
# Optimized for visual search dynamics
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

# Display Parameters
DISPLAY_SIZES = [8, 16, 24, 32]  # Number of items in display
TARGET_PRESENT_PROB = 0.5  # 50% target present trials

# Stimulus Configuration
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["O", "X", "T", "L"]


# Search Types
class SearchType(Enum):
    FEATURE = "feature"  # Single feature search (pop-out)
    CONJUNCTION = "conjunction"  # Feature conjunction
    SPATIAL_CONFIG = "spatial_config"  # Spatial configuration search


@dataclass
class VSItem:
    """Single item in visual search display."""

    shape: str
    color: str
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    is_target: bool = False


@dataclass
class VSTrial:
    """Single Visual Search trial data."""

    trial_number: int
    search_type: SearchType
    display_size: int
    target_present: bool
    items: List[VSItem] = field(default_factory=list)
    target_item: Optional[VSItem] = None
    response: Optional[str] = None  # "present" or "absent"
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class VSDisplay:
    """Generates visual search displays."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        """Reset display generator."""
        self.trial_count = 0

    def _create_positions(self, n: int, min_dist: float = 0.15) -> List[tuple]:
        """Create non-overlapping positions in display."""
        positions = []
        max_attempts = 1000

        for _ in range(n):
            for _ in range(max_attempts):
                x = self.rng.uniform(0.1, 0.9)
                y = self.rng.uniform(0.1, 0.9)

                # Check minimum distance from existing positions
                too_close = False
                for px, py in positions:
                    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                    if dist < min_dist:
                        too_close = True
                        break

                if not too_close:
                    positions.append((x, y))
                    break
            else:
                # Fallback: just add position
                positions.append(
                    (self.rng.uniform(0.1, 0.9), self.rng.uniform(0.1, 0.9))
                )

        return positions

    def create_feature_search(
        self,
        display_size: int,
        target_present: bool,
        target_color: str = "red",
        distractor_color: str = "green",
    ) -> List[VSItem]:
        """
        Create feature search display (e.g., red target among green distractors).
        """
        positions = self._create_positions(display_size)
        items = []

        # Target position
        target_pos = self.rng.randint(0, display_size) if target_present else None

        for i, (x, y) in enumerate(positions):
            is_target = (i == target_pos) and target_present
            item = VSItem(
                shape="O",  # Same shape for all
                color=target_color if is_target else distractor_color,
                x=x,
                y=y,
                is_target=is_target,
            )
            items.append(item)

        return items

    def create_conjunction_search(
        self,
        display_size: int,
        target_present: bool,
    ) -> List[VSItem]:
        """
        Create conjunction search display (e.g., red X among red O and green X).
        Target: red X, Distractors: red O, green X
        """
        positions = self._create_positions(display_size)
        items = []

        target_pos = self.rng.randint(0, display_size) if target_present else None

        for i, (x, y) in enumerate(positions):
            is_target = (i == target_pos) and target_present

            if is_target:
                # Target: red X
                item = VSItem(shape="X", color="red", x=x, y=y, is_target=True)
            else:
                # Distractor: red O or green X
                if self.rng.random() < 0.5:
                    item = VSItem(shape="O", color="red", x=x, y=y)
                else:
                    item = VSItem(shape="X", color="green", x=x, y=y)

            items.append(item)

        return items

    def create_trial(
        self,
        trial_number: int,
        search_type: Optional[SearchType] = None,
        display_size: Optional[int] = None,
    ) -> VSTrial:
        """Create a trial with random parameters."""
        if search_type is None:
            search_type = self.rng.choice([SearchType.FEATURE, SearchType.CONJUNCTION])
        if display_size is None:
            display_size = self.rng.choice(DISPLAY_SIZES)

        target_present = self.rng.random() < TARGET_PRESENT_PROB

        if search_type == SearchType.FEATURE:
            items = self.create_feature_search(display_size, target_present)
        else:
            items = self.create_conjunction_search(display_size, target_present)

        target_item = next((item for item in items if item.is_target), None)

        return VSTrial(
            trial_number=trial_number,
            search_type=search_type,
            display_size=display_size,
            target_present=target_present,
            items=items,
            target_item=target_item,
        )


class VSExperiment:
    """Manages a complete Visual Search experiment session."""

    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.display_gen = VSDisplay(seed=seed)
        self.trials: List[VSTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        """Reset experiment state."""
        self.trials = []
        self.current_trial_idx = 0
        self.display_gen.reset()

    def get_next_trial(
        self,
        search_type: Optional[SearchType] = None,
        display_size: Optional[int] = None,
    ) -> Optional[VSTrial]:
        """Get the next trial."""
        if self.current_trial_idx >= self.num_trials:
            return None

        trial = self.display_gen.create_trial(
            self.current_trial_idx + 1,
            search_type=search_type,
            display_size=display_size,
        )
        return trial

    def run_trial(self, trial: VSTrial, response: str, rt_ms: float) -> VSTrial:
        """Execute a trial with participant response."""
        correct = (response == "present" and trial.target_present) or (
            response == "absent" and not trial.target_present
        )

        import time

        trial.response = response
        trial.correct = correct
        trial.rt_ms = rt_ms
        trial.timestamp = time.time()

        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_search_slope(self, search_type: SearchType, target_present: bool) -> float:
        """
        Calculate search slope (ms/item) for given search type.
        Classic result: conjunction > feature slopes.
        """
        by_size: Dict[int, List[float]] = {size: [] for size in DISPLAY_SIZES}

        for trial in self.trials:
            if (
                trial.search_type == search_type
                and trial.target_present == target_present
            ):
                by_size[trial.display_size].append(trial.rt_ms)

        # Linear regression of RT vs display size
        sizes = []
        rts = []
        for size in DISPLAY_SIZES:
            if by_size[size]:
                sizes.append(size)
                rts.append(np.mean(by_size[size]))

        if len(sizes) < 2:
            return 0.0

        slope, _ = np.polyfit(sizes, rts, 1)
        return slope

    def get_intercept(self, search_type: SearchType, target_present: bool) -> float:
        """Calculate intercept (baseline processing time)."""
        by_size: Dict[int, List[float]] = {size: [] for size in DISPLAY_SIZES}

        for trial in self.trials:
            if (
                trial.search_type == search_type
                and trial.target_present == target_present
            ):
                by_size[trial.display_size].append(trial.rt_ms)

        sizes = []
        rts = []
        for size in DISPLAY_SIZES:
            if by_size[size]:
                sizes.append(size)
                rts.append(np.mean(by_size[size]))

        if len(sizes) < 2:
            return 0.0

        _, intercept = np.polyfit(sizes, rts, 1)
        return intercept

    def get_accuracy(self, search_type: Optional[SearchType] = None) -> float:
        """Calculate accuracy for given search type."""
        trials = [
            t
            for t in self.trials
            if search_type is None or t.search_type == search_type
        ]
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
            "feature_accuracy": self.get_accuracy(SearchType.FEATURE),
            "conjunction_accuracy": self.get_accuracy(SearchType.CONJUNCTION),
            "feature_present_slope": self.get_search_slope(SearchType.FEATURE, True),
            "feature_absent_slope": self.get_search_slope(SearchType.FEATURE, False),
            "conjunction_present_slope": self.get_search_slope(
                SearchType.CONJUNCTION, True
            ),
            "conjunction_absent_slope": self.get_search_slope(
                SearchType.CONJUNCTION, False
            ),
            "mean_rt_ms": np.mean([t.rt_ms for t in self.trials]),
        }

    def save_results(self, filepath: str):
        """Save trial data to JSON file."""
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "search_type": t.search_type.value,
                    "display_size": t.display_size,
                    "target_present": t.target_present,
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
    print("Visual Search - Configuration Verification")
    print("=" * 60)

    print(f"\nDisplay Sizes: {DISPLAY_SIZES}")
    print(f"Target Present Probability: {TARGET_PRESENT_PROB}")
    print(f"Colors: {COLORS}")
    print(f"Shapes: {SHAPES}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    verify_configurations()

# Standardized APGI Parameters Export (READ-ONLY)
# These parameters are used by the AGENT-EDITABLE run file for APGI integration
APGI_PARAMS = {
    # Core identification
    "experiment_name": "visual_search",
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

"""Fixed constants for Simon Effect experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed task configurations and evaluation metrics.
"""
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TIME_BUDGET = 600
NUM_TRIALS = 80

# APGI Integration Parameters (DO NOT MODIFY)
APGI_ENABLED = True
APGI_TAU_S = 0.32
APGI_BETA = 1.4
APGI_THETA_0 = 0.4
APGI_ALPHA = 5.5


class TrialType(Enum):
    CONGRUENT = "congruent"
    INCONGRUENT = "incongruent"


COLORS = ["red", "green", "blue"]
POSITIONS = ["left", "right"]

TRIAL_PROBS = {TrialType.CONGRUENT: 0.5, TrialType.INCONGRUENT: 0.5}


@dataclass
class SimonTrial:
    trial_number: int
    stimulus_color: str
    stimulus_position: str
    trial_type: TrialType
    correct_response: str
    response: Optional[str] = None
    correct: bool = False
    rt_ms: float = 0.0
    timestamp: float = 0.0


class SimonGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> SimonTrial:
        trial_type = self.rng.choice(
            list(TRIAL_PROBS.keys()), p=list(TRIAL_PROBS.values())
        )
        color = self.rng.choice(COLORS)
        position = self.rng.choice(POSITIONS)

        # Color determines response
        color_idx = COLORS.index(color)
        correct_response = POSITIONS[color_idx % len(POSITIONS)]

        return SimonTrial(
            trial_number=trial_number,
            stimulus_color=color,
            stimulus_position=position,
            trial_type=trial_type,
            correct_response=correct_response,
        )


class SimonExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = SimonGenerator(seed=seed)
        self.trials: List[SimonTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self):
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[SimonTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        return trial

    def run_trial(self, trial: SimonTrial, response: str, rt_ms: float) -> SimonTrial:
        trial.response = response
        trial.correct = response == trial.correct_response
        trial.rt_ms = rt_ms
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_mean_rt(self, trial_type: TrialType) -> float:
        trials = [t for t in self.trials if t.trial_type == trial_type and t.correct]
        if not trials:
            return 0.0
        return np.mean([t.rt_ms for t in trials])

    def get_simon_effect(self) -> float:
        """Incongruent - Congruent RT (typically 20-40ms)."""
        return self.get_mean_rt(TrialType.INCONGRUENT) - self.get_mean_rt(
            TrialType.CONGRUENT
        )

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "congruent_rt_ms": self.get_mean_rt(TrialType.CONGRUENT),
            "incongruent_rt_ms": self.get_mean_rt(TrialType.INCONGRUENT),
            "simon_effect_ms": self.get_simon_effect(),
            "accuracy": np.mean([t.correct for t in self.trials]),
        }

    def save_results(self, filepath: str):
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "stimulus_color": t.stimulus_color,
                    "stimulus_position": t.stimulus_position,
                    "trial_type": t.trial_type.value,
                    "correct_response": t.correct_response,
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
    "experiment_name": "simon_effect",
    "enabled": True,
    "tau_s": 0.32,  # Surprise decay time constant (s)
    "beta": 1.4,  # Somatic influence gain
    "theta_0": 0.4,  # Baseline ignition threshold
    "alpha": 5.5,  # Sigmoid steepness
}


def verify():
    print("Simon Effect - Configuration Verification")
    print(f"Colors: {COLORS}")
    print(f"Positions: {POSITIONS}")


if __name__ == "__main__":
    verify()

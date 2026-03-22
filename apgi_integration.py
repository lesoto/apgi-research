"""
APGI Integration Module for Auto-Improvement Experiments

This module provides APGI (Active Predictive Global Ignition) framework integration
for all psychological experiments. It implements core dynamical system equations
that can be applied across different experimental paradigms.

Key Components:
- Foundational Equations (prediction error, precision, z-score)
- Core Ignition System (accumulated signal, ignition probability)
- Dynamical System Equations (signal, threshold, somatic marker dynamics)
- APGI Parameters (validated ranges)
- Running Statistics (for z-score normalization)

Usage in Experiments:
    from apgi_integration import APGIIntegration, APGIParameters
    
    # Initialize APGI for experiment
    apgi = APGIIntegration()
    
    # Compute ignition probability for trial
    ignition_prob = apgi.compute_ignition_probability(
        prediction_error=error,
        precision=Pi,
        somatic_marker=M
    )
    
    # Track APGI metrics alongside primary metrics
    apgi_metrics = apgi.get_trial_metrics()
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# APGI PARAMETERS
# =============================================================================


@dataclass
class APGIParameters:
    """
    APGI dynamical system parameters with validated ranges.

    These parameters control the ignition dynamics and can be tuned
    per experiment to match expected psychological phenomena.
    """

    # Timescales
    tau_S: float = 0.35  # Surprise decay time constant (200-500 ms)
    tau_theta: float = 30.0  # Threshold adaptation time constant (5-60 s)
    tau_M: float = 1.5  # Somatic marker time constant (1-2 s)

    # Threshold parameters
    theta_0: float = 0.5  # Baseline ignition threshold (0.1-1.0)
    alpha: float = 5.5  # Sigmoid steepness (3.0-8.0)

    # Somatic modulation
    beta: float = 1.5  # Somatic influence gain β_som (0.5-2.5)
    M_0: float = 0.0  # Reference somatic marker level

    # Sensitivities
    gamma_M: float = -0.3  # Metabolic sensitivity (-0.5 to 0.5)
    lambda_S: float = 0.1  # Metabolic coupling strength

    # Noise strengths
    sigma_S: float = 0.05  # Surprise noise
    sigma_theta: float = 0.02  # Threshold noise
    sigma_M: float = 0.03  # Somatic marker noise

    # Domain-specific thresholds
    theta_survival: float = 0.3  # Lower threshold for survival-relevant
    theta_neutral: float = 0.7  # Higher threshold for neutral content

    # Reset dynamics
    rho: float = 0.7  # Reset fraction after ignition (0.3-0.9)

    def validate(self) -> List[str]:
        """Validate parameters against physiological constraints."""
        violations = []

        if not (0.2 <= self.tau_S <= 0.5):
            violations.append(f"tau_S={self.tau_S:.3f}s not in [0.2, 0.5]s")
        if not (0.5 <= self.beta <= 2.5):
            violations.append(f"beta={self.beta:.2f} not in [0.5, 2.5]")
        if not (3.0 <= self.alpha <= 8.0):
            violations.append(f"alpha={self.alpha:.1f} not in [3.0, 8.0]")
        if not (0.3 <= self.rho <= 0.9):
            violations.append(f"rho={self.rho:.2f} not in [0.3, 0.9]")

        return violations


# =============================================================================
# RUNNING STATISTICS
# =============================================================================


class RunningStatistics:
    """
    Running statistics for z-score normalization.

    Implements exponential moving average for mean and variance:
        dμ/dt = α_μ(x(t) - μ(t))
        dσ²/dt = α_σ((x(t) - μ(t))² - σ²(t))
    """

    def __init__(self, alpha_mu: float = 0.01, alpha_sigma: float = 0.005):
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.mu = 0.0
        self.variance = 1.0
        self._n_updates = 0

    def update(self, value: float, dt: float = 1.0) -> Tuple[float, float]:
        """Update statistics with new value, return (mean, std)."""
        # Update mean
        dmu = self.alpha_mu * (value - self.mu)
        self.mu += dmu * dt

        # Update variance
        dvar = self.alpha_sigma * ((value - self.mu) ** 2 - self.variance)
        self.variance += dvar * dt
        self.variance = max(0.01, self.variance)

        self._n_updates += 1
        return self.mu, np.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        """Compute z-score for given value."""
        if self._n_updates == 0:
            return 0.0
        std = np.sqrt(self.variance)
        if std <= 0:
            return 0.0
        return (value - self.mu) / std

    def reset(self):
        """Reset statistics to initial state."""
        self.mu = 0.0
        self.variance = 1.0
        self._n_updates = 0


# =============================================================================
# CORE APGI EQUATIONS
# =============================================================================


class CoreEquations:
    """Core APGI equations for prediction error, precision, and ignition."""

    @staticmethod
    def prediction_error(observed: float, predicted: float) -> float:
        """Compute prediction error: ε = x - x̂"""
        return observed - predicted

    @staticmethod
    def precision(variance: float) -> float:
        """Compute precision: Π = 1/σ²"""
        if variance <= 0:
            return 1e6
        return 1.0 / variance

    @staticmethod
    def z_score(error: float, mean: float, std: float) -> float:
        """Compute z-score: z = (ε - μ) / σ"""
        if abs(std) < 1e-10:
            return 0.0
        return (error - mean) / std

    @staticmethod
    def accumulated_signal(
        Pi_e: float, eps_e: float, Pi_i_eff: float, eps_i: float
    ) -> float:
        """
        Compute accumulated signal (surprise):
            S = ½Π^e(ε^e)² + ½Π^i_eff(ε^i)²
        """
        ext_surprise = 0.5 * Pi_e * (eps_e**2)
        int_surprise = 0.5 * Pi_i_eff * (eps_i**2)
        return ext_surprise + int_surprise

    @staticmethod
    def effective_interoceptive_precision(
        Pi_i_baseline: float, M: float, M_0: float, beta: float
    ) -> float:
        """
        Compute effective interoceptive precision with sigmoid modulation:
            Π^i_eff = Π^i_baseline · [1 + β·σ(M - M_0)]
        """
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(-(M - M_0), -500, 500)))
        modulation = 1.0 + beta * sigmoid
        return Pi_i_baseline * modulation

    @staticmethod
    def ignition_probability(S: float, theta: float, alpha: float) -> float:
        """
        Compute ignition probability:
            P(ignite) = σ(α(S - θ)) = 1 / (1 + exp(-α(S - θ)))
        """
        z = alpha * (S - theta)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            z_exp = np.exp(z)
            return z_exp / (1.0 + z_exp)


# =============================================================================
# DYNAMICAL SYSTEM
# =============================================================================


class DynamicalSystem:
    """
    APGI dynamical system for tracking state evolution.

    Implements coupled differential equations for:
    - Accumulated surprise S(t)
    - Dynamic threshold θ(t)
    - Somatic marker M(t)
    """

    def __init__(self, params: Optional[APGIParameters] = None):
        self.params = params or APGIParameters()
        self.rng = np.random.default_rng()

        # State variables
        self.S = 0.0  # Accumulated surprise
        self.theta = self.params.theta_0  # Current threshold
        self.M = 0.0  # Somatic marker

        # Running statistics for errors
        self.stats_exteroceptive = RunningStatistics()
        self.stats_interoceptive = RunningStatistics()

        # History tracking
        self.S_history: List[float] = []
        self.theta_history: List[float] = []
        self.M_history: List[float] = []
        self.ignition_history: List[bool] = []

    def step(
        self,
        prediction_error_ext: float,
        prediction_error_int: float,
        precision_ext: float,
        precision_int_baseline: float,
        dt: float = 0.01,
    ) -> Dict[str, float]:
        """
        Advance dynamical system by one time step.

        Args:
            prediction_error_ext: Exteroceptive prediction error
            prediction_error_int: Interoceptive prediction error
            precision_ext: Exteroceptive precision
            precision_int_baseline: Baseline interoceptive precision
            dt: Time step in seconds

        Returns:
            Dictionary with current state values
        """
        params = self.params

        # Update running statistics
        _, _ = self.stats_exteroceptive.update(prediction_error_ext, dt)
        _, _ = self.stats_interoceptive.update(prediction_error_int, dt)

        # Compute z-scores
        z_e = self.stats_exteroceptive.z_score(prediction_error_ext)
        z_i = self.stats_interoceptive.z_score(prediction_error_int)

        # Compute effective interoceptive precision
        Pi_i_eff = CoreEquations.effective_interoceptive_precision(
            precision_int_baseline, self.M, params.M_0, params.beta
        )

        # Signal dynamics: dS/dt = -τ_S⁻¹S + input + noise
        input_S = CoreEquations.accumulated_signal(
            precision_ext, prediction_error_ext, Pi_i_eff, prediction_error_int
        )
        noise_S = params.sigma_S * self.rng.normal() / np.sqrt(dt)
        dS_dt = -self.S / params.tau_S + input_S + noise_S
        self.S = max(0.0, self.S + dS_dt * dt)

        # Threshold dynamics: dθ/dt = (θ_0 - θ)/τ_θ + γ_M·M + λ·S + noise
        noise_theta = params.sigma_theta * self.rng.normal() / np.sqrt(dt)
        dtheta_dt = (
            (params.theta_0 - self.theta) / params.tau_theta
            + params.gamma_M * self.M
            + params.lambda_S * self.S
            + noise_theta
        )
        self.theta = max(0.01, self.theta + dtheta_dt * dt)

        # Somatic marker dynamics: dM/dt = (tanh(β_M·ε^i) - M)/τ_M + noise
        M_star = np.tanh(params.beta * prediction_error_int)
        noise_M = params.sigma_M * self.rng.normal() / np.sqrt(dt)
        dM_dt = (M_star - self.M) / params.tau_M + noise_M
        self.M = np.clip(self.M + dM_dt * dt, -2.0, 2.0)

        # Compute ignition probability
        ignition_prob = CoreEquations.ignition_probability(
            self.S, self.theta, params.alpha
        )

        # Check for ignition event
        ignited = self.rng.random() < ignition_prob
        if ignited:
            # Reset after ignition
            self.S *= 1.0 - params.rho

        # Record history
        self.S_history.append(self.S)
        self.theta_history.append(self.theta)
        self.M_history.append(self.M)
        self.ignition_history.append(ignited)

        return {
            "S": self.S,
            "theta": self.theta,
            "M": self.M,
            "z_e": z_e,
            "z_i": z_i,
            "Pi_i_eff": Pi_i_eff,
            "ignition_prob": ignition_prob,
            "ignited": ignited,
        }

    def reset(self):
        """Reset dynamical system to initial state."""
        self.S = 0.0
        self.theta = self.params.theta_0
        self.M = 0.0
        self.stats_exteroceptive.reset()
        self.stats_interoceptive.reset()
        self.S_history.clear()
        self.theta_history.clear()
        self.M_history.clear()
        self.ignition_history.clear()

    def get_metabolic_cost(self) -> float:
        """Compute total metabolic cost from surprise history."""
        if len(self.S_history) == 0:
            return 0.0
        return np.trapz(self.S_history, dx=0.01)

    def get_ignition_rate(self) -> float:
        """Compute proportion of trials with ignition."""
        if len(self.ignition_history) == 0:
            return 0.0
        return np.mean(self.ignition_history)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for the session."""
        return {
            "mean_surprise": np.mean(self.S_history) if self.S_history else 0.0,
            "mean_threshold": np.mean(self.theta_history)
            if self.theta_history
            else self.params.theta_0,
            "mean_somatic_marker": np.mean(self.M_history) if self.M_history else 0.0,
            "ignition_rate": self.get_ignition_rate(),
            "metabolic_cost": self.get_metabolic_cost(),
            "final_surprise": self.S,
            "final_threshold": self.theta,
            "final_somatic_marker": self.M,
        }


# =============================================================================
# APGI INTEGRATION FOR EXPERIMENTS
# =============================================================================


class APGIIntegration:
    """
    High-level APGI integration for psychological experiments.

    This class provides a simplified interface for incorporating APGI
    metrics into any experimental paradigm.
    """

    def __init__(self, params: Optional[APGIParameters] = None):
        self.params = params or APGIParameters()
        self.dynamics = DynamicalSystem(self.params)

        # Trial-level tracking
        self.trial_metrics: List[Dict[str, float]] = []

        # Experiment-level summary
        self.experiment_summary: Optional[Dict[str, float]] = None

    def process_trial(
        self,
        observed: float,
        predicted: float,
        trial_type: str = "neutral",
        precision_ext: Optional[float] = None,
        precision_int: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Process a single trial with APGI dynamics.

        Args:
            observed: Observed value (response, RT, etc.)
            predicted: Expected/predicted value
            trial_type: "neutral", "survival", or "congruent"/"incongruent"
            precision_ext: Optional exteroceptive precision override
            precision_int: Optional interoceptive precision override

        Returns:
            Dictionary of APGI metrics for this trial
        """
        # Compute prediction errors
        error_ext = CoreEquations.prediction_error(observed, predicted)
        error_int = error_ext * 0.3  # Interoceptive coupling

        # Default precision based on trial type
        if precision_ext is None:
            precision_ext = 2.0 if trial_type in ["survival", "incongruent"] else 1.0
        if precision_int is None:
            precision_int = 1.5 if trial_type in ["survival", "incongruent"] else 1.0

        # Advance dynamics
        state = self.dynamics.step(
            prediction_error_ext=error_ext,
            prediction_error_int=error_int,
            precision_ext=precision_ext,
            precision_int_baseline=precision_int,
        )

        # Adjust threshold for domain
        if trial_type == "survival":
            state["effective_threshold"] = self.params.theta_survival
        else:
            state["effective_threshold"] = self.params.theta_neutral

        # Store trial metrics
        self.trial_metrics.append(state)

        return state

    def process_rt_trial(
        self, rt: float, expected_rt: float, correct: bool, trial_type: str = "neutral"
    ) -> Dict[str, float]:
        """
        Process a reaction time trial.

        Args:
            rt: Observed reaction time
            expected_rt: Expected/baseline reaction time
            correct: Whether response was correct
            trial_type: Trial type for domain-specific threshold

        Returns:
            Dictionary of APGI metrics
        """
        # Normalize RT to prediction error
        error = (rt - expected_rt) / expected_rt

        # Correctness affects precision
        precision_ext = 3.0 if correct else 1.0

        return self.process_trial(
            observed=error,
            predicted=0.0,  # Expected: no deviation from baseline
            trial_type=trial_type,
            precision_ext=precision_ext,
        )

    def process_choice_trial(
        self,
        choice: int,
        expected_choice: int,
        reward: float,
        trial_type: str = "neutral",
    ) -> Dict[str, float]:
        """
        Process a choice/decision trial.

        Args:
            choice: Made choice (0 or 1)
            expected_choice: Expected choice
            reward: Received reward
            trial_type: Trial type

        Returns:
            Dictionary of APGI metrics
        """
        # Choice prediction error
        choice_error = float(choice != expected_choice)

        # Reward affects interoceptive precision
        precision_int = 1.0 + reward

        return self.process_trial(
            observed=choice_error,
            predicted=0.0,
            trial_type=trial_type,
            precision_int=precision_int,
        )

    def process_detection_trial(
        self,
        detected: bool,
        target_present: bool,
        confidence: Optional[float] = None,
        trial_type: str = "neutral",
    ) -> Dict[str, float]:
        """
        Process a detection trial (e.g., visual search, change blindness).

        Args:
            detected: Whether participant detected target
            target_present: Whether target was present
            confidence: Optional confidence rating
            trial_type: Trial type

        Returns:
            Dictionary of APGI metrics
        """
        # Detection error
        correct = detected == target_present
        error = 0.0 if correct else 1.0

        # Confidence affects precision
        precision_ext = (
            confidence if confidence is not None else (2.0 if correct else 1.0)
        )

        return self.process_trial(
            observed=error,
            predicted=0.0,
            trial_type=trial_type,
            precision_ext=precision_ext,
        )

    def finalize(self) -> Dict[str, float]:
        """
        Finalize experiment and compute summary metrics.

        Returns:
            Dictionary of experiment-level APGI metrics
        """
        self.experiment_summary = self.dynamics.get_summary()

        # Add trial-level aggregates
        if self.trial_metrics:
            self.experiment_summary["mean_ignition_prob"] = np.mean(
                [m["ignition_prob"] for m in self.trial_metrics]
            )
            self.experiment_summary["std_ignition_prob"] = np.std(
                [m["ignition_prob"] for m in self.trial_metrics]
            )

        return self.experiment_summary

    def reset(self):
        """Reset for new experiment."""
        self.dynamics.reset()
        self.trial_metrics.clear()
        self.experiment_summary = None

    def get_report(self) -> str:
        """Generate human-readable APGI report."""
        if self.experiment_summary is None:
            self.finalize()

        s = self.experiment_summary
        report = f"""
APGI Metrics Report
==================
Ignition Dynamics:
  - Ignition Rate: {s['ignition_rate']:.2%}
  - Mean Ignition Probability: {s.get('mean_ignition_prob', 0):.3f}
  
Surprise Accumulation:
  - Mean Surprise: {s['mean_surprise']:.3f}
  - Final Surprise: {s['final_surprise']:.3f}
  
Threshold Dynamics:
  - Mean Threshold: {s['mean_threshold']:.3f}
  - Final Threshold: {s['final_threshold']:.3f}
  
Somatic Markers:
  - Mean Somatic Marker: {s['mean_somatic_marker']:.3f}
  - Final Somatic Marker: {s['final_somatic_marker']:.3f}
  
Metabolic Cost:
  - Total Metabolic Cost: {s['metabolic_cost']:.3f}
"""
        return report


# =============================================================================
# EXPERIMENT-SPECIFIC APGI CONFIGURATIONS
# =============================================================================


def get_apgi_config_for_experiment(experiment_name: str) -> APGIParameters:
    """
    Get experiment-specific APGI parameter configuration.

    Different experiments may benefit from different parameter tuning
    based on the expected psychological phenomena.
    """
    configs = {
        # Attention experiments - higher precision
        "attentional_blink": APGIParameters(
            tau_S=0.25, beta=1.8, theta_0=0.4, alpha=6.0
        ),
        "posner_cueing": APGIParameters(tau_S=0.30, beta=1.5, theta_0=0.35, alpha=5.5),
        "visual_search": APGIParameters(tau_S=0.35, beta=1.3, theta_0=0.5, alpha=5.0),
        "change_blindness": APGIParameters(
            tau_S=0.40, beta=1.2, theta_0=0.6, alpha=4.5
        ),
        "inattentional_blindness": APGIParameters(
            tau_S=0.35, beta=1.5, theta_0=0.55, alpha=5.0
        ),
        # Memory experiments - longer timescales
        "drm_false_memory": APGIParameters(
            tau_S=0.45, beta=1.0, theta_0=0.5, alpha=4.5
        ),
        "sternberg_memory": APGIParameters(
            tau_S=0.40, beta=1.2, theta_0=0.45, alpha=5.0
        ),
        "working_memory_span": APGIParameters(
            tau_S=0.38, beta=1.3, theta_0=0.5, alpha=5.2
        ),
        "dual_n_back": APGIParameters(tau_S=0.35, beta=1.4, theta_0=0.45, alpha=5.5),
        # Decision-making experiments - higher somatic influence
        "iowa_gambling_task": APGIParameters(
            tau_S=0.40, beta=2.0, theta_0=0.4, alpha=5.0
        ),
        "go_no_go": APGIParameters(tau_S=0.30, beta=1.6, theta_0=0.35, alpha=6.0),
        "stop_signal": APGIParameters(tau_S=0.28, beta=1.7, theta_0=0.3, alpha=6.5),
        "simon_effect": APGIParameters(tau_S=0.32, beta=1.4, theta_0=0.4, alpha=5.5),
        "eriksen_flanker": APGIParameters(
            tau_S=0.30, beta=1.5, theta_0=0.35, alpha=5.8
        ),
        # Perception experiments
        "binocular_rivalry": APGIParameters(
            tau_S=0.50, beta=1.0, theta_0=0.6, alpha=4.0
        ),
        "masking": APGIParameters(tau_S=0.25, beta=1.8, theta_0=0.3, alpha=7.0),
        "stroop_effect": APGIParameters(tau_S=0.30, beta=1.6, theta_0=0.35, alpha=6.0),
        "navon_task": APGIParameters(tau_S=0.35, beta=1.3, theta_0=0.45, alpha=5.2),
        # Learning experiments
        "serial_reaction_time": APGIParameters(
            tau_S=0.35, beta=1.2, theta_0=0.5, alpha=5.0
        ),
        "artificial_grammar_learning": APGIParameters(
            tau_S=0.40, beta=1.1, theta_0=0.55, alpha=4.8
        ),
        "probabilistic_category_learning": APGIParameters(
            tau_S=0.38, beta=1.3, theta_0=0.45, alpha=5.2
        ),
        # Interoception experiments
        "interoceptive_gating": APGIParameters(
            tau_S=0.45, beta=2.2, theta_0=0.5, alpha=5.0
        ),
        "somatic_marker_priming": APGIParameters(
            tau_S=0.40, beta=2.0, theta_0=0.45, alpha=5.2
        ),
        "metabolic_cost": APGIParameters(tau_S=0.50, beta=1.8, theta_0=0.6, alpha=4.5),
        # Other experiments
        "time_estimation": APGIParameters(tau_S=0.45, beta=1.2, theta_0=0.5, alpha=5.0),
        "virtual_navigation": APGIParameters(
            tau_S=0.40, beta=1.3, theta_0=0.5, alpha=5.0
        ),
        "multisensory_integration": APGIParameters(
            tau_S=0.35, beta=1.4, theta_0=0.45, alpha=5.5
        ),
        "ai_benchmarking": APGIParameters(tau_S=0.35, beta=1.5, theta_0=0.5, alpha=5.5),
    }

    return configs.get(experiment_name, APGIParameters())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_apgi_enhanced_metric(
    primary_metric: float,
    apgi_summary: Dict[str, float],
    weight_ignition: float = 0.3,
    weight_metabolic: float = 0.2,
) -> float:
    """
    Compute an APGI-enhanced composite metric.

    Combines the primary experiment metric with APGI-derived measures
    to create a more comprehensive performance indicator.

    Args:
        primary_metric: The experiment's primary performance metric
        apgi_summary: APGI summary from finalize()
        weight_ignition: Weight for ignition rate in composite
        weight_metabolic: Weight for metabolic efficiency in composite

    Returns:
        Composite metric incorporating APGI dynamics
    """
    # Normalize ignition rate (higher is generally better for awareness)
    ignition_score = apgi_summary.get("ignition_rate", 0.5)

    # Metabolic efficiency (lower cost is better)
    metabolic_cost = apgi_summary.get("metabolic_cost", 1.0)
    metabolic_efficiency = 1.0 / (1.0 + metabolic_cost)

    # Composite metric
    composite = (
        (1 - weight_ignition - weight_metabolic) * primary_metric
        + weight_ignition * ignition_score
        + weight_metabolic * metabolic_efficiency
    )

    return composite


def format_apgi_output(apgi_summary: Dict[str, float]) -> str:
    """Format APGI metrics for standard output."""
    return (
        f"apgi_ignition_rate: {apgi_summary.get('ignition_rate', 0):.3f}\n"
        f"apgi_metabolic_cost: {apgi_summary.get('metabolic_cost', 0):.3f}\n"
        f"apgi_mean_surprise: {apgi_summary.get('mean_surprise', 0):.3f}\n"
        f"apgi_mean_threshold: {apgi_summary.get('mean_threshold', 0):.3f}\n"
        f"apgi_mean_somatic_marker: {apgi_summary.get('mean_somatic_marker', 0):.3f}"
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "APGIParameters",
    "RunningStatistics",
    "CoreEquations",
    "DynamicalSystem",
    "APGIIntegration",
    "get_apgi_config_for_experiment",
    "compute_apgi_enhanced_metric",
    "format_apgi_output",
]

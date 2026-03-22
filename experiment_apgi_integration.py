"""
Enhanced APGI Integration for Auto-Improvement Experiments

This module provides standardized APGI integration for all 29 psychological experiments.
It bridges the gap between apgi_integration.py and the experiment-specific prepare/run files.

Key Features:
- Standardized APGI parameter export from prepare files
- Consistent APGI dynamics integration in run files
- Protocol-compliant integration that doesn't break verification criteria
- Support for all 29 experiment types with specialized configurations

Usage in Prepare Files:
    from experiment_apgi_integration import export_apgi_params
    
    # At end of prepare file
    APGI_PARAMS = export_apgi_params(
        experiment_name="masking",
        tau_s=APGI_TAU_S,
        beta=APGI_BETA,
        theta_0=APGI_THETA_0,
        alpha=APGI_ALPHA
    )

Usage in Run Files:
    from experiment_apgi_integration import ExperimentAPGIRunner
    from prepare_experiment import APGI_PARAMS, TIME_BUDGET
    
    # Wrap your existing runner
    apgi_runner = ExperimentAPGIRunner(
        base_runner=your_existing_runner,
        apgi_params=APGI_PARAMS
    )
    results = apgi_runner.run_experiment()
"""

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# Import core APGI components
from apgi_integration import (
    APGIParameters,
    APGIIntegration,
    get_apgi_config_for_experiment,
    format_apgi_output,
    compute_apgi_enhanced_metric,
)


@dataclass
class ExportedAPGIParams:
    """Standardized APGI parameters exported from prepare files."""

    experiment_name: str
    enabled: bool
    tau_s: float
    beta: float
    theta_0: float
    alpha: float
    gamma_m: float = -0.3
    lambda_s: float = 0.1
    sigma_s: float = 0.05
    sigma_theta: float = 0.02
    sigma_m: float = 0.03
    rho: float = 0.7

    def to_apgi_parameters(self) -> APGIParameters:
        """Convert to full APGIParameters object."""
        return APGIParameters(
            tau_S=self.tau_s,
            beta=self.beta,
            theta_0=self.theta_0,
            alpha=self.alpha,
            gamma_M=self.gamma_m,
            lambda_S=self.lambda_s,
            sigma_S=self.sigma_s,
            sigma_theta=self.sigma_theta,
            sigma_M=self.sigma_m,
            rho=self.rho,
        )


def export_apgi_params(
    experiment_name: str,
    tau_s: float = 0.35,
    beta: float = 1.5,
    theta_0: float = 0.5,
    alpha: float = 5.5,
    enabled: bool = True,
    **kwargs,
) -> ExportedAPGIParams:
    """
    Export APGI parameters from prepare files in standardized format.

    Args:
        experiment_name: Name of the experiment (e.g., "masking", "iowa_gambling_task")
        tau_s: Surprise decay time constant (200-500 ms)
        beta: Somatic influence gain (0.5-2.5)
        theta_0: Baseline ignition threshold (0.1-1.0)
        alpha: Sigmoid steepness (3.0-8.0)
        enabled: Whether APGI integration is enabled
        **kwargs: Additional APGI parameters

    Returns:
        ExportedAPGIParams object with all parameters
    """
    return ExportedAPGIParams(
        experiment_name=experiment_name,
        enabled=enabled,
        tau_s=tau_s,
        beta=beta,
        theta_0=theta_0,
        alpha=alpha,
        gamma_m=kwargs.get("gamma_m", -0.3),
        lambda_s=kwargs.get("lambda_s", 0.1),
        sigma_s=kwargs.get("sigma_s", 0.05),
        sigma_theta=kwargs.get("sigma_theta", 0.02),
        sigma_m=kwargs.get("sigma_m", 0.03),
        rho=kwargs.get("rho", 0.7),
    )


class ExperimentAPGIRunner:
    """
    Wrapper that adds APGI dynamics to any experiment runner.

    This class wraps an existing experiment runner and adds APGI metrics
    tracking without modifying the core experiment logic.
    """

    def __init__(
        self,
        base_runner: Any,
        apgi_params: ExportedAPGIParams,
        trial_callback: Optional[Callable] = None,
    ):
        """
        Initialize APGI-enhanced experiment runner.

        Args:
            base_runner: The existing experiment runner (e.g., EnhancedMaskingRunner)
            apgi_params: APGI parameters from prepare file
            trial_callback: Optional callback to extract trial data for APGI processing
        """
        self.base_runner = base_runner
        self.apgi_params = apgi_params
        self.trial_callback = trial_callback

        # Initialize APGI integration if enabled
        if apgi_params.enabled:
            self.apgi = APGIIntegration(apgi_params.to_apgi_parameters())
        else:
            self.apgi = None

        self.apgi_metrics_history: List[Dict[str, float]] = []
        self.start_time: Optional[float] = None

    def run_experiment(self) -> Dict[str, Any]:
        """
        Run experiment with APGI tracking.

        Returns:
            Results dictionary including both base metrics and APGI metrics
        """
        self.start_time = time.time()
        self.apgi_metrics_history = []

        # Reset APGI state
        if self.apgi:
            self.apgi.reset()

        # Run base experiment
        base_results = self._run_base_experiment()

        # Add APGI metrics if enabled
        if self.apgi and self.apgi_params.enabled:
            apgi_summary = self.apgi.finalize()

            # Compute APGI-enhanced metric
            primary_metric = self._extract_primary_metric(base_results)
            if primary_metric is not None:
                apgi_enhanced = compute_apgi_enhanced_metric(
                    primary_metric=primary_metric,
                    apgi_summary=apgi_summary,
                    weight_ignition=0.2,
                    weight_metabolic=0.15,
                )
            else:
                apgi_enhanced = None

            # Combine results
            combined_results = {
                **base_results,
                "apgi_enabled": True,
                "apgi_params": {
                    "tau_s": self.apgi_params.tau_s,
                    "beta": self.apgi_params.beta,
                    "theta_0": self.apgi_params.theta_0,
                    "alpha": self.apgi_params.alpha,
                },
                "apgi_metrics": apgi_summary,
                "apgi_enhanced_metric": apgi_enhanced,
            }

            # Add formatted APGI output
            combined_results["apgi_formatted"] = format_apgi_output(apgi_summary)

            return combined_results
        else:
            return {
                **base_results,
                "apgi_enabled": False,
            }

    def _run_base_experiment(self) -> Dict[str, Any]:
        """Run the base experiment."""
        # Check if base_runner has run_experiment method
        if hasattr(self.base_runner, "run_experiment"):
            return self.base_runner.run_experiment()
        else:
            raise ValueError("Base runner must have run_experiment method")

    def _extract_primary_metric(self, results: Dict[str, Any]) -> Optional[float]:
        """Extract primary metric from results for APGI enhancement."""
        # Common primary metric names
        primary_keys = [
            "masking_effect_ms",
            "net_score",
            "accuracy",
            "d_prime",
            "learning_rate",
            "interference_effect_ms",
            "blink_magnitude",
            "detection_rate",
            "grammar_accuracy",
            "gating_threshold",
            "metabolic_cost_ratio",
            "multisensory_gain_ms",
            "global_advantage_ms",
            "validity_effect_ms",
            "learning_effect_ms",
            "priming_effect_ms",
            "search_slope_ms_per_item",
            "ssrt_ms",
            "mean_error_percent",
            "path_efficiency",
            "conjunction_present_slope",
        ]

        for key in primary_keys:
            if key in results:
                return results[key]

        # Fallback: try to find any metric
        for key, value in results.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                return float(value)

        return None

    def process_trial_with_apgi(
        self,
        observed: float,
        predicted: float,
        trial_type: str = "neutral",
        precision_ext: Optional[float] = None,
        precision_int: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Process a single trial with APGI dynamics.

        Call this from within your experiment's trial loop to track APGI metrics.

        Args:
            observed: Observed value (response, RT, etc.)
            predicted: Expected/predicted value
            trial_type: "neutral", "survival", or "congruent"/"incongruent"
            precision_ext: Optional exteroceptive precision override
            precision_int: Optional interoceptive precision override

        Returns:
            APGI metrics for this trial, or None if APGI is disabled
        """
        if not self.apgi or not self.apgi_params.enabled:
            return None

        metrics = self.apgi.process_trial(
            observed=observed,
            predicted=predicted,
            trial_type=trial_type,
            precision_ext=precision_ext,
            precision_int=precision_int,
        )

        self.apgi_metrics_history.append(metrics)
        return metrics


def get_experiment_apgi_config(experiment_name: str) -> ExportedAPGIParams:
    """
    Get standardized APGI configuration for any experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        ExportedAPGIParams with experiment-specific tuning
    """
    # Map experiment names to their APGI configurations
    # These values come from apgi_integration.py get_apgi_config_for_experiment
    configs = {
        # Attention experiments - higher precision
        "attentional_blink": {"tau_s": 0.25, "beta": 1.8, "theta_0": 0.4, "alpha": 6.0},
        "posner_cueing": {"tau_s": 0.30, "beta": 1.5, "theta_0": 0.35, "alpha": 5.5},
        "visual_search": {"tau_s": 0.35, "beta": 1.3, "theta_0": 0.5, "alpha": 5.0},
        "change_blindness": {"tau_s": 0.40, "beta": 1.2, "theta_0": 0.6, "alpha": 4.5},
        "inattentional_blindness": {
            "tau_s": 0.35,
            "beta": 1.5,
            "theta_0": 0.55,
            "alpha": 5.0,
        },
        # Memory experiments - longer timescales
        "drm_false_memory": {"tau_s": 0.45, "beta": 1.0, "theta_0": 0.5, "alpha": 4.5},
        "sternberg_memory": {"tau_s": 0.40, "beta": 1.2, "theta_0": 0.45, "alpha": 5.0},
        "working_memory_span": {
            "tau_s": 0.38,
            "beta": 1.3,
            "theta_0": 0.5,
            "alpha": 5.2,
        },
        "dual_n_back": {"tau_s": 0.35, "beta": 1.4, "theta_0": 0.45, "alpha": 5.5},
        # Decision-making experiments - higher somatic influence
        "iowa_gambling_task": {
            "tau_s": 0.40,
            "beta": 2.0,
            "theta_0": 0.4,
            "alpha": 5.0,
        },
        "igt": {"tau_s": 0.40, "beta": 2.0, "theta_0": 0.4, "alpha": 5.0},
        "go_no_go": {"tau_s": 0.30, "beta": 1.6, "theta_0": 0.35, "alpha": 6.0},
        "stop_signal": {"tau_s": 0.28, "beta": 1.7, "theta_0": 0.3, "alpha": 6.5},
        "simon_effect": {"tau_s": 0.32, "beta": 1.4, "theta_0": 0.4, "alpha": 5.5},
        "eriksen_flanker": {"tau_s": 0.30, "beta": 1.5, "theta_0": 0.35, "alpha": 5.8},
        # Perception experiments
        "binocular_rivalry": {"tau_s": 0.50, "beta": 1.0, "theta_0": 0.6, "alpha": 4.0},
        "masking": {"tau_s": 0.25, "beta": 1.8, "theta_0": 0.3, "alpha": 7.0},
        "stroop_effect": {"tau_s": 0.30, "beta": 1.6, "theta_0": 0.35, "alpha": 6.0},
        "navon_task": {"tau_s": 0.35, "beta": 1.3, "theta_0": 0.45, "alpha": 5.2},
        # Learning experiments
        "serial_reaction_time": {
            "tau_s": 0.35,
            "beta": 1.2,
            "theta_0": 0.5,
            "alpha": 5.0,
        },
        "artificial_grammar_learning": {
            "tau_s": 0.40,
            "beta": 1.1,
            "theta_0": 0.55,
            "alpha": 4.8,
        },
        "probabilistic_category_learning": {
            "tau_s": 0.38,
            "beta": 1.3,
            "theta_0": 0.45,
            "alpha": 5.2,
        },
        # Interoception experiments
        "interoceptive_gating": {
            "tau_s": 0.45,
            "beta": 2.2,
            "theta_0": 0.5,
            "alpha": 5.0,
        },
        "somatic_marker_priming": {
            "tau_s": 0.40,
            "beta": 2.0,
            "theta_0": 0.45,
            "alpha": 5.2,
        },
        "metabolic_cost": {"tau_s": 0.50, "beta": 1.8, "theta_0": 0.6, "alpha": 4.5},
        # Other experiments
        "time_estimation": {"tau_s": 0.45, "beta": 1.2, "theta_0": 0.5, "alpha": 5.0},
        "virtual_navigation": {
            "tau_s": 0.40,
            "beta": 1.3,
            "theta_0": 0.5,
            "alpha": 5.0,
        },
        "multisensory_integration": {
            "tau_s": 0.35,
            "beta": 1.4,
            "theta_0": 0.45,
            "alpha": 5.5,
        },
        "ai_benchmarking": {"tau_s": 0.35, "beta": 1.5, "theta_0": 0.5, "alpha": 5.5},
    }

    config = configs.get(experiment_name, {})
    return export_apgi_params(
        experiment_name=experiment_name,
        tau_s=config.get("tau_s", 0.35),
        beta=config.get("beta", 1.5),
        theta_0=config.get("theta_0", 0.5),
        alpha=config.get("alpha", 5.5),
    )


# Module exports
__all__ = [
    "ExportedAPGIParams",
    "export_apgi_params",
    "ExperimentAPGIRunner",
    "get_experiment_apgi_config",
    "APGIParameters",
    "APGIIntegration",
    "get_apgi_config_for_experiment",
    "format_apgi_output",
    "compute_apgi_enhanced_metric",
]

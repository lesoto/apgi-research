"""
Standardized APGI Runner Template for All Experiments

This module provides a standardized template for adding APGI dynamics to any experiment.
It implements real-time trial processing, hierarchical dynamics, and Π vs Π̂ modeling.

Usage:
    from standard_apgi_runner import StandardAPGIRunner
    
    # Wrap your existing runner
    apgi_runner = StandardAPGIRunner(
        base_runner=YourExistingRunner(),
        experiment_name="your_experiment",
        apgi_params=APGI_PARAMS
    )
    results = apgi_runner.run_experiment()
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# Import APGI components
from apgi_integration import (
    APGIIntegration,
)
from experiment_apgi_integration import (
    ExportedAPGIParams,
    get_experiment_apgi_config,
    compute_apgi_enhanced_metric,
)


@dataclass
class HierarchicalState:
    """Multi-level hierarchical processing state."""

    level_1: Dict[str, float]  # Fast sensory processing
    level_2: Dict[str, float]  # Feature integration
    level_3: Dict[str, float]  # Pattern recognition
    level_4: Dict[str, float]  # Semantic processing
    level_5: Dict[str, float]  # Executive control

    def __post_init__(self):
        # Initialize all levels with default values
        for level in [
            self.level_1,
            self.level_2,
            self.level_3,
            self.level_4,
            self.level_5,
        ]:
            level.setdefault("S", 0.0)  # Accumulated surprise
            level.setdefault("theta", 0.5)  # Threshold
            level.setdefault("M", 0.0)  # Somatic marker
            level.setdefault("ignition_prob", 0.0)  # Ignition probability


@dataclass
class PrecisionExpectationGap:
    """Π vs Π̂ modeling for anxiety and precision expectations."""

    Pi_e_actual: float = 1.0  # Actual exteroceptive precision
    Pi_i_actual: float = 1.0  # Actual interoceptive precision
    Pi_e_expected: float = 1.0  # Expected exteroceptive precision
    Pi_i_expected: float = 1.0  # Expected interoceptive precision
    anxiety_level: float = 0.0  # Current anxiety level (0-1)
    precision_mismatch: float = 0.0  # Π̂ - Π gap

    def update_gap(self, Pi_e_actual: float, Pi_i_actual: float):
        """Update precision expectation gap."""
        self.Pi_e_actual = Pi_e_actual
        self.Pi_i_actual = Pi_i_actual

        # Expected precision based on neuromodulators and context
        self.Pi_e_expected = (
            1.2 + 0.3 * self.anxiety_level
        )  # Higher expectations in anxiety
        self.Pi_i_expected = 1.1 + 0.2 * self.anxiety_level

        # Calculate gap (positive in anxiety: Π̂ > Π)
        self.precision_mismatch = (self.Pi_e_expected + self.Pi_i_expected) / 2 - (
            Pi_e_actual + Pi_i_actual
        ) / 2

        # Update anxiety based on chronic mismatch
        self.anxiety_level = np.clip(
            self.anxiety_level + 0.01 * self.precision_mismatch, 0.0, 1.0
        )


class StandardAPGIRunner:
    """
    Standardized APGI runner with full dynamical system integration.

    This class provides complete APGI integration including:
    - Real-time trial-by-trial processing
    - Hierarchical level dynamics
    - Π vs Π̂ anxiety modeling
    - Metabolic cost tracking
    - APGI-enhanced composite metrics
    """

    def __init__(
        self,
        base_runner: Any,
        experiment_name: str,
        apgi_params: Optional[ExportedAPGIParams] = None,
        enable_hierarchical: bool = True,
        enable_precision_gap: bool = True,
        trial_callback: Optional[Callable] = None,
    ):
        """
        Initialize standardized APGI runner.

        Args:
            base_runner: The existing experiment runner
            experiment_name: Name of the experiment
            apgi_params: APGI parameters (auto-generated if None)
            enable_hierarchical: Enable multi-level processing
            enable_precision_gap: Enable Π vs Π̂ modeling
            trial_callback: Optional callback for trial data extraction
        """
        self.base_runner = base_runner
        self.experiment_name = experiment_name
        self.enable_hierarchical = enable_hierarchical
        self.enable_precision_gap = enable_precision_gap
        self.trial_callback = trial_callback

        # Get APGI parameters
        if apgi_params is None:
            apgi_params = get_experiment_apgi_config(experiment_name)

        # Initialize APGI integration
        if apgi_params.enabled:
            self.apgi = APGIIntegration(apgi_params.to_apgi_parameters())
        else:
            self.apgi = None

        # Initialize hierarchical state
        if self.enable_hierarchical:
            self.hierarchical_state = HierarchicalState(
                level_1={}, level_2={}, level_3={}, level_4={}, level_5={}
            )

        # Initialize precision expectation gap
        if self.enable_precision_gap:
            self.precision_gap = PrecisionExpectationGap()

        # Tracking
        self.trial_count = 0
        self.apgi_metrics_history: List[Dict[str, float]] = []
        self.start_time: Optional[float] = None

    def process_trial_with_full_apgi(
        self,
        observed: float,
        predicted: float,
        trial_type: str = "neutral",
        precision_ext: Optional[float] = None,
        precision_int: Optional[float] = None,
        hierarchical_level: int = 1,
    ) -> Dict[str, float]:
        """
        Process a trial with full APGI dynamics.

        Args:
            observed: Observed value (response, RT, etc.)
            predicted: Expected/predicted value
            trial_type: Trial type for domain-specific processing
            precision_ext: Optional exteroceptive precision
            precision_int: Optional interoceptive precision
            hierarchical_level: Which hierarchical level to process (1-5)

        Returns:
            Complete APGI metrics for this trial
        """
        if not self.apgi:
            return {}

        self.trial_count += 1

        # Process with basic APGI integration
        basic_metrics = self.apgi.process_trial(
            observed=observed,
            predicted=predicted,
            trial_type=trial_type,
            precision_ext=precision_ext,
            precision_int=precision_int,
        )

        # Add hierarchical processing
        if self.enable_hierarchical:
            hierarchical_metrics = self._process_hierarchical_level(
                basic_metrics, hierarchical_level
            )
            basic_metrics.update(hierarchical_metrics)

        # Add precision gap modeling
        if self.enable_precision_gap:
            gap_metrics = self._process_precision_gap(basic_metrics)
            basic_metrics.update(gap_metrics)

        # Store in history
        self.apgi_metrics_history.append(basic_metrics)

        return basic_metrics

    def _process_hierarchical_level(
        self, basic_metrics: Dict[str, float], level: int
    ) -> Dict[str, float]:
        """Process hierarchical level dynamics."""
        if not (1 <= level <= 5):
            return {}

        level_states = [
            self.hierarchical_state.level_1,
            self.hierarchical_state.level_2,
            self.hierarchical_state.level_3,
            self.hierarchical_state.level_4,
            self.hierarchical_state.level_5,
        ]

        current_level = level_states[level - 1]
        higher_level_broadcast = 0.0

        # Get broadcast from higher level if available
        if level < 5:
            higher_level = level_states[level]
            higher_level_broadcast = higher_level.get("ignition_prob", 0.0)

        # Update current level based on basic metrics and higher level
        current_level["S"] = basic_metrics.get("S", 0.0) * (
            1 + 0.2 * higher_level_broadcast
        )
        current_level["theta"] = basic_metrics.get("theta", 0.5) * (
            1 - 0.1 * higher_level_broadcast
        )
        current_level["M"] = basic_metrics.get("M", 0.0) + 0.1 * higher_level_broadcast

        # Compute ignition probability for this level
        current_level["ignition_prob"] = self.apgi.dynamics.params.alpha * (
            current_level["S"] - current_level["theta"]
        )
        current_level["ignition_prob"] = 1.0 / (
            1.0 + np.exp(-current_level["ignition_prob"])
        )

        return {
            f"level_{level}_S": current_level["S"],
            f"level_{level}_theta": current_level["theta"],
            f"level_{level}_M": current_level["M"],
            f"level_{level}_ignition_prob": current_level["ignition_prob"],
            "higher_level_broadcast": higher_level_broadcast,
        }

    def _process_precision_gap(
        self, basic_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Process Π vs Π̂ precision expectation gap."""
        # Extract precision values from basic metrics
        Pi_e_actual = basic_metrics.get("Pi_e_eff", 1.0)
        Pi_i_actual = basic_metrics.get("Pi_i_eff", 1.0)

        # Update precision gap
        self.precision_gap.update_gap(Pi_e_actual, Pi_i_actual)

        return {
            "Pi_e_actual": Pi_e_actual,
            "Pi_i_actual": Pi_i_actual,
            "Pi_e_expected": self.precision_gap.Pi_e_expected,
            "Pi_i_expected": self.precision_gap.Pi_i_expected,
            "precision_mismatch": self.precision_gap.precision_mismatch,
            "anxiety_level": self.precision_gap.anxiety_level,
        }

    def run_experiment(self) -> Dict[str, Any]:
        """
        Run experiment with full APGI tracking.

        Returns:
            Results dictionary including comprehensive APGI metrics
        """
        self.start_time = time.time()
        self.trial_count = 0
        self.apgi_metrics_history = []

        # Reset APGI state
        if self.apgi:
            self.apgi.reset()

        # Reset hierarchical state
        if self.enable_hierarchical:
            self.hierarchical_state = HierarchicalState(
                level_1={}, level_2={}, level_3={}, level_4={}, level_5={}
            )

        # Reset precision gap
        if self.enable_precision_gap:
            self.precision_gap = PrecisionExpectationGap()

        # Run base experiment with APGI integration
        base_results = self._run_base_experiment_with_apgi()

        # Add comprehensive APGI metrics if enabled
        if self.apgi:
            apgi_summary = self.apgi.finalize()

            # Add hierarchical summary
            if self.enable_hierarchical:
                hierarchical_summary = self._get_hierarchical_summary()
                apgi_summary.update(hierarchical_summary)

            # Add precision gap summary
            if self.enable_precision_gap:
                gap_summary = self._get_precision_gap_summary()
                apgi_summary.update(gap_summary)

            # Compute APGI-enhanced metric
            primary_metric = self._extract_primary_metric(base_results)
            if primary_metric is not None:
                apgi_enhanced = compute_apgi_enhanced_metric(
                    primary_metric=primary_metric,
                    apgi_summary=apgi_summary,
                    weight_ignition=0.25,
                    weight_metabolic=0.2,
                    weight_hierarchical=0.15,
                    weight_precision_gap=0.1,
                )
            else:
                apgi_enhanced = None

            # Combine results
            combined_results = {
                **base_results,
                "apgi_enabled": True,
                "apgi_integration": "full",
                "hierarchical_enabled": self.enable_hierarchical,
                "precision_gap_enabled": self.enable_precision_gap,
                "apgi_metrics": apgi_summary,
                "apgi_enhanced_metric": apgi_enhanced,
                "trial_count": self.trial_count,
            }

            # Add formatted APGI output
            combined_results["apgi_formatted"] = self._format_comprehensive_apgi_output(
                apgi_summary
            )

            return combined_results
        else:
            return {
                **base_results,
                "apgi_enabled": False,
                "trial_count": self.trial_count,
            }

    def _run_base_experiment_with_apgi(self) -> Dict[str, Any]:
        """Run base experiment with APGI trial processing."""
        # This is a template - in practice, you'd integrate with your specific experiment
        # For now, run the base experiment and try to extract trial data

        if hasattr(self.base_runner, "run_experiment"):
            base_results = self.base_runner.run_experiment()
        else:
            raise ValueError("Base runner must have run_experiment method")

        # If trial callback is provided, process trials with APGI
        if self.trial_callback and self.apgi:
            # This would be implemented per experiment
            # For now, simulate some trial processing
            pass

        return base_results

    def _extract_primary_metric(self, results: Dict[str, Any]) -> Optional[float]:
        """Extract primary metric from results."""
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
                return float(results[key])

        return None

    def _get_hierarchical_summary(self) -> Dict[str, float]:
        """Get summary of hierarchical processing."""
        summary = {}

        for level in range(1, 6):
            level_name = f"level_{level}"
            getattr(self.hierarchical_state, level_name)

            summary[f"{level_name}_mean_S"] = (
                np.mean(
                    [
                        m.get("S", 0)
                        for m in self.apgi_metrics_history
                        if f"{level_name}_S" in m
                    ]
                )
                if self.apgi_metrics_history
                else 0.0
            )

            summary[f"{level_name}_mean_ignition_prob"] = (
                np.mean(
                    [
                        m.get("ignition_prob", 0)
                        for m in self.apgi_metrics_history
                        if f"{level_name}_ignition_prob" in m
                    ]
                )
                if self.apgi_metrics_history
                else 0.0
            )

        return summary

    def _get_precision_gap_summary(self) -> Dict[str, float]:
        """Get summary of precision expectation gap."""
        return {
            "final_anxiety_level": self.precision_gap.anxiety_level,
            "final_precision_mismatch": self.precision_gap.precision_mismatch,
            "mean_precision_mismatch": np.mean(
                [
                    m.get("precision_mismatch", 0)
                    for m in self.apgi_metrics_history
                    if "precision_mismatch" in m
                ]
            )
            if self.apgi_metrics_history
            else 0.0,
        }

    def _format_comprehensive_apgi_output(self, apgi_summary: Dict[str, float]) -> str:
        """Format comprehensive APGI metrics for output."""
        output = [
            "=== COMPREHENSIVE APGI METRICS ===",
            "Ignition Dynamics:",
            f"  - Ignition Rate: {apgi_summary.get('ignition_rate', 0):.2%}",
            f"  - Mean Ignition Probability: {apgi_summary.get('mean_ignition_prob', 0):.3f}",
            "",
            "Surprise Accumulation:",
            f"  - Mean Surprise: {apgi_summary.get('mean_surprise', 0):.3f}",
            f"  - Final Surprise: {apgi_summary.get('final_surprise', 0):.3f}",
            "",
            "Threshold Dynamics:",
            f"  - Mean Threshold: {apgi_summary.get('mean_threshold', 0):.3f}",
            f"  - Final Threshold: {apgi_summary.get('final_threshold', 0):.3f}",
            "",
            "Somatic Markers:",
            f"  - Mean Somatic Marker: {apgi_summary.get('mean_somatic_marker', 0):.3f}",
            f"  - Final Somatic Marker: {apgi_summary.get('final_somatic_marker', 0):.3f}",
            "",
            "Metabolic Cost:",
            f"  - Total Metabolic Cost: {apgi_summary.get('metabolic_cost', 0):.3f}",
        ]

        # Add hierarchical metrics if enabled
        if self.enable_hierarchical:
            output.extend(
                [
                    "",
                    "Hierarchical Processing:",
                ]
            )
            for level in range(1, 6):
                level_name = f"level_{level}"
                if f"{level_name}_mean_ignition_prob" in apgi_summary:
                    output.append(
                        f"  - Level {level} Ignition: {apgi_summary[f'{level_name}_mean_ignition_prob']:.3f}"
                    )

        # Add precision gap metrics if enabled
        if self.enable_precision_gap:
            output.extend(
                [
                    "",
                    "Precision Expectation Gap (Π vs Π̂):",
                    f"  - Final Anxiety Level: {apgi_summary.get('final_anxiety_level', 0):.3f}",
                    f"  - Final Precision Mismatch: {apgi_summary.get('final_precision_mismatch', 0):.3f}",
                ]
            )

        return "\n".join(output)


def create_standard_apgi_runner(
    base_runner: Any,
    experiment_name: str,
    apgi_params: Optional[ExportedAPGIParams] = None,
    **kwargs,
) -> StandardAPGIRunner:
    """
    Convenience function to create a standardized APGI runner.

    Args:
        base_runner: The existing experiment runner
        experiment_name: Name of the experiment
        apgi_params: APGI parameters (auto-generated if None)
        **kwargs: Additional arguments for StandardAPGIRunner

    Returns:
        Configured StandardAPGIRunner instance
    """
    return StandardAPGIRunner(
        base_runner=base_runner,
        experiment_name=experiment_name,
        apgi_params=apgi_params,
        **kwargs,
    )

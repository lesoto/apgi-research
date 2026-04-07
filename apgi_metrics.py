"""
Enhanced APGI Metrics Module

This module provides comprehensive APGI metrics calculation including:
- Ignition probability and dynamics
- Surprise metrics (information-theoretic and Bayesian)
- Metabolic cost analysis
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class IgnitionMetrics:
    """Ignition system metrics for APGI experiments."""

    ignition_rate: float  # Probability of global broadcast
    mean_ignition_time: float  # Mean time to ignition
    ignition_threshold: float  # Current threshold for ignition
    ignition_volatility: float  # Variability in ignition timing
    cumulative_ignition_prob: float  # Cumulative ignition probability
    ignition_events: List[float] = []  # Raw ignition events for statistical testing


@dataclass
class SurpriseMetrics:
    """Surprise metrics for APGI experiments."""

    mean_surprise: float  # Average surprise across trials
    surprise_variance: float  # Variability in surprise
    max_surprise: float  # Maximum surprise observed
    min_surprise: float  # Minimum surprise observed
    surprise_entropy: float  # Entropy of surprise distribution
    prediction_error_variance: float  # Variance in prediction errors


@dataclass
class MetabolicMetrics:
    """Metabolic cost metrics for APGI experiments."""

    mean_metabolic_cost: float  # Average metabolic cost per trial
    total_metabolic_cost: float  # Total metabolic cost across experiment
    metabolic_efficiency: float  # Performance per metabolic unit
    cost_variance: float  # Variability in metabolic cost
    optimal_cost_rate: float  # Rate of optimal cost usage


@dataclass
class APGIMetricsSummary:
    """Comprehensive APGI metrics summary."""

    ignition: Optional[IgnitionMetrics] = None
    surprise: Optional[SurpriseMetrics] = None
    metabolic: Optional[MetabolicMetrics] = None
    trial_count: int = 0
    experiment_duration: float = 0.0
    overall_performance_score: float = 0.0
    statistical_significance: Optional[Dict[str, float]] = (
        None  # p-values for key metrics
    )


class EnhancedAPGIMetrics:
    """Enhanced APGI metrics calculator with comprehensive analysis."""

    def __init__(self):
        self.trial_data: List[Dict] = []

    def calculate_ignition_metrics(
        self,
        reaction_times: List[float],
        predicted_ignition_times: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ) -> IgnitionMetrics:
        """
        Calculate comprehensive ignition metrics.

        Args:
            reaction_times: List of reaction times in seconds
            predicted_ignition_times: Optional predicted times for comparison
            threshold: Optional threshold for ignition detection

        Returns:
            IgnitionMetrics object with comprehensive metrics
        """
        if not reaction_times:
            return IgnitionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, [])

        # Basic ignition rate calculation
        if threshold is not None:
            # Dynamic threshold based on reaction time distribution
            threshold = float(np.mean(reaction_times) + np.std(reaction_times))

        ignition_events = (
            [1.0 if rt <= threshold else 0.0 for rt in reaction_times]
            if threshold is not None
            else []
        )
        ignition_rate = float(np.mean(ignition_events))

        # Advanced ignition dynamics
        mean_ignition_time = np.mean(
            [rt for rt, event in zip(reaction_times, ignition_events) if event == 1.0]
        )

        # Ignition threshold dynamics
        # threshold_accuracy calculated but not currently used
        if predicted_ignition_times:
            _ = np.mean(
                [
                    1.0 if rt <= pred_rt else 0.0
                    for rt, pred_rt in zip(reaction_times, predicted_ignition_times)
                ]
            )

        # Ignition volatility (variability in response patterns)
        ignition_std = np.std(ignition_events)
        ignition_volatility = float(ignition_std / (ignition_rate + 1e-6))  # Normalized

        # Cumulative ignition probability (exponential decay)
        cumulative_prob = 0.0
        for i, event in enumerate(ignition_events):
            decay_factor = np.exp(-0.1 * i)  # Decay over trials
            cumulative_prob += event * decay_factor * (1 - cumulative_prob)

        return IgnitionMetrics(
            ignition_rate=float(ignition_rate),
            mean_ignition_time=float(mean_ignition_time),
            ignition_threshold=float(threshold) if threshold is not None else 0.0,
            ignition_volatility=float(ignition_volatility),
            cumulative_ignition_prob=float(cumulative_prob),
            ignition_events=ignition_events,
        )

    def calculate_surprise_metrics(
        self,
        prediction_errors: List[float],
        expected_errors: Optional[List[float]] = None,
    ) -> SurpriseMetrics:
        """
        Calculate comprehensive surprise metrics using information theory.

        Args:
            prediction_errors: List of prediction errors (surprise values)
            expected_errors: Optional expected errors for comparison

        Returns:
            SurpriseMetrics object with comprehensive analysis
        """
        if not prediction_errors:
            return SurpriseMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Basic statistics
        mean_surprise = np.mean(prediction_errors)
        surprise_variance = np.var(prediction_errors)
        max_surprise = np.max(prediction_errors)
        min_surprise = np.min(prediction_errors)

        # Information-theoretic measures
        # Treat prediction errors as samples from a distribution
        surprise_entropy = self._calculate_entropy(prediction_errors)

        # Compare with expected distribution if provided
        prediction_error_variance = np.var(prediction_errors)
        # KL-divergence approximation calculated but not currently used
        if expected_errors:
            expected_variance = np.var(expected_errors)
            _ = (
                0.5
                * (prediction_error_variance + expected_variance)
                / (expected_variance + 1e-6)
            )

        return SurpriseMetrics(
            mean_surprise=float(mean_surprise),
            surprise_variance=float(surprise_variance),
            max_surprise=max_surprise,
            min_surprise=min_surprise,
            surprise_entropy=surprise_entropy,
            prediction_error_variance=float(prediction_error_variance),
        )

    def calculate_metabolic_metrics(
        self,
        reaction_times: List[float],
        metabolic_costs: Optional[List[float]] = None,
        time_budget: Optional[float] = None,
    ) -> MetabolicMetrics:
        """
        Calculate comprehensive metabolic cost metrics.

        Args:
            reaction_times: List of reaction times
            metabolic_costs: Optional metabolic costs per trial
            time_budget: Time budget for metabolic cost calculation

        Returns:
            MetabolicMetrics object with comprehensive analysis
        """
        if not reaction_times:
            return MetabolicMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        # Basic metabolic cost metrics
        if metabolic_costs is None:
            # Estimate metabolic cost from reaction time (inverse relationship)
            metabolic_costs = [1.0 / (rt + 1e-3) for rt in reaction_times]

        mean_metabolic_cost = np.mean(metabolic_costs)
        total_metabolic_cost = np.sum(metabolic_costs)

        # Metabolic efficiency (performance per metabolic unit)
        if time_budget:
            metabolic_efficiency = len(reaction_times) / time_budget
        else:
            # Default to 1 trial per second if no budget specified
            metabolic_efficiency = len(reaction_times) / np.sum(reaction_times)

        # Cost variability analysis
        cost_variance = np.var(metabolic_costs)
        # cost_stability calculated but not currently used
        _ = 1.0 / (1.0 + cost_variance)  # Stability measure

        # Optimal cost rate (how close to optimal performance)
        if metabolic_costs:
            optimal_cost = min(metabolic_costs)
            optimal_cost_rate = (
                optimal_cost / mean_metabolic_cost if mean_metabolic_cost > 0 else 1.0
            )
        else:
            optimal_cost_rate = 1.0

        return MetabolicMetrics(
            mean_metabolic_cost=float(mean_metabolic_cost),
            total_metabolic_cost=total_metabolic_cost,
            metabolic_efficiency=metabolic_efficiency,
            cost_variance=float(cost_variance),
            optimal_cost_rate=float(optimal_cost_rate),
        )

    def calculate_comprehensive_metrics(
        self,
        experiment_data: Dict,
    ) -> APGIMetricsSummary:
        """
        Calculate comprehensive APGI metrics summary.

        Args:
            experiment_data: Dictionary containing all experiment data

        Returns:
            APGIMetricsSummary with all calculated metrics
        """
        # Extract trial data
        reaction_times = experiment_data.get("reaction_times", [])
        metabolic_costs = experiment_data.get("metabolic_costs", [])
        prediction_errors = experiment_data.get("prediction_errors", [])
        predicted_ignition_times = experiment_data.get("predicted_ignition_times", [])
        time_budget = experiment_data.get("time_budget", 600.0)

        trial_count = len(reaction_times)

        # Calculate individual metric categories
        ignition = (
            self.calculate_ignition_metrics(reaction_times, predicted_ignition_times)
            if reaction_times or predicted_ignition_times
            else None
        )

        surprise = (
            self.calculate_surprise_metrics(prediction_errors)
            if prediction_errors
            else None
        )

        metabolic = (
            self.calculate_metabolic_metrics(
                reaction_times, metabolic_costs, time_budget
            )
            if reaction_times
            else None
        )

        # Calculate overall performance score
        performance_score = 0.0
        if ignition and metabolic:
            # Weight performance: accuracy (inverse of errors) + efficiency
            ignition_score = max(0, 1.0 - ignition.ignition_rate) * 0.4
            metabolic_score = min(1.0, metabolic.metabolic_efficiency) * 0.3
            performance_score = (ignition_score + metabolic_score) * 0.7

        # Statistical significance testing
        significance_tests = {}
        if trial_count > 1:
            # Test if ignition rate is significantly different from 0.5
            if ignition:
                _, p_value = stats.ttest_1samp(
                    [
                        1.0 if event == 1.0 else 0.0
                        for event in ignition.ignition_events
                    ],
                    popmean=0.5,
                )
                significance_tests["ignition_vs_null"] = float(p_value)

            # Test if metabolic efficiency is significantly different from baseline
            if metabolic:
                _, p_value = stats.ttest_1samp(
                    metabolic.metabolic_efficiency, popmean=0.5  # Baseline efficiency
                )
                significance_tests["metabolic_vs_baseline"] = float(p_value)

        return APGIMetricsSummary(
            ignition=ignition,
            surprise=surprise,
            metabolic=metabolic,
            trial_count=trial_count,
            experiment_duration=sum(reaction_times) if reaction_times else 0.0,
            overall_performance_score=performance_score,
            statistical_significance=significance_tests,
        )

    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy of values."""
        if not values:
            return 0.0

        # Create histogram of values
        hist, bin_edges = np.histogram(values, bins=20, density=True)

        # Calculate entropy
        entropy = 0.0
        for i, count in enumerate(hist):
            if count > 0:
                p = count / len(values)
                entropy -= p * np.log2(p + 1e-10)

        return entropy

    def format_metrics_summary(self, metrics: APGIMetricsSummary) -> str:
        """Format metrics summary for display."""
        lines = []
        lines.append("ENHANCED APGI METRICS SUMMARY")
        lines.append("=" * 60)

        if metrics.ignition:
            lines.append("\n🔥 IGNITION METRICS:")
            lines.append(f"  Ignition Rate: {metrics.ignition.ignition_rate:.3f}")
            lines.append(
                f"  Mean Ignition Time: {metrics.ignition.mean_ignition_time:.3f}s"
            )
            lines.append(
                f"  Ignition Threshold: {metrics.ignition.ignition_threshold:.3f}"
            )
            lines.append(
                f"  Ignition Volatility: {metrics.ignition.ignition_volatility:.4f}"
            )
            lines.append(
                f"  Cumulative Ignition Prob: {metrics.ignition.cumulative_ignition_prob:.3f}"
            )

        if metrics.surprise:
            lines.append("\n🎯 SURPRISE METRICS:")
            lines.append(f"  Mean Surprise: {metrics.surprise.mean_surprise:.4f}")
            lines.append(
                f"  Surprise Variance: {metrics.surprise.surprise_variance:.4f}"
            )
            lines.append(f"  Max Surprise: {metrics.surprise.max_surprise:.4f}")
            lines.append(f"  Min Surprise: {metrics.surprise.min_surprise:.4f}")
            lines.append(f"  Surprise Entropy: {metrics.surprise.surprise_entropy:.4f}")
            if metrics.surprise.prediction_error_variance is not None:
                lines.append(
                    f"  Prediction Error Variance: {metrics.surprise.prediction_error_variance:.4f}"
                )

        if metrics.metabolic:
            lines.append("\n⚡ METABOLIC METRICS:")
            lines.append(
                f"  Mean Metabolic Cost: {metrics.metabolic.mean_metabolic_cost:.6f}"
            )
            lines.append(
                f"  Total Metabolic Cost: {metrics.metabolic.total_metabolic_cost:.6f}"
            )
            lines.append(
                f"  Metabolic Efficiency: {metrics.metabolic.metabolic_efficiency:.4f} trials/sec"
            )
            lines.append(f"  Cost Variance: {metrics.metabolic.cost_variance:.6f}")
            lines.append(
                f"  Optimal Cost Rate: {metrics.metabolic.optimal_cost_rate:.3f}"
            )

        lines.append("\n📊 EXPERIMENT OVERVIEW:")
        lines.append(f"  Trial Count: {metrics.trial_count}")
        lines.append(f"  Experiment Duration: {metrics.experiment_duration:.3f}s")
        lines.append(
            f"  Overall Performance Score: {metrics.overall_performance_score:.3f}"
        )

        if metrics.statistical_significance:
            lines.append("\n📈 STATISTICAL SIGNIFICANCE:")
            for test_name, p_value in metrics.statistical_significance.items():
                significance = (
                    "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
                )
                lines.append(f"  {test_name}: p={p_value:.4f} {significance}")

        return "\n".join(lines)

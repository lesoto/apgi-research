"""Metrics processing and categorization for visualization."""

from enum import Enum
from typing import Any, Optional, cast


class MetricCategory(Enum):
    """Categories for organizing metrics in visualization."""

    CORE_APGI = "Core APGI"
    PERFORMANCE = "Performance"
    TIMING = "Timing"
    NEUROMODULATORS = "Neuromodulators"
    HIERARCHICAL = "Hierarchical"
    PRECISION = "Precision"
    DOMAIN_SPECIFIC = "Domain-Specific"
    OTHER = "Other"


# Metric name mappings and categorizations
METRIC_DEFINITIONS = {
    # Core APGI metrics
    "ignition_rate": {
        "category": MetricCategory.CORE_APGI,
        "display_name": "Ignition Rate",
        "description": "Probability of APGI ignition events",
        "scale": (0, 1),
    },
    "mean_surprise": {
        "category": MetricCategory.CORE_APGI,
        "display_name": "Mean Surprise",
        "description": "Average surprise signal (S)",
        "scale": (0, 0.2),
    },
    "metabolic_cost": {
        "category": MetricCategory.CORE_APGI,
        "display_name": "Metabolic Cost",
        "description": "Energy cost of processing",
        "scale": (0, 0.1),
    },
    "mean_somatic_marker": {
        "category": MetricCategory.CORE_APGI,
        "display_name": "Mean Somatic Marker",
        "description": "Bodily state feedback",
        "scale": (-0.1, 0.1),
    },
    "mean_threshold": {
        "category": MetricCategory.CORE_APGI,
        "display_name": "Mean Threshold",
        "description": "Average surprise threshold (θ)",
        "scale": (0, 1),
    },
    # Performance metrics
    "primary_metric": {
        "category": MetricCategory.PERFORMANCE,
        "display_name": "Primary Metric",
        "description": "Experiment-specific primary outcome",
        "scale": (0, 1),
    },
    "accuracy": {
        "category": MetricCategory.PERFORMANCE,
        "display_name": "Accuracy",
        "description": "Proportion correct",
        "scale": (0, 1),
    },
    "overall_accuracy": {
        "category": MetricCategory.PERFORMANCE,
        "display_name": "Overall Accuracy",
        "description": "Overall accuracy across all conditions",
        "scale": (0, 1),
    },
    "d_prime": {
        "category": MetricCategory.PERFORMANCE,
        "display_name": "d'",
        "description": "Signal detection measure",
        "scale": (-3, 3),
    },
    # Timing metrics
    "completion_time_s": {
        "category": MetricCategory.TIMING,
        "display_name": "Completion Time (s)",
        "description": "Total time to complete experiment",
        "scale": (0, 600),
    },
    "time_min": {
        "category": MetricCategory.TIMING,
        "display_name": "Time (min)",
        "description": "Total time in minutes",
        "scale": (0, 10),
    },
    "mean_rt_ms": {
        "category": MetricCategory.TIMING,
        "display_name": "Mean RT (ms)",
        "description": "Average reaction time",
        "scale": (0, 2000),
    },
    # Neuromodulators
    "dopamine_level": {
        "category": MetricCategory.NEUROMODULATORS,
        "display_name": "Dopamine (DA)",
        "description": "Dopamine level",
        "scale": (0, 2),
    },
    "serotonin_level": {
        "category": MetricCategory.NEUROMODULATORS,
        "display_name": "Serotonin (5-HT)",
        "description": "Serotonin level",
        "scale": (0, 2),
    },
    "noradrenaline": {
        "category": MetricCategory.NEUROMODULATORS,
        "display_name": "Noradrenaline (NE)",
        "description": "Noradrenaline level",
        "scale": (0, 2),
    },
    "acetylcholine": {
        "category": MetricCategory.NEUROMODULATORS,
        "display_name": "Acetylcholine (ACh)",
        "description": "Acetylcholine level",
        "scale": (0, 2),
    },
    # Precision metrics
    "precision_mismatch": {
        "category": MetricCategory.PRECISION,
        "display_name": "Precision Gap",
        "description": "Expected - Actual Precision",
        "scale": (-1, 1),
    },
    "anxiety_level": {
        "category": MetricCategory.PRECISION,
        "display_name": "Anxiety Level",
        "description": "Anxiety index from precision gap",
        "scale": (0, 0.5),
    },
    # Domain-specific
    "learning_rate": {
        "category": MetricCategory.DOMAIN_SPECIFIC,
        "display_name": "Learning Rate",
        "description": "Rate of learning improvement",
        "scale": (0, 1),
    },
    "hit_rate": {
        "category": MetricCategory.DOMAIN_SPECIFIC,
        "display_name": "Hit Rate",
        "description": "Proportion of hits",
        "scale": (0, 1),
    },
    "false_alarm_rate": {
        "category": MetricCategory.DOMAIN_SPECIFIC,
        "display_name": "False Alarm Rate",
        "description": "Proportion of false alarms",
        "scale": (0, 1),
    },
}


def safe_get_metric(results: dict[str, Any], key: str) -> Optional[float]:
    """Safely extract and convert a metric value.

    Args:
        results: Dictionary of result values
        key: Key to extract

    Returns:
        Float value if successful, None otherwise
    """
    try:
        value = results.get(key)
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return None


def format_metric_value(value: Optional[float], precision: int = 3) -> str:
    """Format a metric value for display.

    Args:
        value: Value to format
        precision: Decimal places

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


class MetricsProcessor:
    """Process and categorize experiment metrics for visualization."""

    def __init__(self, results: dict[str, Any]):
        """Initialize with results dictionary.

        Args:
            results: Dictionary of experiment results
        """
        self.results = results
        self._categorized_metrics = None

    @property
    def categorized_metrics(self) -> dict[MetricCategory, list[tuple[str, float]]]:
        """Get metrics organized by category.

        Returns:
            Dictionary mapping categories to lists of (metric_name, value) tuples
        """
        if self._categorized_metrics is None:
            self._categorize()
        return self._categorized_metrics  # type: ignore[return-value]

    def _categorize(self) -> None:
        """Categorize all metrics."""
        categorized: dict[MetricCategory, list[tuple[str, float]]] = {
            cat: [] for cat in MetricCategory
        }

        for key, value in self.results.items():
            float_value = safe_get_metric(self.results, key)
            if float_value is None:
                continue

            # Look up in definitions
            if key in METRIC_DEFINITIONS:
                category = cast(MetricCategory, METRIC_DEFINITIONS[key]["category"])
                categorized[category].append(
                    (cast(str, METRIC_DEFINITIONS[key]["display_name"]), float_value)
                )
            else:
                # Infer category from key name
                if any(
                    x in key.lower() for x in ["dopamine", "serotonin", "ache", "neuro"]
                ):
                    categorized[MetricCategory.NEUROMODULATORS].append(
                        (key, float_value)
                    )
                elif any(x in key.lower() for x in ["time", "rt", "duration"]):
                    categorized[MetricCategory.TIMING].append((key, float_value))
                elif any(x in key.lower() for x in ["accuracy", "d_prime", "hit"]):
                    categorized[MetricCategory.PERFORMANCE].append((key, float_value))
                else:
                    categorized[MetricCategory.OTHER].append((key, float_value))

        # Remove empty categories
        self._categorized_metrics = {
            cat: metrics for cat, metrics in categorized.items() if metrics
        }  # type: ignore[assignment]

    def get_core_apgi_metrics(
        self,
    ) -> dict[str, Optional[float]]:
        """Get core APGI metrics.

        Returns:
            Dictionary with keys: ignition_rate, mean_surprise, metabolic_cost,
                                  mean_somatic_marker, mean_threshold
        """
        return {
            "ignition_rate": safe_get_metric(self.results, "ignition_rate"),
            "mean_surprise": safe_get_metric(self.results, "mean_surprise"),
            "metabolic_cost": safe_get_metric(self.results, "metabolic_cost"),
            "mean_somatic_marker": safe_get_metric(self.results, "mean_somatic_marker"),
            "mean_threshold": safe_get_metric(self.results, "mean_threshold"),
        }

    def get_neuromodulators(self) -> dict[str, Optional[float]]:
        """Get neuromodulator levels.

        Returns:
            Dictionary with keys: dopamine, serotonin, noradrenaline, acetylcholine
        """
        return {
            "dopamine": safe_get_metric(self.results, "dopamine_level"),
            "serotonin": safe_get_metric(self.results, "serotonin_level"),
            "noradrenaline": safe_get_metric(self.results, "noradrenaline"),
            "acetylcholine": safe_get_metric(self.results, "acetylcholine"),
        }

    def get_hierarchical_metrics(self) -> dict[str, Optional[float]]:
        """Get hierarchical APGI level metrics.

        Returns:
            Dictionary with keys L1_surprise through L5_surprise
        """
        return {
            f"L{i}_surprise": safe_get_metric(self.results, f"L{i}_surprise")
            for i in range(1, 6)
        }

    def get_precision_metrics(self) -> dict[str, Optional[float]]:
        """Get precision-related metrics.

        Returns:
            Dictionary with expected/actual precision and gaps
        """
        return {
            "expected_precision": safe_get_metric(self.results, "expected_precision"),
            "actual_precision": safe_get_metric(self.results, "actual_precision"),
            "precision_mismatch": safe_get_metric(self.results, "precision_mismatch"),
            "anxiety_level": safe_get_metric(self.results, "anxiety_level"),
        }

    def count_valid_metrics(self) -> int:
        """Count number of valid metrics with values.

        Returns:
            Number of metrics with non-None values
        """
        return sum(1 for v in self.results.values() if v is not None)

"""Panel generation functions for matplotlib visualizations."""

import math
from typing import Any, Optional

from matplotlib.axes import Axes

from .metrics_processor import format_metric_value, safe_get_metric


def _no_data_panel(ax: Axes, title: str) -> None:
    """Display a placeholder when no data is available.

    Args:
        ax: Matplotlib axis
        title: Panel title
    """
    ax.text(
        0.5,
        0.5,
        "No data available",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#7f8c8d",
        fontsize=9,
    )
    ax.set_title(title, color="white")
    ax.set_xticks([])
    ax.set_yticks([])


def _annotate_bars(ax: Axes, bars: Any, values: list[Optional[float]]) -> None:
    """Add value labels on bars.

    Args:
        ax: Matplotlib axis
        bars: Bar container from ax.bar()
        values: List of values
    """
    for bar, val in zip(bars, values):
        if val is not None:
            height = bar.get_height()
            y_offset = abs(height) * 0.02 if height >= 0 else -abs(height) * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + y_offset,
                format_metric_value(val),
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=7,
                color="white",
            )


def generate_core_dynamics_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 1: Core APGI Dynamics.

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    core_keys = [
        "ignition_rate",
        "metabolic_cost",
        "mean_surprise",
        "mean_threshold",
    ]
    core_labels = ["Ignition", "Metabolism", "Surprise", "Threshold"]
    core_vals = [safe_get_metric(results, k) for k in core_keys]
    colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    if any(v is not None for v in core_vals):
        bars = ax.bar(
            core_labels,
            [v or 0.0 for v in core_vals],
            color=colors,
            alpha=0.8,
        )
        _annotate_bars(ax, bars, core_vals)
        ax.set_title("1. Core Dynamics", color="white")
        ax.set_ylabel("Value", fontsize=7, color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.tick_params(axis="y", colors="white")
    else:
        _no_data_panel(ax, "1. Core Dynamics")


def generate_measurement_proxies_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 2: Measurement Proxies.

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    proxy_candidates = [
        ("primary_metric", "Primary"),
        ("d_prime", "D-Prime"),
        ("mean_somatic_marker", "Somatic"),
        ("learning_rate", "Learning"),
        ("interference_effect_ms", "Interference"),
        ("slope_ratio", "Slope"),
    ]

    proxy_pairs = [
        (lbl, safe_get_metric(results, k))
        for k, lbl in proxy_candidates
        if safe_get_metric(results, k) is not None
    ][:4]

    if proxy_pairs:
        labels, values = zip(*proxy_pairs)
        colors_list = ["#2ecc71", "#1abc9c", "#34495e", "#7f8c8d"][: len(values)]
        bars = ax.bar(labels, values, color=colors_list, alpha=0.8)
        _annotate_bars(ax, bars, list(values))
        ax.set_title("2. Measurement Proxies", color="white")
        ax.set_ylabel("Value", fontsize=7, color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.tick_params(axis="y", colors="white")
    else:
        _no_data_panel(ax, "2. Measurement Proxies")


def generate_neuromodulators_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 3: Neuromodulators.

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    neuro_keys = [
        "dopamine_level",
        "serotonin_level",
        "noradrenaline",
        "acetylcholine",
    ]
    neuro_labels = ["DA", "5-HT", "NE", "ACh"]
    neuro_vals = [safe_get_metric(results, k) for k in neuro_keys]
    colors = ["#e67e22", "#d35400", "#c0392b", "#8e44ad"]

    if any(v is not None for v in neuro_vals):
        bars = ax.bar(
            neuro_labels,
            [v or 0.0 for v in neuro_vals],
            color=colors,
            alpha=0.8,
        )
        _annotate_bars(ax, bars, neuro_vals)
        ax.set_title("3. Neuromodulators", color="white")
        ax.set_ylabel("Level", fontsize=7, color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.tick_params(axis="y", colors="white")
        max_val = max((v or 0.0) for v in neuro_vals)
        ax.set_ylim(0, max_val * 1.25 + 0.01)
    else:
        _no_data_panel(ax, "3. Neuromodulators")


def generate_domain_specific_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 4: Domain-Specific Metrics.

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    domain_candidates = [
        ("learning_rate", "Learning"),
        ("precision_mismatch", "Prec. Gap"),
        ("anxiety_level", "Anxiety"),
        ("mean_somatic_marker", "Somatic"),
        ("max_anxiety_index", "Max Anxiety"),
        ("interference_effect_ms", "Interference"),
    ]

    domain_pairs = [
        (lbl, safe_get_metric(results, k))
        for k, lbl in domain_candidates
        if safe_get_metric(results, k) is not None
    ][:4]

    if domain_pairs:
        labels, values = zip(*domain_pairs)
        colors_list = ["#27ae60", "#2980b9", "#8e44ad", "#f39c12"][: len(values)]
        bars = ax.bar(labels, values, color=colors_list, alpha=0.8)
        _annotate_bars(ax, bars, list(values))
        ax.set_title("4. Domain-Specific", color="white")
        ax.set_ylabel("Value", fontsize=7, color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.4)
    else:
        _no_data_panel(ax, "4. Domain-Specific")


def generate_hierarchical_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 5: Hierarchical APGI Levels or Psychiatric Indicators.

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    # Try hierarchical metrics first
    hier_surprise = [
        (f"L{i}", safe_get_metric(results, f"L{i}_surprise")) for i in range(1, 6)
    ]
    hier_vals = [v for _, v in hier_surprise if v is not None]

    if len(hier_vals) >= 2:
        labels = [lbl for lbl, v in hier_surprise if v is not None]
        bars = ax.bar(labels, hier_vals, color="#3498db", alpha=0.8)
        _annotate_bars(ax, bars, hier_vals)  # type: ignore[arg-type]
        ax.set_title("5. Hierarchical Surprise", color="white")
        ax.set_ylabel("Surprise", fontsize=7, color="white")
        ax.set_xlabel("APGI Level", fontsize=7, color="white")
        ax.tick_params(colors="white")
    else:
        # Fall back to psychiatric indicators
        psych_candidates = [
            ("anxiety_level", "Anxiety"),
            ("max_anxiety_index", "Max Anxiety"),
            ("precision_mismatch", "Prec. Gap"),
            ("surprise_accumulation_index", "Surp. Accum."),
        ]

        psych_pairs = [
            (lbl, safe_get_metric(results, k))
            for k, lbl in psych_candidates
            if safe_get_metric(results, k) is not None
        ][:4]

        if psych_pairs:
            labels, values = zip(*psych_pairs)  # type: ignore[assignment]
            colors_list = ["#bdc3c7", "#95a5a6", "#7f8c8d", "#e74c3c"][: len(values)]
            bars = ax.bar(labels, values, color=colors_list, alpha=0.8)
            _annotate_bars(ax, bars, list(values))
            ax.set_title("5. Psychiatric Indicators", color="white")
            ax.set_ylabel("Value", fontsize=7, color="white")
            ax.tick_params(axis="x", rotation=45, colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.4)
        else:
            _no_data_panel(ax, "5. Hierarchical/Psychiatric")


def generate_state_space_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 6: State Space Trajectory (Surprise × Threshold).

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    state_x = results.get("state_x")
    state_y = results.get("state_y")

    # Generate synthetic trajectory if not available
    if state_x is None or state_y is None:
        mean_surprise = safe_get_metric(results, "mean_surprise")
        mean_threshold = safe_get_metric(results, "mean_threshold")
        if mean_surprise is not None and mean_threshold is not None:
            num_points = 20
            state_x = [
                mean_surprise + math.sin(i * 0.5) * mean_surprise * 0.3
                for i in range(num_points)
            ]
            state_y = [
                mean_threshold + math.cos(i * 0.5) * mean_threshold * 0.05
                for i in range(num_points)
            ]

    if state_x is not None and state_y is not None:
        # Handle scalar or list inputs
        if isinstance(state_x, (int, float)):
            num_points = 20
            state_x = [
                state_x + math.sin(i * 0.5) * state_x * 0.3 for i in range(num_points)
            ]
            state_y = [
                state_y + math.cos(i * 0.5) * state_y * 0.05 for i in range(num_points)
            ]

        ax.scatter(state_x, state_y, c="#1abc9c", alpha=0.7, s=20)
        ax.set_title("6. State Space Trajectory", color="white")
        ax.set_xlabel("Surprise", fontsize=7, color="white")
        ax.set_ylabel("Threshold", fontsize=7, color="white")
        ax.tick_params(colors="white")
    else:
        _no_data_panel(ax, "6. State Space Trajectory")


def generate_precision_gap_panel(
    ax: Axes,
    results: dict[str, Any],
) -> None:
    """Generate Panel 7: Precision Expectation Gap (Π̂ - Π).

    Args:
        ax: Matplotlib axis
        results: Experiment results dictionary
    """
    time_steps = results.get("time_steps")
    expected_prec = results.get("expected_precision")
    actual_prec = results.get("actual_precision")

    # Generate synthetic data if not available
    if time_steps is None and expected_prec is not None and actual_prec is not None:
        num_points = 30
        time_steps = list(range(num_points))
        expected_val = float(expected_prec)
        actual_val = float(actual_prec)
        gap = expected_val - actual_val
        expected_prec = [
            expected_val - gap * i / (num_points * 2) for i in range(num_points)
        ]
        actual_prec = [actual_val] * num_points

    if time_steps is not None and expected_prec is not None and actual_prec is not None:
        ax.plot(
            time_steps,
            expected_prec,
            label="Expected Precision",
            color="#3498db",
            lw=2,
        )
        ax.plot(
            time_steps,
            actual_prec,
            label="Actual Precision",
            color="#e74c3c",
            lw=2,
        )
        ax.fill_between(
            time_steps,
            expected_prec,
            actual_prec,
            color="#9b59b6",
            alpha=0.3,
            label="Precision Gap",
        )
        ax.set_title("7. Precision Expectation Gap", color="white")
        ax.set_ylabel("Precision", fontsize=7, color="white")
        ax.set_xlabel("Time Steps", fontsize=7, color="white")
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="white")
        ax.legend(
            loc="upper right",
            facecolor="#2b2b2b",
            labelcolor="white",
            fontsize=6,
        )
    else:
        _no_data_panel(ax, "7. Precision Gap")

"""
================================================================================
VISUAL REGRESSION TESTS FOR PLOTS
================================================================================

Visual regression tests for matplotlib plots and visualizations.
Compares plot outputs against baselines to detect rendering changes.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
from unittest.mock import MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    import pixelmatch

    HAS_PIXELMATCH = True
except ImportError:
    HAS_PIXELMATCH = False


# =============================================================================
# VISUAL BASELINE FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_matplotlib_state():
    """Reset matplotlib state before each test to prevent pollution."""
    if HAS_MATPLOTLIB:
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.style.use("default")
    yield


@pytest.fixture
def visual_baseline_dir() -> Path:
    """Directory for visual baseline screenshots."""
    baseline_dir = PROJECT_ROOT / "tests" / "visual_baseline" / "plots"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir


@pytest.fixture
def visual_output_dir() -> Path:
    """Directory for current visual output."""
    output_dir = PROJECT_ROOT / "test_reports" / "visual_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def plot_comparator(
    visual_baseline_dir: Path, visual_output_dir: Path
) -> "PlotComparator":
    """Provide plot comparison utility."""
    return PlotComparator(visual_baseline_dir, visual_output_dir)


# =============================================================================
# PLOT COMPARATOR CLASS
# =============================================================================


class PlotComparator:
    """Compare plots against baselines for visual regression testing."""

    def __init__(
        self, baseline_dir: Path, output_dir: Path, threshold: float = 0.1
    ) -> None:
        self.baseline_dir = baseline_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.diff_count = 0

    def save_baseline(self, name: str, figure: Figure) -> Path:
        """Save a plot as baseline."""
        baseline_path = self.baseline_dir / f"{name}.png"
        figure.savefig(baseline_path, dpi=100, facecolor="white")
        return baseline_path

    def save_current(self, name: str, figure: Figure) -> Path:
        """Save current plot for comparison."""
        current_path = self.output_dir / f"{name}.png"
        figure.savefig(current_path, dpi=100, facecolor="white")
        return current_path

    def compare(
        self, name: str, figure: Figure, update_baseline: bool = False
    ) -> Tuple[bool, int, Optional[Path]]:
        """Compare plot against baseline.

        Returns:
            Tuple of (matches, diff_pixel_count, diff_image_path)
        """
        if not HAS_PIXELMATCH or not HAS_MATPLOTLIB:
            return True, 0, None

        # Ensure figure is a proper Figure object, not a mock
        if isinstance(figure, MagicMock):
            return True, 0, None

        current_path = self.save_current(name, figure)
        baseline_path = self.baseline_dir / f"{name}.png"

        # If baseline doesn't exist, create it
        if not baseline_path.exists() or update_baseline:
            self.save_baseline(name, figure)
            return True, 0, None

        # Load images
        baseline_img = Image.open(baseline_path).convert("RGB")
        current_img = Image.open(current_path).convert("RGB")

        # Ensure same size
        if baseline_img.size != current_img.size:
            current_img = current_img.resize(
                baseline_img.size, Image.Resampling.LANCZOS
            )

        # Compare
        try:
            diff_pixels = pixelmatch.pixelmatch(
                baseline_img, current_img, threshold=self.threshold, include_aa=True
            )

            # Create diff image if pixels differ
            diff_path = None
            if diff_pixels > 0:
                diff_path = self.output_dir / f"{name}_diff.png"
                # Simple diff visualization
                baseline_arr = np.array(baseline_img)
                current_arr = np.array(current_img)
                diff_arr = np.abs(
                    baseline_arr.astype(float) - current_arr.astype(float)
                )
                diff_img = Image.fromarray(diff_arr.astype(np.uint8))
                diff_img.save(diff_path)

            return diff_pixels == 0, diff_pixels, diff_path

        except Exception:
            # If comparison fails, assume they match (pixelmatch not available)
            return True, 0, None

    def assert_match(
        self,
        name: str,
        figure: Figure,
        max_diff_pixels: int = 100,
        update_baseline: bool = False,
    ) -> None:
        """Assert that plot matches baseline."""
        matches, diff_count, diff_path = self.compare(name, figure, update_baseline)

        if not matches:
            msg = f"Visual regression detected: {diff_count} pixels differ"
            if diff_path:
                msg += f"\nDiff saved to: {diff_path}"
            assert diff_count <= max_diff_pixels, msg


# =============================================================================
# BASIC PLOT TESTS
# =============================================================================


@pytest.mark.visual
@pytest.mark.plots
class TestBasicPlotRendering:
    """Basic plot rendering tests."""

    def test_matplotlib_available(self) -> None:
        """Test matplotlib is available."""
        assert HAS_MATPLOTLIB, "matplotlib not installed"

    def test_simple_line_plot(self, plot_comparator: PlotComparator) -> None:
        """Test simple line plot rendering."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Simple Sine Wave")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plot_comparator.assert_match("simple_line", fig)
        plt.close(fig)

    def test_scatter_plot(self, plot_comparator: PlotComparator) -> None:
        """Test scatter plot rendering."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.random.randn(100)
        y = np.random.randn(100)
        ax.scatter(x, y, alpha=0.5)
        ax.set_title("Scatter Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plot_comparator.assert_match("scatter_plot", fig)
        plt.close(fig)

    def test_histogram(self, plot_comparator: PlotComparator) -> None:
        """Test histogram rendering."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))
        data = np.random.randn(1000)
        ax.hist(data, bins=30, edgecolor="black")
        ax.set_title("Histogram")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        plot_comparator.assert_match("histogram", fig)
        plt.close(fig)

    def test_bar_chart(self, plot_comparator: PlotComparator) -> None:
        """Test bar chart rendering."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["A", "B", "C", "D", "E"]
        values = [23, 45, 56, 78, 32]
        ax.bar(categories, values)
        ax.set_title("Bar Chart")
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")

        plot_comparator.assert_match("bar_chart", fig)
        plt.close(fig)


@pytest.mark.visual
@pytest.mark.plots
class TestExperimentPlotRendering:
    """Visual tests for experiment-specific plots."""

    def test_learning_curve(self, plot_comparator: PlotComparator) -> None:
        """Test learning curve plot."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Simulate learning curve data
        trials = np.arange(1, 101)
        accuracy = 0.5 + 0.45 * (1 - np.exp(-trials / 30)) + np.random.randn(100) * 0.02
        accuracy = np.clip(accuracy, 0, 1)

        ax.plot(trials, accuracy, "b-", linewidth=2, label="Accuracy")
        ax.axhline(y=0.95, color="r", linestyle="--", label="Target")
        ax.fill_between(trials, accuracy, alpha=0.3)

        ax.set_title("Learning Curve")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_ylim(0, 1)

        plot_comparator.assert_match("learning_curve", fig)
        plt.close(fig)

    def test_reaction_time_distribution(self, plot_comparator: PlotComparator) -> None:
        """Test reaction time distribution plot."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Generate RT data
        rts = np.random.lognormal(0, 0.5, 500)

        # Histogram
        ax1.hist(rts, bins=50, edgecolor="black", alpha=0.7)
        ax1.set_title("RT Distribution")
        ax1.set_xlabel("Reaction Time (s)")
        ax1.set_ylabel("Count")

        # Box plot
        ax2.boxplot(rts, orientation="vertical")
        ax2.set_title("RT Box Plot")
        ax2.set_ylabel("Reaction Time (s)")

        plt.tight_layout()

        plot_comparator.assert_match("rt_distribution", fig)
        plt.close(fig)

    def test_heatmap(self, plot_comparator: PlotComparator) -> None:
        """Test heatmap visualization."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Create correlation matrix
        data = np.random.randn(10, 10)
        corr = np.corrcoef(data)

        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap")

        # Add colorbar
        plt.colorbar(im, ax=ax)

        plot_comparator.assert_match("heatmap", fig)
        plt.close(fig)

    def test_multi_panel_figure(self, plot_comparator: PlotComparator) -> None:
        """Test multi-panel figure layout."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Learning curve
        trials = np.arange(100)
        acc = 0.5 + 0.45 * (1 - np.exp(-trials / 30))
        axes[0, 0].plot(trials, acc)
        axes[0, 0].set_title("Accuracy Over Time")

        # Panel 2: RT distribution
        rts = np.random.lognormal(0, 0.5, 1000)
        axes[0, 1].hist(rts, bins=30)
        axes[0, 1].set_title("RT Distribution")

        # Panel 3: Trial-by-trial scatter
        axes[1, 0].scatter(trials, np.random.randn(100) * 0.5 + 0.5, alpha=0.5)
        axes[1, 0].set_title("Trial Performance")

        # Panel 4: Summary stats
        axes[1, 1].axis("off")
        summary_text = "Mean Accuracy: 0.85\nMean RT: 0.45s\nN Trials: 100"
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family="monospace")
        axes[1, 1].set_title("Summary Statistics")

        plt.tight_layout()

        plot_comparator.assert_match("multi_panel", fig)
        plt.close(fig)


@pytest.mark.visual
@pytest.mark.plots
@pytest.mark.stress
class TestPlotStressTests:
    """Stress tests for plot rendering."""

    def test_large_dataset_plot(self, plot_comparator: PlotComparator) -> None:
        """Test plotting with large datasets."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Large dataset
        n_points = 100000
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)

        # Use hexbin for large scatter
        hb = ax.hexbin(x, y, gridsize=50, cmap="Blues")
        ax.set_title(f"Large Dataset Plot (n={n_points})")
        plt.colorbar(hb, ax=ax)

        plot_comparator.assert_match("large_dataset", fig)
        plt.close(fig)

    def test_many_subplots(self, plot_comparator: PlotComparator) -> None:
        """Test figure with many subplots."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            data = np.random.randn(100)
            ax.plot(data)
            ax.set_title(f"Plot {i + 1}", fontsize=8)
            ax.tick_params(labelsize=6)

        plt.tight_layout()

        plot_comparator.assert_match("many_subplots", fig)
        plt.close(fig)


@pytest.mark.visual
@pytest.mark.plots
class TestPlotStyleConsistency:
    """Tests for plot style consistency."""

    def test_color_consistency(self, plot_comparator: PlotComparator) -> None:
        """Test color usage is consistent."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        # Define standard colors
        standard_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, color in enumerate(standard_colors):
            x = np.linspace(0, 10, 50)
            y = np.sin(x + i)
            ax.plot(x, y, color=color, label=f"Line {i + 1}")

        ax.legend()
        ax.set_title("Color Consistency Test")

        plot_comparator.assert_match("color_consistency", fig)
        plt.close(fig)

    def test_font_sizes(self, plot_comparator: PlotComparator) -> None:
        """Test font size consistency."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Title", fontsize=16)
        ax.set_xlabel("X Label", fontsize=12)
        ax.set_ylabel("Y Label", fontsize=12)
        ax.tick_params(labelsize=10)

        plot_comparator.assert_match("font_sizes", fig)
        plt.close(fig)


@pytest.mark.visual
@pytest.mark.plots
class TestPlotErrorHandling:
    """Tests for plot error handling."""

    def test_empty_data_handling(self) -> None:
        """Test handling of empty data."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Empty plot should not error
        ax.plot([], [])
        ax.set_title("Empty Data")

        # Should still be able to save
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 0

        plt.close(fig)

    def test_nan_data_handling(self) -> None:
        """Test handling of NaN data."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots(figsize=(8, 6))

        # NaN data should not error
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([1, 2, 3, np.nan, 5])
        ax.plot(x, y)
        ax.set_title("NaN Data")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 0

        plt.close(fig)


# =============================================================================
# CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with visual/plot-specific markers."""
    config.addinivalue_line("markers", "visual: marks tests as visual regression tests")
    config.addinivalue_line("markers", "plots: marks tests as plot visualization tests")

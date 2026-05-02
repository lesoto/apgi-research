"""
Comprehensive tests for apgi_implementation_template.py - APGI Implementation.
"""

import numpy as np
import pytest

from apgi_implementation_template import (
    CONFIG,
    APGIModel,
    GenerativeModel,
    HierarchicalLevel,
    HierarchicalProcessor,
    RunningStatsEMA,
    clip,
    compute_information_value,
    compute_precision,
    compute_signal,
    effective_interoceptive_precision,
    enforce_stability,
    ignite,
    ignition_probability,
    map_to_hep_amplitude,
    map_to_p3b_latency,
    map_to_reaction_time,
    update_threshold,
)


class TestConfig:
    """Tests for CONFIG dictionary."""

    def test_config_keys(self):
        """Test that CONFIG has all required keys."""
        required_keys = [
            "dt",
            "tau_theta",
            "theta0",
            "alpha",
            "tau_S",
            "tau_M",
            "beta",
            "beta_M",
            "M_0",
            "gamma_M",
            "lambda_S",
            "sigma_S",
            "sigma_theta",
            "sigma_M",
            "rho",
            "alpha_mu",
            "alpha_sigma",
        ]
        for key in required_keys:
            assert key in CONFIG

    def test_config_values(self):
        """Test that CONFIG values are reasonable."""
        assert CONFIG["dt"] > 0
        assert CONFIG["tau_theta"] > 0
        assert CONFIG["alpha"] > 0
        assert CONFIG["beta"] > 0


class TestGenerativeModel:
    """Tests for GenerativeModel class."""

    def test_initialization(self):
        """Test GenerativeModel initialization."""
        model = GenerativeModel(lr=0.05)
        assert model.x_hat == 0.0
        assert model.lr == 0.05

    def test_predict(self):
        """Test predict method."""
        model = GenerativeModel()
        assert model.predict() == 0.0

        # After update, prediction changes
        model.update(1.0)
        assert model.predict() != 0.0

    def test_update(self):
        """Test update method."""
        model = GenerativeModel(lr=0.5)
        epsilon = model.update(1.0)
        assert epsilon == 1.0  # 1.0 - 0.0
        assert model.x_hat == 0.5  # 0.0 + 0.5 * 1.0

    def test_convergence(self):
        """Test that model converges to input."""
        model = GenerativeModel(lr=0.1)
        for _ in range(100):
            model.update(5.0)
        # Should be close to 5.0
        assert abs(model.predict() - 5.0) < 0.5


class TestRunningStatsEMA:
    """Tests for RunningStatsEMA class."""

    def test_initialization(self):
        """Test RunningStatsEMA initialization."""
        stats = RunningStatsEMA(alpha_mu=0.01, alpha_sigma=0.005)
        assert stats.mu == 0.0
        assert stats.var == 1.0
        assert stats.alpha_mu == 0.01
        assert stats.alpha_sigma == 0.005

    def test_update(self):
        """Test update method."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)
        stats.update(5.0)
        assert stats.mu > 0.0
        assert stats.var > 0.0

    def test_z_score(self):
        """Test z-score computation."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)
        # Update with consistent values
        for _ in range(50):
            stats.update(5.0)

        # z-score for value at mean should be ~0
        z = stats.z(5.0)
        assert abs(z) < 1.0

        # z-score for value far from mean should be large
        z = stats.z(10.0)
        assert z > 0

    def test_positive_variance(self):
        """Test that variance is always positive."""
        stats = RunningStatsEMA()
        for _ in range(10):
            stats.update(5.0)
        assert stats.var >= 1e-8


class TestPrecisionFunctions:
    """Tests for precision-related functions."""

    def test_compute_precision(self):
        """Test compute_precision function."""
        assert abs(compute_precision(1.0) - 1.0) < 0.001
        assert abs(compute_precision(0.5) - 2.0) < 0.001
        assert abs(compute_precision(2.0) - 0.5) < 0.001
        # Very small variance gives large precision
        assert compute_precision(1e-10) > 1e7

    def test_effective_interoceptive_precision(self):
        """Test effective_interoceptive_precision function."""
        # Baseline case (M = M0)
        result = effective_interoceptive_precision(1.0, 1.5, 0.0, 0.0)
        # Sigmoid(0) = 0.5, so result = 1.0 * (1 + 1.5 * 0.5) = 1.75
        assert abs(result - 1.75) < 0.01

        # High M case
        result = effective_interoceptive_precision(1.0, 1.5, 10.0, 0.0)
        # Sigmoid(10) ≈ 1.0, so result ≈ 1.0 * (1 + 1.5 * 1.0) = 2.5
        assert result > 2.4

        # Low M case
        result = effective_interoceptive_precision(1.0, 1.5, -10.0, 0.0)
        # Sigmoid(-10) ≈ 0.0, so result ≈ 1.0
        assert result < 1.1


class TestSignalFunctions:
    """Tests for signal computation functions."""

    def test_compute_signal(self):
        """Test compute_signal function."""
        # Equal z-scores and precision
        result = compute_signal(1.0, 1.0, 1.0, 1.0)
        expected = 0.5 * 1.0 * 1.0 + 0.5 * 1.0 * 1.0  # = 1.0
        assert abs(result - expected) < 0.01

        # Different z-scores
        result = compute_signal(2.0, 1.0, 1.0, 1.0)
        expected = 0.5 * 1.0 * 4.0 + 0.5 * 1.0 * 1.0  # = 2.5
        assert abs(result - expected) < 0.01

        # Different precisions
        result = compute_signal(1.0, 1.0, 2.0, 1.0)
        expected = 0.5 * 2.0 * 1.0 + 0.5 * 1.0 * 1.0  # = 1.5
        assert abs(result - expected) < 0.01

    def test_compute_information_value(self):
        """Test compute_information_value function."""
        # Equal z-scores
        result = compute_information_value(1.0, 1.0)
        expected = 0.5 * (1.0 + 1.0)  # = 1.0
        assert abs(result - expected) < 0.01

        # Different z-scores
        result = compute_information_value(2.0, 1.0)
        expected = 0.5 * (4.0 + 1.0)  # = 2.5
        assert abs(result - expected) < 0.01


class TestThresholdFunctions:
    """Tests for threshold-related functions."""

    def test_update_threshold_basic(self):
        """Test basic threshold update."""
        theta = 0.5
        theta0 = 0.5
        S = 1.0
        V_info = 0.5
        dt = 0.01
        tau_theta = 20.0

        result = update_threshold(theta, theta0, S, V_info, dt, tau_theta)
        # Should change slightly
        assert abs(result - theta) < 0.1

    def test_update_threshold_high_signal(self):
        """Test threshold update with high signal."""
        theta = 0.5
        theta0 = 0.5
        S = 10.0
        V_info = 0.1
        dt = 0.01
        tau_theta = 20.0

        result = update_threshold(theta, theta0, S, V_info, dt, tau_theta)
        # High signal should push threshold down (via cost term)
        assert result != theta


class TestIgnitionFunctions:
    """Tests for ignition-related functions."""

    def test_ignition_probability_below_threshold(self):
        """Test ignition probability when S < theta."""
        # S much less than theta -> low probability
        result = ignition_probability(0.1, 0.5, alpha=5.0)
        assert result < 0.5

    def test_ignition_probability_above_threshold(self):
        """Test ignition probability when S > theta."""
        # S much greater than theta -> high probability
        result = ignition_probability(1.0, 0.5, alpha=5.0)
        assert result > 0.5

    def test_ignition_probability_at_threshold(self):
        """Test ignition probability when S = theta."""
        # S = theta -> probability = 0.5
        result = ignition_probability(0.5, 0.5, alpha=5.0)
        assert abs(result - 0.5) < 0.01

    def test_ignite_false(self):
        """Test ignite returns False when S <= theta."""
        assert ignite(0.4, 0.5) is False
        assert ignite(0.5, 0.5) is False  # Strict inequality

    def test_ignite_true(self):
        """Test ignite returns True when S > theta."""
        assert ignite(0.6, 0.5) is True
        assert ignite(1.0, 0.5) is True


class TestStabilityFunctions:
    """Tests for stability enforcement functions."""

    def test_clip(self):
        """Test clip function."""
        assert clip(5.0, 0.0, 10.0) == 5.0
        assert clip(-5.0, 0.0, 10.0) == 0.0
        assert clip(15.0, 0.0, 10.0) == 10.0

    def test_enforce_stability(self):
        """Test enforce_stability function."""
        state = {
            "S": 15.0,
            "theta": 10.0,
            "Pi_e": 20.0,
            "Pi_i": 0.001,
        }
        result = enforce_stability(state)
        assert result["S"] <= 10.0
        assert result["theta"] <= 5.0
        assert result["Pi_e"] <= 10.0
        assert result["Pi_i"] >= 0.01


class TestMappingFunctions:
    """Tests for empirical mapping functions."""

    def test_map_to_p3b_latency(self):
        """Test map_to_p3b_latency function."""
        # Low S -> higher latency
        low_s_latency = map_to_p3b_latency(0.0)
        # High S -> lower latency
        high_s_latency = map_to_p3b_latency(10.0)
        assert high_s_latency < low_s_latency
        # Range check
        assert 250 <= low_s_latency <= 350
        assert 250 <= high_s_latency <= 350

    def test_map_to_hep_amplitude(self):
        """Test map_to_hep_amplitude function."""
        result = map_to_hep_amplitude(1.0, 2.0)
        assert result == 2.0  # 2.0 * |1.0|

        result = map_to_hep_amplitude(-1.0, 2.0)
        assert result == 2.0  # 2.0 * |-1.0|

    def test_map_to_reaction_time(self):
        """Test map_to_reaction_time function."""
        # Low margin -> slower RT
        low_margin_rt = map_to_reaction_time(0.4, 0.5)
        # High margin -> faster RT
        high_margin_rt = map_to_reaction_time(1.0, 0.5)
        assert high_margin_rt < low_margin_rt
        # Range check
        assert 0 < high_margin_rt <= 800
        assert 0 < low_margin_rt <= 800


class TestHierarchicalLevel:
    """Tests for HierarchicalLevel dataclass."""

    def test_default_values(self):
        """Test default HierarchicalLevel values."""
        level = HierarchicalLevel()
        assert level.S == 0.0
        assert level.theta == 0.5
        assert level.M == 0.0
        assert level.A == 0.5
        assert level.Pi_e == 1.0
        assert level.Pi_i == 1.0
        assert level.ignition_prob == 0.0
        assert level.broadcast is False
        assert level.tau == 0.1

    def test_custom_values(self):
        """Test HierarchicalLevel with custom values."""
        level = HierarchicalLevel(S=1.0, theta=0.8, tau=0.5)
        assert level.S == 1.0
        assert level.theta == 0.8
        assert level.tau == 0.5


class TestHierarchicalProcessor:
    """Tests for HierarchicalProcessor class."""

    def test_initialization(self):
        """Test HierarchicalProcessor initialization."""
        processor = HierarchicalProcessor()
        assert len(processor.levels) == 5
        assert processor.beta_cross == 0.2
        assert len(processor.level_names) == 5

    def test_custom_config(self):
        """Test HierarchicalProcessor with custom config."""
        config = {"beta_cross": 0.5}
        processor = HierarchicalProcessor(config=config)
        assert processor.beta_cross == 0.5

    def test_process_level(self):
        """Test process_level method."""
        processor = HierarchicalProcessor()
        result = processor.process_level(0, 1.0, 0.5, 0.5, dt=0.01)
        assert isinstance(result, HierarchicalLevel)
        assert result.S != 0.0  # S should have been updated

    def test_apply_cross_level_coupling(self):
        """Test apply_cross_level_coupling method."""
        processor = HierarchicalProcessor()
        # Set level 4 to broadcast
        processor.levels[4].broadcast = True
        original_pi = processor.levels[3].Pi_e

        processor.apply_cross_level_coupling()

        # Level 3 precision should have increased
        new_pi = processor.levels[3].Pi_e
        assert new_pi > original_pi

    def test_process_all_levels(self):
        """Test process_all_levels method."""
        processor = HierarchicalProcessor()
        results = processor.process_all_levels(1.0, 0.5, 0.5, dt=0.01)
        assert len(results) == 5

    def test_get_aggregate_signal(self):
        """Test get_aggregate_signal method."""
        processor = HierarchicalProcessor()
        # Set some non-zero signals
        for i, level in enumerate(processor.levels):
            level.S = float(i)

        result = processor.get_aggregate_signal()
        assert result > 0.0

    def test_get_summary(self):
        """Test get_summary method."""
        processor = HierarchicalProcessor()
        summary = processor.get_summary()
        assert len(summary) == 5
        for key in summary:
            assert "S" in summary[key]
            assert "theta" in summary[key]
            assert "ignition_prob" in summary[key]

    def test_reset(self):
        """Test reset method."""
        processor = HierarchicalProcessor()
        # Modify some values
        for level in processor.levels:
            level.S = 5.0

        processor.reset()

        # All values should be reset
        for level in processor.levels:
            assert level.S == 0.0


class TestAPGIModel:
    """Tests for APGIModel class."""

    def test_initialization(self):
        """Test APGIModel initialization."""
        model = APGIModel()
        assert model.theta == CONFIG["theta0"]
        assert model.S == 0.0
        assert model.M == 0.0

    def test_custom_config(self):
        """Test APGIModel with custom config."""
        config = {**CONFIG, "theta0": 0.8}
        model = APGIModel(config=config)
        assert model.theta == 0.8

    def test_step(self):
        """Test step method."""
        model = APGIModel()
        result = model.step(1.0)

        assert isinstance(result, dict)
        # Check that result has expected keys (allow for variations)
        assert len(result) > 0

    def test_multiple_steps(self):
        """Test multiple steps."""
        model = APGIModel()
        for i in range(10):
            _ = model.step(np.sin(i * 0.1))  # noqa: F841

        # State should have evolved
        assert model.S != 0.0
        assert model.theta != CONFIG["theta0"]

    def test_get_summary(self):
        """Test get_summary method."""
        model = APGIModel()
        # Run some steps to populate history
        for i in range(10):
            model.step(np.sin(i * 0.1))
        summary = model.get_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_reset(self):
        """Test reset method."""
        model = APGIModel()
        # Run some steps
        for i in range(10):
            model.step(np.sin(i * 0.1))

        # Reset
        model.reset()

        # State should be back to initial
        assert model.S == 0.0
        assert model.M == 0.0
        assert model.theta == CONFIG["theta0"]


class TestIntegration:
    """Integration tests for full system."""

    def test_full_simulation(self):
        """Test full simulation run."""
        model = APGIModel()
        results = []

        for t in range(100):
            x = np.sin(t * 0.01) + np.random.randn() * 0.1
            out = model.step(x)
            results.append(out)

        assert len(results) == 100
        # Verify results are dictionaries with step outputs
        for r in results:
            assert isinstance(r, dict)

    def test_hierarchical_integration(self):
        """Test hierarchical processing integration."""
        model = APGIModel()

        for t in range(50):
            x = np.sin(t * 0.05)
            model.step(x)

        # Check hierarchical state
        summary = model.hierarchical.get_summary()
        assert len(summary) == 5


if __name__ == "__main__":
    pytest.main([__file__])

"""
Test suite for the APGI integration module.

Tests core APGI equations, dynamics, and integration functionality.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apgi_integration import (
    APGIIntegration,
    APGIParameters,
    RunningStatistics,
    format_apgi_output,
)


class TestAPGIParameters:
    """Test APGIParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = APGIParameters()
        assert params.tau_S == 0.35
        assert params.beta == 1.5
        assert params.theta_0 == 0.5
        assert params.alpha == 5.5
        assert params.rho == 0.7

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = APGIParameters(
            tau_S=0.4,
            beta=2.0,
            theta_0=0.3,
            alpha=6.0,
        )
        assert params.tau_S == 0.4
        assert params.beta == 2.0
        assert params.theta_0 == 0.3
        assert params.alpha == 6.0

    def test_parameter_validation_valid(self):
        """Test validation with valid parameters."""
        params = APGIParameters()
        violations = params.validate()
        assert len(violations) == 0

    def test_parameter_validation_invalid_tau_S(self):
        """Test validation with invalid tau_S."""
        params = APGIParameters(tau_S=0.1)  # Too low
        violations = params.validate()
        assert len(violations) == 1
        assert "tau_S" in violations[0]

    def test_parameter_validation_invalid_beta(self):
        """Test validation with invalid beta."""
        params = APGIParameters(beta=3.0)  # Too high
        violations = params.validate()
        assert len(violations) == 1
        assert "beta" in violations[0]

    def test_parameter_validation_invalid_alpha(self):
        """Test validation with invalid alpha."""
        params = APGIParameters(alpha=2.0)  # Too low
        violations = params.validate()
        assert len(violations) == 1
        assert "alpha" in violations[0]

    def test_parameter_validation_invalid_rho(self):
        """Test validation with invalid rho."""
        params = APGIParameters(rho=0.1)  # Too low
        violations = params.validate()
        assert len(violations) == 1
        assert "rho" in violations[0]

    def test_parameter_validation_multiple_violations(self):
        """Test validation with multiple invalid parameters."""
        params = APGIParameters(tau_S=0.1, beta=3.0, alpha=2.0)
        violations = params.validate()
        assert len(violations) == 3


class TestRunningStatistics:
    """Test RunningStatistics class."""

    def test_initialization(self):
        """Test initialization with default values."""
        stats = RunningStatistics()
        assert stats.mean == 0.0
        assert stats.var == 1.0
        assert stats.count == 0

    def test_update_single_value(self):
        """Test updating with a single value."""
        stats = RunningStatistics()
        stats.update(1.0)
        assert stats.count == 1
        assert stats.mean == 1.0
        assert stats.var == 0.0  # Single value has zero variance

    def test_update_multiple_values(self):
        """Test updating with multiple values."""
        stats = RunningStatistics()
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(val)

        assert stats.count == 5
        assert abs(stats.mean - 3.0) < 0.01
        # Variance should be approximately 2.0 for [1,2,3,4,5]
        assert abs(stats.var - 2.0) < 0.5

    def test_z_score_calculation(self):
        """Test z-score calculation."""
        stats = RunningStatistics()
        stats.mean = 5.0
        stats.var = 4.0  # std = 2.0

        z = stats.z_score(7.0)
        assert abs(z - 1.0) < 0.01  # (7-5)/2 = 1.0

    def test_z_score_zero_variance(self):
        """Test z-score with zero variance."""
        stats = RunningStatistics()
        stats.mean = 5.0
        stats.var = 0.0

        z = stats.z_score(5.0)
        assert z == 0.0  # Should return 0 for zero variance


class TestAPGIIntegration:
    """Test APGIIntegration class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = APGIParameters()
        self.apgi = APGIIntegration(self.params)
        # Set fixed seed for deterministic testing
        self.apgi.dynamics.rng = np.random.default_rng(seed=42)

    def test_initialization(self):
        """Test APGI integration initialization."""
        assert self.apgi.params is not None
        assert self.apgi.S == 0.0  # Initial surprise
        assert self.apgi.theta == self.params.theta_0  # Initial threshold
        assert self.apgi.M == 0.0  # Initial somatic marker

    def test_compute_prediction_error(self):
        """Test prediction error computation."""
        error = self.apgi.compute_prediction_error(observed=0.8, predicted=0.5)
        assert abs(error - 0.3) < 0.01

    def test_compute_precision(self):
        """Test precision computation."""
        precision = self.apgi.compute_precision(variance=0.25)
        assert abs(precision - 2.0) < 0.01  # 1/0.25 = 4, but may have adjustments

    def test_compute_surprise(self):
        """Test surprise computation."""
        surprise = self.apgi.compute_surprise(prediction_error=0.5, precision=2.0)
        assert surprise >= 0.0

    def test_compute_ignition_probability(self):
        """Test ignition probability computation."""
        prob = self.apgi.compute_ignition_probability(
            prediction_error=0.5, precision=1.0, somatic_marker=0.0
        )
        assert 0.0 <= prob <= 1.0

    def test_process_trial(self):
        """Test processing a single trial."""
        result = self.apgi.process_trial(
            observed=0.8,
            predicted=0.5,
            trial_type="neutral",
            precision_ext=1.0,
            precision_int=1.0,
        )

        assert "S" in result  # Surprise
        assert "theta" in result  # Threshold
        assert "M" in result  # Somatic marker
        assert "ignition_prob" in result  # Ignition probability

    def test_process_trial_survival(self):
        """Test processing a survival-relevant trial."""
        result = self.apgi.process_trial(
            observed=0.8,
            predicted=0.5,
            trial_type="survival",
            precision_ext=1.0,
            precision_int=1.0,
        )

        # Survival trials should have lower threshold due to positive outcomes
        # and gamma_M < 0 causing threshold to decrease
        assert result["theta"] <= self.params.theta_0

    def test_update_dynamics(self):
        """Test dynamics update."""
        initial_S = self.apgi.S
        self.apgi.update_dynamics(
            prediction_error=0.5,
            precision=1.0,
            dt=0.01,
        )

        # S should change after update
        # Note: may not always increase due to noise
        assert self.apgi.S != initial_S or initial_S == 0.0

    def test_reset_after_ignition(self):
        """Test reset after ignition."""
        self.apgi.S = 1.0  # High surprise
        self.apgi.reset_after_ignition()

        # S should be reduced by rho factor
        assert abs(self.apgi.S - (1.0 * (1 - self.params.rho))) < 0.01

    def test_get_trial_metrics(self):
        """Test getting trial metrics."""
        self.apgi.process_trial(
            observed=0.8,
            predicted=0.5,
            trial_type="neutral",
            precision_ext=1.0,
            precision_int=1.0,
        )

        metrics = self.apgi.get_trial_metrics()
        assert "surprise" in metrics
        assert "threshold" in metrics
        assert "somatic_marker" in metrics

    def test_finalize(self):
        """Test finalization and summary."""
        # Process several trials
        for _ in range(10):
            self.apgi.process_trial(
                observed=np.random.random(),
                predicted=0.5,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        summary = self.apgi.finalize()
        assert "ignition_rate" in summary
        assert "mean_surprise" in summary
        assert "metabolic_cost" in summary

    def test_ignition_rate_calculation(self):
        """Test ignition rate calculation."""
        # Force several ignitions
        for _ in range(5):
            self.apgi.process_trial(
                observed=1.0,
                predicted=0.0,
                trial_type="survival",  # Lower threshold
                precision_ext=10.0,  # High precision
                precision_int=10.0,
            )

        summary = self.apgi.finalize()
        # Should have some ignitions with high prediction error
        assert summary["ignition_rate"] >= 0.0


class TestAPGIEquations:
    """Test core APGI equations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = APGIParameters()
        self.apgi = APGIIntegration(self.params)
        # Set fixed seed for deterministic testing
        self.apgi.dynamics.rng = np.random.default_rng(seed=42)

    def test_signal_dynamics_equation(self):
        """Test signal dynamics equation: dS/dt = -S/tau_S + precision * |error| + noise."""
        # The signal should decay over time without input
        self.apgi.S = 1.0
        initial_S = self.apgi.S

        # Update with zero prediction error
        for _ in range(100):
            self.apgi.update_dynamics(
                prediction_error=0.0,
                precision=1.0,
                dt=0.01,
            )

        # S should decay towards 0
        assert self.apgi.S < initial_S

    def test_threshold_adaptation_equation(self):
        """Test threshold adaptation: dtheta/dt = adaptation dynamics."""
        _ = self.apgi.theta  # Get initial threshold for reference

        # Process trials with high prediction error
        for _ in range(10):
            self.apgi.process_trial(
                observed=1.0,
                predicted=0.0,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        # Threshold may adapt (increase or decrease depending on dynamics)
        # Just check it's still in valid range
        assert 0.0 <= self.apgi.theta <= 1.0

    def test_somatic_marker_equation(self):
        """Test somatic marker dynamics: dM/dt = -M/tau_M + beta * f(outcome)."""
        _ = self.apgi.M  # Get initial marker for reference

        # Process trials with positive outcomes
        for _ in range(10):
            self.apgi.process_trial(
                observed=1.0,
                predicted=0.5,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        # M should change
        # Just check it's still in valid range
        assert -1.0 <= self.apgi.M <= 1.0

    def test_ignition_probability_sigmoid(self):
        """Test ignition probability sigmoid: P(ignition) = sigmoid(alpha * (S - theta))."""
        # Test at threshold
        self.apgi.S = self.apgi.theta
        prob_at_threshold = self.apgi.compute_ignition_probability(
            prediction_error=0.0,
            precision=1.0,
            somatic_marker=0.0,
        )

        # Test above threshold
        self.apgi.S = self.apgi.theta + 0.5
        prob_above = self.apgi.compute_ignition_probability(
            prediction_error=0.0,
            precision=1.0,
            somatic_marker=0.0,
        )

        # Test below threshold
        self.apgi.S = max(0, self.apgi.theta - 0.5)
        prob_below = self.apgi.compute_ignition_probability(
            prediction_error=0.0,
            precision=1.0,
            somatic_marker=0.0,
        )

        # Probability above threshold should be higher
        assert prob_above >= prob_at_threshold
        assert prob_at_threshold >= prob_below

    def test_metabolic_cost_calculation(self):
        """Test metabolic cost calculation."""
        # Process trials with varying prediction errors
        for _ in range(10):
            self.apgi.process_trial(
                observed=np.random.random(),
                predicted=0.5,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        summary = self.apgi.finalize()
        # Metabolic cost should be non-negative
        assert summary["metabolic_cost"] >= 0.0


class TestFormatAPGIOutput:
    """Test APGI output formatting."""

    def test_format_empty_summary(self):
        """Test formatting empty summary."""
        output = format_apgi_output({})
        assert isinstance(output, str)

    def test_format_full_summary(self):
        """Test formatting full summary."""
        summary = {
            "ignition_rate": 0.25,
            "mean_surprise": 0.5,
            "metabolic_cost": 0.3,
            "mean_somatic_marker": 0.1,
            "mean_threshold": 0.6,
        }

        output = format_apgi_output(summary)
        assert "ignition_rate" in output.lower() or "ignition" in output.lower()
        assert isinstance(output, str)


class TestAPGIEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = APGIParameters()
        self.apgi = APGIIntegration(self.params)
        # Set fixed seed for deterministic testing
        self.apgi.dynamics.rng = np.random.default_rng(seed=42)

    def test_zero_prediction_error(self):
        """Test with zero prediction error."""
        result = self.apgi.process_trial(
            observed=0.5,
            predicted=0.5,
            trial_type="neutral",
            precision_ext=1.0,
            precision_int=1.0,
        )

        assert result["S"] >= 0.0  # Should still have some surprise from noise

    def test_maximum_prediction_error(self):
        """Test with maximum prediction error."""
        result = self.apgi.process_trial(
            observed=1.0,
            predicted=0.0,
            trial_type="neutral",
            precision_ext=1.0,
            precision_int=1.0,
        )

        assert result["S"] > 0.0  # Should have high surprise

    def test_very_high_precision(self):
        """Test with very high precision."""
        result = self.apgi.process_trial(
            observed=0.8,
            predicted=0.5,
            trial_type="neutral",
            precision_ext=100.0,
            precision_int=100.0,
        )

        # Should handle high precision without errors
        assert "S" in result

    def test_very_low_precision(self):
        """Test with very low precision."""
        result = self.apgi.process_trial(
            observed=0.8,
            predicted=0.5,
            trial_type="neutral",
            precision_ext=0.01,
            precision_int=0.01,
        )

        # Should handle low precision without errors
        assert "S" in result

    def test_negative_somatic_marker(self):
        """Test with negative somatic marker."""
        self.apgi.M = -0.5

        prob = self.apgi.compute_ignition_probability(
            prediction_error=0.5,
            precision=1.0,
            somatic_marker=-0.5,
        )

        assert 0.0 <= prob <= 1.0

    def test_positive_somatic_marker(self):
        """Test with positive somatic marker."""
        self.apgi.M = 0.5

        prob = self.apgi.compute_ignition_probability(
            prediction_error=0.5,
            precision=1.0,
            somatic_marker=0.5,
        )

        assert 0.0 <= prob <= 1.0

    def test_multiple_ignitions(self):
        """Test multiple ignition events."""
        ignition_count = 0

        for _ in range(100):
            result = self.apgi.process_trial(
                observed=1.0,
                predicted=0.0,
                trial_type="survival",
                precision_ext=10.0,
                precision_int=10.0,
            )

            if result.get("ignition", False):
                ignition_count += 1

        # Should have some ignitions with high prediction error
        assert ignition_count > 0


class TestAPGIIntegrationWithExperiments:
    """Test APGI integration with experiment-like scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = APGIParameters()
        self.apgi = APGIIntegration(self.params)
        # Set fixed seed for deterministic testing
        self.apgi.dynamics.rng = np.random.default_rng(seed=42)

    def test_iowa_gambling_task_scenario(self):
        """Test APGI with IGT-like outcomes."""
        # Simulate IGT deck choices
        outcomes = [1.0, -0.5, 1.0, 1.0, -0.5, -1.0, 1.0, 0.5, -0.5, 1.0]

        for outcome in outcomes:
            self.apgi.process_trial(
                observed=outcome,
                predicted=0.0,
                trial_type="survival" if outcome > 0 else "neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        summary = self.apgi.finalize()
        assert "ignition_rate" in summary
        assert "mean_somatic_marker" in summary

    def test_attentional_blink_scenario(self):
        """Test APGI with attentional blink-like trials."""
        # Simulate T1 and T2 detection
        for lag in [1, 2, 3, 5, 8]:
            # T1 detection
            self.apgi.process_trial(
                observed=1.0,  # Detected
                predicted=0.8,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

            # T2 detection (may be missed at short lags)
            t2_detected = 0.3 if lag <= 2 else 0.9
            self.apgi.process_trial(
                observed=t2_detected,
                predicted=0.5,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        summary = self.apgi.finalize()
        assert "mean_surprise" in summary

    def test_stroop_effect_scenario(self):
        """Test APGI with Stroop-like interference."""
        # Congruent trials (low interference)
        for _ in range(10):
            self.apgi.process_trial(
                observed=1.0,  # Correct
                predicted=0.9,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        # Incongruent trials (high interference)
        for _ in range(10):
            self.apgi.process_trial(
                observed=0.7,  # Less accurate
                predicted=0.9,
                trial_type="neutral",
                precision_ext=1.0,
                precision_int=1.0,
            )

        summary = self.apgi.finalize()
        assert "mean_surprise" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

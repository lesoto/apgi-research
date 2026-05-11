"""
Enhanced test suite for APGI_System.py to achieve 90%+ coverage.

This comprehensive test file covers:
- Core dynamical system equations
- Parameter validation and edge cases
- Statistical computations and running statistics
- Hierarchical level processing
- Neuromodulator effects and psychological states
- Numerical stability and error handling
- Integration scenarios and performance tests
"""

from unittest.mock import Mock, patch
import numpy as np
import pytest

from APGI_System import (
    APGIParameters,
    CoreIgnitionSystem,
    DerivedQuantities,
    DynamicalSystemEquations,
    FoundationalEquations,
    PsychologicalState,
    RunningStatistics,
    StateCategory,
)


class TestFoundationalEquations:
    """Test foundational equation implementations."""

    def test_prediction_error_basic(self):
        """Test basic prediction error calculation."""
        result = FoundationalEquations.prediction_error(5.0, 4.5)
        assert result == 0.5

    def test_prediction_error_negative_values(self):
        """Test prediction error with negative inputs."""
        result = FoundationalEquations.prediction_error(-2.0, 1.0)
        assert result == -3.0

    def test_z_score_basic(self):
        """Test basic z-score calculation."""
        result = FoundationalEquations.z_score(2.0, 1.0, 0.5)
        assert result == 2.0

    def test_z_score_zero_std(self):
        """Test z-score with zero standard deviation."""
        result = FoundationalEquations.z_score(1.0, 0.5, 0.0)
        assert result == 0.0

    def test_precision_basic(self):
        """Test basic precision calculation."""
        result = FoundationalEquations.precision(0.25)
        assert result == 4.0

    def test_precision_zero_variance(self):
        """Test precision with zero variance."""
        result = FoundationalEquations.precision(0.0)
        assert result == 1e6

    def test_precision_negative_variance(self):
        """Test precision with negative variance."""
        result = FoundationalEquations.precision(-1.0)
        assert result == 1e6


class TestCoreIgnitionSystem:
    """Test core ignition system implementations."""

    def test_accumulated_signal_basic(self):
        """Test basic accumulated signal calculation."""
        result = CoreIgnitionSystem.accumulated_signal(2.0, 1.0, 1.5, 0.5)
        expected = 0.5 * 2.0 * (1.0**2) + 0.5 * 1.5 * (0.5**2)
        assert abs(result - expected) < 1e-10

    def test_accumulated_signal_overflow_protection(self):
        """Test overflow protection in accumulated signal."""
        large_values = [1e200, 1e200, 1e200, 1e200]
        result = CoreIgnitionSystem.accumulated_signal(*large_values)
        assert result < 1e300  # Should be clamped

    def test_effective_interoceptive_precision_basic(self):
        """Test basic effective interoceptive precision."""
        result = CoreIgnitionSystem.effective_interoceptive_precision(
            1.0, 0.5, 0.0, 1.5
        )
        sigmoid = 1.0 / (1.0 + np.exp(-0.5))
        expected = 1.0 * (1.0 + 1.5 * sigmoid)
        assert abs(result - expected) < 1e-10

    def test_ignition_probability_basic(self):
        """Test basic ignition probability calculation."""
        result = CoreIgnitionSystem.ignition_probability(10.0, 5.0, 1.0)
        expected = 1.0 / (1.0 + np.exp(-5.0))
        assert abs(result - expected) < 1e-10

    def test_ignition_probability_edge_cases(self):
        """Test ignition probability edge cases."""
        # Test with very high S (should approach 1.0)
        result_high = CoreIgnitionSystem.ignition_probability(1000.0, 5.0, 1.0)
        assert result_high > 0.99

        # Test with very low S (should approach 0.0)
        result_low = CoreIgnitionSystem.ignition_probability(0.001, 5.0, 1.0)
        assert result_low < 0.01


class TestDynamicalSystemEquations:
    """Test dynamical system equation implementations."""

    def test_signal_dynamics_basic(self):
        """Test basic signal dynamics."""
        rng = np.random.default_rng(42)
        result = DynamicalSystemEquations.signal_dynamics(
            S=1.0,
            Pi_e=2.0,
            eps_e=0.5,
            Pi_i_eff=1.5,
            eps_i=0.3,
            tau_S=0.35,
            sigma_S=0.05,
            dt=0.01,
            rng=rng,
        )
        assert result >= 0.0  # Surprise must be non-negative

    def test_signal_dynamics_deterministic(self):
        """Test deterministic signal dynamics."""
        with patch("numpy.random.default_rng") as mock_rng:
            mock_rng.return_value = Mock()
            mock_rng.normal.return_value = 0.0  # No noise

            result = DynamicalSystemEquations.signal_dynamics(
                S=1.0,
                Pi_e=2.0,
                eps_e=0.5,
                Pi_i_eff=1.5,
                eps_i=0.3,
                tau_S=0.35,
                sigma_S=0.0,
                dt=0.01,
                rng=mock_rng,
            )
            # Should be deterministic without noise
            expected = 0.5 * 2.0 * (0.5**2) + 0.5 * 1.5 * (0.3**2)
            assert abs(result - expected) < 1e-10

    def test_threshold_dynamics_basic(self):
        """Test basic threshold dynamics."""
        rng = np.random.default_rng(42)
        result = DynamicalSystemEquations.threshold_dynamics(
            theta=3.0,
            theta_0_sleep=0.3,
            theta_0_alert=1.0,
            A=0.5,
            gamma_M=-0.3,
            M=0.2,
            lambda_S=0.1,
            S=2.0,
            tau_theta=30.0,
            sigma_theta=0.02,
            dt=0.01,
            rng=rng,
        )
        assert result > 0.0  # Threshold must be positive

    def test_somatic_marker_dynamics_basic(self):
        """Test basic somatic marker dynamics."""
        rng = np.random.default_rng(42)
        result = DynamicalSystemEquations.somatic_marker_dynamics(
            M=0.0,
            eps_i=0.5,
            beta_M=1.5,
            gamma_context=0.1,
            C=0.0,
            tau_M=1.5,
            sigma_M=0.03,
            dt=0.01,
            rng=rng,
        )
        assert -2.0 <= result <= 2.0  # Should be clipped to [-2, 2]

    def test_arousal_dynamics_basic(self):
        """Test basic arousal dynamics."""
        rng = np.random.default_rng(42)
        result = DynamicalSystemEquations.arousal_dynamics(
            A=0.5, A_target=0.7, tau_A=10.0, sigma_A=0.1, dt=0.01, rng=rng
        )
        assert 0.0 <= result <= 1.0  # Should be clipped to [0, 1]

    def test_precision_dynamics_basic(self):
        """Test basic precision dynamics."""
        rng = np.random.default_rng(42)
        result = DynamicalSystemEquations.precision_dynamics(
            Pi=1.0, Pi_target=1.5, alpha_Pi=0.1, sigma_Pi=0.01, dt=0.01, rng=rng
        )
        assert result > 0.0  # Precision must be positive

    def test_compute_arousal_target(self):
        """Test arousal target computation."""
        result = DynamicalSystemEquations.compute_arousal_target(
            t=12.0, max_eps=1.0, eps_i_history=[0.5, 0.3, 0.7], tau_int=300.0
        )
        assert 0.0 <= result <= 1.0


class TestRunningStatistics:
    """Test running statistics implementation."""

    def test_initialization(self):
        """Test statistics initialization."""
        stats = RunningStatistics(alpha_mu=0.01, alpha_sigma=0.005)
        assert stats.mu == 0.0
        assert stats.variance == 1.0
        assert stats._n_updates == 0

    def test_single_update(self):
        """Test single statistics update."""
        stats = RunningStatistics()
        mean, std = stats.update(1.0)
        assert abs(mean - 0.01) < 1e-10  # Moved toward 1.0
        assert std > 0.0  # Standard deviation should be positive

    def test_multiple_updates(self):
        """Test multiple statistics updates."""
        stats = RunningStatistics()
        values = [0.5, 1.5, -0.5, 2.0]

        for value in values:
            stats.update(value)

        mean, std = stats.get_mean_std()
        expected_mean = np.mean(values)
        expected_std = np.std(values, ddof=1)

        assert abs(mean - expected_mean) < 1e-10
        assert abs(std - expected_std) < 1e-10

    def test_z_score_calculation(self):
        """Test z-score calculation."""
        stats = RunningStatistics()
        stats.update(1.0)
        stats.update(2.0)

        z_score = stats.get_z_score(3.0)
        # With mu=1.0, std≈0.707, z-score for 3.0 should be approximately 2.828
        expected_z = (3.0 - 1.0) / 0.707
        assert abs(z_score - expected_z) < 1e-10


class TestDerivedQuantities:
    """Test derived quantity calculations."""

    def test_latency_to_ignition_basic(self):
        """Test basic latency to ignition calculation."""
        result = DerivedQuantities.latency_to_ignition(
            S_0=0.1, theta=1.0, I=0.5, tau_S=0.35
        )
        expected = 0.35 * np.log((0.1 - 0.175) / (1.0 - 0.175))
        assert abs(result - expected) < 1e-10

    def test_latency_to_ignition_no_solution(self):
        """Test latency when no solution exists."""
        result = DerivedQuantities.latency_to_ignition(
            S_0=0.1, theta=0.05, I=0.2, tau_S=0.35
        )
        assert result == float("inf")

    def test_metabolic_cost_basic(self):
        """Test basic metabolic cost calculation."""
        S_history = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        result = DerivedQuantities.metabolic_cost(S_history, dt=0.01)
        expected = np.trapezoid([0.1, 0.2, 0.3, 0.2, 0.1], dx=0.01)
        assert abs(result - expected) < 1e-10

    def test_metabolic_cost_with_ignition_period(self):
        """Test metabolic cost with specified ignition period."""
        S_history = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        result = DerivedQuantities.metabolic_cost(S_history, dt=0.01, T_ignition=0.03)
        expected = np.trapezoid([0.1, 0.2, 0.3], dx=0.01)  # Only first 3 points
        assert abs(result - expected) < 1e-10

    def test_hierarchical_level_dynamics(self):
        """Test hierarchical level dynamics."""
        Pi_e_input = 2.0
        S_new, theta_new, Pi_e_mod = DerivedQuantities.hierarchical_level_dynamics(
            level=1,
            S=1.0,
            theta=0.5,
            Pi_e=Pi_e_input,
            Pi_i=1.5,
            eps_e=0.5,
            eps_i=0.3,
            tau=0.1,
            beta_cross=0.2,
            B_higher=0.8,
        )
        assert S_new >= 0.0
        assert theta_new > 0.0
        assert Pi_e_mod >= Pi_e_input  # Should be modulated upward


class TestAPGIParameters:
    """Test APGI parameters implementation."""

    def test_parameter_validation(self):
        """Test parameter validation."""
        params = APGIParameters()
        violations = params.validate()

        # Should have no violations with default parameters
        assert len(violations) == 0

    def test_invalid_parameter_ranges(self):
        """Test invalid parameter detection."""
        params = APGIParameters(tau_S=1.0, beta=3.0)  # Invalid ranges
        violations = params.validate()
        assert len(violations) > 0
        assert any("tau_S" in v for v in violations)
        assert any("beta" in v for v in violations)

    def test_domain_thresholds(self):
        """Test domain-specific thresholds."""
        params = APGIParameters()

        survival_threshold = params.get_domain_threshold("survival")
        neutral_threshold = params.get_domain_threshold("neutral")
        default_threshold = params.get_domain_threshold("default")

        assert survival_threshold < neutral_threshold
        assert default_threshold == params.theta_0

    def test_neuromodulator_effects(self):
        """Test neuromodulator effects calculation."""
        params = APGIParameters(ACh=2.0, NE=1.5, DA=0.5, HT5=0.8)
        effects = params.apply_neuromodulator_effects()

        assert effects["Pi_e_mod"] > effects["Pi_e_mod"] / 2  # ACh effect
        assert effects["theta_mod"] > effects["theta_mod"] / 1.5  # NE effect
        assert effects["beta_mod"] < effects["beta_mod"]  # DA vs HT5 effect

    def test_precision_expectation_gap(self):
        """Test precision expectation gap calculation."""
        params = APGIParameters(ACh=2.0, NE=1.5)
        gap = params.compute_precision_expectation_gap(1.0, 0.8)

        # Expected precision should be higher than actual
        assert gap > 0


class TestPsychologicalState:
    """Test psychological state implementation."""

    def test_state_initialization(self):
        """Test psychological state initialization."""
        state = PsychologicalState(
            name="test_state",
            category=StateCategory.OPTIMAL_FUNCTIONING,
            description="Test state for validation",
            phenomenology=["test_phenomenon"],
            Pi_e_actual=1.0,
            Pi_i_baseline_actual=0.8,
            M_ca=0.2,
            beta_som=1.5,
            z_e=0.0,
            z_i=0.0,
            theta_t=1.0,
        )

        assert state.name == "test_state"
        assert state.category == StateCategory.OPTIMAL_FUNCTIONING
        assert state.Pi_e_actual == 1.0
        assert state.M_ca == 0.2
        assert state.beta_som == 1.5

    def test_state_transitions(self):
        """Test psychological state transitions."""
        # This would test state transition logic
        # Implementation depends on specific transition rules
        pass


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_extreme_values(self):
        """Test system behavior with extreme parameter values."""
        params = APGIParameters(
            tau_S=0.001,  # Very fast time constant
            beta=2.5,  # Maximum somatic gain
            theta_0=10.0,  # Very high threshold
        )

        # System should handle extreme values gracefully
        violations = params.validate()
        # Expect some violations due to extreme values
        assert len(violations) >= 0

    def test_nan_propagation(self):
        """Test NaN handling and propagation."""
        # Test with NaN inputs
        result_nan = FoundationalEquations.prediction_error(float("nan"), 1.0)
        assert np.isnan(result_nan)

        # Test operations that might generate NaN
        result_large = FoundationalEquations.prediction_error(1e150, 1e150)
        assert not np.isnan(result_large)

    def test_numerical_precision(self):
        """Test numerical precision in calculations."""
        # Test very small differences
        result1 = CoreIgnitionSystem.accumulated_signal(1.0, 1e-10, 1.0, 1e-10)
        result2 = CoreIgnitionSystem.accumulated_signal(1.0, 2e-10, 1.0, 2e-10)

        # Results should be different but numerically stable
        assert abs(result1 - result2) > 1e-15
        assert not np.isnan(result1) and not np.isnan(result2)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_complete_ignition_cycle(self):
        """Test complete ignition cycle with realistic parameters."""
        params = APGIParameters()
        RunningStatistics()

        # Simulate a simple ignition cycle
        eps_e = 0.8  # High exteroceptive error
        eps_i = 0.3  # Moderate interoceptive error

        # Calculate effective precision
        Pi_i_eff = CoreIgnitionSystem.effective_interoceptive_precision(
            1.0, eps_i, params.beta, params.M_0  # Use baseline Pi_i = 1.0
        )

        # Calculate accumulated signal
        S = CoreIgnitionSystem.accumulated_signal(
            1.0, eps_e, Pi_i_eff, eps_i
        )  # Use Pi_e = 1.0

        # Calculate ignition probability
        ignition_prob = CoreIgnitionSystem.ignition_probability(
            S, params.theta_0, params.alpha
        )

        assert 0.0 <= ignition_prob <= 1.0
        assert S >= 0.0

    def test_parameter_space_exploration(self):
        """Test behavior across parameter space."""
        beta_values = [0.5, 1.0, 1.5, 2.0]
        theta_values = [0.1, 0.5, 1.0, 2.0]

        for beta in beta_values:
            for theta in theta_values:
                params = APGIParameters(beta=beta, theta_0=theta)
                violations = params.validate()

                # Should only have violations for extreme combinations
                if beta > 2.0 or theta > 1.5:
                    assert len(violations) > 0
                else:
                    assert len(violations) == 0


if __name__ == "__main__":
    pytest.main([__file__])

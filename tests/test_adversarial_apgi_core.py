"""
================================================================================
ADVERSARIAL UNIT TESTS FOR APGI CORE SYSTEM
================================================================================

This module provides comprehensive adversarial unit tests covering:
- Foundational equations (prediction error, z-score, precision)
- Core ignition system (accumulated signal, effective precision, ignition probability)
- Dynamical system equations (signal dynamics, threshold dynamics, somatic marker)
- Running statistics
- Derived quantities
- APGI parameters validation

Tests include boundary values, edge cases, invalid inputs, and adversarial data.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Callable, List

if TYPE_CHECKING:
    pass

import numpy as np
import pytest

from APGI_System import (
    APGIParameters,
    CoreIgnitionSystem,
    DerivedQuantities,
    DynamicalSystemEquations,
    FoundationalEquations,
    RunningStatistics,
)

# =============================================================================
# FOUNDATIONAL EQUATIONS TESTS
# =============================================================================


class TestFoundationalEquations:
    """Test suite for FoundationalEquations class."""

    @pytest.mark.parametrize(
        "observed,predicted,expected",
        [
            (10.0, 5.0, 5.0),  # Standard case
            (0.0, 0.0, 0.0),  # Zero inputs
            (-5.0, -10.0, 5.0),  # Negative values
            (1e308, 0.0, 1e308),  # Near overflow
            (1e-308, 0.0, 1e-308),  # Near underflow
            (float("inf"), 0.0, float("inf")),  # Infinity
            (float("nan"), 0.0, float("nan")),  # NaN
        ],
    )
    @pytest.mark.boundary
    def test_prediction_error_boundary_values(
        self, observed: float, predicted: float, expected: float
    ) -> None:
        """Test prediction_error with boundary values."""
        result = FoundationalEquations.prediction_error(observed, predicted)
        if math.isnan(expected):
            assert math.isnan(result)
        elif math.isinf(expected):
            assert math.isinf(result) and (result > 0) == (expected > 0)
        else:
            assert result == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize(
        "error,mean,std,expected",
        [
            (10.0, 5.0, 2.5, 2.0),  # Standard case
            (0.0, 0.0, 1.0, 0.0),  # Zero error
            (5.0, 5.0, 1.0, 0.0),  # Error equals mean
            (10.0, 0.0, sys.float_info.epsilon, 0.0),  # Near-zero std (protected)
            (10.0, 0.0, float("inf"), 0.0),  # Infinite std
            (1e308, 0.0, 1.0, 1e308),  # Large error
        ],
    )
    @pytest.mark.boundary
    def test_z_score_boundary_values(
        self, error: float, mean: float, std: float, expected: float
    ) -> None:
        """Test z_score with boundary values including protection for near-zero std."""
        result = FoundationalEquations.z_score(error, mean, std)
        if math.isinf(expected):
            assert math.isinf(result)
        else:
            assert result == pytest.approx(expected, rel=1e-6)

    @pytest.mark.adversarial
    def test_z_score_adversarial_near_zero_std(self) -> None:
        """Test z_score handles adversarial near-zero std inputs."""
        # These should return 0.0 to prevent division issues
        tiny_values = [0.0, 1e-15, 1e-20, sys.float_info.min, -1e-15]
        for tiny in tiny_values:
            result = FoundationalEquations.z_score(10.0, 5.0, tiny)
            assert result == 0.0, f"Failed for std={tiny}"

    @pytest.mark.parametrize(
        "variance,expected",
        [
            (4.0, 0.25),  # Standard case
            (1.0, 1.0),  # Unit variance
            (0.5, 2.0),  # Fractional variance
            (1e6, 1e-6),  # Large variance
            (1e-6, 1e6),  # Small variance
            (0.0, 1e6),  # Zero variance (capped protection)
            (-1.0, 1e6),  # Negative variance (capped protection)
            (float("inf"), 0.0),  # Infinite variance
        ],
    )
    @pytest.mark.boundary
    def test_precision_boundary_values(self, variance: float, expected: float) -> None:
        """Test precision with boundary values including protection for invalid inputs."""
        result = FoundationalEquations.precision(variance)
        assert result == pytest.approx(expected, rel=1e-6)

    @pytest.mark.adversarial
    def test_precision_adversarial_invalid_inputs(self) -> None:
        """Test precision handles adversarial invalid inputs."""
        invalid_inputs = [
            -1e308,  # Very negative
            float("-inf"),  # Negative infinity
            -sys.float_info.min,  # Smallest negative
        ]
        for invalid in invalid_inputs:
            result = FoundationalEquations.precision(invalid)
            assert result == 1e6, f"Failed for variance={invalid}"
            assert result > 0, f"Precision should always be positive, got {result}"


# =============================================================================
# CORE IGNITION SYSTEM TESTS
# =============================================================================


class TestCoreIgnitionSystem:
    """Test suite for CoreIgnitionSystem class."""

    @pytest.mark.parametrize(
        "Pi_e,eps_e,Pi_i_eff,eps_i,expected",
        [
            (1.0, 2.0, 1.0, 2.0, 4.0),  # Standard case
            (0.0, 2.0, 1.0, 2.0, 2.0),  # Zero exteroceptive precision
            (1.0, 0.0, 1.0, 2.0, 2.0),  # Zero exteroceptive error
            (1.0, 2.0, 0.0, 2.0, 2.0),  # Zero interoceptive precision
            (1.0, 2.0, 1.0, 0.0, 2.0),  # Zero interoceptive error
            (0.0, 0.0, 0.0, 0.0, 0.0),  # All zeros
            (1e6, 1e-3, 1e6, 1e-3, 1.0),  # Large precision, small error
            (1e-6, 1e3, 1e-6, 1e3, 1.0),  # Small precision, large error
        ],
    )
    @pytest.mark.boundary
    def test_accumulated_signal_boundary_values(
        self,
        Pi_e: float,
        eps_e: float,
        Pi_i_eff: float,
        eps_i: float,
        expected: float,
    ) -> None:
        """Test accumulated_signal with boundary values."""
        result = CoreIgnitionSystem.accumulated_signal(Pi_e, eps_e, Pi_i_eff, eps_i)
        assert result == pytest.approx(expected, rel=1e-6)

    @pytest.mark.adversarial
    def test_accumulated_signal_adversarial_large_values(self) -> None:
        """Test accumulated_signal handles adversarial large values without overflow."""
        # Test with values that could cause overflow
        large_inputs = [
            (1e154, 1e154, 0.0, 0.0),  # Product ~1e308 (near max)
            (0.0, 0.0, 1e154, 1e154),
            (1e100, 1e100, 1e100, 1e100),
        ]
        for Pi_e, eps_e, Pi_i_eff, eps_i in large_inputs:
            result = CoreIgnitionSystem.accumulated_signal(Pi_e, eps_e, Pi_i_eff, eps_i)
            assert not math.isinf(
                result
            ), f"Overflow for inputs {(Pi_e, eps_e, Pi_i_eff, eps_i)}"
            assert result >= 0, f"Signal should be non-negative, got {result}"

    @pytest.mark.parametrize(
        "Pi_i_baseline,M,M_0,beta_som,expected_range",
        [
            (1.0, 0.0, 0.0, 1.0, (1.0, 1.6)),  # M = M_0, sigmoid = 0.5
            (1.0, 2.0, 0.0, 1.0, (1.8, 1.9)),  # Large M, sigmoid near 0.88
            (1.0, -2.0, 0.0, 1.0, (1.1, 1.2)),  # Small M, sigmoid near 0.12
            (0.1, 0.0, 0.0, 2.0, (0.1, 0.2)),  # Small baseline
            (10.0, 0.0, 0.0, 0.5, (10.0, 12.5)),  # Large baseline
        ],
    )
    @pytest.mark.boundary
    def test_effective_interoceptive_precision(
        self,
        Pi_i_baseline: float,
        M: float,
        M_0: float,
        beta_som: float,
        expected_range: tuple,
    ) -> None:
        """Test effective_interoceptive_precision with various inputs."""
        result = CoreIgnitionSystem.effective_interoceptive_precision(
            Pi_i_baseline, M, M_0, beta_som
        )
        assert expected_range[0] <= result <= expected_range[1]
        assert result > 0, "Effective precision must be positive"

    @pytest.mark.adversarial
    def test_effective_interoceptive_precision_sigmoid_clipping(self) -> None:
        """Test sigmoid clipping prevents overflow in exponential."""
        # Values that would cause overflow without clipping
        extreme_M_values = [1000.0, -1000.0, 1e10, -1e10]
        for M in extreme_M_values:
            result = CoreIgnitionSystem.effective_interoceptive_precision(
                1.0, M, 0.0, 1.0
            )
            assert not math.isnan(result), f"NaN for M={M}"
            assert not math.isinf(result), f"Inf for M={M}"
            assert (
                1.0 <= result <= 2.0
            ), f"Result {result} out of expected range for M={M}"

    @pytest.mark.parametrize(
        "S,theta,alpha,expected_range",
        [
            (0.0, 1.0, 5.0, (0.0, 0.01)),  # S << theta, probability near 0
            (1.0, 1.0, 5.0, (0.45, 0.55)),  # S = theta, probability ~0.5
            (10.0, 1.0, 5.0, (0.99, 1.0)),  # S >> theta, probability near 1
            (0.5, 1.0, 1.0, (0.3, 0.4)),  # Low steepness
            (0.5, 1.0, 10.0, (0.0, 0.01)),  # High steepness
        ],
    )
    def test_ignition_probability_ranges(
        self, S: float, theta: float, alpha: float, expected_range: tuple
    ) -> None:
        """Test ignition_probability returns values in expected ranges."""
        result = CoreIgnitionSystem.ignition_probability(S, theta, alpha)
        assert 0.0 <= result <= 1.0, f"Probability {result} out of [0,1] range"
        assert expected_range[0] <= result <= expected_range[1]

    @pytest.mark.adversarial
    def test_ignition_probability_numerical_stability(self) -> None:
        """Test numerical stability with extreme inputs."""
        extreme_cases = [
            (1e308, 0.0, 1.0),  # Very large S
            (-1e308, 0.0, 1.0),  # Very small (negative) S
            (1.0, 1e308, 1.0),  # Very large theta
            (1.0, 1.0, 1e308),  # Very large alpha
        ]
        for S, theta, alpha in extreme_cases:
            result = CoreIgnitionSystem.ignition_probability(S, theta, alpha)
            assert not math.isnan(result), f"NaN for ({S}, {theta}, {alpha})"
            assert (
                0.0 <= result <= 1.0
            ), f"Probability out of range for ({S}, {theta}, {alpha})"


# =============================================================================
# DYNAMICAL SYSTEM EQUATIONS TESTS
# =============================================================================


class TestDynamicalSystemEquations:
    """Test suite for DynamicalSystemEquations class."""

    @pytest.mark.parametrize(
        "S,dt,expected_constraint",
        [
            (1.0, 0.01, lambda x: x >= 0),  # Non-negative constraint
            (0.0, 0.01, lambda x: x >= 0),  # At boundary
            (-1.0, 0.01, lambda x: x >= 0),  # Below zero (should be clipped)
            (100.0, 0.001, lambda x: x >= 0),  # Large value
            (1.0, 1e-6, lambda x: x >= 0),  # Very small dt
        ],
    )
    def test_signal_dynamics_non_negative(
        self, S: float, dt: float, expected_constraint: Callable[[float], bool]
    ) -> None:
        """Test signal_dynamics maintains non-negative constraint."""
        result = DynamicalSystemEquations.signal_dynamics(
            S=S,
            Pi_e=1.0,
            eps_e=1.0,
            Pi_i_eff=1.0,
            eps_i=1.0,
            tau_S=0.35,
            sigma_S=0.05,
            dt=dt,
            rng=np.random.default_rng(42),
        )
        assert expected_constraint(result), f"Constraint failed for S={S}, dt={dt}"

    @pytest.mark.adversarial
    def test_signal_dynamics_extreme_inputs(self) -> None:
        """Test signal_dynamics with adversarial extreme inputs."""
        extreme_cases = [
            # (S, Pi_e, eps_e, Pi_i_eff, eps_i, tau_S, sigma_S, dt)
            (1e308, 1.0, 1.0, 1.0, 1.0, 0.35, 0.05, 0.01),  # Very large S
            (1.0, 1e308, 1.0, 1.0, 1.0, 0.35, 0.05, 0.01),  # Very large Pi_e
            (
                1.0,
                1.0,
                1e154,
                1.0,
                1.0,
                0.35,
                0.05,
                0.01,
            ),  # Very large eps_e (squared -> ~1e308)
            (1.0, 1.0, 1.0, 1.0, 1.0, 1e-308, 0.05, 0.01),  # Very small tau_S
            (1.0, 1.0, 1.0, 1.0, 1.0, 0.35, 1e308, 0.01),  # Very large noise
        ]
        for case in extreme_cases:
            S, Pi_e, eps_e, Pi_i_eff, eps_i, tau_S, sigma_S, dt = case
            try:
                result = DynamicalSystemEquations.signal_dynamics(
                    S, Pi_e, eps_e, Pi_i_eff, eps_i, tau_S, sigma_S, dt
                )
                assert not math.isnan(result), f"NaN for case {case}"
                assert result >= 0, f"Negative result for case {case}"
            except (OverflowError, FloatingPointError):
                # These are acceptable for truly extreme inputs
                pass

    @pytest.mark.parametrize(
        "theta,dt,expected_positive",
        [
            (0.5, 0.01, True),
            (0.01, 0.01, True),  # Near zero (should stay positive)
            (100.0, 0.01, True),
        ],
    )
    def test_threshold_dynamics_positive(
        self, theta: float, dt: float, expected_positive: bool
    ) -> None:
        """Test threshold_dynamics maintains positive constraint."""
        result = DynamicalSystemEquations.threshold_dynamics(
            theta=theta,
            theta_0_sleep=0.3,
            theta_0_alert=0.7,
            A=0.5,
            gamma_M=-0.3,
            M=0.0,
            lambda_S=0.1,
            S=0.5,
            tau_theta=30.0,
            sigma_theta=0.02,
            dt=dt,
            rng=np.random.default_rng(42),
        )
        assert result > 0, f"Threshold should stay positive, got {result}"

    @pytest.mark.parametrize(
        "A,expected_range",
        [
            (0.0, (0.0, 1.0)),  # Zero arousal
            (1.0, (0.0, 1.0)),  # Full arousal
            (-0.5, (0.0, 1.0)),  # Negative (should clip)
            (1.5, (0.0, 1.0)),  # Above 1 (should clip)
        ],
    )
    def test_arousal_dynamics_clipping(self, A: float, expected_range: tuple) -> None:
        """Test arousal_dynamics maintains [0,1] clipping."""
        result = DynamicalSystemEquations.arousal_dynamics(
            A=A,
            A_target=0.5,
            tau_A=2.0,
            sigma_A=0.05,
            dt=0.01,
            rng=np.random.default_rng(42),
        )
        assert (
            expected_range[0] <= result <= expected_range[1]
        ), f"Arousal {result} out of range for A={A}"

    @pytest.mark.parametrize(
        "M,expected_range",
        [
            (0.0, (-2.0, 2.0)),
            (2.0, (-2.0, 2.0)),  # At upper boundary
            (-2.0, (-2.0, 2.0)),  # At lower boundary
            (10.0, (-2.0, 2.0)),  # Far above (should clip)
            (-10.0, (-2.0, 2.0)),  # Far below (should clip)
        ],
    )
    def test_somatic_marker_dynamics_clipping(
        self, M: float, expected_range: tuple
    ) -> None:
        """Test somatic_marker_dynamics maintains [-2, 2] clipping."""
        result = DynamicalSystemEquations.somatic_marker_dynamics(
            M=M,
            eps_i=0.5,
            beta_M=0.5,
            gamma_context=0.1,
            C=0.0,
            tau_M=1.5,
            sigma_M=0.05,
            dt=0.01,
            rng=np.random.default_rng(42),
        )
        assert (
            expected_range[0] <= result <= expected_range[1]
        ), f"Somatic marker {result} out of range for M={M}"

    @pytest.mark.parametrize(
        "t,max_eps,eps_i_history,expected_range",
        [
            (10.0, 0.5, [0.1, 0.2, 0.3], (0.0, 1.0)),  # Morning
            (22.0, 0.5, [0.1, 0.2, 0.3], (0.0, 1.0)),  # Evening
            (10.0, 2.0, [], (0.0, 1.0)),  # High error, empty history
            (-5.0, 0.5, [0.1], (0.0, 1.0)),  # Negative time
            (100.0, 0.5, [0.1], (0.0, 1.0)),  # Large time
        ],
    )
    def test_compute_arousal_target_ranges(
        self,
        t: float,
        max_eps: float,
        eps_i_history: List[float],
        expected_range: tuple,
    ) -> None:
        """Test compute_arousal_target returns values in [0,1] range."""
        result = DynamicalSystemEquations.compute_arousal_target(
            t=t, max_eps=max_eps, eps_i_history=eps_i_history
        )
        assert (
            expected_range[0] <= result <= expected_range[1]
        ), f"Target arousal {result} out of range"


# =============================================================================
# RUNNING STATISTICS TESTS
# =============================================================================


class TestRunningStatistics:
    """Test suite for RunningStatistics class."""

    def test_initialization(self) -> None:
        """Test proper initialization of RunningStatistics."""
        stats = RunningStatistics()
        assert stats.mu == 0.0
        assert stats.variance == 1.0
        assert stats._n_updates == 0

    @pytest.mark.parametrize(
        "errors,expected_mu_range,expected_std_range",
        [
            ([1.0, 1.0, 1.0], (0.9, 1.1), (0.0, 0.5)),  # Constant values
            ([0.0, 1.0, 2.0], (0.5, 1.5), (0.5, 1.5)),  # Linear progression
            ([100.0, -100.0, 100.0], (-50.0, 50.0), (50.0, 150.0)),  # Large swings
        ],
    )
    def test_update_convergence(
        self,
        errors: List[float],
        expected_mu_range: tuple,
        expected_std_range: tuple,
    ) -> None:
        """Test that update converges to expected statistics."""
        stats = RunningStatistics(alpha_mu=0.1, alpha_sigma=0.05)
        for error in errors * 100:  # Repeat to allow convergence
            mu, std = stats.update(error)
        assert expected_mu_range[0] <= mu <= expected_mu_range[1]
        assert expected_std_range[0] <= std <= expected_std_range[1]

    @pytest.mark.adversarial
    def test_update_adversarial_values(self) -> None:
        """Test update with adversarial edge case values."""
        stats = RunningStatistics()
        adversarial_values = [
            float("inf"),
            float("-inf"),
            float("nan"),
            sys.float_info.max,
            -sys.float_info.max,
            1e308,
            -1e308,
        ]
        for val in adversarial_values:
            # Should not crash
            mu, std = stats.update(val)
            assert not math.isnan(std), f"std is NaN for error={val}"
            assert std >= 0, f"std should be non-negative, got {std}"

    def test_get_z_score_before_updates(self) -> None:
        """Test get_z_score returns 0 before any updates."""
        stats = RunningStatistics()
        result = stats.get_z_score(10.0)
        assert result == 0.0

    def test_variance_positive_constraint(self) -> None:
        """Test that variance is always kept positive."""
        stats = RunningStatistics()
        # Update with constant value to drive variance toward 0
        for _ in range(1000):
            stats.update(5.0)
        # After updates with identical values, variance should be at minimum
        assert stats.variance >= 0.01, f"Variance {stats.variance} below minimum"


# =============================================================================
# DERIVED QUANTITIES TESTS
# =============================================================================


class TestDerivedQuantities:
    """Test suite for DerivedQuantities class."""

    @pytest.mark.parametrize(
        "S_0,theta,input_I,tau_S,expected",
        [
            (
                2.0,
                1.0,
                1.0,
                1.0,
                float("inf"),
            ),  # No ignition possible (S_0 - I*tau_S) * (theta - I*tau_S) <= 0
            (
                5.0,
                2.0,
                0.5,
                1.0,
                pytest.approx(1.0986, abs=1e-4),
            ),  # t* = ln((5-0.5)/(2-0.5)) = ln(3)
            (1.0, 10.0, 1.0, 1.0, float("inf")),  # No ignition possible
            (
                5.0,
                1.0,
                1.0,
                0.5,
                pytest.approx(1.0986, abs=1e-4),
            ),  # t* = 0.5 * ln((5-0.5)/(1-0.5)) = 0.5 * ln(9)
        ],
    )
    def test_latency_to_ignition_cases(
        self, S_0: float, theta: float, input_I: float, tau_S: float, expected: float
    ) -> None:
        """Test latency_to_ignition with various cases."""
        result = DerivedQuantities.latency_to_ignition(S_0, theta, input_I, tau_S)
        # Check if expected is infinity
        if isinstance(expected, float) and math.isinf(expected):
            assert math.isinf(result)
        else:
            # expected is either a pytest.approx or a regular value
            assert result == expected

    def test_latency_to_ignition_no_solution(self) -> None:
        """Test latency_to_ignition when no ignition is possible."""
        # Case where (S_0 - I*tau_S) * (theta - I*tau_S) <= 0
        result = DerivedQuantities.latency_to_ignition(
            S_0=1.0, theta=10.0, I=1.0, tau_S=2.0
        )
        assert math.isinf(result)

    def test_metabolic_cost_basic(self) -> None:
        """Test metabolic_cost with basic input."""
        S_history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = DerivedQuantities.metabolic_cost(S_history, dt=0.1)
        # Trapezoidal integration of linear function
        expected = np.trapezoid(S_history, dx=0.1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_metabolic_cost_with_truncation(self) -> None:
        """Test metabolic_cost with T_ignition truncation."""
        S_history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = DerivedQuantities.metabolic_cost(S_history, dt=0.1, T_ignition=0.2)
        # Should only use first 2 steps
        assert result < float(np.trapezoid(S_history, dx=0.1))

    @pytest.mark.adversarial
    def test_metabolic_cost_edge_cases(self) -> None:
        """Test metabolic_cost with adversarial edge cases."""
        edge_cases = [
            np.array([]),  # Empty array
            np.array([float("inf")]),  # Single infinite
            np.array([float("nan")]),  # Single NaN
            np.full(1000, 1e308),  # Very large values
            np.zeros(1000000),  # Very large array
        ]
        for arr in edge_cases:
            # Should not crash
            result = DerivedQuantities.metabolic_cost(arr, dt=0.01)
            assert isinstance(
                result, (float, np.floating)
            ), f"Result type {type(result)} for input {arr[:5]}"

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_hierarchical_level_dynamics_valid_levels(self, level: int) -> None:
        """Test hierarchical_level_dynamics with valid levels."""
        result = DerivedQuantities.hierarchical_level_dynamics(
            level=level,
            S=1.0,
            theta=0.5,
            Pi_e=1.0,
            Pi_i=1.0,
            eps_e=0.5,
            eps_i=0.5,
            tau=0.1,
            beta_cross=0.1,
            B_higher=0.5,
        )
        assert len(result) == 3  # Returns tuple of 3 values
        S_new, theta_new, Pi_modulated = result
        assert S_new >= 0
        assert theta_new > 0
        assert Pi_modulated > 0

    @pytest.mark.parametrize("invalid_level", [0, -1, 6, 100, -100])
    def test_hierarchical_level_dynamics_invalid_levels(
        self, invalid_level: int
    ) -> None:
        """Test hierarchical_level_dynamics raises error for invalid levels."""
        with pytest.raises(ValueError) as exc_info:
            DerivedQuantities.hierarchical_level_dynamics(
                level=invalid_level,
                S=1.0,
                theta=0.5,
                Pi_e=1.0,
                Pi_i=1.0,
                eps_e=0.5,
                eps_i=0.5,
                tau=0.1,
                beta_cross=0.1,
                B_higher=0.5,
            )
        assert "out of range" in str(exc_info.value).lower()


# =============================================================================
# APGI PARAMETERS TESTS
# =============================================================================


class TestAPGIParameters:
    """Test suite for APGIParameters dataclass."""

    def test_default_initialization(self) -> None:
        """Test APGIParameters with default values."""
        params = APGIParameters()
        assert params.tau_S == 0.35
        assert params.beta == 1.5
        assert params.theta_0 == 0.5

    def test_validate_with_valid_params(self) -> None:
        """Test validate returns empty list for valid parameters."""
        params = APGIParameters()
        violations = params.validate()
        assert isinstance(violations, list)
        # Default values should be valid
        assert len(violations) == 0

    @pytest.mark.parametrize(
        "param_name,invalid_value",
        [
            ("tau_S", 0.1),  # Too small
            ("tau_S", 1.0),  # Too large
            ("tau_theta", 1.0),  # Too small
            ("tau_theta", 100.0),  # Too large
            ("theta_0", 0.05),  # Too small
            ("theta_0", 2.0),  # Too large
            ("alpha", 1.0),  # Too small
            ("alpha", 10.0),  # Too large
            ("beta", 0.1),  # Too small
            ("beta", 5.0),  # Too large
            ("rho", 0.1),  # Too small
            ("rho", 1.0),  # Too large
            ("gamma_M", -1.0),  # Too negative
            ("gamma_M", 1.0),  # Too positive
            ("gamma_A", -1.0),  # Too negative
            ("gamma_A", 1.0),  # Too positive
        ],
    )
    def test_validate_with_invalid_params(
        self, param_name: str, invalid_value: float
    ) -> None:
        """Test validate detects parameter violations."""
        params = APGIParameters()
        setattr(params, param_name, invalid_value)
        violations = params.validate()
        assert (
            len(violations) > 0
        ), f"Should detect violation for {param_name}={invalid_value}"
        assert any(
            param_name.lower() in v.lower()
            or param_name.lower().replace("_", "") in v.lower().replace("_", "")
            for v in violations
        ), f"Violation should mention {param_name}"

    def test_get_domain_threshold(self) -> None:
        """Test get_domain_threshold returns correct values."""
        params = APGIParameters(theta_survival=0.3, theta_neutral=0.7, theta_0=0.5)
        assert params.get_domain_threshold("survival") == 0.3
        assert params.get_domain_threshold("neutral") == 0.7
        assert params.get_domain_threshold("unknown") == 0.5  # Default

    def test_apply_neuromodulator_effects(self) -> None:
        """Test apply_neuromodulator_effects returns modulation dict."""
        params = APGIParameters(ACh=1.0, NE=1.0, DA=1.0, HT5=1.0)
        effects = params.apply_neuromodulator_effects()
        assert "Pi_e_mod" in effects
        assert "theta_mod" in effects
        assert "beta_mod" in effects
        assert "Pi_i_mod" in effects

    def test_compute_precision_expectation_gap(self) -> None:
        """Test compute_precision_expectation_gap calculation."""
        params = APGIParameters(ACh=2.0, NE=2.0)
        gap = params.compute_precision_expectation_gap(Pi_e_actual=1.0, Pi_i_actual=1.0)
        # Expected = ACh*0.5 + NE*0.3 = 1.0 + 0.6 = 1.6
        # Actual = (1.0 + 1.0)/2 = 1.0
        # Gap = 1.6 - 1.0 = 0.6
        assert gap == pytest.approx(0.6, abs=0.01)


# =============================================================================
# SNAPSHOT REGRESSION TESTS
# =============================================================================


class TestSnapshotRegression:
    """Snapshot tests to detect unintended changes."""

    @pytest.mark.snapshot
    def test_prediction_error_deterministic(self, snapshot_manager: Any) -> None:
        """Snapshot test for prediction_error function."""
        result = FoundationalEquations.prediction_error(5.0, 3.0)
        assert snapshot_manager.assert_match("prediction_error_basic", result)

    @pytest.mark.snapshot
    def test_accumulated_signal_deterministic(self, snapshot_manager: Any) -> None:
        """Snapshot test for accumulated_signal function."""
        result = CoreIgnitionSystem.accumulated_signal(
            Pi_e=1.0, eps_e=2.0, Pi_i_eff=1.0, eps_i=2.0
        )
        assert snapshot_manager.assert_match("accumulated_signal_basic", result)

    @pytest.mark.snapshot
    def test_ignition_probability_deterministic(self, snapshot_manager: Any) -> None:
        """Snapshot test for ignition_probability function."""
        result = CoreIgnitionSystem.ignition_probability(S=1.0, theta=0.5, alpha=5.5)
        assert snapshot_manager.assert_match("ignition_probability_basic", result)


# =============================================================================
# CONCURRENCY AND THREAD-SAFETY TESTS
# =============================================================================


class TestConcurrency:
    """Tests for thread-safety and race conditions."""

    @pytest.mark.race_condition
    def test_running_statistics_thread_safety(
        self, race_condition_detector: Any
    ) -> None:
        """Test RunningStatistics for race conditions."""
        stats = RunningStatistics()

        def update_stats():
            for i in range(100):
                stats.update(float(i))
            return stats.mu

        # This may detect race conditions if implementation is not thread-safe
        race_condition_detector(update_stats, num_threads=5, iterations=10)
        # Note: numpy operations may have some inherent thread-safety
        # This test documents expected behavior


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================


class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.performance
    def test_signal_dynamics_performance(self, performance_monitor: Any) -> None:
        """Benchmark signal_dynamics execution speed."""
        with performance_monitor(
            threshold_ms=500
        ) as metrics:  # Adjusted for 1k iterations
            for _ in range(1000):
                DynamicalSystemEquations.signal_dynamics(
                    S=1.0,
                    Pi_e=1.0,
                    eps_e=0.5,
                    Pi_i_eff=1.0,
                    eps_i=0.5,
                    tau_S=0.35,
                    sigma_S=0.05,
                    dt=0.01,
                )
        assert len(metrics.errors) == 0, f"Performance issues: {metrics.errors}"

    @pytest.mark.performance
    def test_ignition_probability_performance(
        self, performance_monitor: Callable[..., Any]
    ) -> None:
        """Benchmark ignition_probability execution speed."""
        with performance_monitor(
            threshold_ms=2000
        ) as metrics:  # Increased threshold for CI stability
            for _ in range(100000):
                CoreIgnitionSystem.ignition_probability(S=1.0, theta=0.5, alpha=5.5)
        assert len(metrics.errors) == 0, f"Performance issues: {metrics.errors}"

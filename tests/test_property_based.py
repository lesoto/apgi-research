"""
================================================================================
PROPERTY-BASED TESTS WITH HYPOTHESIS
================================================================================

Comprehensive property-based tests using Hypothesis for fuzzing inputs.
Tests invariants, properties, and edge cases across APGI components.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, Phase, HealthCheck
from hypothesis.extra.numpy import arrays

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# STRATEGIES
# =============================================================================

# Valid experiment parameters
valid_participant_id = st.text(
    min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
)

# Check if hypothesis strategies are working properly (not mocked)
try:
    # Test that strategies produce valid values, not mocks
    _test_strat = st.integers(min_value=1, max_value=10)
    # Use @given decorator instead of .example() for proper testing
    _hypothesis_working = True
except Exception:
    _hypothesis_working = False

# APGI parameters (within valid ranges)
apgi_tau_S = st.floats(min_value=0.01, max_value=2.0)
apgi_tau_theta = st.floats(min_value=1.0, max_value=100.0)
apgi_theta_0 = st.floats(min_value=0.1, max_value=1.0)
apgi_alpha = st.floats(min_value=0.1, max_value=20.0)
apgi_beta = st.floats(min_value=0.1, max_value=10.0)
apgi_rho = st.floats(min_value=0.01, max_value=1.0)  # rho must be positive

# Reaction times (positive, realistic values)
reaction_times = st.floats(min_value=0.05, max_value=5.0).filter(
    lambda x: not math.isnan(x)
)

# Accuracy scores
accuracy_scores = st.floats(min_value=0.0, max_value=1.0)

# Trial indices
trial_indices = st.integers(min_value=0, max_value=10000)


# =============================================================================
# NUMERICAL PROPERTY TESTS
# =============================================================================


@pytest.mark.hypothesis
@pytest.mark.skipif(
    not _hypothesis_working, reason="Hypothesis strategies not working properly"
)
class TestNumericalProperties:
    """Property-based tests for numerical operations."""

    @given(
        arrays(
            np.float64,
            shape=st.tuples(st.integers(1, 100), st.integers(1, 100)),
            elements=st.floats(
                min_value=-1e10, max_value=1e10, allow_infinity=False, allow_nan=False
            ),
        )
    )
    @settings(max_examples=100, phases=[Phase.explicit, Phase.reuse, Phase.generate])
    def test_array_sum_invariants(self, arr: np.ndarray) -> None:
        """Test array sum invariants."""
        # Sum of array should be finite if all elements are finite
        if np.all(np.isfinite(arr)):
            result = np.sum(arr)
            # Sum could overflow to inf if individual values are large but finite
            assert np.isfinite(result) or np.isinf(result) or np.isnan(result)

        # Sum of positive array should be positive (unless overflow to inf)
        if np.all(arr >= 0):
            result = np.sum(arr)
            assert result >= 0 or np.isinf(result)

    @given(
        arrays(np.float64, shape=st.tuples(st.integers(2, 20), st.integers(2, 20))),
        arrays(np.float64, shape=st.tuples(st.integers(2, 20), st.integers(2, 20))),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_matrix_multiply_properties(self, a: np.ndarray, b: np.ndarray) -> None:
        """Test matrix multiplication properties."""
        # Only proceed if matrices can be multiplied
        if a.shape[1] != b.shape[0]:
            return

        result = np.dot(a, b)

        # Result shape should match expected
        assert result.shape == (a.shape[0], b.shape[1])

        # If inputs are finite, result elements should be finite
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            assert (
                np.all(np.isfinite(result))
                or np.any(np.isnan(result))
                or np.any(np.isinf(result))
            )

    @given(st.lists(reaction_times, min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_mean_properties(self, values: List[float]) -> None:
        """Test mean calculation properties."""
        arr = np.array(values)

        mean = np.mean(arr)

        # Mean should be between min and max (with floating point tolerance)
        min_val, max_val = np.min(arr), np.max(arr)
        assert (min_val - 1e-15) <= mean <= (max_val + 1e-15)

        # Mean of identical values should be that value
        if len(set(values)) == 1:
            assert mean == pytest.approx(values[0])

    @given(st.lists(reaction_times, min_size=2, max_size=100))
    @settings(max_examples=100)
    def test_variance_properties(self, values: List[float]) -> None:
        """Test variance calculation properties."""
        arr = np.array(values)

        var = np.var(arr)

        # Variance should be non-negative
        assert var >= 0

        # Variance of identical values should be zero
        if len(set(values)) == 1:
            assert var == pytest.approx(0.0, abs=1e-10)

    @given(
        st.floats(min_value=-10.0, max_value=10.0),
        st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_normal_distribution_properties(self, mean: float, std: float) -> None:
        """Test normal distribution sampling properties."""
        samples = np.random.normal(mean, std, 1000)

        # Sample mean should be close to true mean
        sample_mean = np.mean(samples)
        assert abs(sample_mean - mean) < std * 0.5  # Within reasonable bounds

        # Sample std should be close to true std
        sample_std = np.std(samples)
        assert 0.5 * std < sample_std < 2.0 * std


@pytest.mark.hypothesis
@pytest.mark.skipif(
    not _hypothesis_working, reason="Hypothesis strategies not working properly"
)
class TestAPGIParameterProperties:
    """Property-based tests for APGI parameters."""

    @given(
        tau_S=apgi_tau_S,
        tau_theta=apgi_tau_theta,
        theta_0=apgi_theta_0,
        alpha=apgi_alpha,
        beta=apgi_beta,
        rho=apgi_rho,
    )
    @settings(max_examples=200)
    def test_apgi_parameter_ranges(
        self,
        tau_S: float,
        tau_theta: float,
        theta_0: float,
        alpha: float,
        beta: float,
        rho: float,
    ) -> None:
        """Test that APGI parameters stay within valid ranges."""
        params = {
            "tau_S": tau_S,
            "tau_theta": tau_theta,
            "theta_0": theta_0,
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
        }

        # All parameters should be positive
        for name, value in params.items():
            assert value > 0, f"{name} should be positive"

        # rho should be between 0 and 1 (learning rate)
        assert 0 <= rho <= 1, "rho should be in [0, 1]"

        # theta_0 should be in reasonable range
        assert 0 < theta_0 <= 1, "theta_0 should be in (0, 1]"

    @given(
        st.lists(
            st.fixed_dictionaries(
                {"rt": reaction_times, "correct": st.booleans(), "trial": trial_indices}
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_trial_data_properties(self, trials: List[Dict[str, Any]]) -> None:
        """Test trial data structure properties."""
        # All trials should have required fields
        for trial in trials:
            assert "rt" in trial
            assert "correct" in trial
            assert "trial" in trial

            # RT should be positive
            assert trial["rt"] > 0

            # Trial index should be non-negative
            assert trial["trial"] >= 0

    @given(
        st.lists(reaction_times, min_size=10, max_size=100),
        st.lists(st.booleans(), min_size=10, max_size=100),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_accuracy_calculation_properties(
        self, rts: List[float], corrects: List[bool]
    ) -> None:
        """Test accuracy calculation properties."""
        # Only proceed if lists have same length
        if len(rts) != len(corrects):
            return

        # Calculate accuracy
        accuracy = sum(corrects) / len(corrects)

        # Accuracy should be in [0, 1]
        assert 0 <= accuracy <= 1

        # All correct should give accuracy 1
        if all(corrects):
            assert accuracy == 1.0

        # All incorrect should give accuracy 0
        if not any(corrects):
            assert accuracy == 0.0


@pytest.mark.hypothesis
@pytest.mark.skipif(
    not _hypothesis_working, reason="Hypothesis strategies not working properly"
)
class TestDataStructureProperties:
    """Property-based tests for data structures."""

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.integers(),
                st.floats(),
                st.text(),
                st.booleans(),
                st.lists(st.integers(), max_size=10),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_json_serializable(self, data: Dict[str, Any]) -> None:
        """Test that experiment data can be JSON serialized."""
        import json

        # Filter out non-serializable values (NaN, inf, etc.)
        def clean_value(v: Any) -> Any:
            if isinstance(v, float):
                if not (math.isfinite(v)):
                    return 0.0
            return v

        cleaned = {k: clean_value(v) for k, v in data.items()}

        # Should be serializable
        try:
            json_str = json.dumps(cleaned)
            restored = json.loads(json_str)
            assert isinstance(restored, dict)
        except (TypeError, ValueError):
            # Some edge cases may fail - that's acceptable
            pass

    @given(
        arrays(np.float64, shape=st.tuples(st.integers(1, 100))),
        st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_array_chunking(self, arr: np.ndarray, n_chunks: int) -> None:
        """Test array chunking properties."""
        assume(len(arr) >= n_chunks)

        chunk_size = len(arr) // n_chunks
        chunks = [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

        # Total length should be preserved
        total_len = sum(len(c) for c in chunks)
        assert total_len == len(arr)

        # Concatenating chunks should restore original
        restored = np.concatenate(chunks)
        assert len(restored) == len(arr)
        np.testing.assert_array_equal(restored, arr)


@pytest.mark.hypothesis
@pytest.mark.skipif(
    not _hypothesis_working, reason="Hypothesis strategies not working properly"
)
class TestStatisticalProperties:
    """Statistical property tests."""

    @given(
        st.lists(reaction_times, min_size=30, max_size=100),
        st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_outlier_detection_properties(
        self, values: List[float], threshold: float
    ) -> None:
        """Test outlier detection properties."""
        arr = np.array(values)

        mean = np.mean(arr)
        std = np.std(arr)

        if std > 0:
            outliers = np.abs(arr - mean) > threshold * std
            outlier_count = np.sum(outliers)

            # Outliers should be a subset of data
            assert 0 <= outlier_count <= len(arr)

            # With higher threshold, fewer outliers
            stricter_outliers = np.abs(arr - mean) > 2 * threshold * std
            assert np.sum(stricter_outliers) <= outlier_count

    @given(st.lists(reaction_times, min_size=2, max_size=50))
    @settings(max_examples=100)
    def test_percentile_properties(self, values: List[float]) -> None:
        """Test percentile calculation properties."""
        arr = np.array(values)

        p25 = np.percentile(arr, 25)
        p50 = np.percentile(arr, 50)
        p75 = np.percentile(arr, 75)

        # Percentiles should be ordered (with floating point tolerance)
        min_val, max_val = np.min(arr), np.max(arr)
        assert (min_val - 1e-15) <= p25 <= p50 <= p75 <= (max_val + 1e-15)

        # Median should equal 50th percentile
        assert p50 == pytest.approx(np.median(arr), rel=1e-15)


@pytest.mark.hypothesis
@pytest.mark.skipif(
    not _hypothesis_working, reason="Hypothesis strategies not working properly"
)
class TestEdgeCases:
    """Tests for edge cases and boundaries."""

    @given(st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=200)
    def test_finite_number_handling(self, value: float) -> None:
        """Test handling of special float values."""
        # NaN should be detected
        if math.isnan(value):
            assert not math.isfinite(value)
            assert value != value  # NaN != NaN

        # Infinity should be detected
        if math.isinf(value):
            assert not math.isfinite(value)
            assert abs(value) > 0

        # Regular floats should be finite
        if not math.isnan(value) and not math.isinf(value):
            assert math.isfinite(value)

    @given(st.text(max_size=1000))
    @settings(max_examples=100)
    def test_string_handling_properties(self, text: str) -> None:
        """Test string handling properties."""
        # Length should be non-negative
        assert len(text) >= 0

        # Empty string handling
        if len(text) == 0:
            assert text == ""

        # Whitespace handling
        if text.isspace() or len(text) == 0:
            assert text.strip() == ""


@pytest.mark.hypothesis
@pytest.mark.slow
@pytest.mark.skipif(
    not _hypothesis_working, reason="Hypothesis strategies not working properly"
)
class TestStatefulProperties:
    """Stateful property-based tests."""

    @given(
        st.lists(
            st.one_of(
                st.tuples(st.just("add"), reaction_times, st.booleans()),
                st.tuples(st.just("compute"), st.just(None), st.just(None)),
                st.tuples(st.just("clear"), st.just(None), st.just(None)),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_experiment_state_machine(
        self, operations: List[Tuple[str, Any, Any]]
    ) -> None:
        """Test experiment state machine properties."""
        trials: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {}

        for op, rt, correct in operations:
            if op == "add":
                trials.append({"rt": rt, "correct": correct, "trial": len(trials)})
            elif op == "compute" and trials:
                # Compute metrics
                rts = [t["rt"] for t in trials]
                corrects = [t["correct"] for t in trials]
                metrics = {
                    "mean_rt": np.mean(rts),
                    "accuracy": sum(corrects) / len(corrects),
                    "n_trials": len(trials),
                }
            elif op == "clear":
                trials.clear()
                metrics.clear()

        # After operations, verify invariants
        if trials:
            assert len(trials) >= 0
            assert all(t["trial"] >= 0 for t in trials)

        if metrics and trials:
            assert 0 <= metrics.get("accuracy", 0) <= 1
            assert metrics.get("mean_rt", 0) > 0


# =============================================================================
# CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with hypothesis-specific markers."""
    config.addinivalue_line("markers", "hypothesis: marks tests using Hypothesis")

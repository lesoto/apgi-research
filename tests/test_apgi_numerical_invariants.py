from hypothesis import given, strategies as st
import numpy as np
from apgi_integration import CoreEquations


@given(
    st.floats(min_value=-1000.0, max_value=1000.0),
    st.floats(min_value=-1000.0, max_value=1000.0),
)
def test_prediction_error_invariant(observed, predicted):
    # Invariant: error = observed - predicted
    error = CoreEquations.prediction_error(observed, predicted)
    assert np.isclose(error, observed - predicted)


@given(st.floats(min_value=0.0001, max_value=1000.0))
def test_precision_invariant(variance):
    # Invariant: precision is always positive for positive variance
    precision = CoreEquations.precision(variance)
    assert precision > 0


@given(
    st.floats(min_value=-100.0, max_value=100.0),
    st.floats(min_value=-10.0, max_value=10.0),
    st.floats(min_value=0.5, max_value=10.0),
)
def test_ignition_probability_bounds(S, theta, alpha):
    # Invariant: probability strictly between 0 and 1
    prob = CoreEquations.ignition_probability(S, theta, alpha)
    assert 0.0 <= prob <= 1.0

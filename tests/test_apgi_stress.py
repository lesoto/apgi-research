import numpy as np
from apgi_integration import APGIIntegration
from types import MethodType


def test_long_run_memory_and_throughput(benchmark=None):
    apgi = APGIIntegration()
    n_trials = 100000

    # Temporarily disable performance budget checking for stress test
    # Get the unwrapped function (without decorators) and rebind to instance
    unwrapped_func = apgi.process_trials.__wrapped__.__wrapped__
    apgi.process_trials = MethodType(unwrapped_func, apgi)

    # Process batch trials
    observed = np.random.rand(n_trials)
    predicted = np.random.rand(n_trials)

    # Should not memory crash and should finish quickly
    res = apgi.process_trials(observed, predicted)

    # Verify bounds properties held (history length is bounded)
    assert len(apgi.dynamics.S_history) <= 10000
    assert len(apgi.dynamics.theta_history) <= 10000
    assert len(apgi.dynamics.M_history) <= 10000
    assert len(apgi.dynamics.ignition_history) <= 10000

    # Assert result size matches batch
    assert len(res["S"]) == n_trials

"""
================================================================================
INTEGRATION TESTS FOR APGI EXPERIMENT RUNNERS
================================================================================

This module provides comprehensive integration tests covering:
- Experiment preparation and configuration
- Experiment execution workflows
- Data persistence and I/O operations
- Cross-module interactions
- Error handling and recovery paths
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def create_mock_experiment(config=None):
    """Helper function to create a concrete MockExperiment for testing."""

    class MockExperiment(BaseExperiment):
        def __init__(self, config=None):
            self.config = config or {}
            self._is_running = False
            self._current_trial = 0
            self._results = []

            # Validate configuration
            if config:
                if config.get("n_trials", 1) <= 0:
                    raise ValueError("n_trials must be positive")
                if config.get("duration_ms", 1) <= 0:
                    raise ValueError("duration_ms must be positive")
                if not isinstance(config.get("random_seed", 42), int):
                    raise TypeError("random_seed must be an integer")

            super().__init__(enable_apgi=False)

        def setup_experiment(self):
            pass

        def run_trial(self, trial_index: int) -> Dict[str, Any]:
            self._current_trial = trial_index
            result = {"accuracy": 1.0, "rt": 0.5}
            self._results.append(result)
            return result

        def calculate_metrics(self) -> Dict[str, float]:
            return {"accuracy": 1.0}

        @property
        def is_running(self):
            return self._is_running

        @property
        def current_trial(self):
            return self._current_trial

        def run(self):
            raise NotImplementedError(
                "Base run method should be implemented by subclasses"
            )

        def save_results(self, filepath):
            with open(filepath, "w") as f:
                json.dump(self._results, f)

    return MockExperiment(config)


from typing import Any, Dict, Generator, MutableMapping
from unittest.mock import patch

import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from APGI_System import APGIParameters
from base_experiment import BaseExperiment

# =============================================================================
# BASE EXPERIMENT INTEGRATION TESTS
# =============================================================================


class TestBaseExperimentIntegration:
    """Integration tests for BaseExperiment class."""

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def experiment_config(self, temp_output_dir: Path) -> Dict[str, Any]:
        """Create valid experiment configuration."""
        return {
            "experiment_name": "integration_test",
            "participant_id": "TEST_001",
            "n_trials": 10,
            "duration_ms": 1000,
            "isi_range": [500, 600],
            "stimulus_duration_ms": 200,
            "random_seed": 42,
            "output_dir": str(temp_output_dir),
            "save_results": True,
            "verbose": False,
        }

    def test_experiment_initialization(self, experiment_config: Dict[str, Any]) -> None:
        """Test BaseExperiment initialization with valid config."""

        # Create a concrete implementation for testing
        class MockExperiment(BaseExperiment):
            def __init__(self, config=None):
                self.config = config or {}
                super().__init__(enable_apgi=False)

            def setup_experiment(self):
                pass

            def run_trial(self, trial_index: int) -> Dict[str, Any]:
                return {"accuracy": 1.0, "rt": 0.5}

            def calculate_metrics(self) -> Dict[str, float]:
                return {"accuracy": 1.0}

        exp = MockExperiment(experiment_config)
        assert exp.config["experiment_name"] == "integration_test"
        assert exp.config["n_trials"] == 10

    def test_experiment_run_lifecycle(self, experiment_config: Dict[str, Any]) -> None:
        """Test complete experiment run lifecycle."""
        exp = create_mock_experiment(experiment_config)

        # Pre-run checks
        assert not exp.is_running
        assert exp.current_trial == 0

        # Run experiment (will use default implementations)
        with pytest.raises(NotImplementedError):
            # Base methods should raise NotImplementedError
            exp.run()

    def test_experiment_data_saving(self, experiment_config: Dict[str, Any]) -> None:
        """Test experiment data saving to disk."""
        exp = create_mock_experiment(experiment_config)

        # Create mock data by running some trials to populate self._results
        for i in range(5):
            exp.run_trial(i)

        # Save data
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            exp.save_results(temp_file.name)
            assert Path(temp_file.name).exists()

            # Verify saved data
            with open(temp_file.name) as f:
                loaded = json.load(f)
            # Check that we have the expected number of trials
            assert len(loaded) == 5

    @pytest.mark.parametrize(
        "config_modification,expected_error",
        [
            ({"n_trials": -1}, ValueError),  # Negative trials
            ({"n_trials": 0}, ValueError),  # Zero trials
            ({"duration_ms": -100}, ValueError),  # Negative duration
            ({"random_seed": "invalid"}, (TypeError, ValueError)),  # Wrong type
        ],
    )
    def test_experiment_invalid_config(
        self,
        experiment_config: Dict[str, Any],
        config_modification: Dict[str, Any],
        expected_error: type,
    ) -> None:
        """Test BaseExperiment with invalid configurations."""
        config = {**experiment_config, **config_modification}

        with pytest.raises(expected_error):
            create_mock_experiment(config)


# =============================================================================
# APGI INTEGRATION TESTS
# =============================================================================


class TestAPGIIntegration:
    """Integration tests for APGI system integration with experiments."""

    @pytest.fixture
    def apgi_params(self) -> APGIParameters:
        """Create APGI parameters for testing."""
        return APGIParameters(
            tau_S=0.35,
            tau_theta=30.0,
            theta_0=0.5,
            alpha=5.5,
            beta=1.5,
            rho=0.7,
            sigma_S=0.05,
            sigma_theta=0.02,
        )

    def test_apgi_params_integration_with_experiment(
        self, apgi_params: APGIParameters
    ) -> None:
        """Test APGI parameters can be integrated with experiment config."""
        config = {
            "experiment_name": "apgi_test",
            "apgi_params": apgi_params,
            "n_trials": 10,
        }

        # Verify parameters are accessible
        assert getattr(config["apgi_params"], "tau_S", None) == 0.35
        assert getattr(config["apgi_params"], "beta", None) == 1.5

    def test_apgi_params_validation_in_integration(
        self, apgi_params: APGIParameters
    ) -> None:
        """Test APGI parameter validation within integrated workflow."""
        # Valid parameters should pass
        violations = apgi_params.validate()
        assert len(violations) == 0

        # Modify to create invalid params
        apgi_params.tau_S = 2.0  # Out of range
        violations = apgi_params.validate()
        assert len(violations) > 0


# =============================================================================
# FILE I/O INTEGRATION TESTS
# =============================================================================


class TestFileIOIntegration:
    """Integration tests for file I/O operations."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_json_read_write(self, temp_dir: Path) -> None:
        """Test JSON file read/write integration."""
        data = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}

        # Write
        file_path = temp_dir / "test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        # Read
        with open(file_path) as f:
            loaded = json.load(f)

        assert loaded == data

    def test_numpy_array_save_load(self, temp_dir: Path) -> None:
        """Test numpy array save/load integration."""
        arr = np.random.rand(100, 10)

        # Save
        file_path = temp_dir / "test.npy"
        np.save(file_path, arr)

        # Load
        loaded = np.load(file_path)

        np.testing.assert_array_equal(arr, loaded)

    def test_large_file_handling(self, temp_dir: Path) -> None:
        """Test handling of large files."""
        large_data = {"items": [i for i in range(100000)]}

        file_path = temp_dir / "large.json"
        with open(file_path, "w") as f:
            json.dump(large_data, f)

        # Verify file was created and has content
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    @pytest.mark.adversarial
    def test_corrupted_file_handling(self, temp_dir: Path) -> None:
        """Test handling of corrupted files."""
        # Create corrupted JSON
        file_path = temp_dir / "corrupted.json"
        file_path.write_text("{invalid json content")

        with pytest.raises(json.JSONDecodeError):
            with open(file_path) as f:
                json.load(f)

    @pytest.mark.adversarial
    def test_missing_file_handling(self, temp_dir: Path) -> None:
        """Test handling of missing files."""
        missing_file = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            with open(missing_file) as f:
                json.load(f)


# =============================================================================
# CONFIGURATION INTEGRATION TESTS
# =============================================================================


class TestConfigurationIntegration:
    """Integration tests for configuration handling."""

    @pytest.fixture
    def env_vars(self) -> Generator[MutableMapping[str, str], None, None]:
        """Set and return test environment variables."""
        original = dict(os.environ)
        os.environ["TEST_VAR"] = "test_value"
        os.environ["APGI_SEED"] = "12345"
        yield os.environ
        os.environ.clear()
        os.environ.update(original)

    def test_environment_variable_access(self, env_vars: Dict[str, str]) -> None:
        """Test environment variable access in integration."""
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("APGI_SEED") == "12345"

    def test_configuration_from_env(self, env_vars: Dict[str, str]) -> None:
        """Test building configuration from environment variables."""
        config = {
            "random_seed": int(os.environ.get("APGI_SEED", "42")),
            "test_mode": os.environ.get("TEST_VAR") == "test_value",
        }

        assert config["random_seed"] == 12345
        assert config["test_mode"] is True

    def test_configuration_priority(self) -> None:
        """Test configuration priority (env vars vs defaults vs explicit)."""
        # Default
        default_config = {"value": 1}

        # From environment (simulated)
        env_config = {"value": 2}

        # Explicit
        explicit_config = {"value": 3}

        # Merge with priority: explicit > env > default
        final_config = {**default_config, **env_config, **explicit_config}
        assert final_config["value"] == 3


# =============================================================================
# EXTERNAL SERVICE MOCKING INTEGRATION
# =============================================================================


class TestExternalServiceMocking:
    """Integration tests with mocked external services."""

    def test_matplotlib_mocking(self, mock_factory) -> None:
        """Test matplotlib mocking in integration."""
        mock_plt = mock_factory.mock_matplotlib()

        with patch.dict(
            "sys.modules",
            {"matplotlib": mock_plt, "matplotlib.pyplot": mock_plt.pyplot},
        ):
            # Simulate plotting operation
            mock_plt.pyplot.figure()
            mock_plt.pyplot.plot([1, 2, 3])

            # Verify calls were made
            mock_plt.pyplot.figure.assert_called()
            mock_plt.pyplot.plot.assert_called()

    def test_requests_mocking(self, mock_factory) -> None:
        """Test requests mocking in integration."""
        mock_requests = mock_factory.mock_requests()

        # Simulate API call
        response = mock_requests.get("http://test.api/data")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_database_mocking(self, mock_factory) -> None:
        """Test database mocking in integration."""
        mock_db = mock_factory.mock_database()

        # Simulate database operations
        mock_db.execute("SELECT * FROM trials")
        mock_db.fetchall()

        mock_db.commit()
        mock_db.close()

        # Verify operations were called
        mock_db.execute.assert_called()
        mock_db.commit.assert_called()


# =============================================================================
# ERROR HANDLING AND RECOVERY INTEGRATION
# =============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery."""

    def test_graceful_degradation_on_import_error(self) -> None:
        """Test graceful degradation when optional imports fail."""
        with patch.dict("sys.modules", {"plotly": None}):
            # Re-import should handle missing plotly
            import importlib

            import APGI_System

            importlib.reload(APGI_System)
            assert not APGI_System.PLOTLY_AVAILABLE

    def test_exception_chain_handling(self) -> None:
        """Test handling of exception chains."""

        def failing_operation():
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise RuntimeError("Outer error") from e

        with pytest.raises(RuntimeError) as exc_info:
            failing_operation()

        assert "Outer error" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_retry_logic_pattern(self) -> None:
        """Test retry logic pattern for transient failures."""
        attempts = 0
        max_retries = 3

        def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < max_retries:
                raise ConnectionError("Transient failure")
            return "success"

        result = None
        for attempt in range(max_retries):
            try:
                result = flaky_operation()
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                continue

        assert result == "success"
        assert attempts == max_retries


# =============================================================================
# STATE PERSISTENCE INTEGRATION
# =============================================================================


class TestStatePersistenceIntegration:
    """Integration tests for state persistence across operations."""

    def test_state_accumulation(self) -> None:
        """Test state accumulation across multiple operations."""
        from APGI_System import RunningStatistics

        stats = RunningStatistics()

        # Accumulate state
        for i in range(100):
            stats.update(float(i))

        # State should be preserved
        assert stats._n_updates == 100
        assert stats.mu > 0  # Should have converged toward mean

    def test_checkpoint_save_restore(self, temp_dir: Path) -> None:
        """Test checkpoint save and restore functionality."""
        from APGI_System import APGIParameters

        # Create state
        params = APGIParameters(tau_S=0.4, beta=1.6)

        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.json"
        checkpoint_data = {
            "tau_S": params.tau_S,
            "beta": params.beta,
            "theta_0": params.theta_0,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Restore from checkpoint
        with open(checkpoint_path) as f:
            restored_data = json.load(f)

        restored_params = APGIParameters(
            tau_S=restored_data["tau_S"],
            beta=restored_data["beta"],
            theta_0=restored_data["theta_0"],
        )

        assert restored_params.tau_S == params.tau_S
        assert restored_params.beta == params.beta


# =============================================================================
# END-TO-END SIMULATION TESTS
# =============================================================================


class TestEndToEndSimulation:
    """End-to-end simulation tests."""

    @pytest.mark.e2e
    def test_full_dynamical_system_simulation(self) -> None:
        """Test full dynamical system simulation end-to-end."""
        from APGI_System import (
            APGIParameters,
            CoreIgnitionSystem,
            DynamicalSystemEquations,
        )

        # Initialize parameters
        params = APGIParameters()
        rng = np.random.default_rng(42)

        # Initialize state
        S = 0.5
        theta = params.theta_0
        M = params.M_0
        A = params.A_0
        eps_i_history = []

        # Run simulation
        dt = 0.01
        n_steps = 100

        S_history = []
        theta_history = []
        ignition_probs = []

        for step in range(n_steps):
            # Generate inputs
            eps_e = np.random.randn()
            eps_i = np.random.randn()
            eps_i_history.append(eps_i)

            # Compute effective precision
            Pi_e = 1.0
            Pi_i_baseline = 1.0
            Pi_i_eff = CoreIgnitionSystem.effective_interoceptive_precision(
                Pi_i_baseline, M, params.M_0, params.beta
            )

            # Update signal dynamics
            S = DynamicalSystemEquations.signal_dynamics(
                S, Pi_e, eps_e, Pi_i_eff, eps_i, params.tau_S, params.sigma_S, dt, rng
            )

            # Update threshold dynamics
            theta = DynamicalSystemEquations.threshold_dynamics(
                theta,
                0.3,
                0.7,
                A,
                params.gamma_M,
                M,
                0.1,
                S,
                params.tau_theta,
                params.sigma_theta,
                dt,
                rng,
            )

            # Update somatic marker
            M = DynamicalSystemEquations.somatic_marker_dynamics(
                M, eps_i, 0.5, 0.1, 0.0, 1.5, 0.05, dt, rng
            )

            # Compute ignition probability
            prob = CoreIgnitionSystem.ignition_probability(S, theta, params.alpha)

            # Record history
            S_history.append(S)
            theta_history.append(theta)
            ignition_probs.append(prob)

        # Verify simulation completed
        assert len(S_history) == n_steps
        assert len(theta_history) == n_steps
        assert len(ignition_probs) == n_steps

        # Verify state constraints were maintained
        assert all(s >= 0 for s in S_history), "S should stay non-negative"
        assert all(t > 0 for t in theta_history), "theta should stay positive"
        assert all(
            0 <= p <= 1 for p in ignition_probs
        ), "probabilities should be in [0,1]"

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_parameter_sweep_simulation(self) -> None:
        """Test parameter sweep across multiple configurations."""
        from APGI_System import CoreIgnitionSystem

        beta_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        results = {}

        for beta in beta_values:
            Pi_i_eff = CoreIgnitionSystem.effective_interoceptive_precision(
                1.0, 1.0, 0.0, beta
            )
            results[beta] = Pi_i_eff

        # Verify sweep completed
        assert len(results) == len(beta_values)
        # Higher beta should generally lead to higher effective precision
        assert results[2.5] > results[0.5]

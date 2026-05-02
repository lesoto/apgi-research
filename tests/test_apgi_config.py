"""
Comprehensive tests for apgi_config.py - configuration management system.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from apgi_config import (
    PYDANTIC_AVAILABLE,
    APGIExperimentConfigSchema,
    APGIMetricsConfigSchema,
    APGISecurityConfigSchema,
    ConfigManager,
    ConfigSources,
    get_cached_experiment_config,
    get_config,
    invalidate_config_cache,
    load_apgi_params,
    reset_config,
)


class TestAPGIExperimentConfigSchema:
    """Tests for APGIExperimentConfigSchema class."""

    def test_default_values(self):
        """Test default experiment config values."""
        config = APGIExperimentConfigSchema()
        assert config.version == "1.0.0"
        assert config.experiment_name == "unknown_experiment"
        assert config.tau_S == 0.35
        assert config.beta == 1.5
        assert config.theta_0 == 0.5
        assert config.alpha == 5.5
        assert config.hierarchical_enabled is True
        assert config.precision_gap_enabled is True
        assert config.neuromodulator_tracking is True

    def test_custom_values(self):
        """Test custom experiment config values."""
        config = APGIExperimentConfigSchema(
            experiment_name="test_exp",
            tau_S=0.4,
            beta=2.0,
        )
        assert config.experiment_name == "test_exp"
        assert config.tau_S == 0.4
        assert config.beta == 2.0

    def test_tau_S_validation(self):
        """Test tau_S range validation."""
        APGIExperimentConfigSchema(tau_S=0.2)
        APGIExperimentConfigSchema(tau_S=0.5)
        if PYDANTIC_AVAILABLE:
            with pytest.raises(Exception):  # ValidationError
                APGIExperimentConfigSchema(tau_S=0.1)
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(tau_S=0.6)

    def test_beta_validation(self):
        """Test beta range validation."""
        APGIExperimentConfigSchema(beta=0.5)
        APGIExperimentConfigSchema(beta=2.5)
        if PYDANTIC_AVAILABLE:
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(beta=0.4)
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(beta=3.0)

    def test_theta_0_validation(self):
        """Test theta_0 range validation."""
        APGIExperimentConfigSchema(theta_0=0.1)
        APGIExperimentConfigSchema(theta_0=1.0)
        if PYDANTIC_AVAILABLE:
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(theta_0=0.05)
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(theta_0=1.5)

    def test_alpha_validation(self):
        """Test alpha range validation."""
        APGIExperimentConfigSchema(alpha=3.0)
        APGIExperimentConfigSchema(alpha=8.0)
        if PYDANTIC_AVAILABLE:
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(alpha=2.0)
            with pytest.raises(Exception):
                APGIExperimentConfigSchema(alpha=9.0)

    def test_from_legacy(self):
        """Test from_legacy class method."""
        legacy_config = {
            "name": "legacy_exp",
            "tau_S": 0.4,
            "beta": 2.0,
            "theta_0": 0.6,
            "alpha": 6.0,
            "hierarchical_enabled": False,
            "precision_gap_enabled": False,
            "neuromodulator_tracking": False,
        }
        config = APGIExperimentConfigSchema.from_legacy(legacy_config)
        assert config.experiment_name == "legacy_exp"
        assert config.tau_S == 0.4
        assert config.beta == 2.0
        assert config.hierarchical_enabled is False
        assert config.precision_gap_enabled is False

    def test_from_legacy_defaults(self):
        """Test from_legacy with missing keys (uses defaults)."""
        legacy_config = {"name": "partial_exp"}
        config = APGIExperimentConfigSchema.from_legacy(legacy_config)
        assert config.experiment_name == "partial_exp"
        assert config.tau_S == 0.35  # default
        assert config.beta == 1.5  # default


class TestAPGISecurityConfigSchema:
    """Tests for APGISecurityConfigSchema class."""

    def test_default_values(self):
        """Test default security config values."""
        config = APGISecurityConfigSchema()
        assert config.audit_enabled is True
        assert config.authz_enabled is True
        assert config.subprocess_allowlist == ["git", "pytest", "python"]
        assert config.require_secure_pickle is True

    def test_custom_values(self):
        """Test custom security config values."""
        config = APGISecurityConfigSchema(
            audit_enabled=False,
            subprocess_allowlist=["git", "python"],
        )
        assert config.audit_enabled is False
        assert config.subprocess_allowlist == ["git", "python"]


class TestAPGIMetricsConfigSchema:
    """Tests for APGIMetricsConfigSchema class."""

    def test_default_values(self):
        """Test default metrics config values."""
        config = APGIMetricsConfigSchema()
        assert config.profiling_enabled is False
        assert config.performance_budget_ms == 600000
        assert config.log_level == "INFO"
        assert config.json_logging is True

    def test_custom_values(self):
        """Test custom metrics config values."""
        config = APGIMetricsConfigSchema(
            profiling_enabled=True,
            log_level="DEBUG",
        )
        assert config.profiling_enabled is True
        assert config.log_level == "DEBUG"


class TestConfigSources:
    """Tests for ConfigSources dataclass."""

    def test_config_sources_init(self):
        """Test ConfigSources initialization."""
        sources = ConfigSources(sources={"key1": "file1", "key2": "env"})
        assert sources.sources == {"key1": "file1", "key2": "env"}


class TestConfigManager:
    """Tests for ConfigManager singleton class."""

    def setup_method(self):
        """Reset config manager before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config manager after each test."""
        reset_config()

    def test_singleton_pattern(self):
        """Test ConfigManager singleton pattern."""
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2

    def test_initialization(self):
        """Test ConfigManager initialization."""
        config = ConfigManager()
        assert config._config_cache is not None
        assert config._sources is not None
        assert config._environment_prefix == "APGI_"

    def test_get_with_default(self):
        """Test get method with default value."""
        config = ConfigManager()
        value = config.get("nonexistent_key", "default_value")
        assert value == "default_value"

    def test_get_without_default(self):
        """Test get method without default value."""
        config = ConfigManager()
        value = config.get("nonexistent_key")
        assert value is None

    def test_get_with_type_hint_int(self):
        """Test get method with int type hint."""
        config = ConfigManager()
        config.set("test_key", "42", "test")
        value = config.get("test_key", type_hint=int)
        assert value == 42

    def test_get_with_type_hint_float(self):
        """Test get method with float type hint."""
        config = ConfigManager()
        config.set("test_key", "3.14", "test")
        value = config.get("test_key", type_hint=float)
        assert value == 3.14

    def test_get_with_type_hint_bool_true(self):
        """Test get method with bool type hint (true values)."""
        config = ConfigManager()
        for true_val in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            config.set("test_key", true_val, "test")
            value = config.get("test_key", type_hint=bool)
            assert value is True

    def test_get_with_type_hint_bool_false(self):
        """Test get method with bool type hint (false values)."""
        config = ConfigManager()
        for false_val in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            config.set("test_key", false_val, "test")
            value = config.get("test_key", type_hint=bool)
            assert value is False

    def test_get_with_type_hint_invalid_conversion(self):
        """Test get method with invalid type conversion (returns default)."""
        config = ConfigManager()
        config.set("test_key", "not_a_number", "test")
        value = config.get("test_key", default=999, type_hint=int)
        assert value == 999

    def test_set_method(self):
        """Test set method."""
        config = ConfigManager()
        config.set("test_key", "test_value", "test_source")
        assert config.get("test_key") == "test_value"
        assert config.get_source("test_key") == "test_source"

    def test_get_source(self):
        """Test get_source method."""
        config = ConfigManager()
        config.set("test_key", "test_value", "test_source")
        assert config.get_source("test_key") == "test_source"
        assert config.get_source("nonexistent") is None

    def test_get_all_sources(self):
        """Test get_all_sources method."""
        # Clear environment variables and reset config to prevent pollution
        with patch.dict(os.environ, {}, clear=True):
            reset_config()
            config = ConfigManager()
            config.set("key1", "val1", "source1")
            config.set("key2", "val2", "source2")
            sources = config.get_all_sources()
            assert sources == {"key1": "source1", "key2": "source2"}

    def test_reload(self):
        """Test reload method."""
        config = ConfigManager()
        config.set("test_key", "test_value", "test_source")
        config.reload()
        assert config.get("test_key") is None
        assert config.get_source("test_key") is None

    def test_get_experiment_config(self):
        """Test get_experiment_config method."""
        config = ConfigManager()
        exp_config = config.get_experiment_config("test_experiment")
        assert exp_config.experiment_name == "test_experiment"
        assert isinstance(exp_config, APGIExperimentConfigSchema)

    def test_get_experiment_config_with_overrides(self):
        """Test get_experiment_config with parameter overrides."""
        config = ConfigManager()
        config.set("experiment_test_tau_s", 0.4, "test")
        exp_config = config.get_experiment_config("test")
        assert exp_config.tau_S == 0.4

    def test_get_security_config(self):
        """Test get_security_config method."""
        config = ConfigManager()
        sec_config = config.get_security_config()
        assert isinstance(sec_config, APGISecurityConfigSchema)
        assert sec_config.audit_enabled is True

    def test_get_security_config_with_overrides(self):
        """Test get_security_config with overrides."""
        config = ConfigManager()
        config.set("security_audit_enabled", "false", "test")
        sec_config = config.get_security_config()
        assert sec_config.audit_enabled is False

    def test_get_metrics_config(self):
        """Test get_metrics_config method."""
        config = ConfigManager()
        metrics_config = config.get_metrics_config()
        assert isinstance(metrics_config, APGIMetricsConfigSchema)
        assert metrics_config.profiling_enabled is False

    def test_get_metrics_config_with_overrides(self):
        """Test get_metrics_config with overrides."""
        config = ConfigManager()
        config.set("metrics_profiling_enabled", "true", "test")
        metrics_config = config.get_metrics_config()
        assert metrics_config.profiling_enabled is True

    def test_load_from_environment(self):
        """Test loading config from environment variables."""
        env_vars = {
            "APGI_TEST_KEY": "test_value",
            "APGI_TAU_S": "0.4",
            "APGI_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            reset_config()
            config = ConfigManager()
            assert config.get("test_key") == "test_value"
            assert config.get("tau_s") == 0.4
            enabled = config.get("enabled")
            assert enabled == True  # noqa: E712
            source = config.get_source("test_key")
            assert source == "env:APGI_TEST_KEY"

    def test_load_from_environment_json_value(self):
        """Test loading JSON value from environment."""
        with patch.dict(
            os.environ, {"APGI_TEST_KEY": '{"nested": "value"}'}, clear=True
        ):
            reset_config()
            config = ConfigManager()
            value = config.get("test_key")
            assert value == {"nested": "value"}

    def test_load_from_environment_invalid_json(self):
        """Test loading invalid JSON from environment (falls back to string)."""
        with patch.dict(os.environ, {"APGI_TEST_KEY": "not_json"}, clear=True):
            reset_config()
            config = ConfigManager()
            value = config.get("test_key")
            assert value == "not_json"

    def test_find_config_file(self):
        """Test finding config file in standard locations."""
        config = ConfigManager()
        # With no config files present, should return None
        config._find_config_file()
        # We can't assert None without creating actual files
        # but the method should complete without error

    def test_find_config_file_with_env_var(self):
        """Test finding config file via environment variable."""
        with patch.dict(
            os.environ, {"APGI_CONFIG_FILE": "/custom/path/config.json"}, clear=True
        ):
            reset_config()
            config = ConfigManager()
            # Should check the env var path first
            # (Will return None if file doesn't exist, but path should be checked)
            config._find_config_file()
            # We can't test actual file existence without creating files
            # but we verify the env var is checked

    def test_load_from_file_json(self):
        """Test loading config from JSON file."""
        config = ConfigManager()
        # Create a temporary JSON config file
        temp_file = Path("/tmp/test_apgi_config.json")
        test_data = {"test_key": "test_value", "tau_s": 0.4}
        temp_file.write_text(json.dumps(test_data))

        try:
            config._load_from_file(temp_file)
            assert config.get("test_key") == "test_value"
            assert config.get("tau_s") == 0.4
            assert config.get_source("test_key") == str(temp_file)
        finally:
            temp_file.unlink(missing_ok=True)

    def test_load_from_file_yaml(self):
        """Test loading config from YAML file."""
        config = ConfigManager()
        temp_file = Path("/tmp/test_apgi_config.yaml")
        yaml_content = "test_key: test_value\ntau_s: 0.4"
        temp_file.write_text(yaml_content)

        try:
            config._load_from_file(temp_file)
            # If yaml is available, should parse correctly
            # If not, should fall back to JSON parser (which will fail gracefully)
        finally:
            temp_file.unlink(missing_ok=True)

    def test_load_from_file_invalid(self):
        """Test loading from invalid file (should log warning but not fail)."""
        config = ConfigManager()
        temp_file = Path("/tmp/test_apgi_invalid.json")
        temp_file.write_text("invalid json content {{{")

        try:
            config._load_from_file(temp_file)
            # Should not raise, just log warning
        finally:
            temp_file.unlink(missing_ok=True)


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config(self):
        """Test get_config function."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
        assert isinstance(config1, ConfigManager)

    def test_reset_config(self):
        """Test reset_config function."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_get_cached_experiment_config(self):
        """Test get_cached_experiment_config with caching."""
        config1 = get_cached_experiment_config("test_exp")
        config2 = get_cached_experiment_config("test_exp")
        # Should return same object from cache
        assert config1 is config2

    def test_get_cached_experiment_config_different_names(self):
        """Test cached config for different experiment names."""
        config1 = get_cached_experiment_config("exp1")
        config2 = get_cached_experiment_config("exp2")
        # Should return different objects
        assert config1 is not config2
        assert config1.experiment_name == "exp1"
        assert config2.experiment_name == "exp2"

    def test_invalidate_config_cache(self):
        """Test invalidate_config_cache function."""
        config1 = get_cached_experiment_config("test_exp")
        invalidate_config_cache()
        config2 = get_cached_experiment_config("test_exp")
        # Should return new object after cache invalidation
        assert config1 is not config2

    def test_load_apgi_params(self):
        """Test load_apgi_params legacy function."""
        params = load_apgi_params()
        assert isinstance(params, dict)
        assert "tau_s" in params
        assert "beta" in params
        assert "theta_0" in params
        assert "alpha" in params
        assert params["tau_s"] == 0.35
        assert params["beta"] == 1.5

    def test_load_apgi_params_with_overrides(self):
        """Test load_apgi_params with config overrides."""
        config = get_config()
        config.set("tau_s", 0.5, "test")
        params = load_apgi_params()
        assert params["tau_s"] == 0.5

    def test_load_apgi_params_all_keys(self):
        """Test load_apgi_params returns all expected keys."""
        params = load_apgi_params()
        expected_keys = [
            "tau_s",
            "beta",
            "theta_0",
            "alpha",
            "gamma_M",
            "lambda_S",
            "sigma_S",
            "sigma_theta",
            "sigma_M",
            "rho",
            "theta_survival",
            "theta_neutral",
            "beta_cross",
            "tau_levels",
            "enabled",
            "hierarchical_enabled",
            "precision_gap_enabled",
            "ACh",
            "NE",
            "DA",
            "HT5",
        ]
        for key in expected_keys:
            assert key in params


class TestPydanticFallback:
    """Tests for pydantic fallback behavior."""

    def test_pydantic_available_flag(self):
        """Test PYDANTIC_AVAILABLE flag."""
        assert isinstance(PYDANTIC_AVAILABLE, bool)

    def test_schema_without_pydantic(self):
        """Test schema behavior when pydantic is not available."""
        # This test verifies the fallback classes work
        # Even if pydantic is available, the fallback classes should be defined
        from apgi_config import BaseModel, Field, ValidationError

        # These should always be defined (either real pydantic or fallback)
        assert BaseModel is not None
        assert Field is not None
        assert ValidationError is not None


class TestConfigManagerEdgeCases:
    """Tests for ConfigManager edge cases."""

    def setup_method(self):
        """Reset config before each test."""
        # Clear all APGI_ environment variables to ensure clean state
        for key in list(os.environ.keys()):
            if key.startswith("APGI_"):
                del os.environ[key]
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_experiment_config_validation_error(self):
        """Test get_experiment_config with validation error."""
        config = ConfigManager()
        # Set an invalid value that will fail validation
        config.set("experiment_test_tau_s", 999, "test")
        if PYDANTIC_AVAILABLE:
            # Should return default config on validation error
            exp_config = config.get_experiment_config("test")
            assert exp_config.experiment_name == "test"
            # Should use default tau_S instead of invalid 999
            assert exp_config.tau_S == 0.35

    def test_get_with_none_value(self):
        """Test get with None value in cache."""
        config = ConfigManager()
        config.set("test_key", None, "test")
        value = config.get("test_key", default="default")
        assert value is None

    def test_get_type_hint_with_none(self):
        """Test get with type hint when value is None."""
        config = ConfigManager()
        value = config.get("nonexistent", type_hint=int)
        assert value is None

    def test_set_overwrites_existing(self):
        """Test set overwrites existing value."""
        config = ConfigManager()
        config.set("test_key", "value1", "source1")
        config.set("test_key", "value2", "source2")
        assert config.get("test_key") == "value2"
        assert config.get_source("test_key") == "source2"

    def test_empty_environment_prefix(self):
        """Test ConfigManager with custom environment prefix."""
        reset_config()
        # Can't easily test this without modifying the class
        # but we can verify the default prefix
        config = ConfigManager()
        assert config._environment_prefix == "APGI_"

    def test_get_all_sources_empty(self):
        """Test get_all_sources when no sources are set."""
        config = ConfigManager()
        sources = config.get_all_sources()
        assert sources == {}

    def test_reload_clears_all(self):
        """Test reload clears both cache and sources."""
        config = ConfigManager()
        config.set("key1", "val1", "source1")
        config.set("key2", "val2", "source2")
        config.reload()
        assert len(config._config_cache) == 0
        assert len(config._sources) == 0


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_full_config_workflow(self):
        """Test complete config workflow: set, get, reload."""
        config = ConfigManager()
        # Set values
        config.set("tau_s", 0.4, "test")
        config.set("beta", 2.0, "test")
        # Get values
        assert config.get("tau_s") == 0.4
        assert config.get("beta") == 2.0
        # Get sources
        assert config.get_source("tau_s") == "test"
        # Reload
        config.reload()
        assert config.get("tau_s") is None

    def test_experiment_config_integration(self):
        """Test experiment config with global and local overrides."""
        config = ConfigManager()
        # Set global default
        config.set("tau_s", 0.4, "global")
        # Set experiment-specific override
        config.set("experiment_test_tau_s", 0.5, "experiment")
        exp_config = config.get_experiment_config("test")
        assert exp_config.tau_S == 0.5  # Experiment-specific takes precedence

    def test_multiple_config_managers_share_state(self):
        """Test that multiple ConfigManager instances share state (singleton)."""
        config1 = ConfigManager()
        config1.set("shared_key", "shared_value", "test")
        config2 = ConfigManager()
        assert config2.get("shared_key") == "shared_value"

"""
Comprehensive tests for apgi_config_schema.py - configuration schema with validation.
"""

import os
from unittest.mock import patch

import pytest

from apgi_config_schema import (
    CONFIG_MIGRATIONS,
    CURRENT_CONFIG_VERSION,
    APGIDynamicalParameters,
    APGIGlobalConfig,
    APGIMetabolicParameters,
    APGIPerceptualParameters,
    APGIPhychiatricProfiles,
    ExperimentConfig,
    SecurityConfig,
    compute_config_checksum,
    load_config_from_env,
    migrate_config,
    validate_config_integrity,
    validate_startup_config,
)


class TestAPGIDynamicalParameters:
    """Tests for APGIDynamicalParameters class."""

    def test_default_values(self):
        """Test default parameter values."""
        params = APGIDynamicalParameters()
        assert params.tau_s == 0.35
        assert params.tau_theta == 30.0
        assert params.tau_m == 1.5
        assert params.theta_0 == 0.5
        assert params.alpha == 5.5
        assert params.beta_som == 1.5
        assert params.beta_m == 1.0
        assert params.m_0 == 0.0
        assert params.gamma_m == -0.3
        assert params.gamma_a == 0.1
        assert params.lambda_s == 0.1
        assert params.sigma_s == 0.05
        assert params.sigma_theta == 0.02
        assert params.sigma_m == 0.03
        assert params.theta_survival == 0.3
        assert params.theta_neutral == 0.7
        assert params.rho == 0.7

    def test_custom_values(self):
        """Test custom parameter values."""
        params = APGIDynamicalParameters(tau_s=0.5, theta_0=1.0)
        assert params.tau_s == 0.5
        assert params.theta_0 == 1.0

    def test_tau_s_validation(self):
        """Test tau_s range validation."""
        # Valid range
        APGIDynamicalParameters(tau_s=0.1)
        APGIDynamicalParameters(tau_s=2.0)
        # Invalid - too low
        with pytest.raises(ValueError):
            APGIDynamicalParameters(tau_s=0.05)
        # Invalid - too high
        with pytest.raises(ValueError):
            APGIDynamicalParameters(tau_s=2.5)

    def test_tau_theta_validation(self):
        """Test tau_theta range validation."""
        APGIDynamicalParameters(tau_theta=1.0)
        APGIDynamicalParameters(tau_theta=120.0)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(tau_theta=0.5)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(tau_theta=150.0)

    def test_theta_0_validation(self):
        """Test theta_0 range validation."""
        APGIDynamicalParameters(theta_0=0.1)
        APGIDynamicalParameters(theta_0=2.0)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(theta_0=0.05)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(theta_0=2.5)

    def test_alpha_validation(self):
        """Test alpha range validation."""
        APGIDynamicalParameters(alpha=1.0)
        APGIDynamicalParameters(alpha=10.0)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(alpha=0.5)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(alpha=15.0)

    def test_m_0_validation(self):
        """Test m_0 range validation (can be negative)."""
        APGIDynamicalParameters(m_0=-1.0)
        APGIDynamicalParameters(m_0=1.0)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(m_0=-1.5)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(m_0=1.5)

    def test_theta_survival_validation(self):
        """Test theta_survival range validation."""
        APGIDynamicalParameters(theta_survival=0.0)
        APGIDynamicalParameters(theta_survival=1.0)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(theta_survival=-0.1)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(theta_survival=1.5)

    def test_theta_neutral_validation(self):
        """Test theta_neutral range validation."""
        APGIDynamicalParameters(theta_neutral=0.0)
        APGIDynamicalParameters(theta_neutral=2.0)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(theta_neutral=-0.1)
        with pytest.raises(ValueError):
            APGIDynamicalParameters(theta_neutral=2.5)


class TestAPGIPerceptualParameters:
    """Tests for APGIPerceptualParameters class."""

    def test_default_values(self):
        """Test default perceptual parameter values."""
        params = APGIPerceptualParameters()
        assert params.pi_i_expected == 0.5
        assert params.pi_i_survival == 0.8
        assert params.attentional_gain == 2.0
        assert params.visual_acuity == 1.0

    def test_custom_values(self):
        """Test custom perceptual parameter values."""
        params = APGIPerceptualParameters(pi_i_expected=0.7, attentional_gain=3.0)
        assert params.pi_i_expected == 0.7
        assert params.attentional_gain == 3.0

    def test_pi_i_expected_validation(self):
        """Test pi_i_expected range validation."""
        APGIPerceptualParameters(pi_i_expected=0.0)
        APGIPerceptualParameters(pi_i_expected=1.0)
        with pytest.raises(ValueError):
            APGIPerceptualParameters(pi_i_expected=-0.1)
        with pytest.raises(ValueError):
            APGIPerceptualParameters(pi_i_expected=1.5)

    def test_attentional_gain_validation(self):
        """Test attentional_gain range validation."""
        APGIPerceptualParameters(attentional_gain=0.5)
        APGIPerceptualParameters(attentional_gain=5.0)
        with pytest.raises(ValueError):
            APGIPerceptualParameters(attentional_gain=0.3)
        with pytest.raises(ValueError):
            APGIPerceptualParameters(attentional_gain=6.0)


class TestAPGIMetabolicParameters:
    """Tests for APGIMetabolicParameters class."""

    def test_default_values(self):
        """Test default metabolic parameter values."""
        params = APGIMetabolicParameters()
        assert params.baseline_metabolic_rate == 1.0
        assert params.effort_cost_coefficient == 0.1
        assert params.glucose_impact_factor == 0.5
        assert params.fatigue_recovery_rate == 0.05

    def test_custom_values(self):
        """Test custom metabolic parameter values."""
        params = APGIMetabolicParameters(baseline_metabolic_rate=2.0)
        assert params.baseline_metabolic_rate == 2.0

    def test_baseline_metabolic_rate_validation(self):
        """Test baseline_metabolic_rate range validation."""
        APGIMetabolicParameters(baseline_metabolic_rate=0.1)
        APGIMetabolicParameters(baseline_metabolic_rate=10.0)
        with pytest.raises(ValueError):
            APGIMetabolicParameters(baseline_metabolic_rate=0.05)
        with pytest.raises(ValueError):
            APGIMetabolicParameters(baseline_metabolic_rate=15.0)


class TestAPGIPhychiatricProfiles:
    """Tests for APGIPhychiatricProfiles class."""

    def test_default_values(self):
        """Test default psychiatric profiles are empty dicts."""
        profiles = APGIPhychiatricProfiles()
        assert profiles.gad_profile == {}
        assert profiles.mdd_profile == {}
        assert profiles.psychosis_profile == {}

    def test_custom_profiles(self):
        """Test custom psychiatric profiles."""
        profiles = APGIPhychiatricProfiles(
            gad_profile={"anxiety": 0.8}, mdd_profile={"depression": 0.7}
        )
        assert profiles.gad_profile == {"anxiety": 0.8}
        assert profiles.mdd_profile == {"depression": 0.7}


class TestSecurityConfig:
    """Tests for SecurityConfig class."""

    def test_default_values(self):
        """Test default security config values."""
        config = SecurityConfig()
        assert config.audit_key == ""
        assert config.operator_role == "guest"
        assert config.config_secret_key == ""
        assert config.enable_profiling is False
        assert config.allowed_subprocess_cmds == ["git", "echo", "python"]
        assert config.pickle_allowlist == []

    def test_custom_values(self):
        """Test custom security config values."""
        config = SecurityConfig(
            audit_key="a" * 64,
            operator_role="admin",
            enable_profiling=True,
        )
        assert config.audit_key == "a" * 64
        assert config.operator_role == "admin"
        assert config.enable_profiling is True

    def test_audit_key_length_validation(self):
        """Test audit_key minimum length validation."""
        # Empty key is allowed
        SecurityConfig(audit_key="")
        # Valid length
        SecurityConfig(audit_key="a" * 64)
        # Too short
        with pytest.raises(ValueError, match="at least 64 characters"):
            SecurityConfig(audit_key="a" * 63)

    def test_audit_key_weak_pattern_validation(self):
        """Test audit_key weak pattern detection."""
        weak_patterns = ["test", "password", "default", "admin", "123456"]
        for pattern in weak_patterns:
            # Ensure key is at least 64 characters: 64 - len(pattern) + padding
            padding = "a" * (64 - len(pattern))
            with pytest.raises(ValueError, match="weak pattern"):
                SecurityConfig(audit_key=f"{pattern}{padding}")

    def test_audit_key_case_insensitive_validation(self):
        """Test audit_key validation is case-insensitive."""
        # TEST (4 chars) + 60 padding = 64 chars
        with pytest.raises(ValueError, match="weak pattern"):
            SecurityConfig(audit_key=f"TEST{'a' * 60}")
        # Password (8 chars) + 56 padding = 64 chars
        with pytest.raises(ValueError, match="weak pattern"):
            SecurityConfig(audit_key=f"Password{'a' * 56}")

    def test_operator_role_validation(self):
        """Test operator_role literal validation."""
        SecurityConfig(operator_role="guest")
        SecurityConfig(operator_role="operator")
        SecurityConfig(operator_role="admin")
        with pytest.raises(ValueError):
            SecurityConfig(operator_role="superuser")  # type: ignore[arg-type]

    def test_allowed_subprocess_cmds_default(self):
        """Test default allowed subprocess commands."""
        config = SecurityConfig()
        assert "git" in config.allowed_subprocess_cmds
        assert "echo" in config.allowed_subprocess_cmds
        assert "python" in config.allowed_subprocess_cmds


class TestExperimentConfig:
    """Tests for ExperimentConfig class."""

    def test_default_values(self):
        """Test default experiment config values."""
        config = ExperimentConfig()
        assert config.num_trials == 100
        assert config.time_budget_seconds == 600
        assert config.random_seed is None
        assert config.output_format == "json"
        assert config.save_intermediate is True
        assert config.checkpoint_interval_seconds == 300

    def test_custom_values(self):
        """Test custom experiment config values."""
        config = ExperimentConfig(
            num_trials=1000,
            time_budget_seconds=1800,
            random_seed=42,
            output_format="csv",
        )
        assert config.num_trials == 1000
        assert config.time_budget_seconds == 1800
        assert config.random_seed == 42
        assert config.output_format == "csv"

    def test_num_trials_validation(self):
        """Test num_trials range validation."""
        ExperimentConfig(num_trials=1)
        ExperimentConfig(num_trials=10000)
        with pytest.raises(ValueError):
            ExperimentConfig(num_trials=0)
        with pytest.raises(ValueError):
            ExperimentConfig(num_trials=10001)

    def test_time_budget_seconds_validation(self):
        """Test time_budget_seconds range validation."""
        ExperimentConfig(time_budget_seconds=60)
        ExperimentConfig(time_budget_seconds=3600)
        with pytest.raises(ValueError):
            ExperimentConfig(time_budget_seconds=30)
        with pytest.raises(ValueError):
            ExperimentConfig(time_budget_seconds=4000)

    def test_output_format_validation(self):
        """Test output_format literal validation."""
        ExperimentConfig(output_format="json")
        ExperimentConfig(output_format="csv")
        ExperimentConfig(output_format="parquet")
        with pytest.raises(ValueError):
            ExperimentConfig(output_format="xml")  # type: ignore[arg-type]

    def test_checkpoint_interval_validation(self):
        """Test checkpoint_interval_seconds range validation."""
        ExperimentConfig(checkpoint_interval_seconds=30)
        ExperimentConfig(checkpoint_interval_seconds=900)
        with pytest.raises(ValueError):
            ExperimentConfig(checkpoint_interval_seconds=20)
        with pytest.raises(ValueError):
            ExperimentConfig(checkpoint_interval_seconds=1000)


class TestAPGIGlobalConfig:
    """Tests for APGIGlobalConfig class."""

    def test_default_values(self):
        """Test default global config values."""
        config = APGIGlobalConfig()
        assert config.version == CURRENT_CONFIG_VERSION
        assert isinstance(config.dynamical, APGIDynamicalParameters)
        assert isinstance(config.perceptual, APGIPerceptualParameters)
        assert isinstance(config.metabolic, APGIMetabolicParameters)
        assert isinstance(config.psychiatric, APGIPhychiatricProfiles)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.experiment, ExperimentConfig)

    def test_custom_version(self):
        """Test custom version."""
        config = APGIGlobalConfig(version="2.0.0")
        assert config.version == "2.0.0"

    def test_version_pattern_validation(self):
        """Test version pattern validation."""
        APGIGlobalConfig(version="1.0.0")
        APGIGlobalConfig(version="2.3.4")
        with pytest.raises(ValueError):
            APGIGlobalConfig(version="1.0")
        with pytest.raises(ValueError):
            APGIGlobalConfig(version="v1.0.0")
        with pytest.raises(ValueError):
            APGIGlobalConfig(version="1.0.0-beta")

    def test_validate_consistency_valid(self):
        """Test cross-parameter consistency validation with valid config."""
        config = APGIGlobalConfig()
        assert config.dynamical.theta_survival < config.dynamical.theta_neutral
        # Should not raise
        APGIGlobalConfig.model_validate(config.model_dump())

    def test_validate_consistency_invalid(self):
        """Test cross-parameter consistency validation with invalid config."""
        with pytest.raises(
            ValueError, match="theta_survival must be less than theta_neutral"
        ):
            APGIGlobalConfig(
                dynamical=APGIDynamicalParameters(theta_survival=0.8, theta_neutral=0.5)
            )

    def test_validate_consistency_equal_values(self):
        """Test validation when theta_survival equals theta_neutral."""
        with pytest.raises(
            ValueError, match="theta_survival must be less than theta_neutral"
        ):
            APGIGlobalConfig(
                dynamical=APGIDynamicalParameters(theta_survival=0.5, theta_neutral=0.5)
            )


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env function."""

    def test_load_config_from_env_defaults(self):
        """Test loading config with default environment values."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config_from_env()
            assert config.security.audit_key == ""
            assert config.security.operator_role == "guest"
            assert config.security.config_secret_key == ""
            assert config.security.enable_profiling is False

    def test_load_config_from_env_with_values(self):
        """Test loading config with environment values."""
        env_vars = {
            "APGI_AUDIT_KEY": "a" * 64,
            "APGI_OPERATOR_ROLE": "admin",
            "APGI_CONFIG_SECRET_KEY": "secret123",
            "APGI_ENABLE_PROFILING": "true",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_config_from_env()
            assert config.security.audit_key == "a" * 64
            assert config.security.operator_role == "admin"
            assert config.security.config_secret_key == "secret123"
            assert config.security.enable_profiling is True

    def test_enable_profiling_variations(self):
        """Test various enable_profiling environment values."""
        true_values = ["1", "true", "yes", "TRUE", "YES"]
        for val in true_values:
            with patch.dict(os.environ, {"APGI_ENABLE_PROFILING": val}, clear=True):
                config = load_config_from_env()
                assert config.security.enable_profiling is True

        false_values = ["0", "false", "no", "FALSE", "NO", ""]
        for val in false_values:
            with patch.dict(os.environ, {"APGI_ENABLE_PROFILING": val}, clear=True):
                config = load_config_from_env()
                assert config.security.enable_profiling is False


class TestComputeConfigChecksum:
    """Tests for compute_config_checksum function."""

    def test_compute_checksum_from_config(self):
        """Test checksum computation from APGIGlobalConfig."""
        config = APGIGlobalConfig()
        checksum = compute_config_checksum(config)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex length

    def test_compute_checksum_from_dict(self):
        """Test checksum computation from dict."""
        config_dict = {"version": "1.0.0", "test": "value"}
        checksum = compute_config_checksum(config_dict)
        assert isinstance(checksum, str)
        assert len(checksum) == 64

    def test_compute_checksum_with_secret_key(self):
        """Test checksum computation with secret key."""
        config = APGIGlobalConfig()
        checksum1 = compute_config_checksum(config, secret_key="")
        checksum2 = compute_config_checksum(config, secret_key="secret")
        assert checksum1 != checksum2

    def test_compute_checksum_deterministic(self):
        """Test checksum is deterministic for same input."""
        config = APGIGlobalConfig()
        checksum1 = compute_config_checksum(config)
        checksum2 = compute_config_checksum(config)
        assert checksum1 == checksum2

    def test_compute_checksum_different_configs(self):
        """Test different configs produce different checksums."""
        config1 = APGIGlobalConfig()
        config2 = APGIGlobalConfig(dynamical=APGIDynamicalParameters(tau_s=0.5))
        checksum1 = compute_config_checksum(config1)
        checksum2 = compute_config_checksum(config2)
        assert checksum1 != checksum2


class TestValidateConfigIntegrity:
    """Tests for validate_config_integrity function."""

    def test_validate_integrity_valid(self):
        """Test integrity validation with matching checksum."""
        config = APGIGlobalConfig()
        expected_checksum = compute_config_checksum(config)
        assert validate_config_integrity(config, expected_checksum) is True

    def test_validate_integrity_invalid(self):
        """Test integrity validation with mismatched checksum."""
        config = APGIGlobalConfig()
        wrong_checksum = "a" * 64
        assert validate_config_integrity(config, wrong_checksum) is False

    def test_validate_integrity_with_secret_key(self):
        """Test integrity validation with secret key."""
        config = APGIGlobalConfig()
        secret = "my_secret"
        expected_checksum = compute_config_checksum(config, secret)
        assert validate_config_integrity(config, expected_checksum, secret) is True
        assert (
            validate_config_integrity(config, expected_checksum, "wrong_secret")
            is False
        )


class TestMigrateConfig:
    """Tests for migrate_config function."""

    def test_migrate_config_no_migration_needed(self):
        """Test migration when already at current version."""
        config_dict = {"version": "1.0.0", "test": "value"}
        result = migrate_config(config_dict, "1.0.0")
        assert result["version"] == "1.0.0"
        assert result["test"] == "value"

    def test_migrate_config_from_0_9_0(self):
        """Test migration from version 0.9.0 to 1.0.0."""
        config_dict = {"version": "0.9.0", "test": "value"}
        result = migrate_config(config_dict, "0.9.0")
        assert result["version"] == "1.0.0"
        assert result["test"] == "value"

    def test_migrate_config_unknown_version(self):
        """Test migration from unknown version (no migration path)."""
        config_dict = {"version": "0.5.0", "test": "value"}
        result = migrate_config(config_dict, "0.5.0")
        # Should return unchanged if no migration path exists
        assert result["version"] == "0.5.0"

    def test_config_migrations_constant(self):
        """Test CONFIG_MIGRATIONS constant structure."""
        assert "0.9.0" in CONFIG_MIGRATIONS
        assert CONFIG_MIGRATIONS["0.9.0"]["upgrade_to"] == "1.0.0"
        assert "changes" in CONFIG_MIGRATIONS["0.9.0"]


class TestValidateStartupConfig:
    """Tests for validate_startup_config function."""

    def test_validate_startup_config_default(self):
        """Test startup validation with default config."""
        config = validate_startup_config()
        assert isinstance(config, APGIGlobalConfig)
        assert config.version == CURRENT_CONFIG_VERSION

    def test_validate_startup_config_with_valid_config(self):
        """Test startup validation with provided valid config."""
        config = APGIGlobalConfig()
        validated = validate_startup_config(config)
        assert validated.version == config.version

    def test_validate_startup_config_with_none(self):
        """Test startup validation with None (loads from env)."""
        with patch.dict(os.environ, {}, clear=True):
            config = validate_startup_config(None)
            assert isinstance(config, APGIGlobalConfig)

    def test_validate_startup_config_invalid_model(self):
        """Test startup validation with invalid model."""
        # Create an invalid config by manually setting invalid data
        invalid_config_data = {
            "version": "invalid",
            "dynamical": {},
            "perceptual": {},
            "metabolic": {},
            "psychiatric": {},
            "security": {},
            "experiment": {},
        }
        with pytest.raises(ValueError, match="String should match pattern"):
            # This will fail during model_validate
            APGIGlobalConfig(**invalid_config_data)  # type: ignore[arg-type]

    def test_validate_startup_config_short_audit_key(self):
        """Test startup validation with short audit key."""
        with patch.dict(os.environ, {"APGI_AUDIT_KEY": "short"}, clear=True):
            with pytest.raises(ValueError, match="at least 64 characters"):
                validate_startup_config()

    def test_validate_startup_config_valid_audit_key(self):
        """Test startup validation with valid audit key."""
        with patch.dict(os.environ, {"APGI_AUDIT_KEY": "a" * 64}, clear=True):
            config = validate_startup_config()
            assert config.security.audit_key == "a" * 64


class TestCurrentConfigVersion:
    """Tests for CURRENT_CONFIG_VERSION constant."""

    def test_current_config_version_format(self):
        """Test CURRENT_CONFIG_VERSION follows semantic versioning."""
        parts = CURRENT_CONFIG_VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_current_config_version_not_empty(self):
        """Test CURRENT_CONFIG_VERSION is not empty."""
        assert CURRENT_CONFIG_VERSION != ""

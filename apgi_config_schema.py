"""
Typed configuration schema with validation for APGI system.

Provides:
- Pydantic-based config models with versioned migrations
- Startup-time validation
- Environment variable integration
- Checksum verification
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Version Management
# ---------------------------------------------------------------------------

CURRENT_CONFIG_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Core Configuration Models
# ---------------------------------------------------------------------------


class APGIDynamicalParameters(BaseModel):
    """Core dynamical system parameters."""

    tau_s: float = Field(
        default=0.35, ge=0.1, le=2.0, description="Surprise decay timescale (s)"
    )
    tau_theta: float = Field(
        default=30.0, ge=1.0, le=120.0, description="Threshold adaptation (s)"
    )
    tau_m: float = Field(
        default=1.5, ge=0.5, le=5.0, description="Somatic marker timescale (s)"
    )
    theta_0: float = Field(
        default=0.5, ge=0.1, le=2.0, description="Baseline ignition threshold"
    )
    alpha: float = Field(default=5.5, ge=1.0, le=10.0, description="Sigmoid steepness")
    beta_som: float = Field(
        default=1.5, ge=0.1, le=5.0, description="Somatic influence coefficient"
    )
    beta_m: float = Field(default=1.0, ge=0.1, le=3.0, description="Marker sensitivity")
    m_0: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Reference somatic level"
    )
    gamma_m: float = Field(
        default=-0.3, ge=-1.0, le=1.0, description="Metabolic sensitivity"
    )
    gamma_a: float = Field(
        default=0.1, ge=-0.5, le=0.5, description="Arousal sensitivity"
    )
    lambda_s: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Metabolic coupling strength"
    )
    sigma_s: float = Field(
        default=0.05, ge=0.0, le=0.5, description="Surprise noise strength"
    )
    sigma_theta: float = Field(
        default=0.02, ge=0.0, le=0.5, description="Threshold noise strength"
    )
    sigma_m: float = Field(
        default=0.03, ge=0.0, le=0.5, description="Somatic marker noise strength"
    )
    theta_survival: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Survival-relevant threshold"
    )
    theta_neutral: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Neutral content threshold"
    )
    rho: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Reset fraction after ignition"
    )


class APGIPerceptualParameters(BaseModel):
    """Perceptual processing parameters."""

    pi_i_expected: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Expected precision of input"
    )
    pi_i_survival: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Survival-relevant precision"
    )
    attentional_gain: float = Field(
        default=2.0, ge=0.5, le=5.0, description="Attention modulation factor"
    )
    visual_acuity: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Visual processing fidelity"
    )


class APGIMetabolicParameters(BaseModel):
    """Metabolic cost and energy budget parameters."""

    baseline_metabolic_rate: float = Field(default=1.0, ge=0.1, le=10.0)
    effort_cost_coefficient: float = Field(default=0.1, ge=0.0, le=1.0)
    glucose_impact_factor: float = Field(default=0.5, ge=0.0, le=2.0)
    fatigue_recovery_rate: float = Field(default=0.05, ge=0.0, le=0.5)


class APGIPhychiatricProfiles(BaseModel):
    """Psychiatric condition simulation profiles."""

    gad_profile: Dict[str, float] = Field(
        default_factory=dict, description="Generalized anxiety disorder"
    )
    mdd_profile: Dict[str, float] = Field(
        default_factory=dict, description="Major depressive disorder"
    )
    psychosis_profile: Dict[str, float] = Field(
        default_factory=dict, description="Psychosis spectrum"
    )


class SecurityConfig(BaseModel):
    """Security and audit configuration."""

    audit_key: str = Field(default="", description="APGI_AUDIT_KEY value")
    operator_role: Literal["guest", "operator", "admin"] = Field(default="guest")
    config_secret_key: str = Field(default="", description="Config validation key")
    enable_profiling: bool = Field(default=False)
    allowed_subprocess_cmds: List[str] = Field(
        default_factory=lambda: ["git", "echo", "python"]
    )
    pickle_allowlist: List[str] = Field(default_factory=list)

    @field_validator("audit_key")
    @classmethod
    def validate_audit_key(cls, v: str) -> str:
        if v and len(v) < 64:
            raise ValueError("APGI_AUDIT_KEY must be at least 64 characters")
        weak_patterns = ["test", "password", "default", "admin", "123456"]
        if v and any(p in v.lower() for p in weak_patterns):
            raise ValueError("APGI_AUDIT_KEY contains weak pattern")
        return v


class ExperimentConfig(BaseModel):
    """Experiment execution configuration."""

    num_trials: int = Field(default=100, ge=1, le=10000)
    time_budget_seconds: int = Field(default=600, ge=60, le=3600)
    random_seed: Optional[int] = Field(default=None)
    output_format: Literal["json", "csv", "parquet"] = Field(default="json")
    save_intermediate: bool = Field(default=True)
    checkpoint_interval_seconds: int = Field(default=300, ge=30, le=900)


class APGIGlobalConfig(BaseModel):
    """Root configuration model for APGI system."""

    version: str = Field(default=CURRENT_CONFIG_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    dynamical: APGIDynamicalParameters = Field(default_factory=APGIDynamicalParameters)
    perceptual: APGIPerceptualParameters = Field(
        default_factory=APGIPerceptualParameters
    )
    metabolic: APGIMetabolicParameters = Field(default_factory=APGIMetabolicParameters)
    psychiatric: APGIPhychiatricProfiles = Field(
        default_factory=APGIPhychiatricProfiles
    )
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @model_validator(mode="after")
    def validate_consistency(self) -> APGIGlobalConfig:
        """Validate cross-parameter consistency."""
        # Ensure survival threshold is lower than neutral (as per theory)
        if self.dynamical.theta_survival >= self.dynamical.theta_neutral:
            raise ValueError("theta_survival must be less than theta_neutral")
        return self


# ---------------------------------------------------------------------------
# Config Loading and Validation
# ---------------------------------------------------------------------------


def load_config_from_env() -> APGIGlobalConfig:
    """Load configuration from environment variables."""
    return APGIGlobalConfig(
        security=SecurityConfig(
            audit_key=os.environ.get("APGI_AUDIT_KEY", ""),
            operator_role=os.environ.get("APGI_OPERATOR_ROLE", "guest"),  # type: ignore
            config_secret_key=os.environ.get("APGI_CONFIG_SECRET_KEY", ""),
            enable_profiling=os.environ.get("APGI_ENABLE_PROFILING", "").lower()
            in ("1", "true", "yes"),
        )
    )


def compute_config_checksum(
    config: Union[APGIGlobalConfig, Dict[str, Any]], secret_key: str = ""
) -> str:
    """Compute SHA-256 checksum of configuration for integrity verification."""
    if isinstance(config, APGIGlobalConfig):
        data = config.model_dump_json()
    else:
        data = json.dumps(config, sort_keys=True)
    return hashlib.sha256((data + secret_key).encode()).hexdigest()


def validate_config_integrity(
    config: APGIGlobalConfig, expected_checksum: str, secret_key: str = ""
) -> bool:
    """Validate configuration against expected checksum."""
    computed = compute_config_checksum(config, secret_key)
    return computed == expected_checksum


# ---------------------------------------------------------------------------
# Migration Support
# ---------------------------------------------------------------------------

CONFIG_MIGRATIONS: Dict[str, Any] = {
    "0.9.0": {
        "upgrade_to": "1.0.0",
        "changes": [
            "Added pi_i_survival to perceptual parameters",
            "Renamed beta to beta_som in dynamical parameters",
            "Added checkpoint_interval_seconds to experiment config",
        ],
    }
}


def migrate_config(config_dict: Dict[str, Any], from_version: str) -> Dict[str, Any]:
    """Migrate configuration from older version to current."""
    current = from_version
    while current in CONFIG_MIGRATIONS:
        migration = CONFIG_MIGRATIONS[current]
        # Apply migration transformations here
        config_dict["version"] = migration["upgrade_to"]
        current = migration["upgrade_to"]
    return config_dict


# ---------------------------------------------------------------------------
# Startup Validation
# ---------------------------------------------------------------------------


def validate_startup_config(
    config: Optional[APGIGlobalConfig] = None,
) -> APGIGlobalConfig:
    """Validate configuration at application startup.

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If required environment variables are missing
    """
    if config is None:
        config = load_config_from_env()

    # Validate the model
    try:
        config.model_validate(config.model_dump())
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e

    # Security checks
    if config.security.audit_key and len(config.security.audit_key) < 64:
        raise RuntimeError("APGI_AUDIT_KEY must be at least 64 characters")

    return config


if __name__ == "__main__":
    # Example usage and validation
    config = validate_startup_config()
    print(f"Config version: {config.version}")
    print(f"Dynamical params: {config.dynamical.model_dump()}")
    print(f"Checksum: {compute_config_checksum(config)}")

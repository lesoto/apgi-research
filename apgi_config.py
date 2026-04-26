"""
APGI Config schemas with Pydantic validation and explicit versions.

Provides centralized runtime configuration with:
- Typed config schemas via Pydantic
- Singleton ConfigManager for unified access
- Environment-aware loading
- Caching for repeated config access
"""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

try:
    from pydantic import BaseModel, Field, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Fallback for environments without pydantic
    class BaseModel:  # type: ignore
        def __init__(self, **data: Any):
            for k, v in data.items():
                setattr(self, k, v)

    class Field:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ValidationError(Exception):  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Pydantic Config Schemas
# ---------------------------------------------------------------------------


class APGIExperimentConfigSchema(BaseModel):
    """Typed APGI experiment configuration with validation."""

    version: str = "1.0.0"
    experiment_name: str = Field(default="unknown_experiment")
    tau_S: float = Field(default=0.35, ge=0.2, le=0.5)
    beta: float = Field(default=1.5, ge=0.5, le=2.5)
    theta_0: float = Field(default=0.5, ge=0.1, le=1.0)
    alpha: float = Field(default=5.5, ge=3.0, le=8.0)

    # Feature flags
    hierarchical_enabled: bool = Field(default=True)
    precision_gap_enabled: bool = Field(default=True)
    neuromodulator_tracking: bool = Field(default=True)

    @classmethod
    def from_legacy(cls, legacy_config: Dict[str, Any]) -> "APGIExperimentConfigSchema":
        """Backward-compatible adapter for older configs."""
        return cls(
            experiment_name=legacy_config.get("name", "unknown_experiment"),
            tau_S=legacy_config.get("tau_S", 0.35),
            beta=legacy_config.get("beta", 1.5),
            theta_0=legacy_config.get("theta_0", 0.5),
            alpha=legacy_config.get("alpha", 5.5),
            hierarchical_enabled=legacy_config.get("hierarchical_enabled", True),
            precision_gap_enabled=legacy_config.get("precision_gap_enabled", True),
            neuromodulator_tracking=legacy_config.get("neuromodulator_tracking", True),
        )


class APGISecurityConfigSchema(BaseModel):
    """Security-related configuration schema."""

    audit_enabled: bool = Field(default=True)
    authz_enabled: bool = Field(default=True)
    subprocess_allowlist: list = Field(
        default_factory=lambda: ["git", "pytest", "python"]
    )
    require_secure_pickle: bool = Field(default=True)


class APGIMetricsConfigSchema(BaseModel):
    """Metrics and monitoring configuration."""

    profiling_enabled: bool = Field(default=False)
    performance_budget_ms: float = Field(default=600000)  # 10 minutes
    log_level: str = Field(default="INFO")
    json_logging: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Config Manager Singleton
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class ConfigSources:
    """Tracks the sources of configuration values for debugging."""

    sources: Dict[str, str]


class ConfigManager:
    """
    Centralized runtime configuration manager (Singleton pattern).

    Provides unified access to all APGI configuration with:
    - Environment variable overrides
    - Config file loading (JSON)
    - Caching for repeated access
    - Typed schemas for validation
    - Source tracking for audit/debug

    Usage:
        from apgi_config import get_config

        config = get_config()
        experiment_config = config.get_experiment_config("stroop_effect")
    """

    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._config_cache: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}
        self._environment_prefix = "APGI_"

        # Load base configuration
        self._load_base_config()
        self._initialized = True

    def _load_base_config(self) -> None:
        """Load configuration from file and environment."""
        # Try to load from config file
        config_file = self._find_config_file()
        if config_file:
            self._load_from_file(config_file)

        # Environment overrides take precedence
        self._load_from_environment()

    def _find_config_file(self) -> Optional[Path]:
        """Find config file in standard locations."""
        search_paths = [
            Path("config/apgi.json"),
            Path("config/apgi.yaml"),
            Path("apgi.json"),
            Path("apgi.yaml"),
            Path.home() / ".apgi" / "config.json",
        ]

        # Also check APGI_CONFIG_FILE env var
        env_config = os.environ.get("APGI_CONFIG_FILE")
        if env_config:
            search_paths.insert(0, Path(env_config))

        for path in search_paths:
            if path.exists():
                return path
        return None

    def _load_from_file(self, path: Path) -> None:
        """Load configuration from JSON/YAML file."""
        try:
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml

                        data = yaml.safe_load(f)
                    except ImportError:
                        # Fall back to JSON if yaml not available
                        data = json.load(f)
                else:
                    data = json.load(f)

            self._config_cache.update(data)
            for key in data.keys():
                self._sources[key] = str(path)
        except Exception as e:
            # Log but don't fail - env vars can provide config
            import logging

            logging.getLogger("apgi.config").warning(
                f"Failed to load config from {path}: {e}"
            )

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self._environment_prefix):
                # Convert APGI_EXPERIMENT_NAME to experiment_name
                config_key = key[len(self._environment_prefix) :].lower()

                # Try to parse as JSON first, then fall back to string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value

                self._config_cache[config_key] = parsed_value
                self._sources[config_key] = f"env:{key}"

    def get(
        self, key: str, default: Optional[T] = None, type_hint: Optional[Type[T]] = None
    ) -> Optional[T]:
        """Get configuration value with optional type conversion."""
        value = self._config_cache.get(key, default)

        if type_hint and value is not None:
            try:
                if type_hint == bool and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                elif callable(type_hint):
                    value = type_hint(value)  # type: ignore[call-arg]
            except (ValueError, TypeError):
                value = default

        return value

    def get_experiment_config(self, experiment_name: str) -> APGIExperimentConfigSchema:
        """Get typed experiment configuration."""
        # Build config from cached values with experiment prefix
        prefix = f"experiment_{experiment_name}_"

        config_dict = {
            "experiment_name": experiment_name,
            "tau_S": self.get(f"{prefix}tau_s", self.get("tau_s", 0.35)),
            "beta": self.get(f"{prefix}beta", self.get("beta", 1.5)),
            "theta_0": self.get(f"{prefix}theta_0", self.get("theta_0", 0.5)),
            "alpha": self.get(f"{prefix}alpha", self.get("alpha", 5.5)),
            "hierarchical_enabled": self.get(
                f"{prefix}hierarchical_enabled", self.get("hierarchical_enabled", True)
            ),
            "precision_gap_enabled": self.get(
                f"{prefix}precision_gap_enabled",
                self.get("precision_gap_enabled", True),
            ),
        }

        if PYDANTIC_AVAILABLE:
            try:
                return APGIExperimentConfigSchema(**config_dict)  # type: ignore[arg-type]
            except ValidationError as e:
                # Return defaults on validation error, but log
                import logging

                logging.getLogger("apgi.config").error(f"Config validation error: {e}")
                return APGIExperimentConfigSchema(experiment_name=experiment_name)
        else:
            return APGIExperimentConfigSchema(**config_dict)  # type: ignore[arg-type]

    def get_security_config(self) -> APGISecurityConfigSchema:
        """Get security configuration."""
        config_dict = {
            "audit_enabled": self.get("security_audit_enabled", True, bool),
            "authz_enabled": self.get("security_authz_enabled", True, bool),
            "subprocess_allowlist": self.get(
                "security_subprocess_allowlist",
                ["git", "pytest", "python", "screencapture"],
            ),
            "require_secure_pickle": self.get(
                "security_require_secure_pickle", True, bool
            ),
        }

        if PYDANTIC_AVAILABLE:
            return APGISecurityConfigSchema(**config_dict)  # type: ignore[arg-type]
        return APGISecurityConfigSchema(**config_dict)  # type: ignore[arg-type]

    def get_metrics_config(self) -> APGIMetricsConfigSchema:
        """Get metrics configuration."""
        config_dict = {
            "profiling_enabled": self.get("metrics_profiling_enabled", False, bool),
            "performance_budget_ms": self.get(
                "metrics_performance_budget_ms", 600000.0, float
            ),
            "log_level": self.get("metrics_log_level", "INFO"),
            "json_logging": self.get("metrics_json_logging", True, bool),
        }

        if PYDANTIC_AVAILABLE:
            return APGIMetricsConfigSchema(**config_dict)  # type: ignore[arg-type]
        return APGIMetricsConfigSchema(**config_dict)  # type: ignore[arg-type]

    def get_source(self, key: str) -> Optional[str]:
        """Get the source of a configuration value."""
        return self._sources.get(key)

    def get_all_sources(self) -> Dict[str, str]:
        """Get all configuration sources for debugging."""
        return self._sources.copy()

    def reload(self) -> None:
        """Reload configuration from all sources."""
        self._config_cache.clear()
        self._sources.clear()
        self._load_base_config()

    def set(self, key: str, value: Any, source: str = "runtime") -> None:
        """Set configuration value at runtime."""
        self._config_cache[key] = value
        self._sources[key] = source


# ---------------------------------------------------------------------------
# Module-Level Functions (Convenience API)
# ---------------------------------------------------------------------------

_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config() -> None:
    """Reset the global ConfigManager (useful for testing)."""
    global _config_manager
    ConfigManager._instance = None
    ConfigManager._initialized = False
    _config_manager = None


@lru_cache(maxsize=128)
def get_cached_experiment_config(experiment_name: str) -> APGIExperimentConfigSchema:
    """
    Get cached experiment configuration.

    Uses LRU cache to avoid repeated config lookups/validation.
    Cache is keyed by experiment name.
    """
    return get_config().get_experiment_config(experiment_name)


def invalidate_config_cache() -> None:
    """Invalidate the cached experiment config (call after config changes)."""
    get_cached_experiment_config.cache_clear()


# ---------------------------------------------------------------------------
# Legacy Compatibility
# ---------------------------------------------------------------------------


def load_apgi_params() -> Dict[str, Any]:
    """
    Legacy-compatible APGI params loader.

    Returns dict format expected by older code while using new config system.
    """
    config = get_config()
    return {
        "tau_s": config.get("tau_s", 0.35),
        "beta": config.get("beta", 1.5),
        "theta_0": config.get("theta_0", 0.5),
        "alpha": config.get("alpha", 5.5),
        "gamma_M": config.get("gamma_m", 0.1),
        "lambda_S": config.get("lambda_s", 0.05),
        "sigma_S": config.get("sigma_s", 0.1),
        "sigma_theta": config.get("sigma_theta", 0.05),
        "sigma_M": config.get("sigma_m", 0.05),
        "rho": config.get("rho", 0.8),
        "theta_survival": config.get("theta_survival", 0.3),
        "theta_neutral": config.get("theta_neutral", 0.5),
        "beta_cross": config.get("beta_cross", 0.2),
        "tau_levels": config.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
        "enabled": config.get("apgi_enabled", True, bool),
        "hierarchical_enabled": config.get("hierarchical_enabled", True, bool),
        "precision_gap_enabled": config.get("precision_gap_enabled", True, bool),
        "ACh": config.get("ach_level", 1.0),
        "NE": config.get("ne_level", 1.0),
        "DA": config.get("da_level", 1.0),
        "HT5": config.get("ht5_level", 1.0),
    }

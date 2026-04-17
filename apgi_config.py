"""
APGI Config schemas with Pydantic validation and explicit versions.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class APGIExperimentConfigSchema(BaseModel):
    version: str = "1.0.0"
    experiment_name: str
    tau_S: float = Field(default=0.35, ge=0.2, le=0.5)
    beta: float = Field(default=1.5, ge=0.5, le=2.5)
    theta_0: float = Field(default=0.5, ge=0.1, le=1.0)
    alpha: float = Field(default=5.5, ge=3.0, le=8.0)

    @classmethod
    def from_legacy(cls, legacy_config: Dict[str, Any]) -> "APGIExperimentConfigSchema":
        """Backward-compatible adapter for older configs."""
        # Convert legacy name param or other deprecated formats safely
        return cls(
            experiment_name=legacy_config.get("name", "unknown_experiment"),
            tau_S=legacy_config.get("tau_S", 0.35),
            beta=legacy_config.get("beta", 1.5),
            theta_0=legacy_config.get("theta_0", 0.5),
            alpha=legacy_config.get("alpha", 5.5),
        )

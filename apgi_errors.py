"""
Standardized error taxonomy for the APGI system.
"""

from typing import Any, Dict, Optional


class APGIError(Exception):
    """Base class for all APGI-related errors."""

    def __init__(
        self,
        message: Optional[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class APGIConfigurationError(APGIError):
    """Raised when there is an error in the APGI configuration or parameters."""

    pass


class APGIRuntimeError(APGIError):
    """Raised when there is a runtime failure during APGI simulation/processing."""

    pass


class APGIDataValidationError(APGIError):
    """Raised when data input to APGI components fails validation."""

    pass


class APGIIntegrationError(APGIError):
    """Raised when there is an error integrating APGI with external systems."""

    pass


class APGITimeoutError(APGIError):
    """Raised when an APGI operation times out."""

    pass

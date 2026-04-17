"""
Standardized error taxonomy for the APGI system.
"""


class APGIError(Exception):
    """Base class for all APGI-related errors."""

    pass


class APGIConfigurationError(APGIError):
    """Raised when there is an error in the APGI configuration or parameters."""

    pass


class APGIRuntimeError(APGIError):
    """Raised when there is a runtime failure during APGI simulation/processing."""

    pass


class APGIDataValidationError(APGIError):
    """Raised when data input to APGI components fails validation."""

    pass

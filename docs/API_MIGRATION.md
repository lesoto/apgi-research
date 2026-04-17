# APGI API Migration Guide

## Version 1.0.0

### Breaking Changes

- `execute()` in base runners is officially deprecated. Please migrate to using `run_experiment()` directly, conforming to the `ExperimentRunnerProtocol`.
- All APGI-specific errors now inherit from `APGIError`. Catch `APGIConfigurationError`, `APGIRuntimeError`, or `APGIDataValidationError` based on failure contexts.

### Protocol Interfaces

We introduced formal `Protocol` classes in `apgi_protocols.py` to support structural subtyping and improve the type-safety of standard runners. This ensures that any object passed as an `ExperimentRunner` implements `run_experiment()`.

### Logging Upgrades

- Logs are now JSON structured for production grading. Use `apgi_logging.get_logger()` with context using `APGIContextLogger` for correlation and trial traceability.

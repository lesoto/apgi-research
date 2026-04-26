"""
Stabilized Public API configurations.
Defines runner interfaces using Protocol and ABC classes.
"""

import warnings
from typing import Any, Dict, Protocol, runtime_checkable


def deprecated(reason: str) -> Any:
    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@runtime_checkable
class ExperimentRunnerProtocol(Protocol):
    """Protocol for experiment runners"""

    def run_experiment(self) -> Dict[str, Any]: ...  # noqa: E704


@runtime_checkable
class APGIModelProtocol(Protocol):
    """Protocol for APGI model implementations"""

    def process_trial(
        self, observed: float, predicted: float
    ) -> Dict[str, float]:  # noqa: E704
        ...

    def reset(self) -> None: ...  # noqa: E704


class BaseAPGIRunner:
    """Compatibility shim for old runner interfaces."""

    @deprecated("Use standard run_experiment() instead")
    def execute(self) -> Dict[str, Any]:
        """Old execution method. Mapped to run_experiment."""
        return self.run_experiment()

    def run_experiment(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement run_experiment")

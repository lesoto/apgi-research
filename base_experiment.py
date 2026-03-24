from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from apgi_integration import APGIExperiment


class BaseExperiment(ABC):
    """
    Abstract base class for all APGI experiments.
    Enforces a standard structure for initialization, running trials,
    and calculating metrics across the 30 experiment variants.
    """

    def __init__(self, enable_apgi: bool = True):
        self.apgi: Optional[APGIExperiment] = self._init_apgi() if enable_apgi else None
        self.setup_experiment()

    def _init_apgi(self) -> APGIExperiment:
        """
        Initializes the underlying APGI dynamical system instance.
        """
        return APGIExperiment(
            experiment_name=self.__class__.__name__,
            description=f"Automated execution of {self.__class__.__name__}",
        )

    @abstractmethod
    def setup_experiment(self) -> None:
        """
        Prepare any experiment-specific resources, stimulus arrays,
        or pre-computations needed before trials start.
        """
        pass

    @abstractmethod
    def run_trial(self, trial_index: int) -> Dict[str, Any]:
        """
        Run a single iteration of the experiment.
        Must return a dictionary containing raw trial metrics (like accuracy, rt, etc).
        """
        pass

    @abstractmethod
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Aggregate trial results into the final required primary metrics.
        """
        pass

import pytest
from apgi_protocols import ExperimentRunnerProtocol, BaseAPGIRunner
from typing import Dict, Any


class ValidRunner:
    def run_experiment(self) -> Dict[str, Any]:
        return {"success": True}


class InvalidRunner:
    def execute(self):
        pass


def test_runner_protocol_compliance():
    # Valid Runner satisfies protocol
    assert isinstance(ValidRunner(), ExperimentRunnerProtocol)

    # Invalid Runner does not
    assert not isinstance(InvalidRunner(), ExperimentRunnerProtocol)


def test_base_apgi_runner_compatibility():
    class DummyRunner(BaseAPGIRunner):
        def run_experiment(self) -> Dict[str, Any]:
            return {"result": 42}

    runner = DummyRunner()
    with pytest.warns(DeprecationWarning, match="execute is deprecated"):
        res = runner.execute()
        assert res["result"] == 42

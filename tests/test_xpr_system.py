import unittest

from autonomous_agent import ExecutionReport, ExperimentPlan
from memory_store import MemoryStore
from xpr_agent_engine import XPRAgentEngine


class TestXPRSystem(unittest.TestCase):
    def test_memory_store_initialization(self):
        store = MemoryStore()
        self.assertIsInstance(store.memory, list)

    def test_agent_engine_initialization(self):
        engine = XPRAgentEngine()
        # Register a test skill to verify skills dictionary works
        engine.register_skill("test_skill", lambda: "test_result")
        self.assertTrue(len(engine.skills) > 0)
        self.assertTrue("test_skill" in engine.skills)

    def test_experiment_plan_dataclass(self):
        plan = ExperimentPlan(
            hypothesis="Test hypothesis",
            success_metrics={"accuracy": "> 90%"},
            constraints=["No file structure changes"],
            steps=["Step 1", "Step 2"],
        )
        self.assertEqual(plan.hypothesis, "Test hypothesis")

    def test_execution_report_dataclass(self):
        report = ExecutionReport(
            experiment_name="test_exp",
            summary="Test summary",
            metric_deltas={"test_metric": 0.1},
            root_causes=[],
            suggested_fixes=[],
            confidence_score=0.9,
        )
        self.assertEqual(report.experiment_name, "test_exp")
        self.assertEqual(report.confidence_score, 0.9)


if __name__ == "__main__":
    unittest.main()

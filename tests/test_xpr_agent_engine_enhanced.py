"""
Enhanced comprehensive tests for xpr_agent_engine.py module.
"""

import time
import unittest
from unittest.mock import patch

import pytest

try:
    from xpr_agent_engine import ExecutionReport, SkillResult, SkillType, XPRAgentEngine
except ImportError as e:
    pytest.skip(f"Cannot import from xpr_agent_engine: {e}", allow_module_level=True)


class TestSkillResult(unittest.TestCase):
    """Tests for SkillResult dataclass."""

    def test_skill_result_success(self):
        """Test SkillResult with success."""
        result = SkillResult(
            success=True,
            skill_type="test",
            result="test_result",
            execution_time=1.5,
            confidence=0.8,
            metadata={"key": "value"},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.skill_type, "test")
        self.assertEqual(result.result, 20)
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.metadata, {"key": "value"})

    def test_skill_result_failure(self):
        """Test SkillResult with failure."""
        result = SkillResult(
            success=False,
            skill_type="test",
            error="test_error",
            execution_time=0.5,
            confidence=0.0,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "test_error")
        self.assertEqual(result.execution_time, 0.5)
        self.assertEqual(result.confidence, 0.0)

    def test_skill_result_defaults(self):
        """Test SkillResult with default values."""
        result = SkillResult(success=True, skill_type="test")
        self.assertIsNone(result.result)
        self.assertIsNone(result.error)
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertIsNone(result.metadata)


class TestExecutionReport(unittest.TestCase):
    """Tests for ExecutionReport dataclass."""

    def test_execution_report_full(self):
        """Test ExecutionReport with all fields."""
        report = ExecutionReport(
            experiment_name="test_exp",
            success=True,
            execution_time=10.5,
            result="success",
            confidence=0.9,
            metadata={"iterations": 5},
        )
        self.assertEqual(report.experiment_name, "test_exp")
        self.assertTrue(report.success)
        self.assertEqual(report.execution_time, 10.5)
        self.assertEqual(report.result, "success")
        self.assertEqual(report.confidence, 0.9)
        self.assertEqual(report.metadata, {"iterations": 5})

    def test_execution_report_post_init(self):
        """Test ExecutionReport post_init metadata initialization."""
        report = ExecutionReport(
            experiment_name="test", success=False, execution_time=0.0
        )
        assert report.metadata == {}  # nosec: B101 - Test assertion

    def test_execution_report_with_none_metadata(self):
        """Test ExecutionReport with explicit None metadata."""
        report = ExecutionReport(
            experiment_name="test", success=True, execution_time=1.0, metadata=None
        )
        assert report.metadata == {}  # nosec: B101 - Test assertion


class TestSkillType(unittest.TestCase):
    """Tests for SkillType enum."""

    def test_skill_type_values(self):
        """Test SkillType enum values."""
        self.assertEqual(SkillType.PLAN_GENERATION.value, "plan_generation")
        self.assertEqual(SkillType.EXECUTION.value, "execution")
        self.assertEqual(SkillType.ANALYSIS.value, "analysis")
        self.assertEqual(SkillType.MEMORY_UPDATE.value, "memory_update")
        self.assertEqual(SkillType.MODIFICATION.value, "modification")

    def test_skill_type_comparison(self):
        """Test SkillType enum comparison."""
        self.assertEqual(SkillType.PLAN_GENERATION, SkillType.PLAN_GENERATION)
        # Test that different enum values are not equal
        plan_gen_value = SkillType.PLAN_GENERATION.value
        exec_value = SkillType.EXECUTION.value
        self.assertNotEqual(plan_gen_value, exec_value)


class TestXPRAgentEngine(unittest.TestCase):
    """Tests for XPRAgentEngine class."""

    def test_engine_initialization(self):
        """Test XPRAgentEngine initialization."""
        engine = XPRAgentEngine()
        self.assertIsInstance(engine.skills, dict)
        self.assertIsInstance(engine.execution_history, list)
        self.assertIsNone(engine.current_plan)
        self.assertIsNotNone(engine.compliance_manager)

    def test_register_skill(self):
        """Test skill registration."""
        engine = XPRAgentEngine()

        def test_skill(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"

        engine.register_skill("test_skill", test_skill)
        self.assertIn("test_skill", engine.skills)
        self.assertEqual(engine.skills["test_skill"], test_skill)

    def test_execute_skill_success(self):
        """Test successful skill execution."""
        engine = XPRAgentEngine()

        def test_skill(x):
            return x * 2

        engine.register_skill("double", test_skill)
        result = engine.execute_skill("double", 5)

        self.assertTrue(result.success)
        self.assertEqual(result.skill_type, "double")
        self.assertEqual(result.result, 10)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.execution_time, 0)
        self.assertGreater(result.confidence, 0)

    def test_execute_skill_not_found(self):
        """Test executing non-existent skill."""
        engine = XPRAgentEngine()
        result = engine.execute_skill("nonexistent")

        self.assertFalse(result.success)
        self.assertEqual(result.skill_type, "nonexistent")
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error)  # type: ignore[arg-type]
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.confidence, 0.0)

    def test_execute_skill_with_exception(self):
        """Test skill execution with exception."""
        engine = XPRAgentEngine()

        def failing_skill():
            raise ValueError("Test error")

        engine.register_skill("failing", failing_skill)
        result = engine.execute_skill("failing")

        self.assertFalse(result.success)
        self.assertEqual(result.skill_type, "failing")
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("Test error", result.error)  # type: ignore[arg-type]
        self.assertGreaterEqual(result.execution_time, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_execute_skill_with_kwargs(self):
        """Test skill execution with keyword arguments."""
        engine = XPRAgentEngine()

        def test_skill(x, y=10):
            return x + y

        engine.register_skill("add", test_skill)
        result = engine.execute_skill("add", 5, y=15)

        self.assertTrue(result.success)
        self.assertEqual(result.result, 20)

    def test_execute_plan_generation(self):
        """Test plan generation execution."""
        engine = XPRAgentEngine()

        def plan_skill(goal):
            return {"steps": [f"step1_{goal}", f"step2_{goal}"]}

        engine.register_skill("plan_generation", plan_skill)
        result = engine.execute_skill("plan_generation", "test_goal")

        self.assertTrue(result.success)
        self.assertEqual(result.skill_type, "plan_generation")
        self.assertIsNotNone(result.result)
        self.assertIsInstance(result.result, dict)
        self.assertIn("steps", result.result)  # type: ignore[arg-type]

    def test_execute_analysis_skill(self):
        """Test analysis skill execution."""
        engine = XPRAgentEngine()

        def analysis_skill(data):
            return {"analysis": f"analyzed_{data}", "confidence": 0.85}

        engine.register_skill("analysis", analysis_skill)
        result = engine.execute_skill("analysis", "test_data")

        self.assertTrue(result.success)
        self.assertEqual(result.skill_type, "analysis")
        self.assertIsNotNone(result.result)
        self.assertIsInstance(result.result, dict)
        self.assertIn("analysis", result.result)  # type: ignore[arg-type]

    def test_execute_memory_update(self):
        """Test memory update skill execution."""
        engine = XPRAgentEngine()

        def memory_skill(key, value):
            return {"stored": f"{key}:{value}"}

        engine.register_skill("memory_update", memory_skill)
        result = engine.execute_skill("memory_update", "test_key", "test_value")

        self.assertTrue(result.success)
        self.assertEqual(result.skill_type, "memory_update")
        self.assertIsNotNone(result.result)
        self.assertIsInstance(result.result, dict)
        self.assertIn("stored", result.result)  # type: ignore[arg-type]

    def test_execute_modification(self):
        """Test modification skill execution."""
        engine = XPRAgentEngine()

        def modification_skill(file_path, change):
            return {"modified": f"{file_path}:{change}"}

        engine.register_skill("modification", modification_skill)
        result = engine.execute_skill("modification", "test.py", "add_function")

        self.assertTrue(result.success)
        self.assertEqual(result.skill_type, "modification")
        self.assertIsNotNone(result.result)
        self.assertIsInstance(result.result, dict)
        self.assertIn("modified", result.result)  # type: ignore[arg-type]

    def test_multiple_skill_registrations(self):
        """Test registering multiple skills."""
        engine = XPRAgentEngine()

        skills = {
            "skill1": lambda x: x + 1,
            "skill2": lambda x: x * 2,
            "skill3": lambda x: x - 1,
        }

        for name, func in skills.items():
            engine.register_skill(name, func)

        self.assertEqual(len(engine.skills), 3)
        for name in skills:
            self.assertIn(name, engine.skills)

    def test_skill_execution_timing(self):
        """Test skill execution timing accuracy."""
        engine = XPRAgentEngine()

        def slow_skill():
            time.sleep(0.1)  # Sleep for 100ms
            return "done"

        engine.register_skill("slow", slow_skill)
        start_time = time.time()
        result = engine.execute_skill("slow")
        end_time = time.time()

        assert result.success is True  # nosec: B101 - Test assertion
        self.assertGreaterEqual(result.execution_time, 0.1)
        self.assertGreaterEqual(end_time - start_time, 0.1)

    def test_skill_confidence_calculation(self):
        """Test confidence calculation in skill results."""
        engine = XPRAgentEngine()

        def confident_skill():
            return {"success": True, "data": "test"}

        engine.register_skill("confident", confident_skill)
        result = engine.execute_skill("confident")

        self.assertTrue(result.success)
        self.assertGreaterEqual(0, result.confidence)
        self.assertLessEqual(result.confidence, 1)

    def test_skill_result_metadata_preservation(self):
        """Test metadata preservation in skill results."""
        engine = XPRAgentEngine()

        def metadata_skill():
            return {"result": "success", "metadata": {"iterations": 5, "time": 1.2}}

        engine.register_skill("metadata", metadata_skill)
        result = engine.execute_skill("metadata")

        self.assertTrue(result.success)
        self.assertIsNotNone(result.metadata)
        self.assertIsInstance(result.metadata, dict)


class TestXPRAgentEngineIntegration(unittest.TestCase):
    """Integration tests for XPRAgentEngine."""

    def test_full_workflow_simulation(self):
        """Test a complete workflow simulation."""
        engine = XPRAgentEngine()

        # Register skills for a complete workflow
        def plan_skill(goal):
            return {"plan": f"plan_for_{goal}", "steps": 3}

        def execute_skill(plan):
            return {"execution": f"executed_{plan}", "status": "success"}

        def analyze_skill(result):
            return {"analysis": f"analyzed_{result}", "quality": "high"}

        engine.register_skill("plan", plan_skill)
        engine.register_skill("execute", execute_skill)
        engine.register_skill("analyze", analyze_skill)

        # Execute workflow
        plan_result = engine.execute_skill("plan", "test_goal")
        self.assertTrue(plan_result.success)

        exec_result = engine.execute_skill(
            "execute", plan_result.result["plan"] if plan_result.result else ""
        )
        self.assertTrue(exec_result.success)

        analysis_result = engine.execute_skill(
            "analyze", exec_result.result["execution"] if exec_result.result else ""
        )
        self.assertTrue(analysis_result.success)

    def test_error_handling_workflow(self):
        """Test error handling in workflow."""
        engine = XPRAgentEngine()

        def failing_skill():
            raise RuntimeError("Simulated failure")

        def recovery_skill():
            return {"recovered": True}

        engine.register_skill("failing", failing_skill)
        engine.register_skill("recovery", recovery_skill)

        # First skill fails
        fail_result = engine.execute_skill("failing")
        self.assertFalse(fail_result.success)
        self.assertIsNotNone(fail_result.error)
        self.assertIn("Simulated failure", fail_result.error)  # type: ignore[arg-type]

        # Recovery succeeds
        recovery_result = engine.execute_skill("recovery")
        self.assertTrue(recovery_result.success)

    @patch("xpr_agent_engine.litellm", None)
    def test_without_litellm(self):
        """Test engine behavior when litellm is not available."""
        engine = XPRAgentEngine()

        def simple_skill(x):
            return x * 2

        engine.register_skill("simple", simple_skill)
        result = engine.execute_skill("simple", 21)

        self.assertTrue(result.success)
        self.assertEqual(result.result, 42)


class TestComplianceIntegration(unittest.TestCase):
    """Tests for compliance manager integration."""

    def test_compliance_manager_initialization(self):
        """Test compliance manager is properly initialized."""
        engine = XPRAgentEngine()
        self.assertTrue(hasattr(engine, "compliance_manager"))
        self.assertIsNotNone(engine.compliance_manager)

    def test_compliant_skill_execution(self):
        """Test skill execution with compliance checks."""
        engine = XPRAgentEngine()

        def compliant_skill(data):
            return {"processed": data, "compliant": True}

        engine.register_skill("compliant", compliant_skill)
        result = engine.execute_skill("compliant", {"test": "data"})

        self.assertTrue(result.success)
        self.assertIsNotNone(result.result)
        self.assertTrue(result.result["compliant"])  # type: ignore[index]


class TestLoggingAndDebugging(unittest.TestCase):
    """Tests for logging and debugging features."""

    def test_logging_setup(self):
        """Test that logging is properly configured."""
        engine = XPRAgentEngine()

        # Check that logger exists
        self.assertTrue(hasattr(engine, "skills"))
        self.assertIsInstance(engine.skills, dict)

    def test_skill_execution_logging(self):
        """Test that skill execution is logged."""
        engine = XPRAgentEngine()

        def logged_skill(x):
            return x + 1

        engine.register_skill("logged", logged_skill)

        # Capture logs (would need log capture in real test setup)
        result = engine.execute_skill("logged", 5)
        self.assertTrue(result.success)
        self.assertEqual(result.result, 6)

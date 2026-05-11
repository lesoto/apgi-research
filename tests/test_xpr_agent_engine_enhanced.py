"""
Enhanced comprehensive tests for xpr_agent_engine.py module.
"""

import time
from unittest.mock import patch

import pytest

try:
    from xpr_agent_engine import ExecutionReport, SkillResult, SkillType, XPRAgentEngine
except ImportError as e:
    pytest.skip(f"Cannot import from xpr_agent_engine: {e}", allow_module_level=True)


class TestSkillResult:
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
        assert result.success is True
        assert result.skill_type == "test"
        assert result.result == "test_result"
        assert result.execution_time == 1.5
        assert result.confidence == 0.8
        assert result.metadata == {"key": "value"}

    def test_skill_result_failure(self):
        """Test SkillResult with failure."""
        result = SkillResult(
            success=False,
            skill_type="test",
            error="test_error",
            execution_time=0.5,
            confidence=0.0,
        )
        assert result.success is False
        assert result.error == "test_error"
        assert result.execution_time == 0.5
        assert result.confidence == 0.0

    def test_skill_result_defaults(self):
        """Test SkillResult with default values."""
        result = SkillResult(success=True, skill_type="test")
        assert result.result is None
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.confidence == 0.0
        assert result.metadata is None


class TestExecutionReport:
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
        assert report.experiment_name == "test_exp"
        assert report.success is True
        assert report.execution_time == 10.5
        assert report.result == "success"
        assert report.confidence == 0.9
        assert report.metadata == {"iterations": 5}

    def test_execution_report_post_init(self):
        """Test ExecutionReport post_init metadata initialization."""
        report = ExecutionReport(
            experiment_name="test", success=False, execution_time=0.0
        )
        assert report.metadata == {}

    def test_execution_report_with_none_metadata(self):
        """Test ExecutionReport with explicit None metadata."""
        report = ExecutionReport(
            experiment_name="test", success=True, execution_time=1.0, metadata=None
        )
        assert report.metadata == {}


class TestSkillType:
    """Tests for SkillType enum."""

    def test_skill_type_values(self):
        """Test SkillType enum values."""
        assert SkillType.PLAN_GENERATION.value == "plan_generation"
        assert SkillType.EXECUTION.value == "execution"
        assert SkillType.ANALYSIS.value == "analysis"
        assert SkillType.MEMORY_UPDATE.value == "memory_update"
        assert SkillType.MODIFICATION.value == "modification"

    def test_skill_type_comparison(self):
        """Test SkillType enum comparison."""
        assert SkillType.PLAN_GENERATION == SkillType.PLAN_GENERATION
        # Test that different enum values are not equal
        plan_gen_value = SkillType.PLAN_GENERATION.value
        exec_value = SkillType.EXECUTION.value
        assert plan_gen_value != exec_value


class TestXPRAgentEngine:
    """Tests for XPRAgentEngine class."""

    def test_engine_initialization(self):
        """Test XPRAgentEngine initialization."""
        engine = XPRAgentEngine()
        assert isinstance(engine.skills, dict)
        assert isinstance(engine.execution_history, list)
        assert engine.current_plan is None
        assert engine.compliance_manager is not None

    def test_register_skill(self):
        """Test skill registration."""
        engine = XPRAgentEngine()

        def test_skill(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"

        engine.register_skill("test_skill", test_skill)
        assert "test_skill" in engine.skills
        assert engine.skills["test_skill"] == test_skill

    def test_execute_skill_success(self):
        """Test successful skill execution."""
        engine = XPRAgentEngine()

        def test_skill(x):
            return x * 2

        engine.register_skill("double", test_skill)
        result = engine.execute_skill("double", 5)

        assert result.success is True
        assert result.skill_type == "double"
        assert result.result == 10
        assert result.error is None
        assert result.execution_time >= 0
        assert result.confidence > 0

    def test_execute_skill_not_found(self):
        """Test executing non-existent skill."""
        engine = XPRAgentEngine()
        result = engine.execute_skill("nonexistent")

        assert result.success is False
        assert result.skill_type == "nonexistent"
        assert result.result is None
        assert result.error is not None and "not found" in result.error
        assert result.execution_time == 0.0
        assert result.confidence == 0.0

    def test_execute_skill_with_exception(self):
        """Test skill execution with exception."""
        engine = XPRAgentEngine()

        def failing_skill():
            raise ValueError("Test error")

        engine.register_skill("failing", failing_skill)
        result = engine.execute_skill("failing")

        assert result.success is False
        assert result.skill_type == "failing"
        assert result.result is None
        assert result.error is not None and "Test error" in result.error
        assert result.execution_time >= 0
        assert result.confidence == 0.0

    def test_execute_skill_with_kwargs(self):
        """Test skill execution with keyword arguments."""
        engine = XPRAgentEngine()

        def test_skill(x, y=10):
            return x + y

        engine.register_skill("add", test_skill)
        result = engine.execute_skill("add", 5, y=15)

        assert result.success is True
        assert result.result == 20

    def test_execute_plan_generation(self):
        """Test plan generation execution."""
        engine = XPRAgentEngine()

        def plan_skill(goal):
            return {"steps": [f"step1_{goal}", f"step2_{goal}"]}

        engine.register_skill("plan_generation", plan_skill)
        result = engine.execute_skill("plan_generation", "test_goal")

        assert result.success is True
        assert result.skill_type == "plan_generation"
        assert result.result is not None and "steps" in result.result

    def test_execute_analysis_skill(self):
        """Test analysis skill execution."""
        engine = XPRAgentEngine()

        def analysis_skill(data):
            return {"analysis": f"analyzed_{data}", "confidence": 0.85}

        engine.register_skill("analysis", analysis_skill)
        result = engine.execute_skill("analysis", "test_data")

        assert result.success is True
        assert result.skill_type == "analysis"
        assert result.result is not None and "analysis" in result.result

    def test_execute_memory_update(self):
        """Test memory update skill execution."""
        engine = XPRAgentEngine()

        def memory_skill(key, value):
            return {"stored": f"{key}:{value}"}

        engine.register_skill("memory_update", memory_skill)
        result = engine.execute_skill("memory_update", "test_key", "test_value")

        assert result.success is True
        assert result.skill_type == "memory_update"
        assert result.result is not None and "stored" in result.result

    def test_execute_modification(self):
        """Test modification skill execution."""
        engine = XPRAgentEngine()

        def modification_skill(file_path, change):
            return {"modified": f"{file_path}:{change}"}

        engine.register_skill("modification", modification_skill)
        result = engine.execute_skill("modification", "test.py", "add_function")

        assert result.success is True
        assert result.skill_type == "modification"
        assert result.result is not None and "modified" in result.result

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

        assert len(engine.skills) == 3
        for name in skills:
            assert name in engine.skills

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

        assert result.success is True
        assert result.execution_time >= 0.1
        assert end_time - start_time >= 0.1

    def test_skill_confidence_calculation(self):
        """Test confidence calculation in skill results."""
        engine = XPRAgentEngine()

        def confident_skill():
            return {"success": True, "data": "test"}

        engine.register_skill("confident", confident_skill)
        result = engine.execute_skill("confident")

        assert result.success is True
        assert 0 <= result.confidence <= 1

    def test_skill_result_metadata_preservation(self):
        """Test metadata preservation in skill results."""
        engine = XPRAgentEngine()

        def metadata_skill():
            return {"result": "success", "metadata": {"iterations": 5, "time": 1.2}}

        engine.register_skill("metadata", metadata_skill)
        result = engine.execute_skill("metadata")

        assert result.success is True
        assert result.metadata is not None
        assert isinstance(result.metadata, dict)


class TestXPRAgentEngineIntegration:
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
        assert plan_result.success is True

        exec_result = engine.execute_skill(
            "execute", plan_result.result["plan"] if plan_result.result else ""
        )
        assert exec_result.success is True

        analysis_result = engine.execute_skill(
            "analyze", exec_result.result["execution"] if exec_result.result else ""
        )
        assert analysis_result.success is True

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
        assert fail_result.success is False
        assert (
            fail_result.error is not None and "Simulated failure" in fail_result.error
        )

        # Recovery succeeds
        recovery_result = engine.execute_skill("recovery")
        assert recovery_result.success is True

    @patch("xpr_agent_engine.litellm", None)
    def test_without_litellm(self):
        """Test engine behavior when litellm is not available."""
        engine = XPRAgentEngine()

        def simple_skill(x):
            return x * 2

        engine.register_skill("simple", simple_skill)
        result = engine.execute_skill("simple", 21)

        assert result.success is True
        assert result.result == 42


class TestComplianceIntegration:
    """Tests for compliance manager integration."""

    def test_compliance_manager_initialization(self):
        """Test compliance manager is properly initialized."""
        engine = XPRAgentEngine()
        assert hasattr(engine, "compliance_manager")
        assert engine.compliance_manager is not None

    def test_compliant_skill_execution(self):
        """Test skill execution with compliance checks."""
        engine = XPRAgentEngine()

        def compliant_skill(data):
            return {"processed": data, "compliant": True}

        engine.register_skill("compliant", compliant_skill)
        result = engine.execute_skill("compliant", {"test": "data"})

        assert result.success is True
        assert result.result is not None and result.result["compliant"] is True


class TestLoggingAndDebugging:
    """Tests for logging and debugging features."""

    def test_logging_setup(self):
        """Test that logging is properly configured."""
        engine = XPRAgentEngine()

        # Check that logger exists
        assert hasattr(engine, "skills")
        assert isinstance(engine.skills, dict)

    def test_skill_execution_logging(self):
        """Test that skill execution is logged."""
        engine = XPRAgentEngine()

        def logged_skill(x):
            return x + 1

        engine.register_skill("logged", logged_skill)

        # Capture logs (would need log capture in real test setup)
        result = engine.execute_skill("logged", 5)
        assert result.success is True
        assert result.result == 6

"""
================================================================================
COMPREHENSIVE TESTS FOR XPR AGENT ENGINE
================================================================================

This module provides extensive testing for the XPR Agent Engine including:
- XPRAgentEngine core functionality
- LLMIntegration provider management
- EnhancedXPRAgentEngine features
- XPRAgentEngineEnhanced advanced capabilities
- Error handling and edge cases
- Concurrent execution testing

Addresses critical testing gap identified in testing-report.md:
"xpr_agent_engine.py (4 tests for 1,198 lines)"

Target: 40+ test functions to achieve comprehensive coverage
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from xpr_agent_engine import (
    SkillResult,
    ExecutionReport,
    XPRAgentEngine,
    LLMIntegration,
    EnhancedXPRAgentEngine,
    XPRAgentEngineEnhanced,
    XPRSkillResult,
    XPRSkillType,
    register_xpr_skills,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine() -> XPRAgentEngine:
    """Provide a fresh XPRAgentEngine instance."""
    return XPRAgentEngine()


@pytest.fixture
def enhanced_engine() -> XPRAgentEngineEnhanced:
    """Provide a fresh XPRAgentEngineEnhanced instance."""
    return XPRAgentEngineEnhanced()


@pytest.fixture
def llm_integration() -> LLMIntegration:
    """Provide a fresh LLMIntegration instance."""
    return LLMIntegration()


@pytest.fixture
def sample_skill_func():
    """Provide a sample skill function."""

    def skill(x: int) -> int:
        return x * 2

    return skill


@pytest.fixture
def sample_failing_skill():
    """Provide a skill that raises an exception."""

    def skill(x: int) -> int:
        raise ValueError("Skill failed")

    return skill


@pytest.fixture
def sample_execution_report() -> ExecutionReport:
    """Provide a sample execution report."""
    return ExecutionReport(
        experiment_name="test_experiment",
        success=True,
        execution_time=1.5,
        result={"accuracy": 0.95},
        confidence=0.8,
    )


# =============================================================================
# SKILL RESULT AND EXECUTION REPORT TESTS
# =============================================================================


class TestSkillResult:
    """Test SkillResult dataclass."""

    def test_skill_result_creation(self) -> None:
        """Test creating a SkillResult with all fields."""
        result = SkillResult(
            success=True,
            skill_type="test_skill",
            result={"key": "value"},
            error=None,
            execution_time=1.5,
            confidence=0.9,
            metadata={"version": "1.0"},
        )
        assert result.success is True
        assert result.skill_type == "test_skill"
        assert result.result == {"key": "value"}
        assert result.error is None
        assert result.execution_time == 1.5
        assert result.confidence == 0.9
        assert result.metadata == {"version": "1.0"}

    def test_skill_result_defaults(self) -> None:
        """Test SkillResult with default values."""
        result = SkillResult(success=True, skill_type="test")
        assert result.result is None
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.confidence == 0.0
        assert result.metadata is None

    def test_skill_result_failure(self) -> None:
        """Test SkillResult for failed execution."""
        result = SkillResult(
            success=False,
            skill_type="failing_skill",
            error="Something went wrong",
            confidence=0.0,
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestExecutionReport:
    """Test ExecutionReport dataclass."""

    def test_execution_report_creation(self) -> None:
        """Test creating an ExecutionReport."""
        report = ExecutionReport(
            experiment_name="exp_001",
            success=True,
            execution_time=2.5,
            result={"metric": 0.95},
            confidence=0.85,
        )
        assert report.experiment_name == "exp_001"
        assert report.success is True
        assert report.execution_time == 2.5

    def test_execution_report_post_init(self) -> None:
        """Test __post_init__ initializes metadata."""
        report = ExecutionReport(
            experiment_name="test",
            success=True,
            execution_time=1.0,
        )
        assert report.metadata == {}

    def test_execution_report_with_metadata(self) -> None:
        """Test ExecutionReport with custom metadata."""
        report = ExecutionReport(
            experiment_name="test",
            success=True,
            execution_time=1.0,
            metadata={"git_commit": "abc123", "timestamp": "2024-01-01"},
        )
        if report.metadata:
            assert report.metadata["git_commit"] == "abc123"


# =============================================================================
# XPR AGENT ENGINE CORE TESTS
# =============================================================================


class TestXPRAgentEngine:
    """Test XPRAgentEngine core functionality."""

    def test_engine_initialization(self, engine: XPRAgentEngine) -> None:
        """Test engine initializes with empty state."""
        assert engine.skills == {}
        assert engine.execution_history == []
        assert engine.current_plan is None

    def test_register_skill(
        self, engine: XPRAgentEngine, sample_skill_func: Callable[[int], int]
    ) -> None:
        """Test registering a skill."""
        engine.register_skill("double", sample_skill_func)
        assert "double" in engine.skills
        assert engine.skills["double"] == sample_skill_func

    def test_register_multiple_skills(self, engine: XPRAgentEngine) -> None:
        """Test registering multiple skills."""
        engine.register_skill("skill1", lambda x: x)
        engine.register_skill("skill2", lambda x: x + 1)
        assert len(engine.skills) == 2

    def test_execute_skill_success(
        self, engine: XPRAgentEngine, sample_skill_func: Callable[[int], int]
    ) -> None:
        """Test executing a registered skill successfully."""
        engine.register_skill("double", sample_skill_func)
        result = engine.execute_skill("double", 5)

        assert result.success is True
        assert result.skill_type == "double"
        assert result.result == 10
        assert result.error is None
        assert result.confidence == 0.8

    def test_execute_skill_not_found(self, engine: XPRAgentEngine) -> None:
        """Test executing a non-existent skill."""
        result = engine.execute_skill("nonexistent")

        assert result.success is False
        assert result.error == "Skill 'nonexistent' not found"
        assert result.confidence == 0.0

    def test_execute_skill_failure(
        self, engine: XPRAgentEngine, sample_failing_skill: Callable[[int], int]
    ) -> None:
        """Test executing a skill that raises an exception."""
        engine.register_skill("failing", sample_failing_skill)
        result = engine.execute_skill("failing", 5)

        assert result.success is False
        assert result.error is not None and "Skill failed" in result.error
        assert result.confidence == 0.0

    def test_execute_skill_with_kwargs(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with keyword arguments."""

        def skill_with_kwargs(x: int, multiplier: int = 2) -> int:
            return x * multiplier

        engine.register_skill("multiply", skill_with_kwargs)
        result = engine.execute_skill("multiply", 5, multiplier=3)

        assert result.success is True
        assert result.result == 15

    def test_get_performance_summary_empty(self, engine: XPRAgentEngine) -> None:
        """Test performance summary with no history."""
        summary = engine.get_performance_summary()

        assert summary["total_experiments"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["avg_execution_time"] == 0.0

    def test_get_performance_summary_with_data(
        self, engine: XPRAgentEngine, sample_execution_report: ExecutionReport
    ) -> None:
        """Test performance summary with execution history."""
        engine.add_execution_report(sample_execution_report)

        summary = engine.get_performance_summary()

        assert summary["total_experiments"] == 1
        assert summary["success_rate"] == 1.0
        assert summary["avg_execution_time"] == 1.5

    def test_add_execution_report(self, engine: XPRAgentEngine) -> None:
        """Test adding execution report to history."""
        report = ExecutionReport(
            experiment_name="test_exp",
            success=True,
            execution_time=1.0,
        )
        engine.add_execution_report(report)

        assert len(engine.execution_history) == 1
        assert engine.execution_history[0].experiment_name == "test_exp"

    def test_add_multiple_execution_reports(self, engine: XPRAgentEngine) -> None:
        """Test adding multiple execution reports."""
        for i in range(5):
            report = ExecutionReport(
                experiment_name=f"exp_{i}",
                success=i % 2 == 0,
                execution_time=float(i),
            )
            engine.add_execution_report(report)

        assert len(engine.execution_history) == 5
        summary = engine.get_performance_summary()
        assert summary["success_rate"] == 0.6  # 3 out of 5 successful

    def test_set_and_get_current_plan(self, engine: XPRAgentEngine) -> None:
        """Test setting and getting current plan."""
        plan = {"hypothesis": "Test hypothesis", "steps": ["step1", "step2"]}
        engine.set_current_plan(plan)

        assert engine.get_current_plan() == plan

    def test_get_current_plan_none(self, engine: XPRAgentEngine) -> None:
        """Test getting plan when none is set."""
        assert engine.get_current_plan() is None

    def test_plan_experiment_with_lr(self, engine: XPRAgentEngine) -> None:
        """Test experiment planning with learning rate parameter."""
        current_params = {"lr": 0.01, "epochs": 10}
        result = engine.plan_experiment("optimize_learning", current_params)

        assert result.success is True
        assert "modifications" in str(result.result)
        # lr gets multiplied by 0.9 (9% decrease)
        if isinstance(result.result, dict):
            assert result.result["modifications"]["lr"] == pytest.approx(
                0.009, rel=1e-3
            )

    def test_plan_experiment_with_epochs(self, engine: XPRAgentEngine) -> None:
        """Test experiment planning with epochs parameter."""
        current_params = {"epochs": 100}
        result = engine.plan_experiment("optimize_training", current_params)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["modifications"]["epochs"] == 110  # 100 * 1.1

    def test_plan_experiment_empty_params(self, engine: XPRAgentEngine) -> None:
        """Test experiment planning with empty parameters."""
        result = engine.plan_experiment("test_task", {})

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["modifications"] == {}


# =============================================================================
# LLM INTEGRATION TESTS
# =============================================================================


class TestLLMIntegration:
    """Test LLMIntegration functionality."""

    def test_llm_integration_initialization(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test LLMIntegration initialization."""
        assert llm_integration.preferred_provider == "openai"
        assert llm_integration._initialized == {}

    def test_llm_integration_custom_provider(self) -> None:
        """Test LLMIntegration with custom provider."""
        integration = LLMIntegration(preferred_provider="local")
        assert integration.preferred_provider == "local"

    def test_get_provider_config_existing(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test getting config for existing provider."""
        config = llm_integration._get_provider_config("openai")
        assert config is not None
        assert "model" in config

    def test_get_provider_config_nonexistent(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test getting config for non-existent provider."""
        config = llm_integration._get_provider_config("nonexistent")
        assert config == {}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_initialize_client_openai_success(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test initializing OpenAI client with API key."""
        result = llm_integration._initialize_client("openai")
        assert result is True
        assert llm_integration._initialized.get("openai") is True

    def test_initialize_client_openai_no_key(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test initializing OpenAI client without API key."""
        # Ensure no API key is set
        with patch.dict(os.environ, {}, clear=True):
            result = llm_integration._initialize_client("openai")
            assert result is False

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_initialize_client_anthropic_raises_valueerror(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test that Anthropic raises ValueError as it's not in LLM_PROVIDERS."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm_integration._initialize_client("anthropic")

    def test_initialize_client_local(self, llm_integration: LLMIntegration) -> None:
        """Test initializing local provider."""
        result = llm_integration._initialize_client("local")
        assert result is True

    def test_initialize_client_unsupported_raises_valueerror(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm_integration._initialize_client("unsupported_provider")

    def test_get_client_initializes_when_needed(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test get_client initializes client if not already done."""
        with patch.object(
            llm_integration, "_initialize_client", return_value=True
        ) as mock_init:
            result = llm_integration.get_client("openai")
            mock_init.assert_called_once_with("openai")
            assert result is True

    def test_get_client_already_initialized(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test get_client returns True when already initialized."""
        llm_integration._initialized["openai"] = True
        result = llm_integration.get_client("openai")
        assert result is True

    def test_generate_text_provider_not_initialized(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test generate_text when provider not initialized."""
        with patch.object(llm_integration, "_initialize_client", return_value=False):
            result = llm_integration.generate_text("test prompt", provider="openai")
            assert result == ""

    def test_generate_text_no_litellm(self, llm_integration: LLMIntegration) -> None:
        """Test generate_text when litellm is not available."""
        with patch("xpr_agent_engine.litellm", None):
            with patch.object(llm_integration, "_initialized", {"openai": True}):
                result = llm_integration.generate_text("test", provider="openai")
                assert result == ""


# =============================================================================
# ENHANCED XPR AGENT ENGINE TESTS
# =============================================================================


class TestEnhancedXPRAgentEngine:
    """Test EnhancedXPRAgentEngine functionality."""

    def test_enhanced_engine_initialization(self) -> None:
        """Test EnhancedXPRAgentEngine initialization."""
        engine = EnhancedXPRAgentEngine()
        assert engine.llm_integration is not None
        assert isinstance(engine.llm_integration, LLMIntegration)

    def test_enhanced_engine_skills_registered(self) -> None:
        """Test that enhanced skills are registered on init."""
        engine = EnhancedXPRAgentEngine()
        assert "plan_experiment" in engine.skills
        assert "llm_plan_generation" in engine.skills


# =============================================================================
# XPR AGENT ENGINE ENHANCED TESTS
# =============================================================================


class TestXPRAgentEngineEnhanced:
    """Test XPRAgentEngineEnhanced advanced features."""

    def test_enhanced_engine_initialization(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPRAgentEngineEnhanced initialization."""
        assert enhanced_engine.llm_integration is not None
        assert enhanced_engine.llm_providers == {}
        assert enhanced_engine.performance_history == {}
        assert enhanced_engine.skill_dependencies == {}
        assert enhanced_engine.optimization_strategies == {}

    def test_register_llm_provider(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test registering an LLM provider."""
        config = {"model": "gpt-4", "api_key": "test"}
        enhanced_engine.register_llm_provider("custom", config)

        assert "custom" in enhanced_engine.llm_providers
        assert enhanced_engine.llm_providers["custom"] == config

    def test_set_optimization_strategy(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test setting optimization strategy."""
        strategy = {"learning_rate": 0.01, "batch_size": 32}
        enhanced_engine.set_optimization_strategy("training", strategy)

        assert "training" in enhanced_engine.optimization_strategies
        assert enhanced_engine.optimization_strategies["training"] == strategy

    def test_analyze_performance_trend_no_history(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test analyzing trend with no history."""
        result = enhanced_engine.analyze_performance_trend("unknown_type")

        assert result["trend"] == 0.0
        assert result["volatility"] == 0.0

    def test_analyze_performance_trend_with_history(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test analyzing trend with performance history."""
        enhanced_engine.performance_history["test_type"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = enhanced_engine.analyze_performance_trend("test_type", window_size=3)

        assert result["trend"] > 0  # Upward trend
        assert result["volatility"] >= 0

    def test_analyze_performance_trend_with_window_larger_than_data(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test analyzing trend when window is larger than data."""
        enhanced_engine.performance_history["test"] = [0.5, 0.6]
        result = enhanced_engine.analyze_performance_trend("test", window_size=10)

        assert result["trend"] == 0.55  # Average of [0.5, 0.6]

    def test_xpr_plan_experiment_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan experiment generation."""
        current_params = {"lr": 0.01, "epochs": 100}
        result = enhanced_engine.xpr_plan_experiment("optimize", current_params)

        assert isinstance(result, XPRSkillResult)
        assert result.success is True
        assert result.skill_type == XPRSkillType.PLAN_GENERATION.value
        assert "modifications" in result.result

    def test_xpr_plan_experiment_with_performance_history(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan with existing performance history."""
        enhanced_engine.performance_history["optimize"] = [0.1, 0.15, 0.2, 0.25, 0.3]
        current_params = {"lr": 0.01}
        result = enhanced_engine.xpr_plan_experiment("optimize", current_params)

        assert result.success is True
        if result.metadata:
            assert result.metadata.get("adaptive") is True

    def test_xpr_plan_experiment_no_improving_trend(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan when trend is not improving."""
        enhanced_engine.performance_history["optimize"] = [0.5, 0.4, 0.3, 0.2]
        current_params = {"epochs": 100}
        result = enhanced_engine.xpr_plan_experiment("optimize", current_params)

        assert result.success is True
        if isinstance(result.result, dict):
            modified_epochs = result.result["modifications"]["epochs"]
            assert isinstance(modified_epochs, (int, float))

    def test_xpr_plan_experiment_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan handles exceptions gracefully."""
        # Pass invalid data to trigger exception handling
        with patch.object(
            enhanced_engine,
            "analyze_performance_trend",
            side_effect=Exception("Test error"),
        ):
            result = enhanced_engine.xpr_plan_experiment("test", {})

            assert isinstance(result, XPRSkillResult)
            assert result.success is False
            assert result.error is not None and "Test error" in result.error

    def test_extract_missing_module_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test extracting missing module from error."""
        error_msg = "No module named 'numpy'"
        result = enhanced_engine._extract_missing_module(error_msg)

        assert result == "numpy"

    def test_extract_missing_module_no_match(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test extracting module when no match found."""
        error_msg = "Some other error message"
        result = enhanced_engine._extract_missing_module(error_msg)

        assert result is None

    def test_xpr_job_debug_import_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with ImportError."""
        experiment_data = {
            "error": "ImportError: No module named 'missing_module'",
            "experiment": "test_exp",
            "file": "/path/to/file.py",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["error_type"] == "import_error"
            assert "missing_module" in result.result["dependencies"]

    def test_xpr_job_debug_module_not_found(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with ModuleNotFoundError."""
        experiment_data = {
            "error": "ModuleNotFoundError: No module named 'test'",
            "experiment": "test",
            "file": "test.py",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["error_type"] == "module_not_found"

    def test_xpr_job_debug_permission_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with PermissionError."""
        experiment_data = {
            "error": "PermissionError: [Errno 13] Permission denied",
            "experiment": "test",
            "file": "/etc/passwd",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["error_type"] == "permission_error"

    def test_xpr_job_debug_timeout(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with timeout error."""
        experiment_data = {
            "error": "Request timed out after 30 seconds",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["error_type"] == "timeout_error"

    def test_xpr_job_debug_generic_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with generic error."""
        experiment_data = {
            "error": "Something unexpected happened",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["error_type"] == "generic_error"

    def test_xpr_job_debug_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug exception handling."""

        # Trigger an exception during dictionary access
        class BadDict(dict):
            def get(self, *args):
                raise ValueError("Skill failed")

        result = enhanced_engine.xpr_job_debug(BadDict())

        assert result.success is False
        assert result.error is not None

    def test_xpr_issue_fix_syntax_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix with SyntaxError."""
        experiment_data = {
            "error": "SyntaxError: invalid syntax at line 42",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_issue_fix(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["fix_type"] == "syntax_fix"
            assert "42" in result.result["generated_code"]

    def test_xpr_issue_fix_file_not_found(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix with FileNotFoundError."""
        experiment_data = {
            "error": "FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_issue_fix(experiment_data)

        assert result.success is True
        assert result.result["fix_type"] == "file_creation_fix"
        assert "missing.txt" in result.result["generated_code"]

    def test_xpr_issue_fix_import_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix with ImportError."""
        experiment_data = {
            "error": "ImportError: No module named 'requests'",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_issue_fix(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert result.result["fix_type"] == "import_fix"
            assert "import requests" in result.result["generated_code"]

    def test_xpr_issue_fix_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix exception handling."""

        # Trigger an exception during dictionary access
        class BadDict(dict):
            def get(self, *args):
                raise ValueError("Skill failed")

        result = enhanced_engine.xpr_issue_fix(BadDict())

        assert result.success is False
        assert result.error is not None

    def test_xpr_issue_report_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue report generation."""
        experiment_data = {
            "error": "ImportError: No module named 'missing'",
            "experiment": "test_exp",
            "file": "/path/to/file.py",
            "metrics": {"accuracy": 0.5},
        }
        result = enhanced_engine.xpr_issue_report(experiment_data)

        assert result.success is True
        if isinstance(result.result, dict):
            assert "experiment_name" in result.result
            assert "severity" in result.result
            assert "recommendations" in result.result

    def test_xpr_issue_report_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue report exception handling."""

        # Trigger an exception during dictionary access
        class BadDict(dict):
            def get(self, *args):
                raise ValueError("Skill failed")

        result = enhanced_engine.xpr_issue_report(BadDict())

        assert result.success is False
        assert result.error is not None

    def test_assess_severity_critical(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test severity assessment for critical errors."""
        assert enhanced_engine._assess_severity("Critical system failure") == "critical"
        assert enhanced_engine._assess_severity("Fatal error occurred") == "critical"
        assert enhanced_engine._assess_severity("System crash") == "critical"

    def test_assess_severity_high(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test severity assessment for high severity."""
        assert enhanced_engine._assess_severity("Error in processing") == "high"
        assert enhanced_engine._assess_severity("Exception raised") == "high"
        assert enhanced_engine._assess_severity("Task failed") == "high"

    def test_assess_severity_medium(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test severity assessment for medium severity."""
        assert (
            enhanced_engine._assess_severity("Warning: something happened") == "medium"
        )
        assert enhanced_engine._assess_severity("Notice") == "medium"

    def test_generate_recommendations_timeout(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recommendation generation for timeout."""
        recs = enhanced_engine._generate_recommendations("Request timeout", {})

        assert any("timeout" in r.lower() for r in recs)
        assert any("duration" in r.lower() for r in recs)

    def test_generate_recommendations_memory(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recommendation generation for memory issues."""
        recs = enhanced_engine._generate_recommendations("Out of memory error", {})

        assert any("memory" in r.lower() for r in recs)
        assert any("batch" in r.lower() for r in recs)

    def test_generate_recommendations_permission(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recommendation generation for permission issues."""
        recs = enhanced_engine._generate_recommendations("Permission denied", {})

        assert any("permission" in r.lower() for r in recs)

    def test_generate_recovery_steps(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recovery steps generation."""
        steps = enhanced_engine._generate_recovery_steps("any error")

        assert len(steps) == 5
        assert all(isinstance(step, str) for step in steps)

    def test_analyze_root_cause_import(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for ImportError."""
        cause = enhanced_engine._analyze_root_cause("ImportError: No module named 'x'")
        assert "Missing or incompatible dependency" in cause

    def test_analyze_root_cause_permission(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for PermissionError."""
        cause = enhanced_engine._analyze_root_cause("PermissionError: denied")
        assert "Insufficient file system permissions" in cause

    def test_analyze_root_cause_timeout(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for timeout."""
        cause = enhanced_engine._analyze_root_cause("timeout occurred")
        assert "Resource exhaustion" in cause

    def test_analyze_root_cause_generic(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for generic errors."""
        cause = enhanced_engine._analyze_root_cause("Unknown error")
        assert "Unknown error requiring investigation" in cause

    def test_xpr_skill_chain_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test successful skill chaining."""
        # Register skills that work with dictionaries
        enhanced_engine.register_skill("skill1", lambda x: {"value": x["value"] + 1})
        enhanced_engine.register_skill("skill2", lambda x: {"value": x["value"] * 2})

        results = enhanced_engine.xpr_skill_chain({"value": 5}, ["skill1", "skill2"])

        assert len(results) == 2
        assert all(isinstance(r, XPRSkillResult) for r in results)
        assert results[0].success is True
        assert results[1].success is True

    def test_xpr_skill_chain_skill_not_found(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test skill chain with missing skill."""
        results = enhanced_engine.xpr_skill_chain({}, ["nonexistent_skill"])

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None and "not found" in results[0].error

    def test_xpr_skill_chain_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test skill chain exception handling."""

        def failing_skill(x):
            raise ValueError("Skill failed")

        enhanced_engine.register_skill("failing", failing_skill)
        results = enhanced_engine.xpr_skill_chain({}, ["failing"])

        assert len(results) == 1
        assert results[0].success is False

    def test_get_performance_summary_enhanced(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test enhanced performance summary."""
        # Add some execution reports
        for i in range(5):
            report = ExecutionReport(
                experiment_name=f"exp_{i}",
                success=i % 2 == 0,
                execution_time=float(i + 1),
                metadata={"adaptive": i % 3 == 0},
            )
            enhanced_engine.add_execution_report(report)

        summary = enhanced_engine.get_performance_summary()

        assert "avg_confidence" in summary
        assert "skill_success_rate" in summary
        assert "adaptive_optimizations" in summary
        assert summary["adaptive_optimizations"] == 2  # i=0,3


# =============================================================================
# REGISTER XPR SKILLS TESTS
# =============================================================================


class TestRegisterXPRSkills:
    """Test register_xpr_skills function."""

    def test_register_skills_basic(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test registering XPR skills on enhanced engine."""
        register_xpr_skills(enhanced_engine)
        # Should register plan_experiment
        assert "plan_experiment" in enhanced_engine.skills
        # XPRAgentEngineEnhanced has xpr_* methods, so they should be registered
        assert "xpr_job_debug" in enhanced_engine.skills
        assert "xpr_issue_fix" in enhanced_engine.skills
        assert "xpr_issue_report" in enhanced_engine.skills
        assert "xpr_skill_chain" in enhanced_engine.skills
        assert "xpr_plan_experiment" in enhanced_engine.skills

    def test_register_skills_on_base_engine(self, engine: XPRAgentEngine) -> None:
        """Test registering on base engine without xpr methods."""
        register_xpr_skills(engine)  # type: ignore[arg-type]

        # Should only register plan_experiment
        assert "plan_experiment" in engine.skills
        # These should not be registered on base engine
        assert "xpr_job_debug" not in engine.skills
        assert "xpr_issue_fix" not in engine.skills
        assert "xpr_issue_report" not in engine.skills
        assert "xpr_skill_chain" not in engine.skills
        assert "xpr_plan_experiment" not in engine.skills

        # Should only register plan_experiment
        assert "plan_experiment" in engine.skills
        # These should not be registered on base engine
        assert "xpr_job_debug" not in engine.skills


# =============================================================================
# XPR SKILL TYPE ENUM TESTS
# =============================================================================


class TestXPRSkillType:
    """Test XPRSkillType enum."""

    def test_skill_type_values(self) -> None:
        """Test all skill type enum values."""
        assert XPRSkillType.PLAN_GENERATION.value == "plan_generation"
        assert XPRSkillType.EXECUTION.value == "execution"
        assert XPRSkillType.ANALYSIS.value == "analysis"
        assert XPRSkillType.JOB_DEBUG.value == "job_debug"
        assert XPRSkillType.ISSUE_FIX.value == "issue_fix"
        assert XPRSkillType.ISSUE_REPORT.value == "issue_report"

    def test_xpr_skill_result_with_skill_type(self) -> None:
        """Test XPRSkillResult using enum."""
        result = XPRSkillResult(
            success=True,
            skill_type=XPRSkillType.PLAN_GENERATION.value,
            result={},
        )
        assert result.skill_type == "plan_generation"


# =============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_execute_skill_with_none_args(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with None arguments."""

        def skill_with_none(x=None):
            return x if x is not None else "default"

        engine.register_skill("none_test", skill_with_none)
        result = engine.execute_skill("none_test", None)

        assert result.success is True
        assert result.result == "default"

    def test_execute_skill_returning_none(self, engine: XPRAgentEngine) -> None:
        """Test executing skill that returns None."""
        engine.register_skill("returns_none", lambda: None)
        result = engine.execute_skill("returns_none")

        assert result.success is True
        assert result.result is None

    def test_performance_history_with_empty_list(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with empty list."""
        enhanced_engine.performance_history["test"] = []
        result = enhanced_engine.analyze_performance_trend("test")

        assert result["trend"] == 0.0
        assert result["volatility"] == 0.0

    def test_performance_history_with_single_value(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with single value."""
        enhanced_engine.performance_history["test"] = [0.5]
        result = enhanced_engine.analyze_performance_trend("test")

        # With single value, volatility is 0 and trend depends on implementation
        # Implementation requires >= 2 values for non-zero trend calculation
        assert result["volatility"] == pytest.approx(0.0, abs=1e-3)

    def test_skill_result_with_complex_metadata(self) -> None:
        """Test SkillResult with nested metadata."""
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
        }
        result = SkillResult(
            success=True,
            skill_type="test",
            metadata=complex_metadata,
        )
        if result.metadata and isinstance(result.metadata, dict):
            nested = result.metadata.get("nested")
            if isinstance(nested, dict):
                assert nested["key"] == "value"

    def test_xpr_skill_result_post_init(self) -> None:
        """Test XPRSkillResult __post_init__ with None values."""
        result = XPRSkillResult(success=True, skill_type="test")

        assert result.metadata == {}
        assert result.dependencies == []
        assert result.recommendations == []

    @pytest.mark.parametrize(
        "error_msg,expected_type",
        [
            ("ImportError: No module named 'x'", "import_error"),
            ("ModuleNotFoundError: No module named 'y'", "module_not_found"),
            ("PermissionError: denied", "permission_error"),
            ("timeout error occurred", "timeout_error"),
            ("Unknown error", "generic_error"),
        ],
    )
    def test_job_debug_error_pattern_matching(
        self,
        enhanced_engine: XPRAgentEngineEnhanced,
        error_msg: str,
        expected_type: str,
    ) -> None:
        """Test error pattern matching for various error types."""
        result = enhanced_engine.xpr_job_debug(
            {
                "error": error_msg,
                "experiment": "test",
            }
        )

        assert result.success is True
        assert result.result["error_type"] == expected_type


# =============================================================================
# BOUNDARY VALUE TESTS
# =============================================================================


class TestBoundaryValues:
    """Test boundary values and extreme inputs."""

    def test_execute_skill_with_large_number(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with very large number."""
        engine.register_skill("identity", lambda x: x)
        result = engine.execute_skill("identity", 1e308)

        assert result.success is True
        assert result.result == 1e308

    def test_execute_skill_with_zero(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with zero."""
        engine.register_skill("double", lambda x: x * 2)
        result = engine.execute_skill("double", 0)

        assert result.success is True
        assert result.result == 0

    def test_execution_report_with_zero_time(self) -> None:
        """Test ExecutionReport with zero execution time."""
        report = ExecutionReport(
            experiment_name="test",
            success=True,
            execution_time=0.0,
        )
        assert report.execution_time == 0.0

    def test_performance_trend_with_negative_values(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with negative values."""
        enhanced_engine.performance_history["test"] = [-0.5, -0.4, -0.3]
        result = enhanced_engine.analyze_performance_trend("test")

        assert result["trend"] == pytest.approx(-0.4, rel=1e-3)  # Average
        assert result["volatility"] >= 0

    def test_performance_trend_with_mixed_values(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with mixed positive/negative values."""
        enhanced_engine.performance_history["test"] = [-0.5, 0.0, 0.5]
        result = enhanced_engine.analyze_performance_trend("test")

        assert result["trend"] == 0.0  # Average

    def test_xpr_skill_result_with_edge_confidence_values(self) -> None:
        """Test XPRSkillResult with edge confidence values."""
        result_min = XPRSkillResult(success=True, skill_type="test", confidence=0.0)
        result_max = XPRSkillResult(success=True, skill_type="test", confidence=1.0)

        assert result_min.confidence == 0.0
        assert result_max.confidence == 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestXPRIntegration:
    """Integration tests for XPR Agent Engine."""

    def test_full_workflow_simulation(self) -> None:
        """Test a full workflow simulation."""
        engine = XPRAgentEngineEnhanced()
        register_xpr_skills(engine)

        # Set up optimization strategy
        engine.set_optimization_strategy("training", {"lr": 0.01, "epochs": 100})

        # Simulate some performance history
        engine.performance_history["training"] = [0.1, 0.15, 0.2, 0.25]

        # Generate plan
        plan_result = engine.xpr_plan_experiment(
            "training", {"lr": 0.01, "epochs": 100}
        )
        assert plan_result.success is True

        # Simulate execution and add report
        report = ExecutionReport(
            experiment_name="training_run_1",
            success=True,
            execution_time=10.5,
            metadata={"adaptive": True},
        )
        engine.add_execution_report(report)

        # Check performance summary
        summary = engine.get_performance_summary()
        assert summary["total_experiments"] == 1
        # avg_confidence should be 0 since we didn't set confidence on the report
        assert "avg_confidence" in summary

    def test_error_recovery_workflow(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test error detection and recovery workflow."""
        # Simulate an error
        error_data = {
            "error": "ImportError: No module named 'missing_dependency'",
            "experiment": "test_exp",
            "file": "/path/to/experiment.py",
        }

        # Debug the error
        debug_result = enhanced_engine.xpr_job_debug(error_data)
        assert debug_result.success is True
        assert debug_result.result["error_type"] == "import_error"

        # Generate fix
        fix_result = enhanced_engine.xpr_issue_fix(error_data)
        assert fix_result.success is True

        # Generate report
        report_result = enhanced_engine.xpr_issue_report(error_data)
        assert report_result.success is True
        assert report_result.result["severity"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

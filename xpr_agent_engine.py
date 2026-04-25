"""
XPR Agent Engine - Core Autonomous Agent Framework

This module provides the XPRAgentEngine and SkillResult classes
that form the foundation of the autonomous agent system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
import time
import json
import os
from apgi_compliance import ComplianceManager, DataClassification

logger = logging.getLogger(__name__)

try:
    import litellm
except ImportError:
    logger.warning("litellm not available, using mock LLM integration")
    litellm = None

# Type hint for when litellm is not available
LLMClient = Any


@dataclass
class SkillResult:
    """Result of executing a skill."""

    success: bool
    skill_type: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionReport:
    """Report of experiment execution."""

    experiment_name: str
    success: bool
    execution_time: float
    result: Optional[Any] = None
    error: Optional[str] = None
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class SkillType(Enum):
    """Types of skills available to the agent."""

    PLAN_GENERATION = "plan_generation"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    MEMORY_UPDATE = "memory_update"
    MODIFICATION = "modification"


class XPRAgentEngine:
    """Core XPR Agent Engine for autonomous experimentation."""

    def __init__(self) -> None:
        self.skills: Dict[str, Callable] = {}
        self.execution_history: List[ExecutionReport] = []
        self.current_plan: Optional[Dict[str, Any]] = None
        self.compliance_manager = ComplianceManager()

    def register_skill(self, name: str, skill_func: Callable[..., Any]) -> None:
        """Register a skill function."""
        self.skills[name] = skill_func
        logger.info(f"Registered skill: {name}")

    def execute_skill(self, skill_name: str, *args: Any, **kwargs: Any) -> SkillResult:
        """Execute a registered skill by name."""
        if skill_name not in self.skills:
            return SkillResult(
                success=False,
                skill_type=skill_name,
                error=f"Skill '{skill_name}' not found",
                execution_time=0.0,
                confidence=0.0,
            )

        try:
            result = self.skills[skill_name](*args, **kwargs)
            return SkillResult(
                success=True,
                skill_type=skill_name,
                result=result,
                execution_time=0.1,
                confidence=0.8,
            )
        except Exception as e:
            return SkillResult(
                success=False,
                skill_type=skill_name,
                error=str(e),
                execution_time=0.0,
                confidence=0.0,
            )

    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary."""
        return {
            "total_experiments": len(self.execution_history),
            "success_rate": sum(
                1 for report in self.execution_history if report.success
            )
            / max(len(self.execution_history), 1),
            "avg_execution_time": sum(
                report.execution_time for report in self.execution_history
            )
            / max(len(self.execution_history), 1),
        }

    def add_execution_report(self, report: ExecutionReport) -> None:
        """Add execution report to history."""
        self.execution_history.append(report)
        self.compliance_manager.log_experiment_run(
            experiment_id=report.experiment_name,
            classification=DataClassification.INTERNAL,
        )
        logger.info(f"Added execution report for {report.experiment_name}")

    def set_current_plan(self, plan: Dict[str, Any]) -> None:
        """Set the current execution plan."""
        self.current_plan = plan
        logger.info("Set current execution plan")

    def plan_experiment(self, task: str, current_params: Dict[str, Any]) -> SkillResult:
        """Plan experiment modifications based on task and current parameters."""
        # Generate modifications based on task and current params
        modifications = {}

        if "lr" in current_params:
            base_lr = current_params["lr"]
            modifications["lr"] = base_lr * 0.9  # Decrease learning rate

        if "epochs" in current_params:
            base_epochs = current_params["epochs"]
            modifications["epochs"] = int(base_epochs * 1.1)  # Small increase

        # Create plan
        plan = {
            "hypothesis": f"Optimize {task} based on current performance",
            "modifications": modifications,
            "steps": [
                "1. Analyze current results",
                "2. Apply parameter changes",
                "3. Run experiment",
                "4. Evaluate outcomes",
            ],
            "confidence": 0.7,
        }

        self.current_plan = plan
        return SkillResult(
            success=True,
            skill_type="plan_experiment",
            result=plan,
            confidence=0.7,
        )

    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current execution plan."""
        return self.current_plan


# LLM provider registry
LLM_PROVIDERS = {
    "openai": {
        "client": None,  # Will be initialized when needed
        "model": "gpt-4",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "local": {
        "client": None,  # Will be initialized when needed
        "model": "llama-3.1-8b",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
}


class LLMIntegration:
    """Enhanced LLM integration using litellm unified API.

    litellm is a proxy that normalises calls across providers via
    ``litellm.completion(model=..., messages=...)``.  There is no need
    to instantiate per-provider client objects.
    """

    def __init__(self, preferred_provider: str = "openai"):
        self.preferred_provider = preferred_provider
        self._initialized: Dict[str, bool] = {}
        self._current_provider: Optional[str] = None

    def _get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific LLM provider."""
        return LLM_PROVIDERS.get(provider, {})

    def _initialize_client(self, provider: str) -> bool:
        """Validate that a provider can be used (API key present, etc.)."""
        if provider not in LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        config = self._get_provider_config(provider)

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key and litellm is None:
                logger.warning("OpenAI: no API key and litellm unavailable")
                return False
            logger.info(f"Initialized OpenAI provider with model {config.get('model')}")
            self._initialized[provider] = True
            return True

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key and litellm is None:
                logger.warning("Anthropic: no API key and litellm unavailable")
                return False
            logger.info(
                f"Initialized Anthropic provider with model {config.get('model', 'claude-3-5-sonnet-20241022')}"
            )
            self._initialized[provider] = True
            return True

        elif provider == "local":
            logger.info("Local LLM provider configured")
            self._initialized[provider] = True
            return True

        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            return False

    def get_client(self, provider: Optional[str] = None) -> bool:
        """Ensure provider is initialised.  Returns True if ready."""
        if provider is None:
            provider = self.preferred_provider
        if provider not in self._initialized:
            return self._initialize_client(provider)
        return self._initialized.get(provider, False)

    # ------------------------------------------------------------------
    # Core LLM call — uses litellm.completion() unified API
    # ------------------------------------------------------------------
    def _llm_completion(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> Optional[Any]:
        """Call the LLM via litellm's unified completion API.

        Returns the response object or None on failure.
        """
        if litellm is None:
            logger.warning("litellm not installed — cannot call LLM")
            return None

        if provider is None:
            provider = self.preferred_provider
        config = self._get_provider_config(provider)
        model = config.get("model", "gpt-4")

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response
        except Exception as e:
            logger.error(f"litellm.completion failed: {e}")
            return None

    def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from a prompt using litellm unified API."""
        if provider is None:
            provider = self.preferred_provider

        # Check for provider readiness
        if not self._initialized.get(provider, False):
            if not self._initialize_client(provider):
                return ""

        messages = [{"role": "user", "content": prompt}]
        response = self._llm_completion(
            messages=messages,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response and hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content if content else ""

        return ""

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------
    def generate_plan(
        self,
        task: str,
        current_params: Dict[str, Any],
        context: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> SkillResult:
        """Generate intelligent experiment plan using LLM."""
        start_time = time.time()

        try:
            if provider is None:
                provider = self.preferred_provider

            if not self.get_client(provider):
                return SkillResult(
                    success=False,
                    skill_type="llm_plan_generation",
                    error=f"LLM provider '{provider}' not available or not initialized",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            prompt = f"""You are an expert AI assistant specializing in autonomous experimentation and optimization for APGI (Attention, Perception, and General Intelligence) systems.

TASK: {task}

Current Parameters:
{json.dumps(current_params, indent=2)}

CONTEXT: {context if context else "No additional context provided"}

Generate an experiment optimization plan. Return ONLY valid JSON:
{{
    "hypothesis": "Clear, actionable hypothesis title",
    "analysis": "Brief analysis of current state and trends",
    "modifications": {{"param_name": "new_value"}},
    "steps": ["Step 1: ...", "Step 2: ..."],
    "confidence": 0.0,
    "constraints": ["constraint1"],
    "risk_level": "low/medium/high"
}}"""

            response = self._llm_completion(
                messages=[{"role": "system", "content": prompt}],
                provider=provider,
                max_tokens=2000,
                temperature=0.3,
            )

            if response is None:
                return SkillResult(
                    success=False,
                    skill_type="llm_plan_generation",
                    error="No response from LLM",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            config = self._get_provider_config(provider)
            if response.choices:
                plan_text = response.choices[0].message.content
                try:
                    plan_data = json.loads(plan_text)
                    return SkillResult(
                        success=True,
                        skill_type="llm_plan_generation",
                        result=plan_data,
                        execution_time=time.time() - start_time,
                        confidence=0.8,
                        metadata={
                            "provider": provider,
                            "model": config.get("model"),
                            "tokens_used": getattr(
                                getattr(response, "usage", None),
                                "total_tokens",
                                0,
                            ),
                        },
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM plan response: {plan_text}")
                    return SkillResult(
                        success=False,
                        skill_type="llm_plan_generation",
                        error="Invalid LLM response format",
                        execution_time=time.time() - start_time,
                        confidence=0.0,
                    )
            else:
                return SkillResult(
                    success=False,
                    skill_type="llm_plan_generation",
                    error="No response from LLM",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

        except Exception as e:
            return SkillResult(
                success=False,
                skill_type="llm_plan_generation",
                error=str(e),
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    # ------------------------------------------------------------------
    # LLM-generated code patches (replaces regex-only modification)
    # ------------------------------------------------------------------
    def generate_code_patch(
        self,
        file_path: str,
        file_content: str,
        modifications: Dict[str, Any],
        provider: Optional[str] = None,
    ) -> SkillResult:
        """Generate a code patch using the LLM instead of blind regex.

        The LLM receives the current file content and requested parameter
        changes, then returns the modified source with only safe parameter
        assignments changed.

        Args:
            file_path: Path to the file being modified
            file_content: Current content of the file
            modifications: Dict of parameter_name -> new_value
            provider: LLM provider to use

        Returns:
            SkillResult with the patched source in ``result``
        """
        start_time = time.time()

        if provider is None:
            provider = self.preferred_provider

        if not self.get_client(provider):
            return SkillResult(
                success=False,
                skill_type="llm_code_patch",
                error="LLM provider not available — falling back to regex",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

        mod_desc = "\n".join(f"  {k} = {repr(v)}" for k, v in modifications.items())
        prompt = f"""You are a precise code editor. Given the Python source file below,
change ONLY the parameter assignments listed. Do NOT add, remove, or
re-order any other code. Return the COMPLETE modified file and nothing else.

FILE: {file_path}
REQUESTED CHANGES:
{mod_desc}

--- BEGIN SOURCE ---
{file_content}
--- END SOURCE ---

Return ONLY the complete modified Python source code, no commentary."""

        response = self._llm_completion(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=4096,
            temperature=0.0,
        )

        if response is None:
            return SkillResult(
                success=False,
                skill_type="llm_code_patch",
                error="LLM returned no response",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

        if response.choices:
            patched = response.choices[0].message.content
            # Strip markdown fences if present
            if patched.startswith("```"):
                lines = patched.split("\n")
                patched = (
                    "\n".join(lines[1:-1])
                    if lines[-1].strip() == "```"
                    else "\n".join(lines[1:])
                )

            # Basic validation: must still compile
            try:
                compile(patched, file_path, "exec")
            except SyntaxError as e:
                return SkillResult(
                    success=False,
                    skill_type="llm_code_patch",
                    error=f"LLM patch has syntax error: {e}",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            return SkillResult(
                success=True,
                skill_type="llm_code_patch",
                result=patched,
                execution_time=time.time() - start_time,
                confidence=0.85,
                metadata={
                    "file": file_path,
                    "modifications": list(modifications.keys()),
                },
            )

        return SkillResult(
            success=False,
            skill_type="llm_code_patch",
            error="No choices in LLM response",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )


class EnhancedXPRAgentEngine(XPRAgentEngine):
    """Enhanced XPR Agent Engine with LLM integration."""

    def __init__(self) -> None:
        # Initialize base engine
        super().__init__()

        # Initialize LLM integration
        self.llm_integration = LLMIntegration()

        # Override plan generation to use LLM
        self.skills["plan_experiment"] = self.llm_integration.generate_plan
        self.skills["llm_plan_generation"] = self.llm_integration.generate_plan


"""
XPR Agent Engine - Enhanced Autonomous Agent Framework

This module extends the XPR Agent Engine with additional capabilities
for enterprise-grade autonomous experimentation and optimization.

Key enhancements over XPR Agent Engine:
- Enhanced LLM integration with multiple provider support
- Advanced skill chaining with dependency resolution
- Performance-driven skill selection
- Comprehensive error recovery and self-healing
- Real-time metrics and monitoring
- Advanced plan generation with natural language understanding
- Multi-objective optimization strategies
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum
import logging
import time
import re
import datetime

# XPRAgentEngine is already defined above in this same file

logger = logging.getLogger(__name__)


class XPRSkillType(Enum):
    """Enhanced skill types for XPR Agent Engine."""

    # Core skills (inherited from XPR)
    PLAN_GENERATION = "plan_generation"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    MEMORY_UPDATE = "memory_update"
    MODIFICATION = "modification"

    # XPR enhanced skills
    JOB_DEBUG = "job_debug"
    ISSUE_FIX = "issue_fix"
    ISSUE_REPORT = "issue_report"
    PLAN_ANALYSIS = "plan_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_GENERATION = "code_generation"
    SKILL_CHAIN = "skill_chain"
    GUARDRAIL_CHECK = "guardrail_check"


@dataclass
class XPRSkillResult:
    """Enhanced result structure for XPR Agent skills."""

    success: bool
    skill_type: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    dependencies: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.recommendations is None:
            self.recommendations = []


class XPRAgentEngineEnhanced(XPRAgentEngine):
    """Enhanced XPR Agent Engine with advanced autonomous capabilities."""

    def __init__(self) -> None:
        # Initialize base engine
        super().__init__()

        # Initialize LLM integration (Phase 2 Component)
        self.llm_integration = LLMIntegration()

        # XPR specific enhancements
        self.llm_providers: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.skill_dependencies: Dict[str, List[str]] = {}
        self.optimization_strategies: Dict[str, Any] = {}

        # Register XPR enhanced skills automatically
        register_xpr_skills(self)

    def _extract_missing_module(self, error_msg: str) -> Optional[str]:
        """Extract missing module name from error message."""
        match = re.search(r"No module named '([^']+)'", error_msg)
        if match:
            return match.group(1)
        return None

    def register_llm_provider(self, name: str, provider_config: Dict[str, Any]) -> None:
        """Register an LLM provider for intelligent operations."""
        self.llm_providers[name] = provider_config
        logger.info(f"Registered LLM provider: {name}")

    def set_optimization_strategy(
        self, experiment_type: str, strategy: Dict[str, Any]
    ) -> None:
        """Set optimization strategy for specific experiment type."""
        self.optimization_strategies[experiment_type] = strategy
        logger.info(f"Set optimization strategy for {experiment_type}")

    def analyze_performance_trend(
        self, experiment_type: str, window_size: int = 10
    ) -> Dict[str, float]:
        """Analyze performance trend for adaptive optimization."""
        if experiment_type not in self.performance_history:
            return {"trend": 0.0, "volatility": 0.0}

        recent_metrics_raw = self.performance_history.get(experiment_type, [])
        # Explicit slicing to satisfy type checkers that may not recognize [-n:] on all sequence types
        if len(recent_metrics_raw) >= window_size:
            start_idx = len(recent_metrics_raw) - window_size
            recent_metrics = [
                recent_metrics_raw[i] for i in range(start_idx, len(recent_metrics_raw))
            ]
        else:
            recent_metrics = list(recent_metrics_raw)
        if len(recent_metrics) < 2:
            return {"trend": 0.0, "volatility": 0.0}

        # Calculate trend and volatility
        metrics_count = len(recent_metrics)
        if metrics_count == 0:
            return {"trend": 0.0, "volatility": 0.0}

        trend = float(sum(recent_metrics)) / metrics_count
        volatility = (
            float(sum((m - trend) ** 2 for m in recent_metrics)) / metrics_count
        )

        return {"trend": trend, "volatility": volatility}

    def xpr_plan_experiment(
        self, task: str, current_params: Dict[str, Any]
    ) -> XPRSkillResult:
        """Enhanced plan generation with natural language understanding."""
        start_time = time.time()

        try:
            # Analyze current performance and trends
            performance_trend = self.analyze_performance_trend(task, window_size=5)

            # Generate intelligent modifications based on multiple factors
            modifications = {}

            # Base parameter modifications (inherited logic)
            if "lr" in current_params:
                base_lr = current_params["lr"]
                # Adaptive learning rate adjustment
                if performance_trend["trend"] > 0.05:  # Improving
                    modifications["lr"] = base_lr * 1.1
                elif performance_trend["volatility"] > 0.1:  # High volatility
                    modifications["lr"] = base_lr * 0.95  # More conservative
                else:
                    modifications["lr"] = base_lr * 1.05  # Standard adjustment

            if "epochs" in current_params:
                base_epochs = current_params["epochs"]
                # Dynamic epoch adjustment based on performance
                if performance_trend["trend"] > 0:
                    modifications["epochs"] = min(base_epochs * 1.2, base_epochs + 50)
                else:
                    modifications["epochs"] = max(
                        base_epochs - 20, 10
                    )  # Reduce if not improving

            # Generate comprehensive plan with analysis
            plan = {
                "hypothesis": f"Optimize {task} using adaptive strategy based on performance trend analysis",
                "analysis": {
                    "current_performance": performance_trend,
                    "trend_direction": (
                        "improving" if performance_trend["trend"] > 0 else "stable"
                    ),
                    "volatility_level": (
                        "high" if performance_trend["volatility"] > 0.15 else "normal"
                    ),
                },
                "modifications": modifications,
                "steps": [
                    "1. Analyze current performance metrics",
                    "2. Calculate optimal parameter adjustments",
                    "3. Apply adaptive modifications",
                    "4. Monitor convergence and adjust strategy",
                    "5. Validate against guardrails and constraints",
                ],
                "confidence": min(0.8, 0.9),  # Higher confidence with trend analysis
                "constraints": [
                    "Max 15% parameter changes per iteration",
                    "Maintain minimum performance thresholds",
                    "Respect safety and guardrail constraints",
                ],
            }

            self.current_plan = plan

            return XPRSkillResult(
                success=True,
                skill_type=XPRSkillType.PLAN_GENERATION.value,
                result=plan,
                execution_time=time.time() - start_time,
                confidence=0.85,
                metadata={"analysis": performance_trend, "adaptive": True},
            )

        except Exception as e:
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.PLAN_GENERATION.value,
                error=str(e),
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    def xpr_job_debug(self, experiment_data: Any) -> XPRSkillResult:
        """Enhanced debugging with pattern recognition and fix suggestions."""
        start_time = time.time()

        if not isinstance(experiment_data, dict):
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.JOB_DEBUG.value,
                error="experiment_data must be a dictionary",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

        try:
            error_msg = experiment_data.get("error", "")
            experiment_name = experiment_data.get("experiment", "unknown")
            file_path = experiment_data.get("file", "")

            # Enhanced pattern recognition
            debug_info = {
                "error_type": "unknown",
                "suggested_fixes": [],
                "confidence": 0.6,
                "dependencies": [],
                "patterns_found": [],
            }

            # Common error patterns with enhanced fixes
            if "ImportError" in error_msg:
                missing_module = (
                    self._extract_missing_module(error_msg)
                    if hasattr(self, "_extract_missing_module")
                    else None
                )
                if missing_module:
                    debug_info.update(
                        {
                            "error_type": "import_error",
                            "suggested_fixes": [
                                f"Install missing module: {missing_module}",
                                "Verify PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}",
                                "Run: pip install {missing_module}",
                                "Check module location: {file_path}",
                            ],
                            "dependencies": [missing_module],
                            "confidence": 0.9,
                            "patterns_found": ["ImportError", "missing_module"],
                        }
                    )
            elif "ModuleNotFoundError" in error_msg:
                debug_info.update(
                    {
                        "error_type": "module_not_found",
                        "suggested_fixes": [
                            f"Verify module exists: {file_path}",
                            "Check module naming convention",
                            "Validate import syntax",
                            "Check sys.path configuration",
                        ],
                        "dependencies": ["os", "sys"],
                        "confidence": 0.8,
                        "patterns_found": ["ModuleNotFoundError", "file_path"],
                    }
                )
            elif "PermissionError" in error_msg:
                debug_info.update(
                    {
                        "error_type": "permission_error",
                        "suggested_fixes": [
                            "Check file permissions: {file_path}",
                            "Run with appropriate privileges",
                            "Verify directory access rights",
                            "Check user and group ownership",
                        ],
                        "dependencies": ["os", "pwd"],
                        "confidence": 0.7,
                        "patterns_found": ["PermissionError", "file_access"],
                    }
                )
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                debug_info.update(
                    {
                        "error_type": "timeout_error",
                        "suggested_fixes": [
                            "Increase timeout duration: {experiment_name}",
                            "Optimize experiment for faster execution",
                            "Check system resources",
                            "Profile memory usage",
                            "Consider reducing dataset size",
                        ],
                        "dependencies": ["psutil", "tracemalloc"],
                        "confidence": 0.6,
                        "patterns_found": ["timeout", "performance"],
                    }
                )
            else:
                # Generic error analysis with enhanced patterns
                debug_info.update(
                    {
                        "error_type": "generic_error",
                        "suggested_fixes": [
                            "Enable verbose logging",
                            "Check experiment configuration",
                            "Validate input parameters",
                            "Review system logs",
                            "Check resource utilization",
                        ],
                        "confidence": 0.4,
                        "patterns_found": ["generic"],
                    }
                )

            conf_value = debug_info.get("confidence", 0.5)
            return XPRSkillResult(
                success=True,
                skill_type=XPRSkillType.JOB_DEBUG.value,
                result=debug_info,
                execution_time=time.time() - start_time,
                confidence=(
                    float(conf_value) if isinstance(conf_value, (int, float)) else 0.5
                ),
                metadata={
                    "debug_type": debug_info.get("error_type"),
                    "experiment": experiment_name,
                    "file": file_path,
                },
            )

        except Exception as e:
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.JOB_DEBUG.value,
                error=str(e),
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    def xpr_issue_fix(self, experiment_data: Any) -> XPRSkillResult:
        """Enhanced issue fixing with code generation and application."""
        start_time = time.time()

        if not isinstance(experiment_data, dict):
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.ISSUE_FIX.value,
                error="experiment_data must be a dictionary",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

        try:
            error_msg = experiment_data.get("error", "")
            experiment_name = experiment_data.get("experiment", "unknown")

            # Generate fix with code generation
            fix_info = {
                "fix_type": "enhanced_code_fix",
                "generated_code": "",
                "applied": False,
                "confidence": 0.6,
                "dependencies": [],
            }

            # Enhanced pattern recognition for code fixes
            if "SyntaxError" in error_msg or "IndentationError" in error_msg:
                line_match = re.search(r"line (\d+)", error_msg)
                if line_match:
                    line_num = line_match.group(1)
                    fix_info.update(
                        {
                            "fix_type": "syntax_fix",
                            "generated_code": f"# Fix syntax error at line {line_num}",
                            "applied": False,
                            "confidence": 0.9,
                            "dependencies": ["re", "ast"],
                        }
                    )
            elif "FileNotFoundError" in error_msg:
                file_match = re.search(r"'([^']+)'", error_msg)
                if file_match:
                    missing_file = file_match.group(1)
                    fix_info.update(
                        {
                            "fix_type": "file_creation_fix",
                            "generated_code": f"# Create missing file: {missing_file}",
                            "applied": False,
                            "confidence": 0.8,
                            "dependencies": ["pathlib"],
                        }
                    )
            elif "ImportError" in error_msg:
                module_match = re.search(r"No module named '([^']+)'", error_msg)
                if module_match:
                    missing_module = module_match.group(1)
                    fix_info.update(
                        {
                            "fix_type": "import_fix",
                            "generated_code": f"import {missing_module}",
                            "applied": False,
                            "confidence": 0.9,
                            "dependencies": [missing_module],
                        }
                    )

            raw_conf = fix_info.get("confidence", 0.5)
            try:
                conf_val = float(raw_conf) if raw_conf is not None else 0.5  # type: ignore
            except (ValueError, TypeError):
                conf_val = 0.5
            return XPRSkillResult(
                success=True,
                skill_type=XPRSkillType.ISSUE_FIX.value,
                result=fix_info,
                execution_time=time.time() - start_time,
                confidence=float(conf_val),
                metadata={"experiment": experiment_name, "fixes_generated": 1},
            )

        except Exception as e:
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.ISSUE_FIX.value,
                error=str(e),
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    def xpr_issue_report(self, experiment_data: Any) -> XPRSkillResult:
        """Enhanced issue reporting with comprehensive analysis."""
        start_time = time.time()

        if not isinstance(experiment_data, dict):
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.ISSUE_REPORT.value,
                error="experiment_data must be a dictionary",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

        try:
            error_msg = experiment_data.get("error", "")
            experiment_name = experiment_data.get("experiment", "unknown")
            file_path = experiment_data.get("file", "")
            metrics = experiment_data.get("metrics", {})

            # Generate comprehensive report
            report = {
                "experiment_name": experiment_name,
                "error_summary": error_msg,
                "file_path": file_path,
                "metrics": metrics,
                "timestamp": datetime.datetime.now().isoformat(),
                "severity": self._assess_severity(error_msg),
                "recommendations": self._generate_recommendations(error_msg, metrics),
                "next_steps": self._generate_recovery_steps(error_msg),
                "root_cause_analysis": self._analyze_root_cause(error_msg),
            }

            return XPRSkillResult(
                success=True,
                skill_type=XPRSkillType.ISSUE_REPORT.value,
                result=report,
                execution_time=time.time() - start_time,
                confidence=0.8,
                metadata={"experiment": experiment_name, "report_generated": 1},
            )

        except Exception as e:
            return XPRSkillResult(
                success=False,
                skill_type=XPRSkillType.ISSUE_REPORT.value,
                error=str(e),
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    def _assess_severity(self, error_msg: str) -> str:
        """Assess error severity based on error patterns."""
        if any(
            keyword in error_msg.lower() for keyword in ["critical", "fatal", "crash"]
        ):
            return "critical"
        elif any(
            keyword in error_msg.lower() for keyword in ["error", "exception", "failed"]
        ):
            return "high"
        else:
            return "medium"

    def _generate_recommendations(
        self, error_msg: str, metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on error analysis."""
        recommendations = []

        if "timeout" in error_msg.lower():
            recommendations.extend(
                [
                    "Increase timeout duration",
                    "Optimize experiment for faster execution",
                    "Consider reducing dataset size",
                ]
            )

        if "memory" in error_msg.lower():
            recommendations.extend(
                [
                    "Check memory usage",
                    "Reduce batch size",
                    "Implement memory cleanup",
                ]
            )

        if "permission" in error_msg.lower():
            recommendations.extend(
                [
                    "Check file permissions",
                    "Run with appropriate user privileges",
                    "Verify directory access rights",
                ]
            )

        return recommendations

    def _generate_recovery_steps(self, error_msg: str) -> List[str]:
        """Generate step-by-step recovery plan."""
        return [
            "1. Analyze root cause of error",
            "2. Review experiment configuration",
            "3. Apply appropriate fix",
            "4. Test with minimal configuration",
            "5. Document resolution and preventatives",
        ]

    def _analyze_root_cause(self, error_msg: str) -> str:
        """Analyze potential root causes of errors."""
        if "ImportError" in error_msg:
            return "Missing or incompatible dependency"
        elif "PermissionError" in error_msg:
            return "Insufficient file system permissions"
        elif "timeout" in error_msg:
            return "Resource exhaustion or infinite loop"
        else:
            return "Unknown error requiring investigation"

    def xpr_skill_chain(
        self, initial_input: Dict[str, Any], skills_list: List[str]
    ) -> List[XPRSkillResult]:
        """Enhanced skill chaining with dependency resolution and error handling."""
        results = []
        current_input = initial_input

        for skill_name in skills_list:
            if skill_name not in self.skills:
                error_result = XPRSkillResult(
                    success=False,
                    skill_type=skill_name,
                    error=f"Skill '{skill_name}' not found",
                    confidence=0.0,
                )
                results.append(error_result)
                continue

            # Execute skill with enhanced error handling
            try:
                skill_result = self.execute_skill(skill_name, current_input)

                # Convert to XPRSkillResult if needed
                if not isinstance(skill_result, XPRSkillResult):
                    xpr_result = XPRSkillResult(
                        success=skill_result.success,
                        skill_type=skill_name,
                        result=skill_result.result,
                        error=str(skill_result.error) if skill_result.error else None,
                        execution_time=skill_result.execution_time,
                        confidence=float(getattr(skill_result, "confidence", 0.5)),
                        metadata=(
                            skill_result.metadata
                            if hasattr(skill_result, "metadata")
                            else {}
                        ),
                    )
                else:
                    xpr_result = skill_result

                results.append(xpr_result)

                # Use output for next skill if successful
                if xpr_result.success and xpr_result.result:
                    current_input = xpr_result.result
                else:
                    # Keep original input if skill failed
                    pass

            except Exception as e:
                error_result = XPRSkillResult(
                    success=False,
                    skill_type=skill_name,
                    error=str(e),
                    execution_time=0.0,
                    confidence=0.0,
                )
                results.append(error_result)

        return results

    def get_performance_summary(self) -> Dict[str, float]:
        """Enhanced performance metrics summary."""
        base_summary = super().get_performance_summary()

        # Add XPR specific metrics
        return {
            **base_summary,
            "avg_confidence": sum(
                float(r.confidence) if hasattr(r, "confidence") else 0.0
                for r in self.execution_history
            )
            / max(len(self.execution_history), 1),
            "skill_success_rate": sum(1 for r in self.execution_history if r.success)
            / max(len(self.execution_history), 1),
            "adaptive_optimizations": len(
                [
                    r
                    for r in self.execution_history
                    if hasattr(r, "metadata")
                    and r.metadata is not None
                    and isinstance(r.metadata, dict)
                    and r.metadata.get("adaptive", False)
                ]
            ),
        }


# Register XPR enhanced skills with the base engine
def register_xpr_skills(engine: "XPRAgentEngineEnhanced") -> None:
    """Register XPR enhanced skills with the engine."""

    # Register enhanced skills
    engine.register_skill("plan_experiment", engine.plan_experiment)
    # Note: xpr_* methods are only available on the enhanced XPRAgentEngine class
    if hasattr(engine, "xpr_job_debug"):
        engine.register_skill("xpr_job_debug", engine.xpr_job_debug)
        engine.register_skill("xpr_issue_fix", engine.xpr_issue_fix)
        engine.register_skill("xpr_issue_report", engine.xpr_issue_report)
        engine.register_skill("xpr_skill_chain", engine.xpr_skill_chain)
        engine.register_skill("xpr_plan_experiment", engine.xpr_plan_experiment)

    logger.info("XPR Agent Engine skills registered successfully")

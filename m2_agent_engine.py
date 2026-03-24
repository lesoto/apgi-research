import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result from a single atomic skill execution."""

    skill_name: str
    success: bool
    output: str
    metadata: Dict[str, Any]


class M2AgentEngine:
    """
    The M2* Agent Engine (Execution Environment + LLM Integration).
    Handles reading logs, profiling runs, formulating hypotheses, and code modifications.
    """

    def __init__(self, llm_api_key: Optional[str] = None):
        """Initialize the M2 Agent Engine with LLM configuration."""
        self.api_key = llm_api_key or os.environ.get("LLM_API_KEY", "dummy_key")
        self.skills: Dict[str, Callable[[Any], SkillResult]] = {
            "read_logs": self.read_logs,
            "patch_source_code": self.patch_source_code,
            "evaluate_metrics": self.evaluate_metrics,
            "job_debug": self.job_debug,
            "issue_fix": self.issue_fix,
            "issue_report": self.issue_report,
            "plan_experiment": self.plan_experiment,
            "analyze_results": self.analyze_results,
        }

    def _call_llm(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Actual LLM inference integration using litellm.
        """
        import json

        try:
            import litellm

            logger.info(f"LLM Prompt: {prompt[:100]}...")
            messages = [
                {
                    "role": "system",
                    "content": "You are the M2* AI Agent. You read execution logs, design experiments, output JSON patches and evaluate metrics.",
                }
            ]
            if context:
                messages.append(
                    {"role": "system", "content": f"Context: {json.dumps(context)}"}
                )
            messages.append({"role": "user", "content": prompt})

            model = os.environ.get("LLM_MODEL", "gpt-4o")
            api_key = self.api_key if self.api_key != "dummy_key" else None

            response = litellm.completion(
                model=model, messages=messages, api_key=api_key
            )
            return response.choices[0].message.content
        except ImportError:
            logger.warning("litellm not installed. Using mock response.")
            return '{"modifications": {"BASE_LEARNING_RATE": 0.05}, "hypothesis": "Mock LLM hypothesis"}'
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return '{"modifications": {}, "hypothesis": "Error making LLM call"}'

    # ==========================================
    # ATOMIC SKILLS
    # ==========================================

    def read_logs(self, log_path: str = "autonomous_agent.log") -> SkillResult:
        """Atomic Skill: Read and parse log files from the environment."""
        try:
            if not os.path.exists(log_path):
                return SkillResult(
                    "read_logs", False, f"File {log_path} not found.", {}
                )

            with open(log_path, "r") as f:
                # Read last 50 lines for context
                lines = f.readlines()[-50:]
                log_content = "".join(lines)

            return SkillResult(
                "read_logs", True, log_content, {"lines_read": len(lines)}
            )
        except Exception as e:
            return SkillResult("read_logs", False, str(e), {})

    def patch_source_code(self, patch_instructions: Dict[str, str]) -> SkillResult:
        """
        Atomic Skill: Apply code modifications with validation.

        patch_instructions keys:
            - filepath (str): Target file to patch.
            - old_code (str, optional): Exact text to replace. If absent, 'content'
              is appended or the LLM is asked to generate a diff.
            - new_code (str, optional): Replacement text for old_code.
            - content (str, optional): Raw LLM output describing the patch (used
              as fallback context for LLM-based diff generation).

        The function:
            1. Creates a .bak backup of the original file.
            2. Applies the text replacement (old_code → new_code).
            3. Validates syntax with py_compile.
            4. Optionally validates style with flake8.
            5. Rolls back the file on any validation failure.
        """
        import py_compile
        import shutil

        filepath = patch_instructions.get("filepath", "")
        old_code = patch_instructions.get("old_code", "")
        new_code = patch_instructions.get("new_code", "")

        # --- Validation -------------------------------------------------
        if not filepath:
            return SkillResult(
                "patch_source_code", False, "No filepath provided", {"status": "error"}
            )

        if not os.path.exists(filepath):
            return SkillResult(
                "patch_source_code",
                False,
                f"File not found: {filepath}",
                {"status": "error"},
            )

        if not filepath.endswith(".py"):
            return SkillResult(
                "patch_source_code",
                False,
                "Only .py files can be patched",
                {"status": "error"},
            )

        # --- Read original -----------------------------------------------
        try:
            with open(filepath, "r") as f:
                original_content = f.read()
        except Exception as e:
            return SkillResult(
                "patch_source_code",
                False,
                f"Failed to read {filepath}: {e}",
                {"status": "error"},
            )

        # --- Build patched content ---------------------------------------
        if old_code and new_code:
            if old_code not in original_content:
                return SkillResult(
                    "patch_source_code",
                    False,
                    f"old_code snippet not found in {filepath}",
                    {"status": "error", "old_code_preview": old_code[:120]},
                )
            patched_content = original_content.replace(old_code, new_code, 1)
        elif new_code:
            # No old_code → append new_code at end of file
            patched_content = original_content.rstrip() + "\n\n" + new_code + "\n"
        else:
            # Fallback: ask LLM to generate a concrete diff
            llm_patch = self._call_llm(
                f"Given this file content, generate a concrete old_code and new_code replacement pair as JSON "
                f"(keys: old_code, new_code) for the following request:\n{patch_instructions.get('content', '')}\n\n"
                f"File ({filepath}) first 200 lines:\n{original_content[:8000]}"
            )
            try:
                import json as _json

                patch_data = _json.loads(llm_patch)
                old_c = patch_data.get("old_code", "")
                new_c = patch_data.get("new_code", "")
                if old_c and new_c and old_c in original_content:
                    patched_content = original_content.replace(old_c, new_c, 1)
                else:
                    return SkillResult(
                        "patch_source_code",
                        False,
                        f"LLM-generated patch could not be applied. LLM output: {llm_patch[:300]}",
                        {"status": "error"},
                    )
            except Exception:
                return SkillResult(
                    "patch_source_code",
                    False,
                    f"Failed to parse LLM patch output: {llm_patch[:300]}",
                    {"status": "error"},
                )

        # --- Create backup -----------------------------------------------
        backup_path = filepath + ".bak"
        try:
            shutil.copy2(filepath, backup_path)
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")

        # --- Write patched file ------------------------------------------
        try:
            with open(filepath, "w") as f:
                f.write(patched_content)
        except Exception as e:
            # Restore from backup
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, filepath)
            return SkillResult(
                "patch_source_code",
                False,
                f"Failed to write patched file: {e}",
                {"status": "error"},
            )

        # --- Validate syntax with py_compile -----------------------------
        try:
            py_compile.compile(filepath, doraise=True)
        except py_compile.PyCompileError as e:
            # Rollback
            logger.error(f"Syntax validation failed, rolling back: {e}")
            shutil.copy2(backup_path, filepath)
            return SkillResult(
                "patch_source_code",
                False,
                f"Syntax error in patched file (rolled back): {e}",
                {"status": "rollback"},
            )

        # --- Optional flake8 validation ----------------------------------
        flake8_issues: List[str] = []
        try:
            flake8_result = subprocess.run(
                ["flake8", "--max-line-length=120", "--count", filepath],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if flake8_result.returncode != 0:
                flake8_issues = flake8_result.stdout.strip().split("\n")[-5:]
                logger.warning(f"flake8 warnings (non-blocking): {flake8_issues}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # flake8 not installed or timed out — skip
            pass

        # --- Cleanup backup on success -----------------------------------
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except OSError:
            pass

        metadata: Dict[str, Any] = {"status": "success", "filepath": filepath}
        if flake8_issues:
            metadata["flake8_warnings"] = flake8_issues

        return SkillResult(
            "patch_source_code",
            True,
            f"Successfully patched {filepath}",
            metadata,
        )

    def evaluate_metrics(self, data: Dict[str, Any]) -> SkillResult:
        """Atomic Skill: Evaluate provided metrics to detect regressions."""
        # Process the dict
        # Call LLM to summarize metric significance
        summary = self._call_llm(f"Evaluate these metrics: {data}")
        return SkillResult("evaluate_metrics", True, summary, {"metrics": data})

    def plan_experiment(self, task: str, current_params: Dict[str, Any]) -> SkillResult:
        """Atomic Skill: Generate an ExperimentPlan."""
        plan_str = self._call_llm(
            f"Task: {task}. Params: {current_params}. Create plan."
        )

        # Mocks generating a JSON plan structure
        modifications = {}
        for k, v in current_params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                modifications[k] = v * 1.05  # simplistic mock change

        plan_dict = {
            "hypothesis": f"Optimizing variables for {task}",
            "modifications": modifications,
            "success_metrics": {"primary_metric": "improvement"},
            "constraints": ["Keep bounds valid"],
            "steps": ["Apply patch", "Run experiment", "Analyze"],
        }
        return SkillResult("plan_experiment", True, plan_str, {"plan": plan_dict})

    def analyze_results(self, metrics: Dict[str, Any]) -> SkillResult:
        """Atomic Skill: Compute anomaly report and write natural language summary.

        Accepts enriched metrics context:
            - delta (float): Change from previous iteration.
            - primary_metric (float): Current metric value.
            - status (str): Experiment status (success/crash/timeout).
            - experiment (str): Experiment name.
            - iteration (int): Current iteration number.
            - modifications (dict): Parameters that were changed.
            - performance_history (list[float]): Recent metric values.
        """
        delta = metrics.get("delta", 0.0)
        status = metrics.get("status", "unknown")
        experiment = metrics.get("experiment", "unknown")
        primary_metric = metrics.get("primary_metric", 0.0)
        iteration = metrics.get("iteration", -1)
        modifications = metrics.get("modifications", {})
        perf_history = metrics.get("performance_history", [])

        # Multi-signal confidence scoring
        if status != "success":
            confidence = 0.1
        elif delta > 0:
            confidence = min(0.95, 0.7 + abs(delta) * 0.5)
        elif delta == 0:
            confidence = 0.5
        else:
            confidence = max(0.1, 0.4 - abs(delta) * 0.3)

        # Detect regression trend (3+ consecutive declines)
        if len(perf_history) >= 3:
            recent = perf_history[-3:]
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                confidence = min(confidence, 0.25)

        # Build rich context for LLM analysis
        analysis_prompt = (
            f"Analyze the outcome of experiment '{experiment}' (iteration {iteration}).\n"
            f"Status: {status}\n"
            f"Primary Metric: {primary_metric:.4f}\n"
            f"Delta from previous: {delta:+.4f}\n"
            f"Modifications applied: {modifications}\n"
            f"Recent performance trend: {perf_history}\n"
            f"Computed confidence: {confidence:.2f}\n\n"
            f"Provide a brief analysis: Is this an improvement? Any anomalies? Recommendations?"
        )
        analysis = self._call_llm(analysis_prompt)

        return SkillResult(
            "analyze_results",
            True,
            analysis,
            {
                "confidence": confidence,
                "delta": delta,
                "primary_metric": primary_metric,
                "status": status,
            },
        )

    # ==========================================
    # HIERARCHICAL SKILL CHAINS
    # ==========================================

    def job_debug(self, context: Dict[str, Any]) -> SkillResult:
        """Chain Skill: Read logs and identify the root cause."""
        log_res = self.read_logs()
        analysis = self._call_llm(
            f"Debug this context: {context}\nLogs: {log_res.output}"
        )
        return SkillResult("job_debug", True, analysis, {"dependencies": ["read_logs"]})

    def issue_fix(self, debug_output: str) -> SkillResult:
        """Chain Skill: Given a root cause, apply a source code patch."""
        patch_plan = self._call_llm(f"Draft patch for: {debug_output}")
        # Normally this would feed into patch_source_code
        patch_res = self.patch_source_code(
            {"filepath": "dummy.py", "content": patch_plan}
        )
        return SkillResult("issue_fix", patch_res.success, patch_res.output, {})

    def issue_report(self, fix_output: str) -> SkillResult:
        """Chain Skill: Document the implemented fixes and their impact."""
        report = self._call_llm(f"Write execution report for fix: {fix_output}")
        return SkillResult("issue_report", True, report, {})

    def run_skill_chain(
        self, initial_input: Any, skills_list: List[str]
    ) -> List[SkillResult]:
        """
        Execute a sequence of skills consecutively, passing the output of one
        as the context for the next.
        """
        results = []
        current_input = initial_input

        for skill_name in skills_list:
            if skill_name not in self.skills:
                logger.error(f"Skill {skill_name} not found.")
                break

            skill_func: Callable[[Any], SkillResult] = self.skills[skill_name]
            result = skill_func(current_input)
            results.append(result)

            if not result.success:
                logger.warning(f"Skill {skill_name} failed. Halting chain.")
                break

            current_input = result.output

        return results

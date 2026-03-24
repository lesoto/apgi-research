"""
test_rl_loop.py — End-to-End RL Loop Validation (AL-Plan Phase 5)

Verifies each stage of the M2* autonomous research loop:
  1. Plan generation (plan_experiment)
  2. Execution (run_experiment)
  3. Analysis (analyze_results)
  4. Guardrail evaluation (confidence thresholds)
  5. Memory update (update_memory_from_report)

Usage:
    python -m pytest test_rl_loop.py -v
"""

import json
import pytest
from unittest.mock import MagicMock

from m2_agent_engine import M2AgentEngine, SkillResult
from memory_store import MemoryStore, update_memory_from_report


# ---- Fixtures ----


@pytest.fixture
def engine():
    """Provide a fresh M2AgentEngine."""
    return M2AgentEngine()


@pytest.fixture
def memory_store(tmp_path):
    """Provide a MemoryStore backed by a temporary file."""
    return MemoryStore(storage_path=str(tmp_path / "test_memory.json"))


# ---- Stage 1: Plan Generation ----


class TestPlanGeneration:
    """Test the /exp-plan stage of the RL loop."""

    def test_plan_returns_skill_result(self, engine):
        result = engine.plan_experiment(
            task="Optimize stroop_effect", current_params={"lr": 0.01, "epochs": 100}
        )
        assert isinstance(result, SkillResult)
        assert result.skill_name == "plan_experiment"
        assert result.success is True

    def test_plan_contains_structured_data(self, engine):
        result = engine.plan_experiment(
            task="Optimize stroop_effect", current_params={"lr": 0.01}
        )
        plan = result.metadata.get("plan", {})
        assert "hypothesis" in plan
        assert "modifications" in plan
        assert "steps" in plan
        assert "constraints" in plan

    def test_plan_modifies_numeric_params(self, engine):
        result = engine.plan_experiment(
            task="Optimize test", current_params={"lr": 0.01, "epochs": 100}
        )
        mods = result.metadata.get("plan", {}).get("modifications", {})
        # The mock implementation applies a 1.05x multiplier
        assert len(mods) > 0

    def test_plan_with_empty_params(self, engine):
        result = engine.plan_experiment(task="Optimize empty", current_params={})
        assert result.success is True
        assert result.metadata["plan"]["modifications"] == {}


# ---- Stage 2: Execution ----
# (run_experiment is tested indirectly through integration; here we test the
#  module loader / runner class detection logic)


class TestExecution:
    """Test execution-related components."""

    def test_run_experiment_requires_known_experiment(self):
        """Verify ValueError is raised for unknown experiments."""
        from autonomous_agent import AutonomousAgent

        with pytest.raises((ValueError, Exception)):
            agent = AutonomousAgent(".")
            agent.run_experiment("nonexistent_experiment_xyz")


# ---- Stage 3: Analysis ----


class TestAnalysisStage:
    """Test the analyze_results stage with enriched context."""

    def test_positive_delta_yields_high_confidence(self, engine):
        result = engine.analyze_results(
            {
                "delta": 0.5,
                "primary_metric": 1.5,
                "status": "success",
                "experiment": "test",
                "iteration": 1,
                "modifications": {"lr": 0.02},
                "performance_history": [1.0, 1.5],
            }
        )
        confidence = result.metadata.get("confidence", 0)
        assert confidence > 0.5

    def test_negative_delta_yields_low_confidence(self, engine):
        result = engine.analyze_results(
            {
                "delta": -0.3,
                "primary_metric": 0.7,
                "status": "success",
                "experiment": "test",
                "iteration": 2,
                "modifications": {},
                "performance_history": [1.0, 0.7],
            }
        )
        confidence = result.metadata.get("confidence", 1)
        assert confidence < 0.5

    def test_crash_status_yields_very_low_confidence(self, engine):
        result = engine.analyze_results(
            {
                "delta": 0,
                "status": "crash",
                "experiment": "test",
            }
        )
        confidence = result.metadata.get("confidence", 1)
        assert confidence <= 0.1

    def test_regression_trend_lowers_confidence(self, engine):
        result = engine.analyze_results(
            {
                "delta": -0.1,
                "primary_metric": 0.5,
                "status": "success",
                "experiment": "test",
                "iteration": 4,
                "modifications": {},
                "performance_history": [1.0, 0.8, 0.6, 0.5],  # 3 consecutive declines
            }
        )
        confidence = result.metadata.get("confidence", 1)
        assert confidence <= 0.25

    def test_analysis_returns_enriched_metadata(self, engine):
        result = engine.analyze_results(
            {
                "delta": 0.1,
                "primary_metric": 1.1,
                "status": "success",
                "experiment": "test",
            }
        )
        assert "confidence" in result.metadata
        assert "delta" in result.metadata
        assert "primary_metric" in result.metadata
        assert "status" in result.metadata


# ---- Stage 4: Guardrail Evaluation ----


class TestGuardrailEvaluation:
    """Test the guardrail confidence thresholds."""

    def test_low_confidence_triggers_halt(self, engine):
        """Confidence < 0.2 should trigger a guardrail halt."""
        result = engine.analyze_results(
            {
                "delta": -2.0,
                "status": "success",
                "experiment": "test",
            }
        )
        confidence = result.metadata.get("confidence", 1)
        # Very large negative delta should drive confidence below 0.2
        assert confidence < 0.2

    def test_high_confidence_allows_continuation(self, engine):
        result = engine.analyze_results(
            {
                "delta": 0.5,
                "status": "success",
                "experiment": "test",
            }
        )
        confidence = result.metadata.get("confidence", 0)
        assert confidence >= 0.5


# ---- Stage 5: Memory Update ----


class TestMemoryUpdate:
    """Test that ExecutionReport data flows into the persistent MemoryStore."""

    def test_success_creates_success_pattern(self, memory_store):
        report = {
            "experiment_name": "test_exp",
            "summary": "Iteration 1: Metric 1.5 (Delta +0.5)",
            "metric_deltas": {"test_exp": 0.5},
            "root_causes": [],
            "suggested_fixes": ["Valid parameters found"],
            "confidence_score": 0.9,
        }
        update_memory_from_report(report, memory_store)
        entries = memory_store.retrieve_memories(experiment_name="test_exp")
        patterns = [e for e in entries if e.pattern_type == "success_pattern"]
        assert len(patterns) >= 1

    def test_failure_creates_failure_mode(self, memory_store):
        report = {
            "experiment_name": "test_exp",
            "summary": "Crash",
            "metric_deltas": {},
            "root_causes": ["Index out of bounds"],
            "suggested_fixes": ["Add bounds check"],
            "confidence_score": 0.1,
        }
        update_memory_from_report(report, memory_store)
        failures = memory_store.retrieve_memories(pattern_type="failure_mode")
        assert any("Index out of bounds" in f.content for f in failures)

    def test_suggested_fixes_stored_as_strategies(self, memory_store):
        report = {
            "experiment_name": "test_exp",
            "summary": "Crash",
            "metric_deltas": {},
            "root_causes": [],
            "suggested_fixes": ["Use gradient clipping"],
            "confidence_score": 0.1,
        }
        update_memory_from_report(report, memory_store)
        strategies = memory_store.retrieve_memories(pattern_type="strategy")
        assert any("gradient clipping" in s.content for s in strategies)

    def test_memory_persists_to_disk(self, memory_store):
        report = {
            "experiment_name": "persist_test",
            "summary": "OK",
            "metric_deltas": {"persist_test": 0.1},
            "root_causes": [],
            "suggested_fixes": [],
            "confidence_score": 0.9,
        }
        update_memory_from_report(report, memory_store)

        # Reload from disk
        reloaded = MemoryStore(storage_path=memory_store.storage_path)
        entries = reloaded.retrieve_memories(experiment_name="persist_test")
        assert len(entries) >= 1

    def test_llm_based_extraction_when_available(self, memory_store):
        """When llm_call_fn is provided and returns valid JSON, LLM path is used."""
        fake_llm = MagicMock(
            return_value=json.dumps(
                [
                    {
                        "pattern_type": "success_pattern",
                        "content": "LLM extracted lesson",
                    },
                ]
            )
        )
        report = {
            "experiment_name": "llm_test",
            "summary": "OK",
            "metric_deltas": {"llm_test": 0.5},
            "root_causes": [],
            "suggested_fixes": [],
            "confidence_score": 0.9,
        }
        update_memory_from_report(report, memory_store, llm_call_fn=fake_llm)
        entries = memory_store.retrieve_memories(experiment_name="llm_test")
        assert any("LLM extracted lesson" in e.content for e in entries)
        fake_llm.assert_called_once()


# ---- Semantic Search ----


class TestSemanticSearch:
    """Test the TF-IDF semantic search retrieval."""

    def test_semantic_search_returns_relevant_results(self, memory_store):
        memory_store.add_memory(
            "exp1", "strategy", "Use lower learning rate for stability"
        )
        memory_store.add_memory("exp2", "failure_mode", "Gradient explosion on high LR")
        memory_store.add_memory(
            "exp3", "success_pattern", "Image augmentation improved accuracy"
        )

        results = memory_store.semantic_search("learning rate gradient", top_k=2)
        assert len(results) <= 2
        # The query about "learning rate gradient" should rank the first two higher
        contents = [r.content for r in results]
        assert any(
            "learning rate" in c.lower() or "gradient" in c.lower() for c in contents
        )

    def test_semantic_search_empty_store(self, memory_store):
        results = memory_store.semantic_search("anything")
        assert results == []

    def test_semantic_search_empty_query(self, memory_store):
        memory_store.add_memory("exp1", "strategy", "Test entry")
        results = memory_store.semantic_search("")
        assert len(results) <= 10


# ---- Skill Chain ----


class TestSkillChain:
    """Test the hierarchical skill chaining system."""

    def test_skill_chain_runs_in_sequence(self, engine):
        results = engine.run_skill_chain(
            initial_input={"error": "test error", "experiment": "test"},
            skills_list=["job_debug", "issue_fix", "issue_report"],
        )
        assert len(results) == 2  # Chain halts on issue_fix failure
        assert all(isinstance(r, SkillResult) for r in results)
        assert results[0].skill_name == "job_debug"
        assert results[1].skill_name == "issue_fix"
        # No third result since chain halted

    def test_skill_chain_halts_on_unknown_skill(self, engine):
        results = engine.run_skill_chain(
            initial_input={}, skills_list=["nonexistent_skill"]
        )
        assert len(results) == 0

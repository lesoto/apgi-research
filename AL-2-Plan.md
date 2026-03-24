# M2* Algorithm Alignment Evaluation & Implementation Plan

## Evaluation: 15 / 100

The current codebase (specifically `autonomous_agent.py` and `GUI-auto_improve_experiments.py`) attempts to create an autonomous loop, but it fundamentally functions as a hyperparameter tuning script (random value mutation, numerical metric tracking, and Git rollback) rather than the true "M2* Model Iteration System" specified in `ALGORITH-2.md`. 

### Key Gaps:
1. **Missing Human Control Layer:** The algorithm explicitly requires continuous human steering (defining constraints, reviewing task outputs, approving/rejecting results via a dashboard). The current system executes blindly and automatically once started, with zero human intervention logic (`decision = human.review(result)`).
2. **Missing True Agentic Engine & Skill Chaining:** The M2* Engine requires the agent to read logs, profile runs, formulate hypotheses, and execute multi-step tool chains, e.g., `debug -> fix -> report`. The current agent (`ParameterOptimizer`) merely increments floats and integers via `numpy` (`mutation_strength`, `exploration_rate`).
3. **Absence of Cognitive Memory:** The spec requires extracting "patterns, successful strategies, and failure modes" and storing them in an indexed Knowledge Base (Vector DB). Currently, the system only records the highest numeric configuration scores in `optimization_results.json`.
4. **Missing Experiment Design Step:** The `/exp-plan` generation (creating hypotheses, defining success metrics, setting constraints, defining steps) is entirely non-existent.

To align with `ALGORITH-2.md`, the platform requires a sweeping architectural shift from automated parameter tweaking to an LLM-orchestrated autonomous researcher loop.

---

## Actionable Implementation Steps

### Phase 1: Establish M2* Memory & Data Structures
We need proper abstraction representations to move past simple numeric tracking.

1. **Implement Core Abstractions:**
   - Define an `ExperimentPlan` dataclass holding `hypothesis`, `success_metrics`, `constraints`, and `steps`.
   - Define an `ExecutionReport` dataclass holding `summary`, `metric_deltas`, `root_causes`, and `suggested_fixes`.
   - Replace the current numerical `ExperimentResult` with the broader M2* experiment model.
2. **Build the Memory Update Algorithm:**
   - Develop `memory_store.py` (representing a simple Vector DB or indexed JSON system) to store textual "success patterns" and "failure modes."
   - Create the `update_memory(report)` function to extract lessons learned via LLM after each iteration.

### Phase 2: Construct the Human Control Layer (Control System)
We must implement the Human-in-the-Loop decision gates.

1. **Build the Steering Dashboard:**
   - Update `GUI-auto_improve_experiments.py` to allow the user to `configure_if_needed()` (setting bounds/constraints before a run).
   - Display the LLM-generated `ExperimentPlan` objects for human review before execution.
2. **Build the Review & Decision Gates:**
   - The GUI must catch `ExecutionReport` outputs.
   - Implement the strict `{Approve, Modify, Reject}` logic interface. 
   - `Approve` triggers `deploy_or_store()`.
   - `Modify` triggers a new `/issue-fix` chain.
   - `Reject` prompts the human for next iteration priorities.

### Phase 3: The Agent Harness & Skill Chaining
Replace the numerical `OptimizationStrategy` with an LLM-driven modular skill execution system.

1. **Integrate the LLM Intelligence:**
   - Add a module (e.g., `m2_agent_engine.py`) to connect an LLM capable of generating strategies and analyzing codebase output.
2. **Develop the Skill Chain Functionality:**
   - Build individual atomic "skills" (e.g., `read_logs()`, `patch_source_code()`, `evaluate_metrics()`).
   - Implement `run_skill_chain(input)` so the agent can string these together (e.g., executing the `/job-debug` sequence autonomously).

### Phase 4: Construct the M2* Experiment Loop
Wire the architecture into the recursive RL-style engine described in the core algorithm.

1. **Step 1: Plan Generation (`/exp-plan`)**
   - The LLM receives a user task (e.g., "Improve RL reward by 5%") and drafts an `ExperimentPlan` object.
2. **Step 2: Execution Engine**
   - Trigger the script modifications drafted by the agent. 
   - Introduce auto-failback logic: if a run fails mid-execution, the agent attempts to profile the issue and run again (Self-Healing).
3. **Step 3: Analyze & Report**
   - Stop tracking merely `is_improvement()`. Use the Agent Engine to process logs and compute anomaly reports into an `ExecutionReport`.
4. **Step 4: Guardrails Evaluation**
   - Implement the `escalate_to_human()` trigger if execution confidence drops below a defined threshold or regression happens. This ensures runaway execution is halted.

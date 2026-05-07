# APGI Research Platform — Comprehensive Audit Report

**Project:** `autoresearch` (APGI Autonomous Research Swarm)
**Audit Date:** 2026-05-07
**Auditor:** Claude Code (AI-assisted production audit)
**Overall Grade:** **C+** — Functional core with significant security, testing, and architecture gaps
**Overall Score:** **57 / 100**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [KPI Scores](#2-kpi-scores)
3. [Codebase Overview](#3-codebase-overview)
4. [Bug Inventory — CRITICAL](#4-bug-inventory--critical)
5. [Bug Inventory — HIGH](#5-bug-inventory--high)
6. [Bug Inventory — MEDIUM](#6-bug-inventory--medium)
7. [Bug Inventory — LOW](#7-bug-inventory--low)
8. [Missing Features & Incomplete Implementations](#8-missing-features--incomplete-implementations)
9. [Architecture Issues](#9-architecture-issues)
10. [Testing Infrastructure Analysis](#10-testing-infrastructure-analysis)
11. [Cross-Platform Compatibility](#11-cross-platform-compatibility)
12. [Remediation Roadmap to 100/100](#12-remediation-roadmap-to-100100)

---

## 1. Executive Summary

The **APGI Research Platform** is a sophisticated autonomous cognitive-science research system that combines a Tkinter/CustomTkinter GUI, an autonomous agent loop, 29 cognitive experiment runners, and the APGI (Autonomous Predictive Generative Intelligence) framework. The codebase demonstrates strong architectural *intent* — it has dedicated security modules, RBAC authorization, immutable audit trails, GDPR-compliant data retention stubs, and a Pydantic-validated config schema. However, the execution contains **4 critical bugs** (including a live `eval()` injection vulnerability and a testing infrastructure bomb), **12 high-severity issues**, and a test coverage of only **~16%** against 25,864 lines of production Python.

**Key concerns:**

- A **code injection vulnerability** exists in `autonomous_agent.py` where an AI-generated string is passed directly to `eval()`.
- The **CI pipeline systematically fails** due to a 1-byte fallback secret that violates the audit module's own 32-byte minimum enforcement.
- A **test file calls `sys.exit()` at module level**, causing pytest to abort the entire test collection run.
- **Real data deletion is stubbed** — the GDPR "right to erasure" is logged but never executed.
- The GUI `ExperimentRunnerGUI` is a **2,645-line god class** with 59 methods that needs decomposition.

**Remediation priority:** Fix the 4 critical issues within 1 sprint (1 week), address the 12 high issues in the following 2 sprints, then tackle architecture and coverage.

---

## 2. KPI Scores

| KPI | Score | Rationale |
|-----|-------|-----------|
| **Functional Completeness** | 62 / 100 | Core APGI math, experiment runners, and GUI are functional. Several stub implementations (data deletion, config deletion, KMS). Missing root README, Docker support. |
| **UI/UX Consistency** | 68 / 100 | CustomTkinter dark-mode GUI is cohesive. Monkey-patching of DropdownMenu is a brittle workaround. No accessibility considerations. Guardrail dashboard is well-designed. |
| **Responsiveness & Performance** | 61 / 100 | GUI uses threads for subprocess I/O (correct). Global numpy warning suppression hides real performance signals. No caching on expensive APGI calculations. Module-level singleton instantiation slows imports. |
| **Error Handling & Resilience** | 44 / 100 | Good error taxonomy (`apgi_errors.py`). However: `eval()` on untrusted data, raw `pickle.load` from disk, stub deletions that silently succeed, and module-level globals that crash on missing env vars all undermine resilience. |
| **Overall Implementation Quality** | 51 / 100 | Strong security *design* (RBAC, audit, adapters) undermined by gaps in *execution* (bypassed wrappers, hardcoded salt, 16% coverage, god classes). |
| **OVERALL** | **57 / 100** | |

---

## 3. Codebase Overview

| Metric | Value |
|--------|-------|
| Total Python files | ~130 |
| Total lines of Python | 25,864 |
| Test files | 52 |
| Test functions | 2,278 |
| Measured test coverage | **~16%** (6,628 / 37,776 measured lines) |
| Largest file | `APGI_System.py` — 4,048 lines |
| Largest class | `ExperimentRunnerGUI` — 2,645 lines, 59 methods |
| Python version target | 3.10 – 3.12 |
| Key dependencies | `customtkinter`, `litellm`, `torch`, `numpy`, `scipy`, `GitPython`, `matplotlib`, `pydantic` (test-only) |
| CI/CD | GitHub Actions (security scan, unit, benchmark, integration, lint) |

**Architecture layers:**

```
GUI Layer              → GUI_auto_improve_experiments.py
Autonomous Agent       → autonomous_agent.py, xpr_agent_engine.py
Orchestration Kernel   → apgi_orchestration_kernel.py
Experiment Runners     → experiments/run_*.py (29 experiments)
APGI Core              → APGI_System.py, apgi_integration.py
Security/Auth          → apgi_security.py, apgi_security_adapters.py, apgi_authz.py, apgi_audit.py
Config/Validation      → apgi_config.py, apgi_config_schema.py, apgi_validation.py
Data/Compliance        → apgi_data_retention.py, apgi_compliance.py
Utilities              → apgi_logging.py, apgi_metrics.py, apgi_profiler.py, memory_store.py
```

---

## 4. Bug Inventory — CRITICAL

### BUG-C001 — `eval()` on AI-generated file content (Code Injection)
**Severity:** 🔴 CRITICAL
**File:** `autonomous_agent.py:1706`
**Category:** Security — CWE-95 (Improper Neutralization of Directives in Dynamically Evaluated Code)

**Description:**
The autonomous agent reads Python source files and extracts parameter values using `eval()` on the raw string. Since the source files can be AI-generated or modified by the agent itself, this creates a direct arbitrary code execution pathway.

```python
# VULNERABLE — autonomous_agent.py:1704-1708
try:
    parameters[param_name] = eval(param_value)  # ← CRITICAL: exec arbitrary code
except (ValueError, SyntaxError, NameError, TypeError):
    parameters[param_name] = param_value
```

**Affected Component:** `AutonomousAgent._extract_parameters_from_file()`
**Expected Behavior:** Parameter values parsed safely using `ast.literal_eval()` or typed parsing
**Actual Behavior:** Arbitrary Python expressions executed at runtime

**Reproduction Steps:**
1. Create a run_*.py file containing a parameter assignment like `PARAM = __import__('os').system('rm -rf /')`
2. Point the autonomous agent at that file
3. `eval()` executes the malicious expression

**Fix:**
```python
import ast
try:
    parameters[param_name] = ast.literal_eval(param_value)
except (ValueError, SyntaxError):
    parameters[param_name] = param_value
```

---

### BUG-C002 — `sys.exit()` at module level in test file (Testing Infrastructure Bomb)
**Severity:** 🔴 CRITICAL
**File:** `tests/test_gui_all_options.py:817`
**Category:** Testing Infrastructure

**Description:**
`test_gui_all_options.py` calls `sys.exit()` unconditionally at the module's top level (not inside a `if __name__ == '__main__'` guard). When pytest imports this file during collection, it triggers a `SystemExit` exception that propagates as `INTERNALERROR`, aborting the **entire** test collection run.

```python
# tests/test_gui_all_options.py:817 — TOP LEVEL (no guard)
sys.exit(0 if failed_val == 0 else 1)
```

**Affected Component:** Entire pytest test suite
**Expected Behavior:** pytest collects and runs all 2,278 tests
**Actual Behavior:** pytest aborts with `INTERNALERROR: SystemExit: 1` before any tests run

**Reproduction Steps:**
```bash
pytest tests/ -m "not slow"
# → INTERNALERROR> SystemExit: 1
```

**Fix:** Wrap in `if __name__ == '__main__':` guard or restructure as proper pytest tests.

---

### BUG-C003 — CI pipeline uses trivially-weak fallback secret that breaks audit module
**Severity:** 🔴 CRITICAL
**File:** `.github/workflows/ci.yml:13–15`
**Category:** Security / CI Infrastructure

**Description:**
The CI workflow provides `'x'` (1 byte) as a fallback for `APGI_AUDIT_KEY`. The `ImmutableAuditSink` validates that the key is at least 32 bytes. This means:
1. If the GitHub secret is not configured, CI always fails with `RuntimeError` at test collection time.
2. If CI *does* pass with `'x'`, it means the entropy check was bypassed or the module wasn't exercised.

```yaml
# .github/workflows/ci.yml:13
APGI_AUDIT_KEY: ${{ secrets.APGI_AUDIT_KEY || 'x' }}
```

```python
# apgi_audit.py:148–151 — contradicts the fallback above
if len(key.encode()) < 32:
    raise RuntimeError(
        f"APGI_AUDIT_KEY has insufficient entropy ({len(key.encode())} bytes). "
        "Minimum required: 32 bytes"
    )
```

**Affected Component:** Entire CI pipeline; `test_apgi_data_retention.py`; `test_apgi_orchestration_kernel.py`
**Expected Behavior:** CI uses a properly provisioned secret
**Actual Behavior:** CI either fails on collection or silently bypasses audit signing

**Fix:** Remove the `|| 'x'` fallback entirely. Document that `APGI_AUDIT_KEY` must be provisioned as a GitHub repository secret using `openssl rand -hex 32`.

---

### BUG-C004 — Hardcoded cryptographic salt for participant pseudonymization
**Severity:** 🔴 CRITICAL
**File:** `apgi_compliance.py:118`
**Category:** Security / Privacy — CWE-321 (Use of Hard-coded Cryptographic Key)

**Description:**
The participant pseudonymization function uses a hardcoded default salt. Any attacker who knows the salt (it's in the public repository) can build a rainbow table to reverse participant IDs, completely defeating the purpose of pseudonymization.

```python
# apgi_compliance.py:118
def pseudonymize_participant(
    participant_id: str, salt: str = "apgi_default_salt_x9Z"  # ← HARDCODED
) -> str:
```

**Affected Component:** `ComplianceManager`, all experiment data pipelines that call this function
**Expected Behavior:** Salt sourced from secure environment variable per deployment
**Actual Behavior:** Deterministic, reversible pseudonymization with a known public salt

**Fix:**
```python
import os

def pseudonymize_participant(participant_id: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = os.environ.get("APGI_PSEUDONYM_SALT")
        if not salt:
            raise ValueError("APGI_PSEUDONYM_SALT environment variable required")
    pipeline_input = f"{participant_id}:{salt}".encode("utf-8")
    return hashlib.sha256(pipeline_input).hexdigest()
```

---

## 5. Bug Inventory — HIGH

### BUG-H001 — Raw `pickle.load` from disk bypasses security architecture
**Severity:** 🟠 HIGH
**File:** `progress_tracking.py:494, 508`
**Category:** Security — CWE-502 (Deserialization of Untrusted Data)

**Description:**
`ProgressTracker` saves and loads experiment state using raw `pickle.dump`/`pickle.load` on disk files. The codebase has an entire `apgi_security.py` module with `SecurePickleWrapper` to prevent exactly this, but `progress_tracking.py` bypasses it. A malicious `.pkl` file in the output directory can execute arbitrary code on load.

```python
# progress_tracking.py:493-508 — bypasses SecurePickleWrapper entirely
with open(pickle_file, "wb") as f:
    pickle.dump(self.progress, f)   # Saves raw pickle
# ...
with open(pickle_file, "rb") as f:
    self.progress = pickle.load(f)  # Loads raw pickle from disk
```

**Fix:** Replace with JSON serialization (already saved alongside) or use `secure_load`/`secure_dump` from `apgi_security.py`. Remove the redundant pickle copy.

---

### BUG-H002 — GUI uses raw `subprocess.Popen` bypassing `SecureSubprocessWrapper`
**Severity:** 🟠 HIGH
**File:** `GUI_auto_improve_experiments.py:1464, 2252`
**Category:** Security — Architecture Bypass

**Description:**
The GUI spawns experiment processes and installs packages using raw `subprocess.Popen` instead of the `secure_popen()` wrapper defined in `apgi_security.py`. This means:
- No allowlist enforcement on which commands can be launched
- No audit logging of subprocess calls
- Any experiment script that modifies the GUI's callback can execute arbitrary commands

```python
# GUI_auto_improve_experiments.py:1464
proc = subprocess.Popen(   # ← Should use secure_popen()
    [sys.executable, "-m", module_name], ...
)
```

**Fix:** Replace `subprocess.Popen(...)` calls with `secure_popen(...)` from `apgi_security` with appropriate allowlist.

---

### BUG-H003 — `ComplianceManager._execute_deletion()` is a stub — GDPR erasure broken
**Severity:** 🟠 HIGH
**File:** `apgi_compliance.py:109-114`
**Category:** Functional / Compliance — GDPR Article 17

**Description:**
The compliance manager's deletion routine logs "Applying deletion routine" but performs **no actual data deletion**. The comment says "In a real system, this would call out to database delete, crypto key trash, etc." — but this *is* the production system.

```python
# apgi_compliance.py:109-114
def _execute_deletion(self, record: dict, routine: str) -> None:
    """Simulates deletion routines based on classification."""
    logger.info(
        f"Applying deletion routine '{routine}' to record ..."
    )
    # In a real system, this would call out to database delete ...
    # ← NOTHING HAPPENS HERE
```

**Affected Component:** `ComplianceManager.enforce_retention()`, GDPR/CCPA/HIPAA compliance workflows
**Expected Behavior:** Data classified as expired is actually deleted per its deletion routine
**Actual Behavior:** Deletion is silently "completed" without any data being removed

---

### BUG-H004 — Module-level global instantiation crashes on missing env vars
**Severity:** 🟠 HIGH
**Files:** `apgi_orchestration_kernel.py:344`, `apgi_data_retention.py:359`, `apgi_security_adapters.py:364`
**Category:** Operational Resilience

**Description:**
Three modules instantiate their global singleton objects at **import time** (not inside `get_*()` accessor functions). `APGIOrchestrationKernel.__init__` calls `get_audit_sink()` which calls `ImmutableAuditSink()` which requires `APGI_AUDIT_KEY`. Any code that `import`s these modules without the env var set will crash immediately with a `RuntimeError`.

```python
# apgi_orchestration_kernel.py:344 — runs at import time
_kernel = APGIOrchestrationKernel()   # ← Will crash if APGI_AUDIT_KEY unset

# apgi_data_retention.py:359 — runs at import time
_retention_scheduler = RetentionJobScheduler(RetentionConfig())  # ← Same
```

**Fix:** Move instantiation inside the `get_*()` accessor function (lazy initialization), matching the pattern used in `apgi_audit.py` which already does lazy init correctly.

---

### BUG-H005 — `ImmutableAuditSink` is purely in-memory — audit trail lost on crash
**Severity:** 🟠 HIGH
**File:** `apgi_audit.py`
**Category:** Data Integrity / Compliance

**Description:**
Despite claiming "immutable" and "append-only" semantics, `ImmutableAuditSink` stores all events in a Python `List[AuditEvent]`. If the process crashes, is killed, or the machine reboots, **all audit trail data is permanently lost**. This violates the compliance goal of the module.

**Fix:** Add a persistence backend. Minimal viable: append each signed event as a JSON line to a file. Production: write to an append-only database table.

---

### BUG-H006 — `pydantic` missing from production dependencies — config validation silently disabled
**Severity:** 🟠 HIGH
**File:** `pyproject.toml`, `apgi_config.py:18-38`
**Category:** Functional — Silent Data Corruption

**Description:**
`pydantic` is listed in `requirements-test.txt` but **not** in `pyproject.toml`'s `[project.dependencies]`. The `apgi_config.py` module has a fallback stub that silently replaces `BaseModel` and `Field` when pydantic is absent. The stub's `Field()` ignores all constraints (`ge=`, `le=`, `min_length=`), meaning parameter validation is entirely disabled in any environment that doesn't have pydantic installed.

```python
# apgi_config.py:31-32 — stub ignores all validation constraints
class Field:   # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass  # ← ALL validation constraints silently dropped
```

**Fix:** Add `pydantic>=2.0.0` to `[project.dependencies]` in `pyproject.toml`. Remove the fallback stub.

---

### BUG-H007 — `validate_config_checksum` uses SHA256 with concatenation (length extension attack)
**Severity:** 🟠 HIGH
**File:** `apgi_security.py:294`
**Category:** Security — CWE-327 (Use of Broken/Risky Cryptographic Algorithm)

**Description:**
The function constructs the hash as `SHA256(config_json + secret_key)` via string concatenation. SHA-256 is vulnerable to length-extension attacks when used this way — an attacker who knows the hash of a config can compute the hash of any config with additional keys appended without knowing the secret. The `apgi_security_adapters.py` module correctly uses `hmac.new()` for the same purpose, making this inconsistency especially dangerous.

```python
# apgi_security.py:294 — VULNERABLE (not HMAC)
config_str = json.dumps(config_dict, sort_keys=True) + secret_key
computed_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
```

**Fix:** Replace with `hmac.new(secret_key.encode(), config_str.encode(), hashlib.sha256).hexdigest()` to match `apgi_security_adapters.py`.

---

### BUG-H008 — `AuthorizationManager` has no authentication — RBAC is identity-claim only
**Severity:** 🟠 HIGH
**File:** `apgi_authz.py`
**Category:** Security — CWE-287 (Improper Authentication)

**Description:**
The RBAC system defines roles and permissions correctly, but there is no authentication mechanism. `register_operator()` creates an `OperatorIdentity` with whatever role the caller requests. Any code can call `register_operator("admin", Role.ADMIN)` and instantly receive admin permissions. There is no password verification, token validation, or session management.

**Fix:** Add at minimum: operator tokens (UUID secrets), token-based `authenticate_operator(token: str)` method, and enforce that `get_operator()` requires token presentation rather than just operator ID.

---

### BUG-H009 — Two test files fail at collection due to env var requirements (blocks CI)
**Severity:** 🟠 HIGH
**Files:** `tests/test_apgi_data_retention.py`, `tests/test_apgi_orchestration_kernel.py`
**Category:** Testing Infrastructure

**Description:**
Both test files import modules that trigger module-level singleton instantiation (BUG-H004), which requires `APGI_AUDIT_KEY`. Running `pytest tests/` without this env var set causes a `RuntimeError` during collection, and pytest aborts with 2 collection errors before any test can run.

**Fix:** Set `APGI_AUDIT_KEY` in the CI environment (see BUG-C003 fix) **and** fix BUG-H004 (lazy init) so tests that mock the audit sink work without the env var.

---

### BUG-H010 — Test coverage is 16% — critical paths are untested
**Severity:** 🟠 HIGH
**Category:** Testing Quality

**Description:**
Running the full test suite against the codebase yields only **~16% line coverage** (6,628 of 37,776 measured lines). The autonomous agent, GUI interactions, compliance deletion, and most experiment runners have 0% coverage. At 16% coverage, entire subsystems can be broken with no test signal.

**Key uncovered modules:**
| Module | Coverage |
|--------|----------|
| `autonomous_agent.py` (1,796 lines) | ~0% branch coverage |
| `GUI_auto_improve_experiments.py` (2,741 lines) | ~0% functional coverage |
| `apgi_data_retention.py` | 0% (collection failure) |
| `apgi_orchestration_kernel.py` | 0% (collection failure) |
| `train.py` (932 lines) | <10% |
| `xpr_agent_engine.py` (1,319 lines) | <15% |

---

### BUG-H011 — GUI `ExperimentRunnerGUI` is a 2,645-line, 59-method god class
**Severity:** 🟠 HIGH
**File:** `GUI_auto_improve_experiments.py:91`
**Category:** Architecture / Maintainability

**Description:**
`ExperimentRunnerGUI` violates the Single Responsibility Principle at a severe scale. It handles: window/menu creation, sidebar navigation, experiment card management, subprocess spawning, stdout/stderr buffering, metric parsing (30+ regex patterns), LLM hypothesis generation, guardrail state management, visualization (matplotlib), hypothesis approval board, dependency checking, and package installation. This makes the class extremely difficult to test, extend, or debug.

---

### BUG-H012 — `datetime.utcnow()` deprecated in Python 3.12+ used in 12 critical locations
**Severity:** 🟠 HIGH
**Files:** `apgi_authz.py:99,100,125`, `apgi_data_retention.py:54,55,72,77,82`, `apgi_security_adapters.py:49,64`, `apgi_audit.py:45,258`
**Category:** Compatibility / Python 3.12+

**Description:**
`datetime.utcnow()` is deprecated since Python 3.12 and scheduled for removal. All 12 callsites are in the security and compliance layer — the most critical paths. In Python 3.12 these produce `DeprecationWarning` on every event logged; in a future Python release they will raise `AttributeError`.

**Fix:** Replace all `datetime.utcnow()` with `datetime.now(tz=timezone.utc)` and add `from datetime import timezone` imports.

---

## 6. Bug Inventory — MEDIUM

### BUG-M001 — `ConfigManager` singleton + `lru_cache` interaction: reload does not invalidate cache
**Severity:** 🟡 MEDIUM
**File:** `apgi_config.py:343–356`

`ConfigManager.reload()` clears `_config_cache` but `get_cached_experiment_config` is decorated with `@lru_cache`. Calling `reload()` followed by `get_cached_experiment_config("stroop")` returns the pre-reload cached value. `invalidate_config_cache()` must be called explicitly — but no code calls both together.

**Fix:** Call `invalidate_config_cache()` inside `ConfigManager.reload()`.

---

### BUG-M002 — Global numpy `RuntimeWarning` suppression hides real errors
**Severity:** 🟡 MEDIUM
**File:** `APGI_System.py:98`

```python
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
```

This is module-level and applies process-wide. Legitimate `RuntimeWarning`s from overflow, invalid value, or division-by-zero in APGI calculations are silently discarded. Scientific computing errors that should surface for investigation are hidden.

**Fix:** Use a context manager (`with warnings.catch_warnings():`) scoped to specific functions where overflow is expected.

---

### BUG-M003 — Duplicate `import warnings` in `APGI_System.py`
**Severity:** 🟡 MEDIUM
**File:** `APGI_System.py:82, 92`

`import warnings` appears twice. While harmless in Python, it indicates the file was assembled without review and may contain other duplicate imports.

---

### BUG-M004 — `CustomTkinter` `DropdownMenu` monkey-patched at import time
**Severity:** 🟡 MEDIUM
**File:** `GUI_auto_improve_experiments.py:27–48`

The GUI patches `ctk.windows.widgets.core_widget_classes.dropdown_menu.DropdownMenu._add_menu_commands` at module import time. This patches a private internal method of a third-party library via deep attribute path access. Any update to `customtkinter` that renames or restructures this path will cause an `AttributeError` crash on startup with no clear error message.

**Fix:** Pin the exact `customtkinter` version this patch was written for, add a version assertion, and file a bug upstream.

---

### BUG-M005 — Custom `TimeoutError` shadows built-in `TimeoutError`
**Severity:** 🟡 MEDIUM
**File:** `autonomous_agent.py:66`

```python
class TimeoutError(Exception):  # Shadows builtins.TimeoutError
    pass
```

Python 3.3+ has a built-in `TimeoutError`. The custom class shadows it within the module, causing any `except TimeoutError` in other modules that catches the built-in to miss the custom one, and vice versa. This creates subtle unhandled-exception bugs in async contexts.

**Fix:** Rename to `APGITimeoutError` (class already exists in `apgi_errors.py`) and delete the duplicate.

---

### BUG-M006 — `delete_config_data()` in `DeletionExecutor` does nothing except log
**Severity:** 🟡 MEDIUM
**File:** `apgi_data_retention.py:138–161`

The function logs "Deleted config data" and audits "success" but performs no actual deletion. Unlike `_execute_deletion()` in `ComplianceManager` which at least has a comment acknowledging the gap, this one falsely reports success to the audit trail.

---

### BUG-M007 — `apgi_validation.py` whitelists `"pickle"` as a safe import module
**Severity:** 🟡 MEDIUM
**File:** `apgi_validation.py:55`

The `SAFE_IMPORT_MODULES` set includes `"pickle"`, which contradicts the rest of the security architecture's goal of blocking pickle deserialization. This allows AI-generated code to `import pickle` without triggering the dangerous import pattern validator.

---

### BUG-M008 — Competing dual singletons: `ConfigManager._instance` + module-level `_config_manager`
**Severity:** 🟡 MEDIUM
**File:** `apgi_config.py:324–341`

The module implements two independent singleton mechanisms: a class-level `_instance` (via `__new__`) and a module-level `_config_manager` variable. `reset_config()` resets both, but code that calls `ConfigManager()` directly bypasses the module-level reset. This creates confusing initialization behavior in tests.

---

### BUG-M009 — Threading in GUI without proper widget-thread safety
**Severity:** 🟡 MEDIUM
**File:** `GUI_auto_improve_experiments.py:1425, 1494-1502`

Tkinter is not thread-safe. The GUI spawns background threads (`threading.Thread`) that call `self._log()` which updates the CTkTextbox widget directly. Direct widget updates from background threads cause intermittent crashes and corrupted display state, particularly on macOS.

**Fix:** Use `self.after(0, lambda: self._log(message))` to marshal updates back to the main thread.

---

### BUG-M010 — `APGI_System.py` `from __future__ import annotations` placement non-standard
**Severity:** 🟡 MEDIUM
**File:** `APGI_System.py:1`

`from __future__ import annotations` is on line 1, but it's followed immediately by a triple-quoted docstring on line 4 and then *another* module-level docstring on line 35. PEP 257 requires the module docstring to be the first statement; the `__future__` import must come before it. This creates confusing documentation structure.

---

### BUG-M011 — `apgi_security.py` and `apgi_security_adapters.py` duplicate security logic
**Severity:** 🟡 MEDIUM
**Files:** `apgi_security.py`, `apgi_security_adapters.py`

Both modules independently implement subprocess allowlist enforcement, pickle control, and HMAC-based config validation. Two separate implementations of the same security policy create drift risk — one module uses SHA256 concatenation (BUG-H007), the other uses proper HMAC.

---

### BUG-M012 — `apgi_version.py` version is hardcoded, not linked to `pyproject.toml`
**Severity:** 🟡 MEDIUM
**File:** `apgi_version.py:5`, `pyproject.toml:3`

Both define `"1.0.0"` independently. These will inevitably diverge.

**Fix:** Read version from `importlib.metadata.version("autoresearch")` at runtime.

---

## 7. Bug Inventory — LOW

### BUG-L001 — `progress_tracking.py` saves redundant pickle alongside JSON
**Severity:** 🔵 LOW
**File:** `progress_tracking.py:488-494`
Both JSON and pickle are saved "for faster loading". The pickle is never loaded if JSON is present first. This redundancy adds attack surface with no performance benefit — JSON parsing at this scale is negligible.

---

### BUG-L002 — 30+ hardcoded `[DEBUG]` string prefixes in GUI instead of logging module
**Severity:** 🔵 LOW
**File:** `GUI_auto_improve_experiments.py:1628–2047`
Parse failures emit `self._log(f"[DEBUG] Failed to parse ...")` strings rather than using `logging.debug()`. These are always emitted regardless of configured log level.

---

### BUG-L003 — `delete_pycache.py` (937 lines) should not be a root-level production module
**Severity:** 🔵 LOW
**File:** `delete_pycache.py`
This is an operational utility script that has grown to 937 lines and includes a full CLI, progress tracking, and error reporting. It is imported by test files and clutters the root namespace. Should be moved to `tools/` or `scripts/`.

---

### BUG-L004 — `apgi_config_schema.py` is redundant with `apgi_config.py`
**Severity:** 🔵 LOW
**Files:** `apgi_config_schema.py`, `apgi_config.py`
Both files define Pydantic schemas for APGI configuration with overlapping content. Maintaining two sources of truth for the same schemas will cause drift.

---

### BUG-L005 — `matplotlib.use("Agg")` at module level in `APGI_System.py` conflicts with GUI's `TkAgg`
**Severity:** 🔵 LOW
**File:** `APGI_System.py:91`, `GUI_auto_improve_experiments.py:67`
`APGI_System.py` calls `matplotlib.use("Agg")` at module level. The GUI calls `matplotlib.use("TkAgg")`. Whichever module is imported first wins. If `APGI_System` is imported before the GUI sets TkAgg, embedded matplotlib charts in the GUI will render to a non-interactive backend.

---

### BUG-L006 — `apgi_authz.py` authorization log is unbounded in-memory list
**Severity:** 🔵 LOW
**File:** `apgi_authz.py:135`
`authorization_log: List[Dict] = []` grows without bound. Unlike `SecurityMetrics.audit_events` which caps at 10,000, the authorization log has no limit. Long-running sessions will accumulate memory indefinitely.

---

### BUG-L007 — CI runs only on `main`/`master` — feature branches have no CI
**Severity:** 🔵 LOW
**File:** `.github/workflows/ci.yml:4-6`
CI triggers only on push/PR to `main`/`master`. Feature branch development gets no CI feedback until merge time.

---

### BUG-L008 — No root-level `README.md`
**Severity:** 🔵 LOW
**File:** `pyproject.toml:5` (`readme = "README.md"`)
`pyproject.toml` references `README.md` but it does not exist at the root. The README lives at `docs/README.md`. `pip install` will fail with a warning and PyPI uploads will have no description.

---

## 8. Missing Features & Incomplete Implementations

| # | Feature | Affected File(s) | Status | Impact |
|---|---------|-----------------|--------|--------|
| MF-01 | Root-level `README.md` | `pyproject.toml` | Missing entirely | Build warning; no project description for PyPI |
| MF-02 | GDPR right to erasure (actual deletion) | `apgi_compliance.py:109` | Stub — logs only | Compliance violation |
| MF-03 | Config data deletion | `apgi_data_retention.py:138` | Stub — logs only | Compliance violation |
| MF-04 | KMS key destruction | `apgi_data_retention.py:163` | Conditional on callback; no real KMS integration | Compliance gap |
| MF-05 | Audit trail persistence | `apgi_audit.py` | In-memory only | Data loss on crash |
| MF-06 | Authentication backend for RBAC | `apgi_authz.py` | Role assignment with no credential check | Security gap |
| MF-07 | Docker / containerization | Root directory | Not present | Deployment gap |
| MF-08 | Health check endpoints | All modules | Not present | Operational monitoring gap |
| MF-09 | Rate limiting on autonomous agent API calls | `autonomous_agent.py` | Partial (rate limiter class exists but not wired to LLM calls) | Cost/DoS risk |
| MF-10 | Secrets management (vault integration) | All env-var consumers | Env var suggestion only | Security gap |
| MF-11 | Cross-platform macOS/Windows GUI testing | `tests/test_gui_*.py` | macOS-only; Linux/Windows untested | Portability gap |
| MF-12 | API documentation (OpenAPI/Swagger) | `apgi_cli.py` | No OpenAPI spec generated | Usability gap |
| MF-13 | Automated backup of experiment results | `progress_tracking.py` | No backup mechanism | Data loss risk |
| MF-14 | `pydantic` in production dependencies | `pyproject.toml` | In test deps only | Silent validation bypass |
| MF-15 | `setup.cfg` / `requirements.txt` for non-uv users | Root directory | `pyproject.toml` uses uv-specific syntax | Portability gap |

---

## 9. Architecture Issues

### A-01 — God Class: `ExperimentRunnerGUI` (2,645 lines, 59 methods)
The entire GUI application is a single class. Responsibilities to extract:
- `ExperimentManager` — discovery, launch, status tracking
- `HypothesisUIController` — approval board integration
- `GuardrailDashboard` — guardrail state display
- `MetricParser` — 30+ regex patterns for output parsing
- `VisualizationPanel` — matplotlib embedding
- `DependencyChecker` — startup checks

### A-02 — God Class: `AutonomousAgent` (1,010 lines, 17 methods)
Mixing: experiment scheduling, parameter extraction, LLM calls, Git operations, overnight loop logic, rate limiting.

### A-03 — Two competing compliance modules
`apgi_compliance.py` and `apgi_data_retention.py` both implement retention policies and data deletion with overlapping but inconsistent APIs. Consolidate into one.

### A-04 — Two competing security modules
`apgi_security.py` and `apgi_security_adapters.py` both implement subprocess allowlists and serialization security. One uses SHA256-concat (vulnerable); the other uses HMAC (correct). Consolidate into one.

### A-05 — Module-level global singletons create hidden import ordering constraints
Eight modules have module-level singletons: `_authz_manager`, `_audit_sink`, `_retention_scheduler`, `_kernel`, `_security_factory`, `_config_manager`, `_default_subprocess_wrapper`, `_default_pickle_wrapper`. Their initialization order is import-order-dependent and untestable without complete env var setup.

### A-06 — `APGIStateLibrary` is a 1,036-line data class
The class contains 51 psychological state definitions as hardcoded Python dictionaries. This data should be externalized to a JSON/YAML configuration file, making it editable without code changes and testable in isolation.

---

## 10. Testing Infrastructure Analysis

| Metric | Value | Target |
|--------|-------|--------|
| Total test functions | 2,278 | — |
| Test collection errors (without env vars) | 2 files | 0 |
| Files with `sys.exit()` at module level | 1 (`test_gui_all_options.py`) | 0 |
| Test coverage (measured) | **~16%** | ≥ 80% |
| Tests passing (clean run, fast subset) | ~117 | 2,278 |
| Tests with flaky markers | Present | 0 |
| GUI test automation | Script-style (not pytest) | Pytest-based |

**Specific test infrastructure problems:**

1. `test_gui_all_options.py:817` — `sys.exit()` at module level (BUG-C002)
2. `test_apgi_data_retention.py` + `test_apgi_orchestration_kernel.py` — fail at collection without `APGI_AUDIT_KEY` (BUG-C003 + BUG-H004)
3. `tests/coverage_config.py` + `tests/mutation_testing.py` — infrastructure code in the test directory is loaded by pytest but not proper tests
4. Many `test_*_coverage.py` files testing the same modules as `test_*.py` — suggests coverage gaps were patched by adding new files rather than improving existing tests
5. No `conftest.py`-level fixture for setting required env vars, meaning any test that imports security modules must set env vars manually

---

## 11. Cross-Platform Compatibility

| Platform | Status | Issues |
|----------|--------|--------|
| **macOS (Apple Silicon)** | ✅ Primary target | macOS-specific subprocess fork safety workaround in GUI (correct). PyTorch routed to CPU for arm64 (correct). |
| **macOS (Intel)** | ✅ Likely functional | Untested explicitly |
| **Linux** | ⚠️ Partial | No Linux-specific CI runner. PyTorch CUDA routing requires NVIDIA GPU. `screencapture` in subprocess allowlist is macOS-only. |
| **Windows** | ❌ Untested | No Windows CI. `screencapture` command not available. Path separator assumptions. `multiprocessing.set_start_method("spawn")` block is macOS-only conditional. |

**Specific cross-platform issues:**
- `DEFAULT_ALLOWED_SUBPROCESS_CMDS` in `apgi_security.py:21` includes `"screencapture"` — macOS-only command
- CI uses `ubuntu-latest` runners but the primary dev platform is macOS — matplotlib backend differences may affect test results
- No Windows path handling (all uses of `os.path` appear correct, but untested)

---

## 12. Remediation Roadmap to 100/100

### Sprint 1 (Week 1) — Critical Fixes (Target: 57 → 70)

| Task | File | BUG |
|------|------|-----|
| Replace `eval()` with `ast.literal_eval()` | `autonomous_agent.py:1706` | C001 |
| Wrap `sys.exit()` in `if __name__ == '__main__':` | `tests/test_gui_all_options.py:817` | C002 |
| Remove `|| 'x'` fallback in CI; provision proper secret | `.github/workflows/ci.yml:13-15` | C003 |
| Replace hardcoded salt with env var | `apgi_compliance.py:118` | C004 |
| Replace `validate_config_checksum` SHA256-concat with HMAC | `apgi_security.py:294` | H007 |
| Fix lazy init for 3 module-level singletons | `apgi_orchestration_kernel.py:344`, `apgi_data_retention.py:359` | H004 |
| Add `conftest.py` fixture setting `APGI_AUDIT_KEY` | `tests/conftest.py` | H009 |

---

### Sprint 2 (Weeks 2–3) — High Severity (Target: 70 → 82)

| Task | File | BUG |
|------|------|-----|
| Replace raw `pickle.load/dump` with JSON in `progress_tracking.py` | `progress_tracking.py` | H001 |
| Replace raw `subprocess.Popen` with `secure_popen()` in GUI | `GUI_auto_improve_experiments.py:1464,2252` | H002 |
| Implement real data deletion in `_execute_deletion()` | `apgi_compliance.py:109` | H003 |
| Add `pydantic>=2.0.0` to production deps | `pyproject.toml` | H006 |
| Replace all `datetime.utcnow()` with `datetime.now(tz=timezone.utc)` | 12 locations | H012 |
| Add authentication tokens to `AuthorizationManager` | `apgi_authz.py` | H008 |
| Add file-backed persistence to `ImmutableAuditSink` | `apgi_audit.py` | H005 |
| Fix `ConfigManager.reload()` to call `invalidate_config_cache()` | `apgi_config.py` | M001 |
| Marshal GUI widget updates to main thread | `GUI_auto_improve_experiments.py` | M009 |
| Add unbounded list cap to `authorization_log` | `apgi_authz.py:135` | L006 |

---

### Sprint 3 (Weeks 4–6) — Architecture & Coverage (Target: 82 → 95)

| Task | Details |
|------|---------|
| Decompose `ExperimentRunnerGUI` | Extract: `ExperimentManager`, `MetricParser`, `GuardrailDashboard`, `VisualizationPanel`, `HypothesisUIController` |
| Consolidate security modules | Merge `apgi_security.py` + `apgi_security_adapters.py` into one authoritative module |
| Consolidate compliance modules | Merge `apgi_compliance.py` + `apgi_data_retention.py` |
| Externalize `APGIStateLibrary` data | Move 51 psychological state definitions to `data/states.json` |
| Raise test coverage to ≥ 80% | Focus on: `autonomous_agent.py`, `apgi_compliance.py`, `apgi_data_retention.py`, `apgi_orchestration_kernel.py` |
| Add root `README.md` | Move/update `docs/README.md` to root |
| Add `Dockerfile` | Containerize for reproducible research environments |
| Fix matplotlib backend conflict | Scope `matplotlib.use("Agg")` to non-GUI contexts |
| Remove `delete_pycache.py` from root | Move to `tools/` |
| Merge `apgi_version.py` → `importlib.metadata` | Single source of truth for version |
| Scope numpy warning suppression | Replace global with context managers |

---

### Sprint 4 (Week 7) — Polish to 100 (Target: 95 → 100)

| Task |
|------|
| Add health-check endpoints / monitoring hooks |
| Add Windows CI runner |
| Implement real KMS key destruction integration |
| Add OpenAPI spec generation for `apgi_cli.py` |
| Implement rate limiting on all LLM API calls in autonomous agent |
| Add `APGI_PSEUDONYM_SALT` to documented required env vars |
| Pin `customtkinter` version for monkey-patch compatibility |
| Add `CHANGELOG.md` and version management automation |
| Add `pre-commit` hooks: `black`, `ruff`, `mypy` |
| CI: add feature-branch triggers |

---

## Appendix — Environment Variables Required for Production

| Variable | Used In | Description | Required |
|----------|---------|-------------|----------|
| `APGI_AUDIT_KEY` | `apgi_audit.py` | HMAC signing key for audit trail (≥32 bytes, use `openssl rand -hex 32`) | **Mandatory** |
| `APGI_KMS_KEY` | `apgi_security_adapters.py` | KMS key for config checksum validation | **Mandatory** if checksum adapter used |
| `APGI_CONFIG_SECRET_KEY` | `apgi_security.py` | Config validation secret | **Mandatory** if config checksums used |
| `APGI_PSEUDONYM_SALT` | `apgi_compliance.py` | Pseudonymization salt (replaces hardcoded value after C004 fix) | **Mandatory** after fix |
| `APGI_ALLOWED_SUBPROCESS_CMDS` | `apgi_security.py` | Comma-separated subprocess allowlist override | Optional |
| `APGI_CONFIG_FILE` | `apgi_config.py` | Path to JSON/YAML config file | Optional |
| `APGI_OPERATOR_ROLE` | CI workflow | Operator role for CI test runs | Optional |

---

*Report generated by automated audit on 2026-05-07. Total issues: 4 Critical, 12 High, 12 Medium, 8 Low = **36 total**.*

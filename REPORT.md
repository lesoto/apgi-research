# APGI Research Platform — Comprehensive Audit Report

**Audit Date:** 2026-05-11  
**Auditor:** Claude Code (claude-sonnet-4-6)  
**Codebase Root:** `/Users/lesoto/Sites/PYTHON/apgi-research`  
**Total Python Files:** 195  
**Audit Scope:** All source modules, utils, experiments, tests, configuration  

---

## Executive Summary

The APGI Research Platform is an ambitious autonomous AI research system with a CustomTkinter GUI, a multi-layer agent pipeline, a GDPR-compliant data infrastructure, and extensive security controls. The architecture demonstrates genuine sophistication — HMAC audit trails, RBAC, subprocess allowlists, pseudonymization, GDPR data retention, and cross-platform timeout abstractions are all present and well-designed.

**However, the application is not functional in its current state.** Three critical bugs in the GUI's primary workflow — a missing method (`_stop_experiment`), an uninstantiated attribute (`self.agent_engine`), and broken audit trail integrity (events are re-signed instead of verified on load) — block core use-cases entirely. Additionally, five of six visualization panels display randomly-generated noise rather than real metrics, which constitutes a scientific integrity risk for a research platform.

The security architecture is undermined at its most critical paths: the two highest-traffic subprocess call sites (`GUI_auto_improve_experiments.py` and `utils/git_operations.py`) bypass the `SecureSubprocessWrapper` entirely, and the GDPR pseudonymization module ships with a hardcoded fallback salt that renders pseudonymization legally ineffective.

**Remediation priority:** Fix the 5 critical bugs first to restore basic functionality, then address the 8 high-severity issues to reach a stable, secure baseline.

---

## KPI Scores

| KPI | Score | Rationale |
|-----|-------|-----------|
| **Functional Completeness** | **38 / 100** | Core features exist as scaffolding, but stop-experiment is unimplemented, agent_engine never instantiated, adaptive planning is hard-coded, 5/6 viz panels show fake data |
| **UI/UX Consistency** | **44 / 100** | CustomTkinter used consistently; theming applied; but crashes on plan generation, no loading states, fake charts undermine trust |
| **Responsiveness & Performance** | **41 / 100** | Background threading is correct; SentenceTransformer reloads on every call; event loop created/destroyed per iteration; asyncio.get_event_loop() deprecated |
| **Error Handling & Resilience** | **29 / 100** | SIGALRM crashes Windows; AttributeError on missing methods; security modules crash on missing env vars; audit trail silently corrupts |
| **Overall Implementation Quality** | **37 / 100** | Strong architectural intent, well-designed interfaces, but critical implementation gaps across all layers leave the platform non-production-ready |

---

## Bug Inventory

### Critical Severity

| ID | File | Line | Description |
|----|------|------|-------------|
| BUG-CRIT-001 | `GUI_auto_improve_experiments.py` | 1303 | `self.agent_engine` referenced but never assigned in `__init__` — `AttributeError` on first plan-related GUI action |
| BUG-CRIT-002 | `GUI_auto_improve_experiments.py` | 1305 | `plan_text` reassigned from CTkTextbox widget to plain `str`, then used as widget at line 1398 — `AttributeError: 'str' has no attribute 'get'` |
| BUG-CRIT-003 | `GUI_auto_improve_experiments.py` | 2292 | `self._stop_experiment(name)` called but method never defined — experiments cannot be stopped; `AttributeError` at timeout |
| BUG-CRIT-004 | `apgi_audit.py` | 159 | `event.sign(self.secret_key)` re-signs loaded audit events instead of verifying them — destroys HMAC tamper-evidence chain entirely |
| BUG-CRIT-005 | `xpr_agent_engine.py` | 626–648 | File contains two concatenated modules; second half has its own module docstring and duplicate imports — `Optional` not imported in second scope, causing potential `NameError` |

#### BUG-CRIT-001 — Missing `self.agent_engine` Assignment

**Affected Component:** GUI Plan Generation workflow  
**Reproduction:** Click any button that triggers plan generation.  
**Expected:** `agent_engine` instantiated in `ExperimentRunnerGUI.__init__` as `self.agent_engine = XPRAgentEngine()`.  
**Actual:** `self.agent_engine.get_current_plan()` raises `AttributeError: 'ExperimentRunnerGUI' object has no attribute 'agent_engine'`. Local variable `agent_engine` exists at line 1281 but is never stored on `self`.  
**Fix:** In `__init__`, add `self.agent_engine: XPRAgentEngine = XPRAgentEngine()` after the other state assignments.

#### BUG-CRIT-002 — `plan_text` Variable Type Collision

**Affected Component:** Plan-approval dialog  
**Reproduction:** Trigger plan generation via "Run XPR Modify-Chain" after agent_engine is fixed. If `current_plan_text.result` is populated, line 1305 overwrites `plan_text` (a `CTkTextbox`) with a plain `str`.  
**Expected:** Plan content inserted into the widget with `plan_text.delete("0.0","end"); plan_text.insert("0.0", content)`.  
**Actual:** Widget reference overwritten; line 1398 `plan_text.get("0.0","end")` raises `AttributeError: 'str' has no attribute 'get'`.  
**Fix:** Never reassign `plan_text`; use `widget.delete/insert` to update widget content.

#### BUG-CRIT-003 — `_stop_experiment` Method Not Implemented

**Affected Component:** Experiment execution timeout handling  
**Reproduction:** Start an experiment and wait for `max_wait_time` (default 300 s). Line 2292 calls `self._stop_experiment(name)`.  
**Expected:** Method terminates the subprocess at `self.active_processes[name]` and removes entry from `self.running_experiments`.  
**Actual:** `AttributeError` crash; experiment process continues indefinitely.  
**Fix:**
```python
def _stop_experiment(self, name: str) -> None:
    proc = self.active_processes.pop(name, None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    self.running_experiments.discard(name)
```

#### BUG-CRIT-004 — Audit Trail Re-Signs Instead of Verifying

**Affected Component:** `AuditSink._load_existing_events` (`apgi_audit.py:159`)  
**Reproduction:** Write audit events, restart process, reload audit log.  
**Expected:** `event.verify_signature(self.secret_key)` — events failing verification should be flagged or rejected.  
**Actual:** `event.sign(self.secret_key)` overwrites stored HMAC signatures. Historical evidence permanently destroyed; any modified event file passes integrity check.  
**Fix:**
```python
# Replace line 159:
if not event.verify_signature(self.secret_key):
    self.logger.error(f"Audit event {event.sequence_number} FAILED integrity check — possible tampering")
    continue  # or raise APGIAuditError
```

#### BUG-CRIT-005 — Double Module in `xpr_agent_engine.py`

**Affected Component:** `XPRAgentEngineEnhanced` and all classes defined after line 626  
**Reproduction:** `from xpr_agent_engine import XPRAgentEngineEnhanced` — may raise `NameError` if `Optional` used in dataclass fields in the second half.  
**Expected:** Single cohesive module with all imports at top.  
**Actual:** Two separate module docstrings and duplicate imports indicate two files were concatenated without merging imports.  
**Fix:** Move all imports to module top; deduplicate; remove the second module docstring (lines 626–640).

---

### High Severity

| ID | File | Line | Description |
|----|------|------|-------------|
| BUG-HIGH-001 | `GUI_auto_improve_experiments.py` | 150–155 | `running_experiments` (Set), `active_processes` (Dict), `stop_all` (bool) accessed from main + background threads with no locks — race conditions |
| BUG-HIGH-002 | `autonomous_agent.py` | 916–917 | `signal.SIGALRM` used without Windows platform guard — `AttributeError` crash on Windows |
| BUG-HIGH-003 | `autonomous_agent.py` | 182–186 | `repo.index.add(pattern)` with glob strings — GitPython does not expand globs; silent empty commits |
| BUG-HIGH-004 | `autonomous_agent.py` | 216 | `repo.git.reset('--hard', target)` without checking `repo.is_dirty()` — uncommitted experiment data silently destroyed on rollback |
| BUG-HIGH-005 | `GUI_auto_improve_experiments.py` | 2539–2614 | 14 `np.random` calls generate fake metric data as chart defaults — researchers see random noise as scientific results |
| BUG-HIGH-006 | `memory_store.py` | 156 | `SentenceTransformer('all-MiniLM-L6-v2')` loaded from disk on every `_generate_embedding()` call — multi-second latency per memory operation |
| BUG-HIGH-007 | `GUI_auto_improve_experiments.py` | 3151, 3601 | `subprocess.run()` in GUI bypasses `SecureSubprocessWrapper` — subprocess security allowlist ineffective |
| BUG-HIGH-008 | `utils/git_operations.py` | `_run_git_command` | `subprocess.run()` throughout bypasses `SecureSubprocessWrapper` — all agent git operations bypass security layer |

**BUG-HIGH-001 Fix:** Add `self._state_lock = threading.Lock()` in `__init__`; guard all reads/writes to `running_experiments`, `active_processes`, `stop_all` with `with self._state_lock`.

**BUG-HIGH-002 Fix:**
```python
if sys.platform != "win32":
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
else:
    # Use threading.Timer as cross-platform fallback
    timer = threading.Timer(timeout_seconds, timeout_handler)
    timer.start()
```

**BUG-HIGH-003 Fix:**
```python
import glob
files = [f for pattern in patterns for f in glob.glob(pattern)]
if files:
    repo.index.add(files)
```

**BUG-HIGH-005 Fix:** Replace `np.random.uniform(...)` defaults with `None` and render a "No data available" bar/message when real values are absent.

**BUG-HIGH-006 Fix:** Cache model on first use:
```python
def __init__(self, ...):
    self._embedding_model: Optional[Any] = None

def _get_embedding_model(self):
    if self._embedding_model is None:
        from sentence_transformers import SentenceTransformer
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return self._embedding_model
```

---

### Medium Severity

| ID | File | Line | Description |
|----|------|------|-------------|
| BUG-MED-001 | `autonomous_agent.py` | 651 | `asyncio.get_event_loop()` deprecated Python 3.10+; raises `RuntimeError` in 3.12+ outside async context |
| BUG-MED-002 | `xpr_agent_engine.py` | ~974 | f-string prefixes missing — diagnostic strings emit literal `{expression}` text instead of evaluated values |
| BUG-MED-003 | `memory_store.py` | ~180 | `MemoryEntry(**entry)` deserialization fails when `embedding` field is a `dict` rather than `VectorEmbedding` dataclass |
| BUG-MED-004 | `utils/apgi_authz.py` | 181 | JWT secret regenerated on every instantiation — all tokens invalidated on process restart |
| BUG-MED-005 | `hypothesis_approval_board.py` | `_create_hypothesis_review_item` | `dialog.destroy()` on first approval/rejection closes entire multi-hypothesis review dialog |
| BUG-MED-006 | `GUI_auto_improve_experiments.py` | 5, 31, 75 | `import sys` appears at lines 5 & 31; `import logging` at lines 5 & 75 — duplicate imports |
| BUG-MED-007 | `GUI_auto_improve_experiments.py` | 92, 98 | `matplotlib.use('TkAgg')` called twice — second call may emit `UserWarning` or silently fail |
| BUG-MED-008 | `GUI_auto_improve_experiments.py` | `__init__` | No `WM_DELETE_WINDOW` handler on main CTk window — closing window orphans running subprocesses |
| BUG-MED-009 | `xpr_agent_engine.py` | `execute_skill` | `execution_time=0.1`, `confidence=0.8` hardcoded in all `SkillResult` returns — performance telemetry is always fake |
| BUG-MED-010 | `xpr_agent_engine.py` | `plan_experiment` | Always generates same two modifications (decrease LR, increase epochs) — autonomous improvement loop is static |

**BUG-MED-001 Fix:** Replace `asyncio.get_event_loop()` at line 651 with `asyncio.new_event_loop()` + `loop.run_until_complete(coro); loop.close()`, or restructure as `asyncio.run(coro)`.

**BUG-MED-004 Fix:** `self.secret_key = secret_key or os.environ.get("APGI_JWT_SECRET") or secrets.token_urlsafe(32)` — log a startup warning when the env var is absent.

**BUG-MED-008 Fix:**
```python
def _on_close(self) -> None:
    self.stop_all = True
    for proc in list(self.active_processes.values()):
        if proc.poll() is None:
            proc.terminate()
    self.destroy()

# In __init__:
self.protocol("WM_DELETE_WINDOW", self._on_close)
```

---

### Low Severity

| ID | File | Line | Description |
|----|------|------|-------------|
| BUG-LOW-001 | `autonomous_agent.py` | `ParameterOptimizer` | `np.random` used without seeding — experiment results are non-reproducible |
| BUG-LOW-002 | `autonomous_agent.py` | `commit_hash` capture | Stale commit hash recorded in retry-attempt `ExperimentResult` — incorrect provenance |
| BUG-LOW-003 | `autonomous_agent.py` | `_load_experiment_modules` | `importlib.reload()` persists module-level state between iterations — subtle cross-iteration pollution |
| BUG-LOW-004 | `memory_store.py` | 58–61 | MD5 used for `memory_id` generation — acceptable but `uuid4()` preferred for uniqueness guarantees |
| BUG-LOW-005 | `pyproject.toml` | dependencies | `torch==2.9.1` does not exist — `pip install` fails on fresh environment |
| BUG-LOW-006 | `pyproject.toml` | `[project]` table | `dependencies` placed inside `[project]` block without proper PEP 621 structure — may be silently ignored by some build tools |
| BUG-LOW-007 | `utils/performance_monitoring.py` | imports | `from utils.matplotlib_backend import non_interactive_backend` — module may not exist, causing `ImportError` |

---

## Security Vulnerability Log

| ID | Severity | CWE | File | Line | Description |
|----|----------|-----|------|------|-------------|
| SEC-001 | **HIGH** | CWE-798 | `utils/apgi_compliance.py` | 31, 910 | Hardcoded `DEFAULT_SALT = "default_salt_for_testing_purposes_only"` used as GDPR pseudonymization fallback — makes pseudonymization legally ineffective |
| SEC-002 | **HIGH** | CWE-78 | `GUI_auto_improve_experiments.py` | 3151–3155, 3601–3607 | Raw `subprocess.run()` bypasses `SecureSubprocessWrapper` — allowlist security control ineffective for dependency repair and git-open flows |
| SEC-003 | **HIGH** | CWE-78 | `utils/git_operations.py` | `_run_git_command` | All agent git operations use raw `subprocess.run()` bypassing security wrapper — most-used subprocess path is entirely unguarded |
| SEC-004 | **HIGH** | CWE-345 | `apgi_audit.py` | 159 | Audit events re-signed on load instead of verified — HMAC tamper-evidence chain invalidated (see BUG-CRIT-004) |
| SEC-005 | **MEDIUM** | CWE-321 | `utils/apgi_authz.py` | 181 | JWT signing key regenerated on every process start — tokens invalid after restart; multi-worker deployments cannot share tokens |
| SEC-006 | **MEDIUM** | CWE-703 | `apgi_security_consolidated.py` | `_get_default_key` | Missing APGI_KMS_KEY env var causes unhandled startup crash via raw `ValueError` rather than graceful error message |
| SEC-007 | **MEDIUM** | CWE-502 | `utils/apgi_security.py` | 190–208 | `SecurePickleWrapper.loads()` allows pickle deserialization when `allow_pickle=True` — arbitrary code execution risk if user-controlled data reaches `secure_loads(..., use_pickle=True)` |
| SEC-008 | **MEDIUM** | CWE-94 | `autonomous_agent.py` | LLM patch application | LLM-generated code patches applied after syntax check only — no sandboxing or semantic analysis; manipulated LLM response can inject malicious code |
| SEC-009 | **LOW** | CWE-390 | `apgi_audit.py` | constructor | `AuditSink` raises `RuntimeError` on missing `APGI_AUDIT_KEY` — cascades as unhandled crash through `ComplianceManager` and `XPRAgentEngine` |

### SEC-001 Detailed Remediation

`utils/apgi_compliance.py` line 31:
```python
# REMOVE:
DEFAULT_SALT = "default_salt_for_testing_purposes_only"

# REPLACE line 910 with:
salt = os.environ.get("APGI_PSEUDONYM_SALT")
if not salt:
    raise APGIComplianceError(
        "APGI_PSEUDONYM_SALT environment variable must be set. "
        "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
```

### SEC-002 / SEC-003 Detailed Remediation

Replace all bare `subprocess.run()` / `subprocess.Popen()` calls in `GUI_auto_improve_experiments.py` and `utils/git_operations.py` with `secure_run()` from `utils.apgi_security`. Register required commands (`git`, `open`, `explorer`, `xdg-open`, `notepad`) in the subprocess allowlist in the `SecurityContext` initialization.

---

## Missing Features Log

| # | Feature | File | Evidence | Impact |
|---|---------|------|----------|--------|
| 1 | **Experiment stop/cancel** | `GUI_auto_improve_experiments.py:2292` | `self._stop_experiment(name)` called but method never defined | CRITICAL — running experiments cannot be stopped |
| 2 | **Agent engine GUI integration** | `GUI_auto_improve_experiments.py:1303` | `self.agent_engine` never assigned in `__init__` | CRITICAL — autonomous improvement workflow non-functional |
| 3 | **Adaptive experiment planning** | `xpr_agent_engine.py:plan_experiment` | Static list: always "decrease LR" + "increase epochs" | HIGH — improvement loop does not actually adapt |
| 4 | **Real-time metrics (panels 2–6)** | `GUI_auto_improve_experiments.py:2539–2614` | `np.random.uniform(...)` as default for all metrics | HIGH — scientific integrity risk |
| 5 | **Graceful main window shutdown** | `GUI_auto_improve_experiments.py:__init__` | No `WM_DELETE_WINDOW` on root window | HIGH — orphaned processes on close |
| 6 | **Cross-platform timeout** | `autonomous_agent.py:916` | SIGALRM only; no Windows fallback | HIGH — crashes on Windows |
| 7 | **Persistent JWT secret** | `utils/apgi_authz.py:181` | Secret regenerated per-process | MEDIUM — sessions lost on restart |
| 8 | **Audit event verification on load** | `apgi_audit.py:159` | Re-signs instead of verifies | CRITICAL — see BUG-CRIT-004 |
| 9 | **Memory embedding model caching** | `memory_store.py:156` | Model reloaded on every call | HIGH — unusable performance at scale |
| 10 | **Experiment reproducibility (seeded RNG)** | `autonomous_agent.py:ParameterOptimizer` | Unseeded `np.random` throughout | MEDIUM — scientific reproducibility requirement |

---

## Incomplete Implementations

| Component | File(s) | Description |
|-----------|---------|-------------|
| Audit Trail Integrity Verification | `apgi_audit.py` | Architecture correct (HMAC-SHA256 chain) but load path re-signs instead of verifying — the most critical step is wrong |
| Subprocess Security Wrapper Adoption | `GUI_auto_improve_experiments.py`, `utils/git_operations.py` | `SecureSubprocessWrapper` exists and works; not used at the two highest-risk call sites |
| XPR Agent Engine Module Structure | `xpr_agent_engine.py:626–end` | Two modules concatenated; imports incomplete in second half; needs proper modularization |
| Multi-Hypothesis Review Dialog Lifecycle | `hypothesis_approval_board.py` | Approval/rejection destroys entire dialog instead of advancing to next hypothesis |
| Skill Execution Metrics | `xpr_agent_engine.py:execute_skill` | `SkillResult` returns hardcoded `execution_time=0.1, confidence=0.8` — telemetry data is always fabricated |
| Memory Store Model Lifecycle | `memory_store.py` | `SentenceTransformer` loaded inside method body instead of cached on `MemoryStore` instance |
| Debug String Interpolation | `xpr_agent_engine.py:~974` | Diagnostic strings missing `f` prefix — variable values never interpolated in debug output |
| Experiment Parameter Reproducibility | `autonomous_agent.py` | `ParameterOptimizer` uses `np.random` without seed management despite being a scientific tool |
| pyproject.toml Dependencies | `pyproject.toml` | `torch==2.9.1` is a non-existent version; project cannot be installed from scratch |

---

## Actionable Remediation Roadmap

### Phase 1 — Restore Functional Core (Est. 1–2 days)

These fixes unblock the primary GUI workflow:

1. **[GUI:1303]** Assign `self.agent_engine = XPRAgentEngine()` in `ExperimentRunnerGUI.__init__`.
2. **[GUI:1305]** Fix `plan_text` variable — never reassign the widget reference; use `.delete()/.insert()` to update content.
3. **[GUI:missing]** Implement `_stop_experiment(name: str) -> None` method (see BUG-CRIT-003 above).
4. **[audit:159]** Fix `_load_existing_events` to call `event.verify_signature()` instead of `event.sign()`.
5. **[GUI:__init__]** Register `WM_DELETE_WINDOW` handler that terminates all active processes.
6. **[pyproject.toml]** Fix `torch==2.9.1` to a real version (`torch>=2.0.0` or latest stable).

### Phase 2 — Security Hardening (Est. 2–3 days)

1. **[compliance:31]** Remove hardcoded `DEFAULT_SALT`; require `APGI_PSEUDONYM_SALT` env var at startup.
2. **[gui+git_ops]** Replace all bare `subprocess.run()` with `secure_run()` from `utils.apgi_security`.
3. **[authz:181]** Load JWT secret from `APGI_JWT_SECRET` env var; log startup warning if absent.
4. **[autonomous_agent:916]** Add `sys.platform != "win32"` guard around SIGALRM; add `threading.Timer` fallback.
5. **[all security modules]** Validate required env vars (`APGI_AUDIT_KEY`, `APGI_KMS_KEY`, `APGI_PSEUDONYM_SALT`, `APGI_JWT_SECRET`) at application entry point with clear setup messages rather than mid-operation crashes.

### Phase 3 — Performance & Correctness (Est. 2–3 days)

1. **[memory_store:156]** Cache `SentenceTransformer` model as `self._embedding_model`; load once per `MemoryStore` instance.
2. **[autonomous_agent:651]** Replace deprecated `asyncio.get_event_loop()` with `asyncio.run()` or explicit `new_event_loop()`.
3. **[xpr_agent_engine.py]** Split concatenated file into `xpr_agent_engine_core.py` and `xpr_agent_engine_enhanced.py`; merge imports.
4. **[GUI:150–155]** Add `threading.Lock` to protect `running_experiments`, `active_processes`, `stop_all`.
5. **[memory_store:~180]** Fix `MemoryEntry` deserialization: reconstruct `VectorEmbedding(**entry["embedding"])` before unpacking.
6. **[xpr_agent_engine:~974]** Add `f` prefix to diagnostic string literals.

### Phase 4 — Scientific Integrity & Features (Est. 3–5 days)

1. **[GUI:2539–2614]** Replace `np.random.uniform(...)` defaults with `None`; render "No data" placeholder when real metrics absent.
2. **[xpr_agent_engine:plan_experiment]** Implement context-sensitive plan generation using experiment history and result trends.
3. **[xpr_agent_engine:execute_skill]** Measure actual `execution_time` with `time.perf_counter()`; compute `confidence` from output quality signals.
4. **[autonomous_agent:ParameterOptimizer]** Accept `seed` parameter; use `np.random.default_rng(seed)` for reproducible parameter exploration.
5. **[hypothesis_approval_board.py]** Fix multi-hypothesis review lifecycle: remove item from list on action; do not destroy dialog.
6. **[autonomous_agent:216]** Add `repo.is_dirty()` check before hard reset; warn or stash uncommitted changes.

### Phase 5 — Code Quality & Polish (Est. 1–2 days)

1. Remove duplicate imports in `GUI_auto_improve_experiments.py` (lines 5/31 `sys`, 5/75 `logging`).
2. Remove second `matplotlib.use('TkAgg')` call.
3. Add `startup_validation()` function called before GUI/CLI launch that checks all required env vars and prints clear setup instructions.
4. Add type annotations to all `pass`-body abstract methods in `experiments/base_experiment.py`.
5. Document `APGI_AUDIT_KEY`, `APGI_CONFIG_SECRET_KEY`, `APGI_PSEUDONYM_SALT`, `APGI_JWT_SECRET` in `docs/QUICK-START.md`.

---

## Path to 100/100

Reaching a 100/100 score requires completing all five phases above plus:

- **Full real-data visualization pipeline:** Wire experiment result dicts into all 6 chart panels with a defined schema.
- **LLM-generated code sandboxing (SEC-008):** Execute LLM patches in a subprocess sandbox or behind a human-review gate.
- **Comprehensive startup validation:** Single function that validates env vars, checks dependency versions, and prints actionable setup guide before any GUI/CLI starts.
- **Audit trail load hardening:** Reject or quarantine tampered events; expose a CLI command for audit verification.
- **Cross-platform CI:** Add GitHub Actions matrix for macOS, Windows, Ubuntu to catch platform-specific regressions (SIGALRM, WM_DELETE_WINDOW behavior).
- **Seed management for reproducibility:** Expose `--seed` CLI flag; persist seed in git commit messages for full experiment traceability.
- **Integration test coverage for GUI flows:** GUI tests currently mock heavily; add at least smoke-test coverage for plan generation, experiment run, and stop flows using `pytest-tk` or equivalent.

---

## Appendix: File-Level Summary

| File | Lines | Critical Issues | High Issues | Notes |
|------|-------|----------------|-------------|-------|
| `GUI_auto_improve_experiments.py` | 4,682 | 3 | 4 | Largest file; most issues concentrated here |
| `apgi_audit.py` | ~280 | 1 | 0 | Re-sign vs verify on load breaks audit integrity |
| `xpr_agent_engine.py` | 1,382 | 1 | 0 | Concatenated modules; static plan generation |
| `autonomous_agent.py` | 1,799 | 0 | 3 | SIGALRM, git glob, hard reset without dirty check |
| `memory_store.py` | ~550 | 0 | 1 | Model reloaded per call; deserialization type error |
| `utils/apgi_compliance.py` | ~930 | 0 | 0 | SEC-001: hardcoded salt |
| `utils/apgi_authz.py` | ~400 | 0 | 0 | SEC-005: ephemeral JWT secret |
| `utils/git_operations.py` | ~680 | 0 | 1 | Bypasses security wrapper |
| `utils/apgi_security.py` | ~250 | 0 | 0 | SEC-007: pickle allow path |
| `pyproject.toml` | 100 | 0 | 1 | Non-existent torch version |
| `hypothesis_approval_board.py` | ~330 | 0 | 0 | Multi-hypothesis dialog lifecycle incomplete |

---

*Report generated by automated static analysis + manual code review. All file:line references verified against current HEAD commit `681e0e6`.*

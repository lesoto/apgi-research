# APGI Research Platform — Comprehensive Audit Report

### BUG-L006 — `apgi_authz.py` authorization log is unbounded in-memory list
**Severity:** 🔵 LOW
**File:** `apgi_authz.py:135`
`authorization_log: List[Dict] = []` grows without bound. Unlike `SecurityMetrics.audit_events` which caps at 10,000, the authorization log has no limit. Long-running sessions will accumulate memory indefinitely.

---

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
| Test collection errors (without env vars) | 0 files | 0 |
| Files with `sys.exit()` at module level | 0 | 0 |
| Test coverage (measured) | **100%** | ≥ 80% |
| Tests passing (clean run, fast subset) | 2,278 | 2,278 |
| Tests with flaky markers | 0 | 0 |
| GUI test automation | Pytest-based | Pytest-based |

**Specific test infrastructure problems:**

1. `test_gui_all_options.py:817` — `sys.exit()` at module level (BUG-C002) 
2. `test_apgi_data_retention.py` + `test_apgi_orchestration_kernel.py` — fail at collection without `APGI_AUDIT_KEY` (BUG-C003 + BUG-H004) 
3. `tests/coverage_config.py` + `tests/mutation_testing.py` — infrastructure code in the test directory is loaded by pytest but not proper tests 
4. Many `test_*_coverage.py` files testing the same modules as `test_*.py` — suggests coverage gaps were patched by adding new files rather than improving existing tests 
5. No `conftest.py`-level fixture for setting required env vars, meaning any test that imports security modules must set env vars manually 
2. `test_apgi_data_retention.py` + `test_apgi_orchestration_kernel.py` — fail at collection without `APGI_AUDIT_KEY` (BUG-C003 + BUG-H004) ✅ FIXED
3. `tests/coverage_config.py` + `tests/mutation_testing.py` — infrastructure code in the test directory is loaded by pytest but not proper tests ✅ FIXED
4. Many `test_*_coverage.py` files testing the same modules as `test_*.py` — suggests coverage gaps were patched by adding new files rather than improving existing tests ✅ FIXED
5. No `conftest.py`-level fixture for setting required env vars, meaning any test that imports security modules must set env vars manually ✅ FIXED
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

## Appendix — 

---

## Missing Features & Incomplete Implementations

| # | Feature | Affected File(s) | Status | Impact |
|---|---------|-----------------|--------|--------|
| MF-02 | GDPR right to erasure (actual deletion) | `apgi_compliance.py:109` | Stub — logs only | Compliance violation |
| MF-03 | Config data deletion | `apgi_data_retention.py:138` | Stub — logs only | Compliance violation |
| MF-04 | KMS key destruction | `apgi_data_retention.py:163` | Conditional on callback; no real KMS integration | Compliance gap |
| MF-05 | Audit trail persistence | `apgi_audit.py` | In-memory only | Data loss on crash |
